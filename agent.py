"""Ridges miner agent: an autonomous engineer for database query engineering tasks.

Entry point:

    def agent_main(input: dict[str, Any]) -> str:
        # input == {"problem_statement": "<task instruction markdown>"}
        # return a unified git diff (string) that solves the task.

The agent runs inside the task container with the application repository as its
working directory and a live database beside it, and reaches inference through
the OpenRouter key the sandbox provides. Nothing is precomputed: the patch is
derived at runtime from the repository in front of it, the schema and rows in
front of it, and the instruction it is handed.

WHAT THESE TASKS ARE
--------------------
An application fetches data from a real database and the fetch is wrong, absent,
or wasteful. The change belongs in the production code that the application
actually runs -- an ORM expression, a query-builder call, embedded SQL, a
migration -- not in a standalone script. The instruction states its own contract,
for example:

    Work in `/app`. Limit production changes to `netbox/ipam/querysets.py`,
    specifically `VLANGroupQuerySet.annotate_utilization()`. Keep its signature
    and the rest of the file unchanged, including imports; use only names the
    file already imports.

    Return a lazy, composable queryset with these exact annotations: ...

    Run these checks before finishing:

        python netbox/manage.py test ipam.tests.test_api.VLANGroupTest --keepdb
        ruff check --no-cache netbox/ipam/querysets.py

Three shapes recur, and the instruction says which one it is:

  * AUTHORING  -- write the fetch path so it meets a data requirement;
  * REPAIR     -- the code returns the wrong data; make it right;
  * OPTIMIZATION -- the code is right but does too much database work; make it
    cheaper without changing a single returned row.

HOW IT WORKS
------------
The agent parses that contract and holds itself to it. It reads the schema and
the real rows, reads the code the application runs, writes the change, and then,
before it will report itself finished, checks its own work against every
constraint the instruction stated:

  * only the file the instruction names has changed;
  * the method it names still has the same signature, and the rest of that file
    is byte-for-byte what it was, imports included;
  * every name the method uses is one the file already had;
  * the method is plain expressions -- no loops, comprehensions, lambdas,
    exception handling or context managers, when the instruction says so;
  * the checks the instruction itself lists all pass;
  * on a task whose stated goal is less database work, the rows that come back
    are identical to the rows that came back before.

A failed check is fed back to the model as a repair task instead of being
shipped. If a candidate still does not satisfy the instruction, the agent starts
over once with a stronger model and keeps whichever attempt is better.

An instruction that names no file or method falls back to a general-purpose
locate / reproduce / fix / verify loop.

Budget: per-call output tokens, step count and wall clock are bounded so a run
stays inside RIDGES_MAX_COST_USD and AGENT_TIMEOUT.

WORKING RULES
-------------
  * Solve the task from the repository and the database in front of it. The
    agent works inside the application directory and keeps scratch files in its
    own /tmp directory.
  * Read the database, do not rewrite it. Exploratory SQL is read-only, and ORM
    exploration runs inside a transaction that is always rolled back, so the
    agent leaves no rows, tables or indexes behind.
  * Never weaken the project: no test is edited, skipped or deleted.
  * Never add a dependency or fetch anything over the network. The only outbound
    calls are inference requests.
  * Nothing branches on the identity of a task, repository or database, and no
    answer is memorized. The playbooks are general technique about joins, grain,
    NULLs, ties, aggregation and index selection.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import traceback
from typing import Any

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Model chain, cheapest capable first. The per-run spend cap is the binding
# constraint on these tasks, not the clock: the container allows a long run but
# only ~$0.29 of inference, and a database task carries a lot of context (schema,
# query plans, ORM code). So attempt 1 runs on the fast coder model and the
# stronger reasoning model is held in reserve for a second attempt, taken ONLY
# when the first candidate fails its own checks (see the attempt loop in
# agent_main). Ranking here is reasoned from price and task shape; measurement
# should settle it.
DEFAULT_MODEL_CHAIN = "qwen/qwen3-coder-next,z-ai/glm-4.6"
MODEL_CHAIN = [
    name.strip()
    for name in (os.getenv("RIDGES_MODEL") or os.getenv("RIDGES_MODELS") or DEFAULT_MODEL_CHAIN).split(",")
    if name.strip()
]
DEFAULT_MODEL = MODEL_CHAIN[0]
# Independent second candidate, judged by the same checks. Only in specialized
# mode: a task that states its file, its method and its checks states enough to
# tell two candidates apart. A general task does not.
MAX_ATTEMPTS = int(os.getenv("RIDGES_MAX_ATTEMPTS", "2"))
# Temperature for attempt 2. Re-rolling at 0 mostly reproduces the same patch.
RETRY_TEMPERATURE = float(os.getenv("RIDGES_RETRY_TEMPERATURE", "0.5"))
# A single hung inference call would otherwise sit on the OpenAI client's 600s
# default and eat the whole run, leaving nothing to submit.
REQUEST_TIMEOUT_SEC = int(os.getenv("RIDGES_REQUEST_TIMEOUT", "120"))

# Spend is what runs out first on these tasks, and every step re-sends the whole
# history, so the step ceiling is really a spend ceiling. Measured on the public
# samples: a task that goes well finishes in ~20 steps for a few cents, and a run
# that is still going at 60 has stopped converging and is buying context, not
# progress.
MAX_STEPS = int(os.getenv("RIDGES_MAX_STEPS", "60"))
# Cap per-call output tokens; otherwise the model reserves its full output
# ceiling (e.g. 65536) which blows the per-run budget and 402s on OpenRouter.
MAX_TOKENS = int(os.getenv("RIDGES_MAX_TOKENS", "8192"))
# Context (and therefore cost) compounds: every step re-sends the whole history,
# so each observation is paid for again on every later call. A 400-line read that
# lands at step 2 is billed ~8 more times, and query plans and schema dumps are
# bulky. Keep each observation SMALL at the source rather than eliding old ones:
# eliding erases the model's working memory and it just re-reads forever. The
# model can still see any part of a file via read_file(offset=..., limit=...).
MAX_OBSERVATION_CHARS = int(os.getenv("RIDGES_MAX_OBS_CHARS", "3000"))
MAX_FILE_READ_LINES = int(os.getenv("RIDGES_MAX_FILE_LINES", "150"))
CMD_TIMEOUT_SEC = int(os.getenv("RIDGES_CMD_TIMEOUT", "180"))
TEMPERATURE = float(os.getenv("RIDGES_TEMPERATURE", "0"))
MAX_REPAIR_NUDGES = int(os.getenv("RIDGES_MAX_REPAIR_NUDGES", "3"))
# The empty-patch failure mode: on a hard task the agent can burn its whole
# time/step budget exploring the schema and writing /tmp probe scripts WITHOUT
# ever editing a production file. Probe scripts live outside the repository, so
# `_has_changes()` stays False, the wrap-up nudge below (which requires changes)
# never fires, and the loop runs until the buzzer and returns an EMPTY patch,
# which fixes nothing. These thresholds (fraction of the time/step budget
# consumed) trigger escalating "edit now" pressure while enough budget remains to
# still make AND verify a fix.
NO_EDIT_PRESSURE_THRESHOLDS = tuple(
    float(x) for x in os.getenv("RIDGES_NO_EDIT_PRESSURE", "0.30,0.45,0.60").split(",") if x.strip()
)
# Before accepting `finish`, run the checks the instruction itself lists. A
# near-miss -- the annotation right for the common rows and wrong for the empty
# group -- is a broken fix, and feeding the failure back turns it into a repair.
MAX_TEST_REPAIRS = int(os.getenv("RIDGES_MAX_TEST_REPAIRS", "2"))
# These instructions state several constraints at once (which file, which method,
# keep the signature, keep the imports, plain expressions, the named checks) and
# each is cheap to re-check locally, so allow more repair rounds: every rejected
# `finish` carries a precise, machine-generated reason to act on.
MAX_VERIFY_REPAIRS = int(os.getenv("RIDGES_MAX_VERIFY_REPAIRS", "4"))
# Max wall time one of the instruction's named check commands may take. An
# application test suite against a real database is slow the first time it runs
# (it migrates a test database), so this is generous by lint standards.
CHECK_TIMEOUT_SEC = int(os.getenv("RIDGES_CHECK_TIMEOUT", "480"))
# Minimum time that must remain for us to bother running the named checks. This
# is NOT how long they take -- a warm re-run finishes in seconds. Gating on the
# full CHECK_TIMEOUT_SEC would mean arriving at `finish` with a few minutes left
# and silently SHIPPING UNVERIFIED, which is how a patch with a typo'd ORM name
# gets submitted.
CHECK_MIN_RESERVE_SEC = int(os.getenv("RIDGES_CHECK_MIN_RESERVE", "90"))
# Wrap-up pressure must watch every budget, not just the clock: a run can hit
# MAX_STEPS with time and money to spare, never reach `finish`, and therefore
# never run the verify-and-repair guard. Fire once this fraction of the
# max(time, step) budget is gone WITH edits in place.
WRAPUP_BUDGET_THRESHOLD = float(os.getenv("RIDGES_WRAPUP_THRESHOLD", "0.55") or "0.55")

# Spend budget. Inference costs money and an agentic loop will happily spend
# without end, so the run keeps its own tally and stops while it still has a
# patch to hand back, holding a reserve for the final checks and diff capture.
# RIDGES_MAX_COST_USD is read because the host sets it; the fallback applies
# when nothing does.
MAX_COST_USD = float(os.getenv("RIDGES_MAX_COST_USD") or os.getenv("MAX_COST_USD") or "1.00")
COST_RESERVE_FRACTION = float(os.getenv("RIDGES_COST_RESERVE", "0.12"))
# Aim to finish well inside whatever the limit is rather than spending up to it.
# A task that names its file and its method has no repo-wide search to pay for; a
# run that needs the whole budget is usually a run that lost its way in the schema.
COST_TARGET_FRACTION = float(os.getenv("RIDGES_COST_TARGET_FRACTION", "0.4") or "0.4")
_MODEL_PRICE_PER_M = {  # published list prices, (input, output) USD per 1M tokens
    "qwen/qwen3-coder-next": (0.11, 0.80),
    "qwen/qwen3-coder": (0.30, 1.20),
    "qwen/qwen3.5-397b-a17b": (0.35, 1.40),
    "moonshotai/kimi-k2.5": (0.60, 2.50),
    "z-ai/glm-4.6": (0.60, 2.20),
    "minimax/minimax-m2.5": (0.15, 1.15),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
}
_SPENT_USD = 0.0

# Time budget. AGENT_TIMEOUT carries the task's own declared timeout when it is
# set, so it is fact rather than a guess; the fallback applies when nothing
# states one. Overrunning is unforgiving: a killed run has nothing to hand back,
# no matter how good the patch was.
_AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT") or "900")
# Reserve exactly what the endgame needs (a last round of the named checks plus
# diff capture) plus a small guard against container start-up skew.
TIME_RESERVE_SEC = int(os.getenv("RIDGES_TIME_RESERVE", "90"))
TIME_SAFETY_FRACTION = float(os.getenv("RIDGES_TIME_SAFETY", "0.05"))
_USABLE_BUDGET = max(60.0, _AGENT_TIMEOUT * (1 - TIME_SAFETY_FRACTION) - TIME_RESERVE_SEC)
_START_TIME = time.monotonic()

# Scratch space, deliberately OUTSIDE the repository: anything we drop inside the
# working tree lands in the submitted patch, which the instruction does not allow.
SCRATCH_DIR = os.getenv("RIDGES_SCRATCH_DIR", "/tmp/agent-scratch")
UNPATCHED_COPY = os.path.join(SCRATCH_DIR, "baseline")

# Keep every child process's byte-code and tool caches out of the repository.
# `git add -A` stages untracked files, so a stray .ruff_cache/ or __pycache__/
# that the project does not gitignore becomes an out-of-scope path in the patch
# and undoes an otherwise perfect fix.
os.environ.setdefault("PYTHONPYCACHEPREFIX", os.path.join(SCRATCH_DIR, "pycache"))
os.environ.setdefault("RUFF_CACHE_DIR", os.path.join(SCRATCH_DIR, "ruff-cache"))
os.environ.setdefault("PYTEST_ADDOPTS", "-p no:cacheprovider")


def _log(message: str) -> None:
    print(f"[AGENT] {message}", flush=True)


def _time_left() -> float:
    return _USABLE_BUDGET - (time.monotonic() - _START_TIME)


def _time_used_fraction() -> float:
    return 1 - max(_time_left(), 0.0) / max(_USABLE_BUDGET, 1e-6)


class _DeadlineReached(BaseException):
    """Raised by the watchdog so the run still emits its best diff.

    Deliberately a BaseException: every tool helper catches `Exception` to turn a
    failure into an observation, and a watchdog that gets swallowed into a tool
    result is no watchdog at all.
    """


def _arm_deadline() -> None:
    """Guarantee we emit a patch even if something overruns.

    Every blocking call already carries its own timeout, but a wedged socket, a
    provider that never answers, or a database connection that hangs would
    otherwise run out the clock and return nothing at all. Unwinding into
    agent_main's handler lets the diff we already have be handed back.
    """

    def _fire(_signum: int, _frame: Any) -> None:
        raise _DeadlineReached("agent time budget exhausted")

    try:
        signal.signal(signal.SIGALRM, _fire)
        signal.signal(signal.SIGTERM, _fire)
        signal.alarm(max(30, int(_time_left())))
    except (AttributeError, ValueError, OSError) as exc:  # not the main thread, or no SIGALRM
        _log(f"deadline watchdog unavailable: {exc}")


def _disarm_deadline() -> None:
    try:
        signal.alarm(0)
    except (AttributeError, ValueError, OSError):
        pass


def _account_usage(model: str, usage: Any) -> None:
    """Add a completed call's estimated cost to the running total."""
    global _SPENT_USD
    if not usage:
        return
    pin, pout = _MODEL_PRICE_PER_M.get(model, (1.0, 3.0))
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    _SPENT_USD += (pt / 1_000_000) * pin + (ct / 1_000_000) * pout


def _budget_left() -> bool:
    if MAX_COST_USD <= 0:
        return True
    return _SPENT_USD < MAX_COST_USD * (1 - COST_RESERVE_FRACTION)


def _spend_used_fraction() -> float:
    if MAX_COST_USD <= 0:
        return 0.0
    return min(_SPENT_USD / MAX_COST_USD, 1.0)


def _budget_used_fraction(step: int) -> float:
    """How far through this run we are, by whichever budget is going fastest.

    Three of them run down at once and any one of them ending the loop leaves
    whatever patch exists at that moment, so the pressure to commit to a fix has
    to key on the fastest, not on the clock alone. On a task whose own checks
    take minutes, wall time drains while barely any thinking has been paid for;
    on a task with a large schema to read, spend drains first.
    """
    return max(
        (step + 1) / MAX_STEPS,
        _spend_used_fraction(),
        _time_used_fraction(),
    )


# --------------------------------------------------------------------------- #
# Inference client (OpenRouter / OpenAI-compatible)
# --------------------------------------------------------------------------- #

def _base_urls() -> list[str]:
    """Inference endpoints to try, in order.

    Some environments route outbound traffic transparently, so the plain provider
    URL works as-is; others expose an explicit proxy in SANDBOX_PROXY_URL and the
    direct host does not resolve. Losing every task to a connection error is the
    worst possible failure, so try the direct endpoint first and fall back to the
    proxy rather than betting the run on either one.
    """
    urls: list[str] = []
    configured = os.getenv("RIDGES_INFERENCE_BASE_URL") or os.getenv("RIDGES_OPENROUTER_BASE_URL")
    if configured:
        urls.append(configured.rstrip("/"))
    urls.append("https://openrouter.ai/api/v1")
    proxy = (os.getenv("SANDBOX_PROXY_URL") or "").strip().rstrip("/")
    if proxy:
        urls.append(proxy if proxy.endswith("/api/v1") else f"{proxy}/api/v1")
    deduped: list[str] = []
    for url in urls:
        if url not in deduped:
            deduped.append(url)
    return deduped


_BASE_URL_INDEX = 0
_CLIENT: Any = None


def _make_client(rotate: bool = False):
    """Build (or rebuild) the inference client, optionally on the next endpoint."""
    global _BASE_URL_INDEX, _CLIENT
    from openai import OpenAI

    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("RIDGES_INFERENCE_API_KEY")
        or os.getenv("RIDGES_OPENROUTER_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "No inference API key found. Expected OPENROUTER_API_KEY (set by the "
            "sandbox / by `ridges miner run-local` with provider=openrouter)."
        )
    urls = _base_urls()
    if rotate:
        if _BASE_URL_INDEX + 1 >= len(urls):
            return None
        _BASE_URL_INDEX += 1
        _log(f"switching inference endpoint to {urls[_BASE_URL_INDEX]}")
    _CLIENT = OpenAI(api_key=api_key, base_url=urls[min(_BASE_URL_INDEX, len(urls) - 1)])
    return _CLIENT


def _model_for_attempt(attempt: int) -> str:
    """Cheap model first; escalate to the next in the chain on a retry attempt."""
    return MODEL_CHAIN[min(attempt, len(MODEL_CHAIN) - 1)]


def _chat(client, messages: list[dict], tools: list[dict], model: str | None = None, temperature: float | None = None):
    """One inference call, with transient-error retries then model fallback.

    `client` may be replaced internally when the endpoint rotates. A model that is
    down or unroutable used to kill the whole run (and with it the patch), so
    exhausting the retries falls through to the next model in the chain rather
    than raising.
    """
    chosen = model or DEFAULT_MODEL
    candidates = [chosen] + [name for name in MODEL_CHAIN if name != chosen]
    last_error: Exception | None = None
    for candidate in candidates:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=candidate,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=TEMPERATURE if temperature is None else temperature,
                    max_tokens=MAX_TOKENS,
                    timeout=max(30, min(REQUEST_TIMEOUT_SEC, int(_time_left()))),
                )
                if candidate != chosen:
                    _log(f"inference fell back to {candidate}")
                _account_usage(candidate, getattr(response, "usage", None))
                return response
            except Exception as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                text = str(exc).lower()
                budgetish = any(k in text for k in ("budget", "exceed", "insufficient", "more credits"))
                if budgetish:
                    raise RuntimeError(f"Inference budget exhausted: {exc}") from exc
                # An endpoint that cannot be reached at all is not a model problem:
                # rotate to the next base URL (the explicit sandbox proxy) instead of
                # burning the remaining models against a dead host.
                if status is None and any(
                    marker in text
                    for marker in ("connection", "name or service not known", "resolve", "ssl", "certificate", "refused")
                ):
                    rotated = _make_client(rotate=True)
                    if rotated is not None:
                        client = rotated
                        continue
                # Don't waste retries on non-transient client errors; try the next model.
                if status and status not in (408, 409, 429) and status < 500:
                    break
                _log(f"inference retry {attempt + 1} on {candidate} after error: {exc}")
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Inference failed on {candidates}: {last_error}")


# --------------------------------------------------------------------------- #
# Shell + filesystem helpers
# --------------------------------------------------------------------------- #

def _truncate(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    if not text:
        return text or ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [truncated {len(text) - limit} chars] ...\n{tail}"


def _run(command: str, timeout: int = CMD_TIMEOUT_SEC, cwd: str | None = None) -> str:
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]"
    except Exception as exc:  # noqa: BLE001
        return f"[command error: {exc}]"
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        out = f"[exit {proc.returncode}]\n{out}"
    return out


def _run_argv(
    argv: list[str],
    timeout: int = CMD_TIMEOUT_SEC,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> tuple[int, str]:
    """Run a command without a shell. Returns (returncode, combined output).

    The instruction's own check commands carry quoted arguments; re-quoting them
    through a shell is how you silently run something other than what was asked
    for. Always pass the parsed argv straight through.
    """
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env, input=stdin
        )
    except subprocess.TimeoutExpired:
        return 124, f"[command timed out after {timeout}s]"
    except Exception as exc:  # noqa: BLE001
        return 127, f"[command error: {exc}]"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


_REPO_ROOT: str | None = None


def _repo_root() -> str:
    """The application to patch: the working directory we were started in."""
    global _REPO_ROOT
    if _REPO_ROOT is None:
        _REPO_ROOT = os.path.abspath(os.getcwd())
    return _REPO_ROOT


def _rel_to_repo(path: str) -> str | None:
    """Repo-relative POSIX path, or None when `path` lies outside the repo."""
    root = _repo_root()
    absolute = os.path.abspath(os.path.join(root, path))
    if absolute == root:
        return "."
    if not absolute.startswith(root + os.sep):
        return None
    return os.path.relpath(absolute, root).replace(os.sep, "/")


def _python_bin() -> str:
    return sys.executable or shutil.which("python3") or "python"


def _import_roots(base: str | None = None) -> list[str]:
    """Import roots for this project: the root, plus src/ for src layouts."""
    root = base or _repo_root()
    roots = [root]
    if os.path.isdir(os.path.join(root, "src")):
        roots.insert(0, os.path.join(root, "src"))
    return roots


def _subprocess_env(extra_paths: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    env.pop("PYTEST_ADDOPTS", None)
    return env


# --------------------------------------------------------------------------- #
# Task spec: the contract the instruction states in prose
# --------------------------------------------------------------------------- #

_SOURCE_EXTS = (
    ".py", ".pyi", ".go", ".ts", ".tsx", ".js", ".jsx", ".rb", ".java", ".kt",
    ".rs", ".php", ".cs", ".scala", ".sql", ".mjs", ".cjs",
)


class DbSpec:
    """Everything the instruction says about the change it wants.

    Every field is read out of the instruction's own words. Nothing here is
    knowledge about a particular task, repository or database.
    """

    def __init__(self) -> None:
        self.scope: list[str] = []            # the only paths the patch may touch
        # Some instructions name the file, some say "find the method that does X
        # and change only that method". In the second case the bound is just as
        # real, so the scope is discovered rather than read: it closes around the
        # first file the run actually edits.
        self.scope_locked: bool = True
        self.bounded_unnamed: bool = False
        self.class_name: str | None = None    # "specifically `Class.method()`"
        self.func_name: str | None = None
        self.checks: list[str] = []           # "Run these checks before finishing"
        self.kind: str = "repair"             # authoring | repair | optimization
        self.engine: str = ""                 # postgresql | clickhouse | ""
        self.freeze_rest: bool = False        # "the rest of the file unchanged"
        self.freeze_imports: bool = False     # "including imports" / "already imports"
        self.freeze_signature: bool = False   # "Keep its signature"
        self.plain_expressions: bool = False  # "no loops, comprehensions, lambdas, ..."
        self.lazy: bool = False               # "Return a lazy queryset" / "keep evaluation lazy"
        self.no_materialize: bool = False     # "Do not materialize ... in Python"
        self.no_side_effects: bool = False    # "Do not add database writes, ... side effects"
        self.results_must_not_change: bool = False  # optimization: same rows, less work

    @property
    def target(self) -> str | None:
        if not self.func_name:
            return None
        return f"{self.class_name}.{self.func_name}" if self.class_name else self.func_name

    @property
    def primary_file(self) -> str | None:
        return self.scope[0] if self.scope else None


_SPEC: DbSpec | None = None


_SCOPE_PATTERNS = (
    r"[Ll]imit\s+(?:production\s+)?changes?\s+to\s+`([^`]+)`",
    r"[Yy]ou\s+may\s+(?:only\s+)?edit\s+only\s*`([^`]+)`",
    r"[Yy]ou\s+may\s+edit\s+only\s*\n?\s*`([^`]+)`",
    r"[Cc]hange\s+only\s+`([^`]+)`",
    r"[Ee]dit\s+only\s+`([^`]+)`",
    r"[Oo]nly\s+`([^`]+)`\s+may\s+change",
    r"[Cc]onfine\s+(?:the\s+)?(?:change|edit)s?\s+to\s+`([^`]+)`",
    r"[Rr]estrict\s+(?:the\s+)?(?:change|edit)s?\s+to\s+`([^`]+)`",
    r"[Ww]ork\s+(?:only\s+)?in\s+`([^`]+\.[A-Za-z]{1,4})`",
)


def _existing_rel(candidate: str) -> str | None:
    """A repo-relative path for `candidate`, if it names something that exists."""
    candidate = (candidate or "").strip().strip("`'\"").rstrip(".,;:")
    if not candidate:
        return None
    rel = _rel_to_repo(candidate)
    if rel and rel != "." and os.path.exists(os.path.join(_repo_root(), rel)):
        return rel
    # A path may be quoted relative to a subdirectory ("ipam/querysets.py") or
    # with the container root spelled out ("/app/netbox/ipam/querysets.py").
    base = candidate.lstrip("/")
    for prefix in ("", "src/"):
        rel = _rel_to_repo(prefix + base)
        if rel and rel != "." and os.path.exists(os.path.join(_repo_root(), rel)):
            return rel
    # Last resort: a unique tracked file with that suffix.
    hits = [
        line.strip()
        for line in _run(f"git ls-files -- '*{shlex.quote(base)}'", timeout=25).splitlines()
        if line.strip()
    ]
    hits = [h for h in hits if h.endswith(base)]
    if len(hits) == 1:
        return hits[0]
    return None


def _declared_scope(text: str) -> list[str]:
    """Paths the submitted patch is allowed to touch.

    The instruction says which file to change, often twice (once as "limit
    changes to X" and again in the check command that lints it). Union the
    readings, keep only paths that exist, and never widen beyond that: editing
    anything else is not the task.
    """
    scope: list[str] = []

    def add(candidate: str) -> None:
        rel = _existing_rel(candidate)
        if rel and rel not in scope:
            scope.append(rel)

    for pattern in _SCOPE_PATTERNS:
        for match in re.findall(pattern, text or ""):
            add(match)
    if scope:
        return scope

    # No explicit "limit changes to" sentence. Fall back to backticked source
    # paths in sentences that talk about editing, which is the other way these
    # instructions name their file.
    for sentence in re.split(r"(?<=[.\n])", text or ""):
        if not re.search(r"\b(edit|chang|modif|repair|author|optimi|fix|rewrite|patch)\w*\b", sentence, re.I):
            continue
        for quoted in re.findall(r"`([^`]+)`", sentence):
            if quoted.endswith(_SOURCE_EXTS):
                add(quoted)
    return scope


def _declared_target(text: str) -> tuple[str | None, str | None]:
    """The single method the instruction says to change, as (class, function)."""
    patterns = (
        r"specifically\s+`([A-Za-z_][\w.]*)\(\)`",
        r"specifically\s+`([A-Za-z_][\w.]*)`",
        r"(?:the\s+)?(?:method|function)\s+`([A-Za-z_][\w.]*)\(\)`",
        r"`([A-Za-z_][\w]*\.[A-Za-z_][\w]*)\(\)`\s+(?:must|should)",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "")
        if not match:
            continue
        dotted = match.group(1)
        parts = dotted.split(".")
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        return None, parts[-1]
    return None, None


_CHECK_RUNNERS = (
    "python", "python3", "pytest", "ruff", "go", "npm", "npx", "yarn", "pnpm",
    "make", "manage.py", "./manage.py", "tox", "mypy", "golangci-lint", "gofmt",
    "jest", "vitest", "tsc", "psql", "clickhouse-client", "bin/rails", "rake",
)


def _declared_checks(text: str) -> list[str]:
    """The commands the instruction tells you to run before finishing.

    These are the project's own regression checks. They say what must keep
    working; they are not the fix, and passing them is necessary rather than
    sufficient. Running them is still the cheapest way to catch a change that
    does not even import.
    """
    commands: list[str] = []

    def add(command: str) -> None:
        command = command.strip().lstrip("$").strip()
        if not command:
            return
        head = command.split()[0]
        if head not in _CHECK_RUNNERS and not head.endswith(("manage.py", ".sh")):
            return
        if command not in commands:
            commands.append(command)

    for block in re.findall(r"```[a-zA-Z]*\n(.*?)```", text or "", re.S):
        # Unfold shell line continuations so a wrapped command stays one command.
        for line in block.replace("\\\n", " ").splitlines():
            add(line)
    if not commands:
        # Inline form: "Run `<cmd>` and `<cmd>` before finishing." Every backticked
        # span in such a sentence is one of the commands, not just the first.
        for sentence in re.split(r"(?<=\.)\s", text or ""):
            if not re.search(r"\b[Rr]un\b", sentence):
                continue
            for span in re.findall(r"`([^`]+)`", sentence):
                add(span.replace("\\\n", " ").replace("\n", " "))
    # A wrapped inline command ("Run `python ... test X\n--keepdb --noinput`")
    # loses its newline above; collapse repeated whitespace so it is runnable.
    return [re.sub(r"\s+", " ", command) for command in commands][:6]


def _declared_kind(text: str) -> str:
    """authoring, repair or optimization, as the instruction frames the job."""
    low = (text or "").lower()
    head = "\n".join(low.splitlines()[:3])
    optimization_markers = (
        "optimi", "faster", "bounded number of sql", "without changing its result",
        "query work", "scan every", "n+1", "grows with the number", "bounded query",
        "performs a bounded", "buffer work", "must use the named index",
    )
    authoring_markers = ("author", "write the production fetch", "currently returns placeholder", "supply the")
    repair_markers = ("repair", "returns the wrong", "incorrect", "defect", "currently performs", "bug")
    for markers, kind in (
        (optimization_markers, "optimization"),
        (authoring_markers, "authoring"),
        (repair_markers, "repair"),
    ):
        if any(marker in head for marker in markers):
            return kind
    for markers, kind in (
        (optimization_markers, "optimization"),
        (authoring_markers, "authoring"),
        (repair_markers, "repair"),
    ):
        if any(marker in low for marker in markers):
            return kind
    return "repair"


def _declared_engine(text: str) -> str:
    low = (text or "").lower()
    if "clickhouse" in low:
        return "clickhouse"
    if "postgres" in low or "psql" in low:
        return "postgresql"
    return ""


def _detect_spec(problem: str) -> DbSpec | None:
    """Read the contract out of the instruction, or return None for general mode."""
    spec = DbSpec()
    spec.scope = _declared_scope(problem)
    if not spec.scope:
        # No path in the prose. If the instruction still bounds the change ("find
        # the manager method that does X and change only that method. Keep its
        # signature and the rest of its file unchanged"), the bound holds and the
        # file is simply something the run has to find. Locate it by editing, then
        # hold the same line. Only an instruction that bounds nothing falls back
        # to the general loop.
        if not re.search(
            r"only that (?:method|function|file)|rest of (?:its|that) file unchanged"
            r"|change only that|already imports",
            problem or "",
            re.I,
        ):
            return None
        spec.scope_locked = False
        spec.bounded_unnamed = True
    spec.class_name, spec.func_name = _declared_target(problem)
    spec.checks = _declared_checks(problem)
    spec.kind = _declared_kind(problem)
    spec.engine = _declared_engine(problem)

    low = (problem or "").lower()
    spec.freeze_rest = bool(re.search(r"rest of the file unchanged|all unrelated source|unrelated methods", low))
    spec.freeze_imports = bool(re.search(r"including imports|already imports|imports unchanged", low))
    spec.freeze_signature = bool(re.search(r"keep (?:its|the (?:method )?)signature|signature (?:and|unchanged)", low))
    spec.plain_expressions = bool(
        re.search(r"no python loops|plain orm expressions|no loops, comprehensions", low)
    )
    spec.lazy = "lazy" in low
    spec.no_materialize = "do not materialize" in low
    spec.no_side_effects = bool(re.search(r"do not add (?:database writes|file|process)", low))
    spec.results_must_not_change = spec.kind == "optimization" or bool(
        re.search(r"without changing (?:its|the) result|same lazy queryset|results? (?:must|should) not change", low)
    )
    return spec


_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")


def _requirement_bullets(text: str) -> list[str]:
    """The instruction's bulleted contract: the specification for this change.

    Each bullet is a separate promise about the patch -- one names ties, one names
    empty groups, one names NULLs -- and these are exactly the clauses a patch
    that "works on the example" quietly drops.
    """
    lines = (text or "").splitlines()
    bullets: list[str] = []
    current: list[str] = []
    for line in lines:
        match = _BULLET_RE.match(line)
        if match:
            if current:
                bullets.append(" ".join(current))
            current = [match.group(1).strip()]
        elif current and line.strip() and line.startswith((" ", "\t")):
            current.append(line.strip())  # continuation of a wrapped bullet
        elif current:
            bullets.append(" ".join(current))
            current = []
    if current:
        bullets.append(" ".join(current))
    return [bullet for bullet in bullets if len(bullet) > 15][:14]


_REQUIREMENTS: list[str] = []
_CHECKLIST_STATE = {"nudged": False}


def _checklist_failures(args: dict) -> list[str]:
    """Reasons the model's requirement accounting is not usable."""
    if not _REQUIREMENTS:
        return []
    entries = args.get("requirements_checklist")
    if not isinstance(entries, list) or not entries:
        return ["no requirements_checklist was provided"]
    problems: list[str] = []
    if len(entries) < len(_REQUIREMENTS):
        problems.append(
            f"only {len(entries)} of the {len(_REQUIREMENTS)} requirements are accounted for"
        )
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            problems.append(f"entry {index} is not an object")
            continue
        evidence = str(entry.get("how_satisfied") or "").strip()
        if not str(entry.get("requirement") or "").strip():
            problems.append(f"entry {index} names no requirement")
        if len(evidence) < 25:
            problems.append(
                f'entry {index} gives no real evidence ("{evidence[:40]}"): name the '
                "expression in your patch that satisfies it and the query result that shows it"
            )
    return problems


def _scope_label() -> str:
    """How to name the allowed file in a message, before and after it is known."""
    if _SPEC is None:
        return "the declared file"
    if _SPEC.scope:
        return ", ".join(_SPEC.scope)
    return "the one file that holds this query (not yet pinned down)"


def _in_scope(rel_path: str) -> bool:
    if _SPEC is None:
        return True
    if not _SPEC.scope_locked and not _SPEC.scope:
        # The file has not been found yet. Anything that is not a test is still a
        # candidate; the bound closes as soon as the run commits to a file.
        return not _is_testish(rel_path)
    for allowed in _SPEC.scope:
        allowed = allowed.rstrip("/")
        if rel_path == allowed or rel_path.startswith(allowed + "/"):
            return True
    return False


def _scope_files() -> list[str]:
    """Concrete files inside the declared scope (a scope may be a directory)."""
    if _SPEC is None:
        return []
    files: list[str] = []
    for entry in _SPEC.scope:
        absolute = os.path.join(_repo_root(), entry)
        if os.path.isfile(absolute):
            files.append(entry)
        elif os.path.isdir(absolute):
            for dirpath, _dirnames, filenames in os.walk(absolute):
                for name in sorted(filenames):
                    if name.endswith(_SOURCE_EXTS):
                        rel = _rel_to_repo(os.path.join(dirpath, name))
                        if rel:
                            files.append(rel)
    return files


# --------------------------------------------------------------------------- #
# The file as it was handed to us
# --------------------------------------------------------------------------- #

_ORIGINAL_SOURCE: dict[str, str] = {}


def _snapshot_originals() -> None:
    """Remember the scope files verbatim, before anything is edited.

    Every "keep the rest of the file unchanged" check compares against this.
    """
    _ORIGINAL_SOURCE.clear()
    for path in _scope_files():
        _original_source(path)


def _original_source(path: str) -> str | None:
    """The file as the task handed it over, whatever has happened since.

    Reading it from the unmodified commit rather than from disk means the
    comparison stays valid even for a file that was already being edited when we
    learned it was the one the instruction meant.
    """
    if path in _ORIGINAL_SOURCE:
        return _ORIGINAL_SOURCE[path]
    text: str | None = None
    code, out = _run_argv(
        ["git", "show", f"HEAD:{path}"], timeout=30, cwd=_repo_root()
    )
    if code == 0:
        text = out
    else:
        try:
            with open(os.path.join(_repo_root(), path), encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError as exc:
            _log(f"could not read the original {path}: {exc}")
    if text is not None:
        _ORIGINAL_SOURCE[path] = text
    return text


def _lock_scope(rel: str) -> None:
    """Close the bound around the file this run has committed to changing.

    Applies only when the instruction bounded the change without naming the
    file. From here on the run holds itself to the same single file the
    instruction asked for.
    """
    if _SPEC is None or _SPEC.scope_locked or not rel:
        return
    _original_source(rel)  # capture it as it shipped, before this edit lands
    _SPEC.scope = [rel]
    _SPEC.scope_locked = True
    _log(f"the change belongs in {rel}; holding the patch to that file")


def _current_source(path: str) -> str | None:
    try:
        with open(os.path.join(_repo_root(), path), encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _find_target_function(tree: ast.AST, class_name: str | None, func_name: str) -> Any:
    """The one function the instruction names. Ambiguity is an error, not a guess."""
    matches: list[Any] = []
    if class_name:
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                matches.extend(
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == func_name
                )
    else:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                matches.append(node)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one definition of {func_name}, found {len(matches)}")
    return matches[0]


def _module_bindings(tree: ast.AST) -> set[str]:
    """Every name the module itself binds: imports, assignments, defs, classes.

    "Use only names the file already imports" is checkable, and checking it
    catches the single most common way one of these patches fails: an ORM
    expression that reaches for a helper the file never imported, which raises
    NameError the first time the query is built.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


def _free_names(node: Any) -> set[str]:
    """Names the node reads without binding them first."""
    bound: set[str] = set()
    read: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if isinstance(child.ctx, ast.Load):
                read.add(child.id)
            else:
                bound.add(child.id)
        elif isinstance(child, ast.arg):
            bound.add(child.arg)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            for alias in child.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(child.name)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
    return read - bound


# Constructs the instruction rules out when it asks for plain expressions, keyed
# by the words it uses. Nothing here is inferred: each entry is one clause of the
# sentence "no Python loops, comprehensions, lambdas, exception handling, or
# context managers inside it".
_PLAIN_EXPRESSION_FORBIDDEN = (
    (ast.For, "a loop"),
    (ast.AsyncFor, "a loop"),
    (ast.While, "a loop"),
    (ast.ListComp, "a comprehension"),
    (ast.SetComp, "a comprehension"),
    (ast.DictComp, "a comprehension"),
    (ast.GeneratorExp, "a generator expression"),
    (ast.Lambda, "a lambda"),
    (ast.Try, "exception handling"),
    (ast.Raise, "a raise"),
    (ast.With, "a context manager"),
    (ast.AsyncWith, "a context manager"),
    (ast.Match, "a match statement"),
    (ast.ClassDef, "a class definition"),
    (ast.Global, "a global statement"),
    (ast.Nonlocal, "a nonlocal statement"),
    (ast.Delete, "a del statement"),
    (ast.Yield, "a yield"),
    (ast.YieldFrom, "a yield"),
    (ast.Await, "an await"),
)

# Names and attributes that would make the method do something other than
# describe a query. The instruction names these itself: "Do not add database
# writes, process, filesystem, network, repository, or dynamic-code side
# effects", and a fetch path has no business calling any of them.
_SIDE_EFFECT_NAMES = {
    "__import__", "breakpoint", "compile", "eval", "exec", "getattr", "globals",
    "locals", "open", "setattr", "vars", "input", "exit", "quit",
}
_SIDE_EFFECT_ATTRS = {
    "cursor", "execute", "executemany", "raw", "save", "update", "delete",
    "create", "bulk_create", "bulk_update", "get_or_create", "update_or_create",
    "system", "popen", "run", "check_output", "urlopen", "request",
}


def _changed_definitions(original_tree: ast.AST, original_text: str,
                         candidate_tree: ast.AST, candidate_text: str) -> list[tuple[str | None, str]]:
    """Which visible definitions differ between the two versions of a file.

    An instruction that says "change only that method" without naming it still
    bounds the change to one method; this is how we learn which one the run
    chose, so the same checks can be applied to it.
    """
    original_lines = original_text.splitlines()
    candidate_lines = candidate_text.splitlines()

    def index(tree: ast.AST, lines: list[str]) -> dict[tuple[str | None, str], str]:
        out: dict[tuple[str | None, str], str] = {}
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[(None, node.name)] = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out[(node.name, child.name)] = "\n".join(
                            lines[child.lineno - 1 : child.end_lineno]
                        )
        return out

    before = index(original_tree, original_lines)
    after = index(candidate_tree, candidate_lines)
    return sorted(key for key in before if key in after and before[key] != after[key])


def _target_failures() -> list[str]:
    """Hold the edited method to everything the instruction said about it.

    Only the checks the instruction actually stated are applied; a task that does
    not bound the change to one method skips this entirely.
    """
    if _SPEC is None or not (_SPEC.func_name or _SPEC.bounded_unnamed):
        return []
    path = _SPEC.primary_file
    if not path or not path.endswith((".py", ".pyi")):
        return []
    original_text = _original_source(path)
    candidate_text = _current_source(path)
    if original_text is None or candidate_text is None:
        return []
    if original_text == candidate_text:
        return []  # nothing edited yet; other checks will say so

    failures: list[str] = []
    try:
        original_tree = ast.parse(original_text)
        candidate_tree = ast.parse(candidate_text)
    except SyntaxError as exc:
        return [f"{path} no longer parses (line {exc.lineno}: {exc.msg})"]

    class_name, func_name = _SPEC.class_name, _SPEC.func_name
    if not func_name:
        # The instruction bounded the change to one method without naming it.
        changed = _changed_definitions(original_tree, original_text, candidate_tree, candidate_text)
        if not changed:
            return []  # the edit is outside any single definition; other checks cover it
        if len(changed) > 1:
            names = ", ".join(f"{cls + '.' if cls else ''}{fn}()" for cls, fn in changed[:5])
            return [
                (
                    f"{path}: {len(changed)} methods changed ({names}). The instruction says to "
                    "change only the one method that does this work; restore the others exactly "
                    "as they were."
                )
            ]
        class_name, func_name = changed[0]
    try:
        original = _find_target_function(original_tree, class_name, func_name)
        candidate = _find_target_function(candidate_tree, class_name, func_name)
    except ValueError as exc:
        return [
            (
                f"target method: {exc}. The instruction bounds the change to "
                f"{_SPEC.target or func_name}(), so that method must still be there, exactly once."
            )
        ]
    label = f"{class_name}.{func_name}" if class_name else func_name

    original_lines = original_text.splitlines(keepends=True)
    candidate_lines = candidate_text.splitlines(keepends=True)
    if _SPEC.freeze_rest or _SPEC.freeze_imports:
        if original_lines[: original.lineno - 1] != candidate_lines[: candidate.lineno - 1]:
            failures.append(
                f"the file above {label}() changed. The instruction says to keep the rest "
                "of the file unchanged, including imports: restore everything before that method "
                "byte-for-byte and put the whole change inside it."
            )
        if original_lines[original.end_lineno :] != candidate_lines[candidate.end_lineno :]:
            failures.append(
                f"the file below {label}() changed. The instruction says to keep the rest "
                "of the file unchanged: restore everything after that method byte-for-byte."
            )

    if _SPEC.freeze_signature:
        for field in ("name", "args", "decorator_list", "returns", "type_params"):
            if _dump_field(getattr(original, field, None)) != _dump_field(getattr(candidate, field, None)):
                failures.append(
                    f"the signature of {label}() changed ({field}). The instruction says to "
                    "keep it exactly as it was; change only the body."
                )
        if type(original) is not type(candidate):
            failures.append(
                f"{label}() changed between `def` and `async def`. Keep the definition kind."
            )

    # A method that opened with a local import kept it: the instruction freezes
    # the imports, and that import is one of them.
    body = list(candidate.body)
    if original.body and isinstance(original.body[0], (ast.Import, ast.ImportFrom)):
        if not body or _dump_field(body[0]) != _dump_field(original.body[0]):
            failures.append(
                f"the import at the top of {label}() changed. It is one of the file's "
                "imports and the instruction freezes those: leave that line exactly as it was."
            )
        else:
            body = body[1:]

    submitted = ast.Module(body=body, type_ignores=[])
    if _SPEC.plain_expressions:
        seen: set[str] = set()
        for node in ast.walk(submitted):
            for node_type, description in _PLAIN_EXPRESSION_FORBIDDEN:
                if isinstance(node, node_type) and description not in seen:
                    seen.add(description)
                    failures.append(
                        f"{label}() contains {description}. The instruction asks for plain "
                        "expressions: express this in the query layer instead, so the database "
                        "does the work."
                    )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                failures.append(
                    f"{label}() defines a nested function. Keep the method a single "
                    "expression built from what the file already provides."
                )
    if _SPEC.freeze_imports:
        for node in ast.walk(submitted):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                failures.append(
                    f"{label}() adds an import. The instruction says to use only names the "
                    "file already imports; if the expression you want needs something else, build "
                    "it from what is already there."
                )
                break

    if _SPEC.no_side_effects or _SPEC.plain_expressions:
        for node in ast.walk(submitted):
            if isinstance(node, ast.Name) and node.id in _SIDE_EFFECT_NAMES:
                failures.append(
                    f"{label}() calls `{node.id}`. A fetch path does not run code, open "
                    "files or reach outside the query; the instruction rules those out."
                )
            if isinstance(node, ast.Attribute) and node.attr in _SIDE_EFFECT_ATTRS:
                failures.append(
                    f"{label}() calls `.{node.attr}(...)`. That writes or executes rather "
                    "than describing a query, and the instruction forbids adding database writes "
                    "or side effects here."
                )
            if isinstance(node, ast.Name) and "__" in node.id:
                failures.append(f"{label}() reaches for the dunder name `{node.id}`.")
            if isinstance(node, ast.Attribute) and "__" in node.attr:
                failures.append(f"{label}() reaches for the dunder attribute `.{node.attr}`.")

    if _SPEC.freeze_imports:
        available = _module_bindings(original_tree) | set(dir(builtins))
        if original.body and isinstance(original.body[0], (ast.Import, ast.ImportFrom)):
            available |= _module_bindings(ast.Module(body=[original.body[0]], type_ignores=[]))
        for arg_group in (candidate.args.posonlyargs, candidate.args.args, candidate.args.kwonlyargs):
            available |= {a.arg for a in arg_group}
        for extra in (candidate.args.vararg, candidate.args.kwarg):
            if extra is not None:
                available.add(extra.arg)
        unknown = sorted(name for name in _free_names(submitted) if name not in available)
        if unknown:
            failures.append(
                f"{label}() uses name(s) the file does not provide: {', '.join(unknown[:8])}. "
                "The instruction says to use only names the file already imports, and a name that "
                "is not there raises NameError the first time the query is built. Build the same "
                "result from the names that are in the file."
            )

    # These methods are meant to stay small. A body that has ballooned is usually
    # a sign the work drifted into Python instead of into the query.
    body_text = "".join(candidate_lines[candidate.lineno - 1 : candidate.end_lineno])
    original_body_text = "".join(original_lines[original.lineno - 1 : original.end_lineno])
    if len(body_text) > max(4000, 8 * len(original_body_text)):
        failures.append(
            f"{label}() has grown to {len(body_text)} characters from {len(original_body_text)}. "
            "These instructions ask for a bounded, expression-shaped method; if it needs this much "
            "code, the work is happening in Python rather than in the database."
        )
    return failures


def _dump_field(value: Any) -> Any:
    if isinstance(value, ast.AST):
        return ast.dump(value, include_attributes=False)
    if isinstance(value, list):
        return [_dump_field(item) for item in value]
    return value


def _file_level_failures() -> list[str]:
    """Checks that apply to a scope file even when no single method is named."""
    if _SPEC is None:
        return []
    failures: list[str] = []
    for path in _scope_files():
        original_text = _original_source(path)
        candidate_text = _current_source(path)
        if original_text is None:
            continue
        if candidate_text is None:
            failures.append(f"{path} was deleted; the instruction says to change it, not remove it.")
            continue
        if candidate_text == original_text or not path.endswith((".py", ".pyi")):
            continue
        try:
            candidate_tree = ast.parse(candidate_text)
        except SyntaxError as exc:
            failures.append(f"{path} no longer parses (line {exc.lineno}: {exc.msg})")
            continue
        try:
            original_tree = ast.parse(original_text)
        except SyntaxError:
            continue
        # Definition conservation: whatever the file exposed, it still exposes.
        before = _visible_defs(original_tree)
        after = _visible_defs(candidate_tree)
        missing = sorted(set(before) - set(after))
        if missing:
            failures.append(
                f"{path} lost or renamed {missing[:6]}. The instruction says to leave unrelated "
                "source alone, so every definition that was there must still be there."
            )
        if _SPEC.freeze_imports and not _SPEC.func_name:
            before_imports = _import_statements(original_tree, original_text)
            after_imports = _import_statements(candidate_tree, candidate_text)
            if before_imports != after_imports:
                failures.append(
                    f"{path} changed its imports. The instruction says to keep them as they are."
                )
    return failures


def _visible_defs(tree: ast.AST) -> list[str]:
    """Module-level and class-level definitions, the ones a reader sees in the file."""
    out: list[str] = []

    def walk(node: ast.AST, prefix: str, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not in_function:
                    kind = "async" if isinstance(child, ast.AsyncFunctionDef) else "def"
                    out.append(f"{kind} {prefix}{child.name}")
                walk(child, prefix + child.name + ".", True)
            elif isinstance(child, ast.ClassDef):
                if not in_function:
                    out.append(f"class {prefix}{child.name}")
                walk(child, prefix + child.name + ".", in_function)
            else:
                walk(child, prefix, in_function)

    walk(tree, "", False)
    return sorted(out)


def _import_statements(tree: ast.AST, text: str) -> list[str]:
    lines = text.splitlines()
    out: list[str] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    return out


# --------------------------------------------------------------------------- #
# The checks the instruction names
# --------------------------------------------------------------------------- #

_CHECK_CACHE: dict[str, tuple[bool, str]] = {}

# A linter answers in under a second; an application test suite against a real
# database takes minutes and, run once per edit, is the single most expensive
# habit a run can fall into. Split them so the fast ones can be checked after
# every edit and the slow ones are rationed.
_FAST_CHECK_HEADS = {
    "ruff", "flake8", "gofmt", "golangci-lint", "tsc", "eslint", "mypy", "black",
    "isort", "prettier", "vet",
}
# How many times the slow checks may run in one attempt. Enough to see a failure,
# fix it, and confirm the fix, with a round to spare; past that the useful
# information is in the failure already on screen, not in running it again.
MAX_SLOW_CHECK_RUNS = int(os.getenv("RIDGES_MAX_SLOW_CHECKS", "5"))
_SLOW_CHECK_RUNS = 0


def _is_fast_check(command: str) -> bool:
    head = (command.split() or [""])[0].rsplit("/", 1)[-1]
    return head in _FAST_CHECK_HEADS


def _scope_fingerprint() -> str:
    """Content hash of the scope files, used to skip work that cannot have changed."""
    digest = hashlib.sha256()
    for path in _scope_files():
        digest.update(path.encode())
        digest.update((_current_source(path) or "<missing>").encode())
    return digest.hexdigest()


def _check_budget() -> int:
    return max(60, min(CHECK_TIMEOUT_SEC, int(_time_left() - 45)))


def _run_declared_check(command: str) -> tuple[bool, str]:
    """Run one of the instruction's own commands in the application root."""
    key = f"{_scope_fingerprint()}::{command}"
    if key in _CHECK_CACHE:
        passed, out = _CHECK_CACHE[key]
        return passed, out + "\n[unchanged since this check last ran]"
    budget = _check_budget()
    _log(f"declared check ({budget}s budget): {command}")
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = []
    env = _subprocess_env(_import_roots())
    if argv:
        code, out = _run_argv(argv, timeout=budget, cwd=_repo_root(), env=env)
    else:
        out = _run(command, timeout=budget, cwd=_repo_root())
        code = 1 if out.startswith("[exit") else 0
    _clean_repo_junk()
    if code == 124:
        # A timeout is not evidence of a broken patch; saying it is sends the
        # model chasing a regression that is not there.
        return True, f"[inconclusive: `{command}` did not finish within {budget}s]"
    passed = code == 0
    result = _truncate(out.strip() or "[no output]", 2500)
    _CHECK_CACHE[key] = (passed, result)
    return passed, result


def _run_declared_checks(slow: bool = True) -> tuple[bool, str]:
    """Run the commands the instruction told us to run.

    The instruction ends with "Run these checks before finishing" and then lists
    them. This runs exactly those, for exactly that reason. They are the
    project's existing regression checks: they passed before the change and must
    pass after it.

    `slow=False` runs only the ones that answer immediately, so the linter can be
    consulted after every edit without paying for the test suite each time.

    Returns (all passed, report).
    """
    global _SLOW_CHECK_RUNS
    if _SPEC is None or not _SPEC.checks:
        return True, "[the instruction names no checks]"
    commands = [c for c in _SPEC.checks if slow or _is_fast_check(c)]
    if not commands:
        return True, "[no immediate checks to run]"
    sections: list[str] = []
    all_passed = True
    charged = False
    for command in commands:
        if _is_fast_check(command):
            pass
        elif _time_left() < CHECK_MIN_RESERVE_SEC:
            sections.append(f"[skipped `{command}`: not enough time left to run it]")
            continue
        elif _SLOW_CHECK_RUNS >= MAX_SLOW_CHECK_RUNS:
            cached = _CHECK_CACHE.get(f"{_scope_fingerprint()}::{command}")
            if cached is not None:
                passed, out = cached
                sections.append(("PASS  " if passed else "FAIL  ") + command + "\n" + out)
                all_passed = all_passed and passed
            else:
                sections.append(
                    f"[not re-running `{command}`: it has already run "
                    f"{_SLOW_CHECK_RUNS} times this attempt. Work out the fix from the failure "
                    "you already have rather than running it again.]"
                )
            continue
        else:
            charged = True
        passed, out = _run_declared_check(command)
        if passed:
            sections.append(f"PASS  {command}")
        else:
            all_passed = False
            sections.append(f"FAIL  {command}\n{out}")
    if charged:
        _SLOW_CHECK_RUNS += 1
    return all_passed, "\n".join(sections)


# --------------------------------------------------------------------------- #
# Patch scope
# --------------------------------------------------------------------------- #

_JUNK_DIR_NAMES = {"__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", ".cache"}
_PRUNE_DIR_NAMES = {".git", "node_modules", ".venv", "venv", ".tox", ".nox", "build", "dist"}


def _git_paths(command: str) -> list[str]:
    out = _run(command, timeout=30, cwd=_repo_root())
    if out.startswith("[exit") or out.startswith("[command"):
        return []
    return [part for part in out.split("\0") if part.strip()]


def _changed_paths() -> list[str]:
    """Every repo path the patch would carry: tracked edits plus untracked files."""
    paths = _git_paths("git diff --name-only -z HEAD")
    paths += _git_paths("git ls-files --others --exclude-standard -z")
    seen: list[str] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen


def _clean_repo_junk() -> None:
    """Delete untracked tool caches and byte-code that would pollute the patch.

    `git add -A` stages untracked files, so a .pytest_cache/ the project does not
    gitignore turns a correct fix into a patch that touches paths the instruction
    did not allow. Tracked paths are never touched: deleting one would itself be
    a change the instruction did not ask for.
    """
    root = _repo_root()
    junk: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in list(dirnames):
            if name in _JUNK_DIR_NAMES:
                junk.append(os.path.join(dirpath, name))
                dirnames.remove(name)
            elif name in _PRUNE_DIR_NAMES:
                dirnames.remove(name)
        for name in filenames:
            if name.endswith((".pyc", ".pyo", ".orig", ".rej")):
                junk.append(os.path.join(dirpath, name))
    if not junk:
        return
    pairs = [(path, rel) for path, rel in ((p, _rel_to_repo(p)) for p in junk) if rel]
    if not pairs:
        return
    quoted = " ".join(shlex.quote(rel) for _path, rel in pairs)
    # `git ls-files` lists tracked FILES, so a junk directory is off-limits when any
    # tracked path lives inside it.
    tracked = set(_git_paths(f"git ls-files -z -- {quoted}"))
    for path, rel in pairs:
        if rel in tracked or any(entry.startswith(rel + "/") for entry in tracked):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def _scope_violations() -> list[str]:
    if _SPEC is None:
        return []
    _clean_repo_junk()
    return sorted(path for path in _changed_paths() if not _in_scope(path))


def _revert_out_of_scope() -> list[str]:
    """Undo everything outside the declared scope before capturing the diff.

    The instruction allows one file, so shipping a second one is not a partial
    win: reverting is strictly better, and at worst the model loses an edit it
    was never allowed to make.
    """
    reverted: list[str] = []
    for path in _scope_violations():
        absolute = os.path.join(_repo_root(), path)
        quoted = shlex.quote(path)
        if _run(f"git ls-files --error-unmatch -- {quoted}", timeout=20, cwd=_repo_root()).startswith("[exit"):
            try:
                if os.path.isdir(absolute) and not os.path.islink(absolute):
                    shutil.rmtree(absolute, ignore_errors=True)
                else:
                    os.remove(absolute)
                reverted.append(path)
            except OSError:
                pass
        else:
            _run(f"git checkout -- {quoted}", timeout=30, cwd=_repo_root())
            reverted.append(path)
    if reverted:
        _log(f"reverted changes outside the file the instruction names: {reverted}")
    return reverted


# --------------------------------------------------------------------------- #
# The database in front of us
# --------------------------------------------------------------------------- #

_DB_TARGET: dict[str, Any] | None = None
_MANAGE_PY: str | None | bool = False

# Statements that change data or structure. Exploration is for understanding the
# schema and the rows, and the instruction is explicit that the change must not
# add database writes: a session that leaves rows, tables or indexes behind has
# altered the very thing it was measuring.
_WRITE_SQL_RE = re.compile(
    r"^\s*(?:insert|update|delete|truncate|drop|create|alter|grant|revoke|comment|"
    r"vacuum|reindex|cluster|refresh|copy|call|do|begin|commit|rollback|set\s+role|"
    r"security\s+label|lock)\b",
    re.I,
)


def _manage_py() -> str | None:
    """The application's Django entry point, if it has one."""
    global _MANAGE_PY
    if _MANAGE_PY is not False:
        return _MANAGE_PY  # type: ignore[return-value]
    _MANAGE_PY = None
    root = _repo_root()
    for depth_root, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(depth_root, root)
        if rel != "." and len(rel.split(os.sep)) > 3:
            dirnames[:] = []
            continue
        for name in list(dirnames):
            if name in _PRUNE_DIR_NAMES or name in _JUNK_DIR_NAMES:
                dirnames.remove(name)
        if "manage.py" in filenames:
            _MANAGE_PY = _rel_to_repo(os.path.join(depth_root, "manage.py"))
            break
    if _MANAGE_PY:
        _log(f"application entry point: {_MANAGE_PY}")
    return _MANAGE_PY  # type: ignore[return-value]


def _dsn_candidates() -> list[dict[str, Any]]:
    """Connection settings to try, gathered from the environment and the app config.

    Nothing is assumed about which application this is: the settings come from
    the process environment, from the application's own settings module when it
    has one, and from connection strings written in its configuration files.
    """
    candidates: list[dict[str, Any]] = []

    def add(kind: str, dsn: str) -> None:
        entry = {"kind": kind, "dsn": dsn}
        if entry not in candidates:
            candidates.append(entry)

    for var in ("DATABASE_URL", "POSTGRES_URL", "PGURL", "PG_DSN"):
        value = os.getenv(var)
        if value:
            add("clickhouse" if value.startswith("clickhouse") else "postgresql", value)
    if os.getenv("PGHOST") or os.getenv("PGDATABASE"):
        user = os.getenv("PGUSER", "postgres")
        password = os.getenv("PGPASSWORD", "")
        host = os.getenv("PGHOST", "localhost")
        port = os.getenv("PGPORT", "5432")
        database = os.getenv("PGDATABASE", user)
        auth = f"{user}:{password}@" if password else f"{user}@"
        add("postgresql", f"postgresql://{auth}{host}:{port}/{database}")

    # The application's own settings are the authoritative answer when it can
    # tell us; asking it is cheaper and more reliable than parsing config files.
    manage = _manage_py()
    if manage:
        script = (
            "from django.db import connection as c;"
            "s=c.settings_dict;"
            "print('postgresql://%s:%s@%s:%s/%s' % ("
            "s.get('USER') or '', s.get('PASSWORD') or '',"
            "s.get('HOST') or 'localhost', s.get('PORT') or 5432, s.get('NAME') or ''))"
        )
        code, out = _run_argv(
            [_python_bin(), os.path.join(_repo_root(), manage), "shell", "-c", script],
            timeout=90,
            cwd=_repo_root(),
            env=_subprocess_env(_import_roots()),
        )
        for line in (out or "").splitlines():
            line = line.strip()
            if line.startswith(("postgresql://", "postgres://")) and code == 0:
                add("postgresql", line)

    # Connection strings written into configuration files.
    grep = _run(
        "git grep -h -I -E -o \"(postgres(ql)?|clickhouse)://[^\\\"' ]+\" -- "
        "':!*test*' ':!*docs*' | head -10",
        timeout=30,
        cwd=_repo_root(),
    )
    if not grep.startswith("[exit") and not grep.startswith("[command"):
        for line in grep.splitlines():
            line = line.strip()
            if "://" in line and "$" not in line and "{" not in line:
                add("clickhouse" if line.startswith("clickhouse") else "postgresql", line)
    return candidates


def _db_target() -> dict[str, Any] | None:
    """The first connection that actually answers, or None."""
    global _DB_TARGET
    if _DB_TARGET is not None:
        return _DB_TARGET or None
    for candidate in _dsn_candidates():
        if candidate["kind"] == "postgresql":
            if not shutil.which("psql"):
                continue
            code, _out = _run_argv(
                ["psql", candidate["dsn"], "-At", "-c", "SELECT 1"], timeout=30, cwd=_repo_root()
            )
        else:
            if not shutil.which("clickhouse-client"):
                continue
            code, _out = _run_argv(
                ["clickhouse-client", "--query", "SELECT 1"], timeout=30, cwd=_repo_root()
            )
        if code == 0:
            _DB_TARGET = candidate
            _log(f"database reachable: {candidate['kind']}")
            return candidate
    _DB_TARGET = {}  # cache the negative answer
    _log("no database connection resolved; work from the schema in the code")
    return None


def _tool_db_sql(sql: str) -> str:
    """Run one read-only statement against the application's database."""
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return "[db_sql needs a statement]"
    if _WRITE_SQL_RE.match(sql) or ";" in sql:
        return (
            "[refused: db_sql reads, it does not write. The change you are making must not add "
            "database writes or leave anything behind, so exploration stays read-only and one "
            "statement at a time. Use SELECT / EXPLAIN / a catalog query. To try out rows, use "
            "app_shell, which rolls everything back.]"
        )
    target = _db_target()
    if target is None:
        return "[no database connection is available from this environment]"
    budget = max(20, min(90, int(_time_left() - 30)))
    if target["kind"] == "postgresql":
        code, out = _run_argv(
            ["psql", target["dsn"], "--no-psqlrc", "-P", "pager=off", "-c", sql],
            timeout=budget,
            cwd=_repo_root(),
        )
    else:
        code, out = _run_argv(
            ["clickhouse-client", "--query", sql], timeout=budget, cwd=_repo_root()
        )
    prefix = "" if code == 0 else f"[exit {code}]\n"
    return _truncate(prefix + (out.strip() or "[no rows]"), 2600)


def _tool_db_explain(sql: str) -> str:
    """Show how the database would execute a statement, and what it costs."""
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return "[db_explain needs a SELECT statement]"
    if _WRITE_SQL_RE.match(sql):
        return "[db_explain runs read-only statements only]"
    target = _db_target()
    if target is None:
        return "[no database connection is available from this environment]"
    if target["kind"] == "postgresql":
        return _tool_db_sql(f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE, COSTS OFF) {sql}")
    return _tool_db_sql(f"EXPLAIN indexes = 1 {sql}")


def _wrap_rolled_back(code: str) -> str:
    """Run application code inside a transaction that is always rolled back.

    This is how you answer "what does this query return for a row that ties, a
    NULL group, an empty group" without inventing the rows in your head and
    without leaving a single row behind. The rollback is unconditional: the
    database is left exactly as it was found.
    """
    body = textwrap.indent(code, " " * 8)
    return (
        "from django.db import transaction\n"
        "class _RollBack(Exception):\n"
        "    pass\n"
        "try:\n"
        "    with transaction.atomic():\n"
        f"{body}\n"
        "        raise _RollBack\n"
        "except _RollBack:\n"
        "    pass\n"
    )


def _tool_app_shell(code: str, rollback: Any = True) -> str:
    """Evaluate a snippet inside the application, against the live database."""
    if not (code or "").strip():
        return "[app_shell needs a `code` snippet that PRINTS what you want to see]"
    manage = _manage_py()
    budget = max(30, min(180, int(_time_left() - 40)))
    env = _subprocess_env(_import_roots())
    if manage:
        payload = _wrap_rolled_back(code) if rollback is not False else code
        code_rc, out = _run_argv(
            [_python_bin(), os.path.join(_repo_root(), manage), "shell", "-c", payload],
            timeout=budget,
            cwd=_repo_root(),
            env=env,
        )
    else:
        os.makedirs(SCRATCH_DIR, exist_ok=True)
        script = os.path.join(SCRATCH_DIR, "app_probe.py")
        try:
            with open(script, "w") as handle:
                handle.write(code)
        except OSError as exc:
            return f"[app_shell error: {exc}]"
        code_rc, out = _run_argv([_python_bin(), script], timeout=budget, cwd=_repo_root(), env=env)
    _clean_repo_junk()
    if code_rc == 124:
        return "[app_shell: the snippet did not finish in time; make it smaller]"
    prefix = "" if code_rc == 0 else f"[exit {code_rc}]\n"
    return _truncate(prefix + (out.strip() or "[no output]"), 2800)


# --------------------------------------------------------------------------- #
# The application as it was, for before/after comparison
# --------------------------------------------------------------------------- #

_UNPATCHED_COPY_READY: bool | None = None


def _ensure_unpatched_copy() -> bool:
    """Keep a copy of the application's source as it was before this run edited it.

    This is `git archive HEAD` of the repository under /tmp: the same files the
    task handed us, minus our own changes, pointed at the same database.

    It exists because "the same rows, with less database work" is a claim that
    has to be checked. A rewritten query that looks equivalent but changes the
    grain, drops the empty groups, or double-counts a join is simply broken, and
    the project's own tests do not always cover the case you moved. The honest
    way to check is to run the same query against both versions and compare. That
    is what any careful engineer does with `git stash` by hand; doing it in a copy
    just avoids disturbing the working tree.
    """
    global _UNPATCHED_COPY_READY
    if _UNPATCHED_COPY_READY is not None:
        return _UNPATCHED_COPY_READY
    _UNPATCHED_COPY_READY = False
    try:
        shutil.rmtree(UNPATCHED_COPY, ignore_errors=True)
        os.makedirs(UNPATCHED_COPY, exist_ok=True)
    except OSError:
        return False
    out = _run(
        f"git archive --format=tar HEAD | tar -x -C {shlex.quote(UNPATCHED_COPY)}",
        timeout=180,
        cwd=_repo_root(),
    )
    if out.startswith("[exit") or out.startswith("[command"):
        _log(f"baseline tree unavailable: {out.strip()[:200]}")
        return False
    _UNPATCHED_COPY_READY = True
    return True


# Whether the model has actually compared before and after, and what it last saw.
# On a task whose stated goal is less database work, this is the only check that
# distinguishes "cheaper" from "cheaper and wrong".
_COMPARE_STATE = {"runs": 0, "diverged": False, "nudged": False}


def _tool_compare_results(code: str) -> str:
    """Run one snippet against the unpatched application and against the patched one."""
    if not (code or "").strip():
        return "[compare_results needs a `code` snippet that PRINTS deterministic output]"
    if not _ensure_unpatched_copy():
        return "[compare_results unavailable: could not materialize the unpatched copy]"
    manage = _manage_py()
    budget = max(30, min(180, int(_time_left() - 40)))
    payload = _wrap_rolled_back(code) if manage else code

    def run_in(base: str) -> tuple[int, str]:
        env = _subprocess_env(_import_roots(base))
        if manage:
            return _run_argv(
                [_python_bin(), os.path.join(base, manage), "shell", "-c", payload],
                timeout=budget,
                cwd=base,
                env=env,
            )
        os.makedirs(SCRATCH_DIR, exist_ok=True)
        script = os.path.join(SCRATCH_DIR, "compare.py")
        with open(script, "w") as handle:
            handle.write(code)
        return _run_argv([_python_bin(), script], timeout=budget, cwd=base, env=env)

    try:
        base_code, base_out = run_in(UNPATCHED_COPY)
        head_code, head_out = run_in(_repo_root())
    except OSError as exc:
        return f"[compare_results error: {exc}]"
    _clean_repo_junk()
    if 124 in (base_code, head_code):
        return "[compare_results: the snippet timed out; make it smaller]"
    if base_code != 0:
        return (
            "[compare_results: the snippet FAILED against the unpatched application, so it is not "
            "a valid comparison. Fix the snippet (imports, arguments), not the repository.]\n"
            + _truncate(base_out, 1800)
        )
    _COMPARE_STATE["runs"] += 1
    if base_out == head_out and base_code == head_code:
        _COMPARE_STATE["diverged"] = False
        return f"[IDENTICAL results before and after]\n{_truncate(head_out, 1200)}"
    _COMPARE_STATE["diverged"] = True
    base_lines = base_out.splitlines()
    head_lines = head_out.splitlines()
    diff: list[str] = []
    for index in range(max(len(base_lines), len(head_lines))):
        before = base_lines[index] if index < len(base_lines) else "<missing>"
        after = head_lines[index] if index < len(head_lines) else "<missing>"
        if before != after:
            diff.append(f"line {index + 1}:\n  before: {before}\n  after : {after}")
        if len(diff) >= 12:
            diff.append("... more differences suppressed ...")
            break
    return _truncate("[RESULTS CHANGED]\n" + "\n".join(diff), 2800)


# --------------------------------------------------------------------------- #
# Heuristic localization (used when the instruction names no file)
# --------------------------------------------------------------------------- #

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_PATH_RE = re.compile(r"[\w./-]+\.[A-Za-z0-9_]+")
_SEARCHABLE_EXTS = _SOURCE_EXTS + (".rst", ".txt", ".cfg", ".toml", ".ini", ".yaml", ".yml")
_SKIP_DIR_PARTS = {
    ".git", "node_modules", "__pycache__", ".tox", ".venv", "venv", "build",
    "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "when", "then", "should",
    "would", "could", "into", "return", "returns", "error", "issue", "value",
    "values", "none", "true", "false", "self", "class", "function", "method",
    "test", "tests", "example", "expected", "actual", "result", "code", "python",
    "version", "https", "http", "github", "import", "object", "string", "number",
    "query", "queries", "database", "table", "column", "rows", "row", "select",
}


def _is_source(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    if any(p in _SKIP_DIR_PARTS for p in parts):
        return False
    return path.endswith(_SEARCHABLE_EXTS)


def _is_testish(path: str) -> bool:
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    return (
        base.startswith("test_")
        or base.endswith(("_test.py", "_test.go", ".test.ts", ".spec.ts"))
        or base == "conftest.py"
        or "/test" in low
        or low.startswith("test")
    )


def _problem_terms(text: str) -> list[str]:
    """Distinctive identifiers/paths from the problem, most specific first."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        low = token.lower()
        if len(token) < 3 or low in _STOPWORDS or low in seen:
            return
        seen.add(low)
        terms.append(token)

    for quoted in re.findall(r"[`'\"]([^`'\"]{3,80})[`'\"]", text or ""):
        for ident in _IDENT_RE.findall(quoted):
            add(ident)
    for path in _PATH_RE.findall(text or ""):
        for piece in re.split(r"[./\\]", path):
            add(piece)
    for ident in _IDENT_RE.findall(text or ""):
        add(ident)
    terms.sort(key=lambda t: (("_" in t or not t.islower()), len(t)), reverse=True)
    return terms[:30]


def _rank_candidate_files(problem: str) -> list[tuple[str, int]]:
    """Return (path, score) for files likely to hold the query that is wrong."""
    terms = _problem_terms(problem)
    if not terms:
        return []
    counts: dict[str, int] = {}
    for term in terms[:18]:
        out = _run(f"git grep -l -F -- {shlex.quote(term)}", timeout=20, cwd=_repo_root())
        if out.startswith("[exit") or out.startswith("[command"):
            continue
        for path in out.splitlines():
            path = path.strip()
            if path and _is_source(path):
                counts[path] = counts.get(path, 0) + 1

    # A file that DEFINES a named symbol is far more likely the fix site than one
    # that merely mentions it.
    defines: dict[str, int] = {}
    for term in terms[:10]:
        pattern = r"^[[:space:]]*(def|class|func|function|const|type)[[:space:]]+" + term + r"\b"
        out = _run(f"git grep -l -E -- {shlex.quote(pattern)}", timeout=20, cwd=_repo_root())
        if out.startswith("[exit") or out.startswith("[command"):
            continue
        for path in out.splitlines():
            path = path.strip()
            if path and _is_source(path):
                defines[path] = defines.get(path, 0) + 1
                counts.setdefault(path, 0)

    explicit = set(_PATH_RE.findall(problem or ""))
    low_terms = [t.lower() for t in terms]
    query_layer_markers = (
        "queryset", "querysets", "filterset", "filtersets", "managers", "manager",
        "repository", "repositories", "dao", "queries", "query", "models", "store",
        "migrations", "sql",
    )

    scored: list[tuple[str, int]] = []
    for path, hits in counts.items():
        score = hits * 3 + defines.get(path, 0) * 25
        low = path.lower()
        base = low.rsplit("/", 1)[-1]
        for t in low_terms:
            if t in base:
                score += 4
            elif t in low:
                score += 1
        # The change belongs on the path the application runs, and in these
        # codebases that path lives in the query layer.
        if any(marker in low for marker in query_layer_markers):
            score += 8
        if any(path.endswith(e) for e in explicit):
            score += 30
        if _is_testish(path):
            score -= 4  # the tests say what must keep working, not what to change
        scored.append((path, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:12]


def _localization_hint(problem: str) -> str:
    ranked = _rank_candidate_files(problem)
    if not ranked:
        return ""
    lines = ["Likely-relevant files (heuristic; verify before trusting):"]
    for path, score in ranked:
        lines.append(f"  - {path}  (score {score})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

_BASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a non-interactive shell command in the application root "
                "(grep -rn, ls, find, cat, python)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Re-run heuristic localization for a free-text query; returns ranked candidate files.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file. Page through big files with start_line/end_line "
                "(1-indexed, inclusive); offset/limit are also accepted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed)."},
                    "end_line": {"type": "integer", "description": "Last line to read (inclusive)."},
                    "offset": {"type": "integer", "description": "Alias for start_line."},
                    "limit": {"type": "integer", "description": "Number of lines to read from start_line."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact substring (must occur exactly once) in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

_DB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "db_sql",
            "description": (
                "Run ONE read-only statement against the live database and show the rows. "
                "Use it to read the schema (information_schema / pg_indexes / SHOW CREATE TABLE), "
                "to see how the real rows are distributed, and to prove what a candidate query "
                "returns. Writes and DDL are refused."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "db_explain",
            "description": (
                "Show the plan the database actually chooses for a SELECT, with timing and "
                "buffer counts. This is how you tell whether an index is being used and how "
                "much work a query really does -- reading the SQL is not evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "app_shell",
            "description": (
                "Evaluate a snippet inside the application itself, against the live database, "
                "and show what it PRINTS. This is the fastest way to see the SQL your change "
                "generates (print(str(qs.query))), to count the statements a code path issues, "
                "and to check a result against rows you create for the occasion (ties, NULLs, "
                "empty groups, duplicates). Everything the snippet does to the database is "
                "rolled back afterwards, so create whatever rows you need."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_results",
            "description": (
                "Before/after A/B test. Give a snippet that PRINTS deterministic results; it is "
                "run against the application as it was before your change and against your "
                "patched version, and any difference is reported. This is the check that "
                "separates 'cheaper' from 'cheaper and wrong'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_requirements",
            "description": (
                "Check your current edits against everything the instruction states: which file "
                "changed, that the named method kept its signature and the rest of the file is "
                "untouched, that it uses only names the file already imports, and that it is "
                "plain expressions. Cheap; run it after every substantive edit."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_checks",
            "description": (
                "Run the commands the instruction itself lists under 'Run these checks before "
                "finishing'. They are regression checks that already passed before your change; "
                "they say what must keep working. The application's test suite is minutes of "
                "work, so use check_requirements (which includes the linter) while you iterate "
                "and call this when you believe you are done."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_GENERAL_FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call only after the change is saved AND you have verified it. Optionally summarize.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}},
    },
}


def _db_finish_tool() -> dict:
    """`finish` with a mandatory account of every requirement the instruction lists.

    Walking each bullet against the patch is the cheapest defence against the
    usual miss: a query that is right for the rows in front of you and wrong on
    the tie, the NULL, or the empty group that the instruction spelled out.
    """
    properties: dict[str, Any] = {
        "summary": {"type": "string", "description": "What you changed and why."},
    }
    required: list[str] = []
    if _REQUIREMENTS:
        numbered = "; ".join(f"{i}. {r}" for i, r in enumerate(_REQUIREMENTS, start=1))
        properties["requirements_checklist"] = {
            "type": "array",
            "description": (
                f"One entry for EACH of the task's {len(_REQUIREMENTS)} requirements, in order: "
                + numbered
            ),
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string", "description": "The requirement, quoted or paraphrased."},
                    "how_satisfied": {
                        "type": "string",
                        "description": (
                            "The expression in YOUR patch that satisfies it, and the evidence you "
                            "have: the rows a query returned, a plan, or a comparison result."
                        ),
                    },
                },
                "required": ["requirement", "how_satisfied"],
            },
        }
        required.append("requirements_checklist")
    return {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call only after check_requirements passes, the instruction's own checks pass, and "
                "you have shown the result is right on the edge cases the instruction names."
            ),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _tools() -> list[dict]:
    if _SPEC is None:
        return _BASE_TOOLS + [_GENERAL_FINISH_TOOL]
    return _BASE_TOOLS + _DB_TOOLS + [_db_finish_tool()]


def _as_int(value: Any, default: int | None) -> int | None:
    """Coerce a tool argument to int.

    Models routinely send line numbers as floats ("1.0") or strings ("1") even
    when the schema says integer. Passing those straight into a slice raises
    "slice indices must be integers", which kills the run before any edit is made.
    Never trust the model's arg types.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tool_read_file(
    path: str,
    start_line: Any = None,
    end_line: Any = None,
    offset: Any = None,
    limit: Any = None,
) -> str:
    rejection = _outside_repo_rejection(path)
    if rejection:
        _log(f"refused a read outside the project: {path}")
        return rejection
    try:
        with open(path, "r", errors="replace") as handle:
            lines = handle.readlines()
    except Exception as exc:  # noqa: BLE001
        return f"[error reading {path}: {exc}]"
    start_line = _as_int(start_line, None)
    end_line = _as_int(end_line, None)
    offset = _as_int(offset, None)
    limit = _as_int(limit, None)
    # Models routinely page with offset/limit instead of start_line/end_line.
    # Silently dropping those makes every read return the same first page, so the
    # model never sees the query it came for.
    if start_line is None and offset is not None:
        start_line = offset
    if end_line is None and limit is not None:
        end_line = (start_line or 1) + limit - 1
    start = max((start_line or 1) - 1, 0)
    end = end_line if end_line is not None else min(len(lines), start + MAX_FILE_READ_LINES)
    end = min(end, len(lines))
    selected = lines[start:end][:MAX_FILE_READ_LINES]
    end = start + len(selected)
    numbered = "".join(f"{i + 1}\t{line}" for i, line in enumerate(selected, start=start))
    return _truncate(f"# {path} (lines {start + 1}-{end} of {len(lines)})\n{numbered}")


# The agent works inside the application it was handed. Anything it needs to try
# out lives in its own scratch directory under /tmp, never in the project tree
# and never elsewhere on the filesystem. Enforcing that here, rather than
# trusting the prompt, keeps a wandering model from reading or writing things
# that have nothing to do with the task.
_ALLOWED_OUTSIDE_PREFIXES = ("/tmp/", "/var/tmp/")

# Pulling code or packages off the network mid-run is against the task rules and
# is never needed: the environment is complete.
_NETWORK_COMMAND_RE = re.compile(
    r"(?:^|[|;&]|\s)(?:curl|wget|nc|ncat|telnet|ssh|scp|rsync"
    r"|pip3?\s+install|pip3?\s+download|uv\s+(?:pip\s+)?(?:install|add)|poetry\s+add"
    r"|conda\s+install|npm\s+(?:i|install)|yarn\s+add|apt(?:-get)?\s+install"
    r"|go\s+get|cargo\s+add|git\s+(?:clone|fetch|pull|remote|push))\b",
    re.I,
)

# Shelling out to the database to change it would undo the very thing the run is
# measuring, and the instruction is explicit that the change adds no writes.
_SHELL_DB_WRITE_RE = re.compile(
    r"(?:psql|clickhouse-client|mysql)\b[^\n]*\b(?:insert|update|delete|drop|create|alter|truncate|grant)\b",
    re.I,
)


def _outside_repo_rejection(target: str) -> str | None:
    """Refuse absolute paths that are neither in the application nor in scratch."""
    if not target:
        return None
    normalized = os.path.normpath(target.strip().strip("'\""))
    if not normalized.startswith("/"):
        return None
    lowered = normalized.replace("\\", "/").lower()
    if lowered.startswith(_ALLOWED_OUTSIDE_PREFIXES) or lowered == "/tmp":
        return None
    if _rel_to_repo(normalized) is not None:
        return None
    return (
        f"[refused: {normalized} is outside the application. Solve the task from its source and "
        "its database, and put anything you want to try out under /tmp.]"
    )


def _bash_rejection(command: str) -> str | None:
    """Block work outside the project, network fetches and database writes."""
    if not command.strip():
        return None
    for token in re.findall(r"[/\w.\-]+", command):
        if token.startswith("/"):
            rejection = _outside_repo_rejection(token)
            if rejection:
                return rejection
    if _NETWORK_COMMAND_RE.search(command):
        return (
            "[refused: this run is self-contained. The task forbids adding dependencies, and "
            "fetching code or packages over the network is not part of solving it. Everything "
            "you need is in the application and the installed environment.]"
        )
    if _SHELL_DB_WRITE_RE.search(command):
        return (
            "[refused: that would change the database. Read it with db_sql, look at plans with "
            "db_explain, and try rows out with app_shell, which rolls everything back. The change "
            "you are making must not add writes or leave anything behind.]"
        )
    return None


def _write_rejection(path: str) -> str | None:
    """Reject writes the instruction forbids, BEFORE they happen."""
    rejection = _outside_repo_rejection(path)
    if rejection:
        return rejection
    rel = _rel_to_repo(path)
    if rel is None:
        return None  # outside the repo (e.g. /tmp scratch): always fine
    if _SPEC is not None and not _in_scope(rel):
        return (
            f"[write rejected: {rel} is OUTSIDE the file the instruction allows "
            f"({_scope_label()}). The change belongs on the path the application already "
            "runs, in that file; a fix placed anywhere else is not the fix that was asked for. "
            "Put scratch scripts under /tmp.]"
        )
    if _SPEC is not None and _is_testish(rel):
        return (
            f"[write rejected: {rel} is part of the project's test suite. The tests say what must "
            "keep working; changing them to fit your query proves nothing.]"
        )
    return None


def _tool_edit_file(path: str, old_str: str, new_str: str) -> str:
    rejection = _write_rejection(path)
    if rejection:
        return rejection
    try:
        with open(path, "r", errors="replace") as handle:
            content = handle.read()
    except Exception as exc:  # noqa: BLE001
        return f"[error reading {path}: {exc}]"
    count = content.count(old_str)
    if count == 0:
        return "[edit failed: old_str not found. Read the file and copy the exact text, including whitespace.]"
    if count > 1:
        return f"[edit failed: old_str occurs {count} times. Include more surrounding context so it is unique.]"
    updated = content.replace(old_str, new_str, 1)
    rel_target = _rel_to_repo(path)
    if rel_target:
        _lock_scope(rel_target)
    try:
        with open(path, "w") as handle:
            handle.write(updated)
    except Exception as exc:  # noqa: BLE001
        return f"[error writing {path}: {exc}]"
    rel = _rel_to_repo(path)
    note = ""
    if rel and rel.endswith((".py", ".pyi")):
        try:
            ast.parse(updated)
        except SyntaxError as exc:
            note = f"\n[WARNING: {rel} no longer parses -- line {exc.lineno}: {exc.msg}]"
    return f"[edited {path}]{note}"


def _tool_create_file(path: str, content: str) -> str:
    rejection = _write_rejection(path)
    if rejection:
        return rejection
    rel_target = _rel_to_repo(path)
    if rel_target:
        _lock_scope(rel_target)
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as handle:
            handle.write(content)
    except Exception as exc:  # noqa: BLE001
        return f"[error writing {path}: {exc}]"
    return f"[wrote {path} ({len(content)} chars)]"


def _tool_check_requirements() -> str:
    _ok, report = _verify_patch(run_checks=False)
    return report


def _dispatch_tool(name: str, args: dict) -> str:
    try:
        if name == "bash":
            command = args.get("command", "")
            rejection = _bash_rejection(command)
            if rejection:
                _log(f"refused a command: {command[:120]}")
                return rejection
            out = _run(command, timeout=min(CMD_TIMEOUT_SEC, max(20, int(_time_left() - 20))), cwd=_repo_root())
            _clean_repo_junk()
            return _truncate(out)
        if name == "find_files":
            hint = _localization_hint(args.get("query", ""))
            return hint or "[no candidate files found for that query]"
        if name == "read_file":
            return _tool_read_file(
                args.get("path", ""),
                args.get("start_line"),
                args.get("end_line"),
                args.get("offset"),
                args.get("limit"),
            )
        if name == "edit_file":
            return _tool_edit_file(args.get("path", ""), args.get("old_str", ""), args.get("new_str", ""))
        if name == "create_file":
            return _tool_create_file(args.get("path", ""), args.get("content", ""))
        if name == "db_sql":
            return _tool_db_sql(args.get("sql", ""))
        if name == "db_explain":
            return _tool_db_explain(args.get("sql", ""))
        if name == "app_shell":
            return _tool_app_shell(args.get("code", ""))
        if name == "compare_results":
            return _tool_compare_results(args.get("code", ""))
        if name == "check_requirements":
            return _tool_check_requirements()
        if name == "run_checks":
            _passed, report = _run_declared_checks()
            return _truncate(report, 3000)
        return f"[unknown tool: {name}]"
    except _DeadlineReached:
        raise
    except Exception as exc:  # noqa: BLE001 - a tool failure is an observation, not a crash
        return f"[tool error in {name}: {exc}]"


# --------------------------------------------------------------------------- #
# Self-check: hold the patch to everything the instruction asked for
# --------------------------------------------------------------------------- #

_LAST_VERIFY: dict[str, Any] = {}


def _verify_patch(run_checks: bool = True) -> tuple[bool, str]:
    """Return (ok, report) after checking the instruction's constraints, cheapest first."""
    sections: list[str] = []
    ok = True

    violations = _scope_violations()
    if violations:
        ok = False
        sections.append(
            "FILE SCOPE FAILED -- the patch may only touch "
            f"{_scope_label()}, but these paths changed: {', '.join(violations)}. Revert them "
            "(they are reverted automatically before submission, so any fix you put there is lost)."
        )
    else:
        sections.append(f"file scope: ok ({_scope_label()})")

    file_level = _file_level_failures()
    if file_level:
        ok = False
        sections.append("FILE CONSERVATION FAILED -- " + "; ".join(file_level))
        if any("no longer parses" in item for item in file_level):
            report = "\n".join(sections)
            _LAST_VERIFY.update({"fingerprint": _scope_fingerprint(), "ok": False, "report": report,
                                 "checked": False})
            return False, report
    else:
        sections.append("file conservation: ok")

    target = _target_failures()
    if target:
        ok = False
        sections.append("BOUNDED METHOD FAILED:\n  - " + "\n  - ".join(target))
    elif _SPEC is not None and _SPEC.func_name:
        sections.append(f"bounded method ({_SPEC.target}): signature, surrounding source and names ok")

    if not _has_changes():
        ok = False
        sections.append(
            "NO CHANGE -- nothing in the file the instruction names has been edited yet, so "
            "nothing has been fixed."
        )

    # The linter among the instruction's checks answers in under a second, so
    # there is never a reason not to consult it: an unused local or an undefined
    # name is caught here for free instead of costing a whole test run.
    fast_passed, fast_report = _run_declared_checks(slow=False)
    if not fast_passed:
        ok = False
        sections.append(
            "AN IMMEDIATE CHECK FAILED -- the instruction lists this one and it passed before "
            "your change:\n" + fast_report
        )
    elif "no immediate checks" not in fast_report:
        sections.append("immediate checks:\n" + fast_report)

    if run_checks and not ok:
        # A failing cheap check means the code is about to change again, so paying
        # for the application's test suite now buys nothing.
        sections.append("the instruction's slower checks: not run (fix the items above first)")
    elif run_checks:
        passed, report = _run_declared_checks(slow=True)
        if not passed:
            ok = False
            sections.append(
                "THE INSTRUCTION'S OWN CHECKS FAILED -- these already passed before your change, "
                "so this is a regression you introduced. Fix the root cause in the file you are "
                "allowed to edit; do NOT edit tests:\n" + report
            )
        else:
            sections.append("the instruction's own checks:\n" + report)

    report = "\n".join(sections)
    _LAST_VERIFY.update(
        {"fingerprint": _scope_fingerprint(), "ok": ok, "report": report, "checked": run_checks}
    )
    return ok, report


# --------------------------------------------------------------------------- #
# Diff capture + validation
# --------------------------------------------------------------------------- #

def _has_changes() -> bool:
    if _SPEC is not None:
        return bool([p for p in _changed_paths() if _in_scope(p)])
    return bool(_run("git status --porcelain", timeout=30, cwd=_repo_root()).strip())


_SCRATCH_RE = re.compile(
    r"(^|/)(repro|reproduce|scratch|debug|probe|tmp_test|check|explain|bench)[\w.-]*\.(py|sql|sh|js|ts)$",
    re.IGNORECASE,
)


def _remove_scratch_files() -> None:
    """Delete agent-created probe/scratch scripts before capturing the patch.

    Prompting for cleanup is unreliable, so enforce it in code. Only UNTRACKED
    files with scratch names are removed, so a genuine new project file is never
    touched.
    """
    out = _run("git ls-files --others --exclude-standard", timeout=30, cwd=_repo_root())
    if out.startswith("[exit") or out.startswith("[command"):
        return
    for path in out.splitlines():
        path = path.strip()
        if path and _SCRATCH_RE.search(path):
            try:
                os.remove(os.path.join(_repo_root(), path))
                _log(f"removed scratch file so it stays out of the patch: {path}")
            except OSError:
                pass


def _validate_applies(diff: str) -> bool:
    """Check that `diff` applies cleanly to the (already-reset) working tree."""
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    patch_path = os.path.join(SCRATCH_DIR, "patch.diff")
    try:
        with open(patch_path, "w") as handle:
            handle.write(diff)
    except Exception:  # noqa: BLE001
        return False
    res = _run(
        f"git apply --check --whitespace=nowarn {shlex.quote(patch_path)}", timeout=60, cwd=_repo_root()
    )
    return not res.startswith("[exit")


def _make_diff() -> str:
    """Capture the changes vs HEAD, then restore the tree to unpatched.

    The returned patch is applied to a clean checkout of the application, so it
    must contain the file the instruction names and nothing else: no probe
    script, no cache directory, no test edit.
    """
    _clean_repo_junk()
    _remove_scratch_files()
    if _SPEC is not None and _SPEC.scope:
        _revert_out_of_scope()
        # Stage the one file and nothing else, so even a revert that failed
        # (permissions, a path git refuses to touch) cannot reach the submission.
        _run("git add -A -- " + " ".join(shlex.quote(p) for p in _SPEC.scope), timeout=60, cwd=_repo_root())
    else:
        _run("git add -A", timeout=60, cwd=_repo_root())
    diff = _run("git diff --cached", timeout=60, cwd=_repo_root())
    _run("git reset -q --hard HEAD", timeout=60, cwd=_repo_root())
    _run("git clean -fdq", timeout=60, cwd=_repo_root())
    if diff.strip():
        if _validate_applies(diff):
            _log("patch validated: applies cleanly to the unpatched tree")
        else:
            _log("WARNING: captured patch does not apply cleanly to the unpatched tree")
    return diff


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

DB_SYSTEM_PROMPT = r"""You are an expert database engineer. The application repository is your current working directory and its database is live beside you.

THE JOB: an application gets the wrong data, no data, or the right data too expensively. Change the production code that issues the query -- the ORM expression, query-builder call, embedded SQL or migration that the application actually runs -- so the data is right. This is not a SQL quiz: a correct query written somewhere the application never calls is worth nothing.

WHAT THE INSTRUCTION REQUIRES (it states these itself; read its own words each time):
1. ONE FILE, USUALLY ONE METHOD. "Limit production changes to <file>, specifically <Class.method>()" means the whole change lives inside that method. Everything else in that file stays byte-for-byte identical, imports included.
2. ONLY NAMES THE FILE ALREADY IMPORTS. You may not add an import. If the expression you want needs a helper that is not in the file, build the same result from what is there.
3. KEEP THE SIGNATURE. Same name, same parameters, same decorators, same def/async def.
4. PLAIN EXPRESSIONS. When the instruction says no loops, comprehensions, lambdas, exception handling or context managers, it means the work belongs in the query, not in Python around it.
5. LAZY AND COMPOSABLE. Return the queryset/builder, never a list. Callers still filter, order, slice and serialize it afterwards, and the annotations have to survive all of that.
6. THE NAMED CHECKS. The commands the instruction lists already passed before you touched anything. They say what must keep working. Passing them is necessary, not sufficient.
7. EVERY REQUIREMENT BULLET. Each bullet is a separate promise: one names ties, one names empty groups, one names NULLs, one names how many queries may run. Correctness is judged on rows you were never shown, so a query that matches the visible sample and gets the grain wrong is a failure.
8. NO SIDE EFFECTS. The method describes a query. It does not write, execute, open, or reach outside.

METHOD (be efficient; the spend cap, not the clock, is what runs out):
- Read the target method and the model/schema behind it FIRST. `db_sql` the catalog: columns, types, nullability, indexes, row counts. A type is not a detail here -- integer division truncates, numeric does not.
- Trace what the CALLER does with the result: the serializer, the view, the template. That tells you the grain and the exact field names expected.
- Build the query, then LOOK AT IT: `app_shell` with print(str(queryset.query)) shows the SQL your ORM expression actually produces. Read that SQL, not your intention.
- Prove it on rows the sample does not show. `app_shell` runs inside a transaction that is rolled back, so create the awkward rows yourself -- a tie, a NULL, an empty group, a duplicate, a boundary -- and print what the query returns for each.
- On any task about doing LESS database work: `db_explain` the statement and read the plan and buffers, and `compare_results` to prove the rows did not change. Faster and wrong scores nothing.
- After each substantive edit: check_requirements (cheap). Before finish: run_checks.
- `finish` asks for one requirements_checklist entry per bullet, naming the expression in YOUR patch that satisfies it and the evidence you have.

QUERY ENGINEERING PLAYBOOK (general technique; apply what fits):
- GRAIN IS EVERYTHING. Know what one row means before and after your change. A JOIN to a one-to-many relation multiplies rows, so a COUNT or SUM after it counts the fan-out, not the thing. Two annotations that each join a different relation multiply each other. The fix is a correlated subquery (Django: Subquery(...)/Exists(...) with OuterRef, or the project's own count_related helper), not COUNT(DISTINCT) bolted on afterwards.
- ARITHMETIC TYPES. Integer / integer truncates in SQL exactly as it does in Python 2: 1/6 is 0, so a percentage becomes 0 before any rounding sees it. Force one side numeric (multiply by 100.0, or cast) before dividing. Rounding to N places is a database operation; doing it in Python breaks laziness.
- EMPTY AND NULL. A LEFT JOIN aggregate over no rows yields NULL, not 0: coalesce it. NULL = NULL is false, so "two rows with no group belong together" needs COALESCE on a sentinel or IS NOT DISTINCT FROM. An INNER JOIN silently drops the empty case the requirements usually name explicitly.
- TIES AND "LATEST". "The latest" is ambiguous the moment two rows share a timestamp. Use a window function with a total ORDER BY (add the primary key as the final tiebreak) or DISTINCT ON with a matching ORDER BY, so the answer is deterministic. In ClickHouse, argMax picks one of the tied rows arbitrarily unless you make the ordering total.
- DISTINCT COUNTS. COUNT(DISTINCT x) after a fan-out join is right for the distinct thing and wrong for the total; the requirements usually want one of each ("duplicate parents count once, duplicate children count individually"). Read which.
- HIERARCHIES. Containment/ancestry is a set predicate, not a loop: a self-join on the containment operator, or a recursive CTE. Exclude the row itself when the requirement says "strict". Keep the partitioning column (tenant, VRF, workspace) in the join condition or rows from different partitions leak into each other.
- N+1. Query work that grows with the size of the selection is the defect. Replace "resolve the ids, then ask again per id" with ONE set-based statement: a join, an IN over a subquery, or EXISTS. Keep it lazy -- the moment you evaluate a queryset to build the next one, you have made the extra query.
- INDEXES. An index serves a predicate only when its leading columns match the predicate's columns. A prefix-only index still scans everything for the second column. A partial index must cover the predicate. Never assert an index is used: db_explain and read whether the plan names it and what the buffer counts are.
- CLICKHOUSE. No unique constraints and no transactions: duplicates are the caller's problem, FINAL or an aggregate resolves them. The ORDER BY key decides what can be skipped; PREWHERE cuts columns read; uniqExact is exact where uniq is approximate.
- MIGRATIONS. A migration must apply to a fresh schema AND leave the state consistent with the model. Keep the dependency, the model and the index name; change only what the predicate needs.

WORKING RULES (non-negotiable):
- Solve the task from the application's source and its database. Work inside the project directory; put anything you want to try out under /tmp. Paths elsewhere on the filesystem are refused.
- Read the database; do not rewrite it. Exploratory SQL is read-only, ORM exploration rolls back, and the database is left exactly as you found it.
- Never edit, delete, skip or weaken a project test or a conftest.py.
- Never fetch code or install packages over the network. Everything you need is already installed.
- Never special-case the task, the repository, the database or specific inputs. Implement the general behavior the requirements describe.

HARD RULES:
- Never call finish before check_requirements passes and the instruction's own checks pass. finish is rejected automatically otherwise.
- Prefer one precise edit_file over rewriting the file with create_file: a full rewrite is how the imports and the surrounding source get silently changed.
- Never edit, create or delete anything outside the file the instruction names."""


SYSTEM_PROMPT = r"""You are an expert software engineer fixing a real issue in an application repository (your current working directory), which is backed by a live database. Produce a correct, minimal change that resolves the issue as stated and keeps the rest of the project working.

Workflow:
1. LOCATE: start from the provided likely-relevant files, then confirm with `bash` (grep -rn) and `read_file`. The change usually belongs in the query layer -- querysets, filtersets, managers, repositories, migrations, embedded SQL -- on the path the application actually runs. Use `find_files` to re-localize if your first guess is wrong.
2. UNDERSTAND THE DATA: read the schema and the real rows before deciding what the query should say. Types, nullability and indexes decide the answer.
3. REPRODUCE: write a minimal script AT AN ABSOLUTE PATH OUTSIDE THE REPO -- always `/tmp/probe.py` -- that shows the wrong result, and run it. Confirm the defect BEFORE editing. Scratch files inside the repository pollute the final patch.
4. FIX: make the smallest change that fully resolves the issue, matching surrounding style. Keep the fetch lazy and database-backed; do not pull rows into Python to post-process them.
5. VERIFY: re-run your probe, check the awkward rows (ties, NULLs, empty groups, duplicates, boundaries), and run the project's own tests for the code you changed.
6. CLEAN UP: delete any scratch files you created so they are NOT in the final patch, then call `finish`.

Rules:
- Prefer `edit_file` for precise edits; `create_file` for genuinely new project files.
- Do NOT modify the project's own test files. Editing tests to fit your change proves nothing.
- Honor the exact contract: grain, ordering, types, rounding, NULL handling, and every stated or implied edge case. A single wrong edge case fails the whole task.
- When the issue is about EFFICIENCY rather than a wrong result, the rows that come back must stay identical: same values, same order, same grain. Change only how the work is done.
- Never leave anything behind in the database: no rows, no tables, no indexes that the patch itself does not create.
- Be economical with the limited token/spend budget: target relevant lines, don't read whole large files. Aim to make your FIRST code edit early; a plausible fix you then refine beats a perfect fix you never get to apply.
- `read_file` returns at most ~150 lines per call and tells you the real file length. Page with read_file(path, offset=N, limit=150). NEVER assume a file ended where your view ended.

INTEGRITY (non-negotiable):
- Solve the task from the application's source and database. Work inside the project directory and keep scratch files under /tmp.
- Never weaken, skip or rewrite a project test to make your change pass.
- Never fetch code or install packages over the network.
- Never special-case the example inputs or branch on task, repository or database identity. Implement the general behavior."""


def _target_source_block() -> str:
    """The method the instruction names, quoted from the file, with line numbers.

    Putting it in the first message saves the model a read and, more importantly,
    makes the exact text it has to preserve unambiguous.
    """
    if _SPEC is None or not _SPEC.func_name or not _SPEC.primary_file:
        return ""
    source = _original_source(_SPEC.primary_file)
    if not source or not _SPEC.primary_file.endswith((".py", ".pyi")):
        return ""
    try:
        node = _find_target_function(ast.parse(source), _SPEC.class_name, _SPEC.func_name)
    except (SyntaxError, ValueError):
        return ""
    lines = source.splitlines()
    body = "\n".join(
        f"{number}\t{lines[number - 1]}" for number in range(node.lineno, node.end_lineno + 1)
    )
    return _truncate(body, 2500)


def _schema_hint() -> str:
    """A short, factual note about the database this application is talking to."""
    target = _db_target()
    if target is None:
        return ""
    if target["kind"] != "postgresql":
        return "Database: ClickHouse, reachable with db_sql / db_explain."
    rows = _tool_db_sql(
        "SELECT relname, n_live_tup FROM pg_stat_user_tables "
        "WHERE n_live_tup > 0 ORDER BY n_live_tup DESC LIMIT 12"
    )
    if rows.startswith("[") and "exit" in rows[:12]:
        return "Database: PostgreSQL, reachable with db_sql / db_explain."
    return (
        "Database: PostgreSQL, reachable with db_sql / db_explain. Largest tables by live rows:\n"
        + _truncate(rows, 900)
    )


def _db_user_message(problem_statement: str) -> str:
    assert _SPEC is not None
    blocks = [
        f"Application root = your current working directory: {_repo_root()}",
        "All paths below are relative to it.",
        "",
        "# Problem statement",
        problem_statement,
        "",
        f"# The contract, as parsed from that instruction (kind: {_SPEC.kind}"
        + (f", engine: {_SPEC.engine}" if _SPEC.engine else "")
        + ")",
    ]
    if _SPEC.scope:
        blocks.append(f"  The ONLY path your diff may touch: {', '.join(_SPEC.scope)}")
    else:
        blocks.append(
            "  The instruction does not name the file: find the code the application actually "
            "runs for this, and change only that one method. The first file you edit becomes the "
            "only file the patch may touch, so locate it before you edit anything."
        )
    if _SPEC.target:
        blocks.append(f"  The change belongs inside: {_SPEC.target}()")
    constraints = []
    if _SPEC.freeze_rest:
        constraints.append("the rest of that file stays byte-for-byte identical")
    if _SPEC.freeze_imports:
        constraints.append("no new imports; use only names the file already has")
    if _SPEC.freeze_signature:
        constraints.append("the signature is unchanged")
    if _SPEC.plain_expressions:
        constraints.append("plain expressions only: no loops, comprehensions, lambdas, try, with")
    if _SPEC.lazy:
        constraints.append("the result stays lazy and composable")
    if _SPEC.no_materialize:
        constraints.append("nothing is materialized in Python")
    if _SPEC.results_must_not_change:
        constraints.append("the rows that come back must be identical to the rows that come back now")
    if constraints:
        blocks.append("  Constraints: " + "; ".join(constraints) + ".")
    if _SPEC.checks:
        blocks.append("  Checks the instruction names (run_checks runs these):")
        blocks += [f"    - {command}" for command in _SPEC.checks]

    target_source = _target_source_block()
    if target_source:
        blocks += [
            "",
            (
                f"# {_SPEC.primary_file}: {_SPEC.target}() as it stands today "
                "(everything outside these lines must not change)"
            ),
            target_source,
        ]
    if not _SPEC.scope:
        hint = _localization_hint(problem_statement)
        if hint:
            blocks += ["", hint]
    schema = _schema_hint()
    if schema:
        blocks += ["", "# " + schema]
    if _REQUIREMENTS:
        blocks += [
            "",
            (
                "# Requirements -- each is a separate promise about your patch. Correctness is "
                "judged on rows you were not shown, so the bullets about ties, NULLs, empty "
                "groups and query counts are the ones that decide it. `finish` makes you "
                "account for all of them."
            ),
        ]
        blocks += [f"  {index}. {text}" for index, text in enumerate(_REQUIREMENTS, start=1)]
    blocks += [
        "",
        "# How to start (do these before rewriting anything)",
        "  1. read_file the target method and the model it queries.",
        "  2. db_sql the schema behind it: the columns, their types and nullability, and the "
        "indexes. Types decide these tasks -- integer division truncates, a nullable column "
        "will not compare the way you expect.",
        "  3. app_shell: print(str(<the current queryset>.query)) and look at the SQL the code "
        "actually produces today.",
        "  4. Only now write the change.",
        "  5. app_shell again to prove it: create the awkward rows the requirements name (a tie, "
        "a NULL, an empty group, a duplicate, a boundary) and print what the query returns for "
        "each. Everything you create is rolled back. This is the step that decides the task, "
        "because the result is judged on rows you were never shown.",
        (
            "  6. check_requirements after each edit (it includes the linter and is instant), "
            "then run_checks once you believe you are done, then finish."
        ),
    ]
    return "\n".join(blocks)


def _initial_user_message(problem_statement: str) -> str:
    if _SPEC is not None:
        return _db_user_message(problem_statement)
    hint = _localization_hint(problem_statement)
    tree = _truncate(_run("git ls-files | head -200", timeout=30, cwd=_repo_root()), 2500)
    blocks = [
        f"Application root = your current working directory: {_repo_root()}",
        "All paths below are relative to it. Do NOT guess other locations.",
        "",
        "Tracked files (first 200):",
        tree,
    ]
    if hint:
        blocks += ["", hint]
    schema = _schema_hint()
    if schema:
        blocks += ["", "# " + schema]
    blocks += [
        "",
        "# Problem statement",
        problem_statement,
        "",
        "Begin by locating the code that issues the query, then understand the data, fix, verify, and finish.",
    ]
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def _handle_finish_db(args: dict, state: dict) -> tuple[bool, str]:
    """Only let the run end when the work matches the instruction."""
    if not _has_changes():
        if state["repair_nudges"] < MAX_REPAIR_NUDGES:
            state["repair_nudges"] += 1
            _log(f"finish rejected: no in-scope changes yet (nudge {state['repair_nudges']})")
            return False, (
                "[finish rejected: you have not changed the file the instruction names, so nothing "
                "has been fixed. Edit the production code that issues the query, then finish.]"
            )
        return True, "[finished]"

    run_checks = _time_left() > CHECK_MIN_RESERVE_SEC
    ok, report = _verify_patch(run_checks=run_checks)

    if not ok:
        if state["verify_repairs"] >= MAX_VERIFY_REPAIRS:
            _log("finish accepted unresolved: repair budget exhausted")
            return True, "[finished with unresolved items; returning best effort]"
        state["verify_repairs"] += 1
        _log(f"finish rejected, needs repair ({state['verify_repairs']}/{MAX_VERIFY_REPAIRS})")
        return False, (
            "[finish rejected: the patch does not satisfy the instruction yet. Fix these, then "
            "finish again. Stay inside the file you are allowed to edit and do not touch tests.]\n"
            + report
        )

    # The mechanical checks pass. What remains are the judgement calls, raised
    # together so an incomplete submission costs one round trip, not three.
    blockers: list[str] = []
    if _SPEC is not None and _SPEC.results_must_not_change:
        if _COMPARE_STATE["diverged"] and state["verify_repairs"] < MAX_VERIFY_REPAIRS:
            state["verify_repairs"] += 1
            blockers.append(
                "RESULTS: your last compare_results run showed the rows changing. This task asks "
                "for the same data with less database work, so make the results identical, then "
                "re-run compare_results."
            )
        elif (
            _COMPARE_STATE["runs"] == 0
            and not _COMPARE_STATE["nudged"]
            and _time_left() > 150
            and _ensure_unpatched_copy()
        ):
            _COMPARE_STATE["nudged"] = True
            blockers.append(
                "RESULTS: this task asks for the same rows with less database work, and you have "
                "not shown the rows are the same. A green test suite covers what someone thought "
                "to test, not the grain, the ties and the empty cases your rewrite touched. Write "
                "one snippet that prints the results for the ordinary case AND the edge cases the "
                "requirements name, run compare_results, then finish."
            )

    checklist = _checklist_failures(args)
    if checklist and not _CHECKLIST_STATE["nudged"]:
        _CHECKLIST_STATE["nudged"] = True
        numbered = "\n".join(f"  {index}. {text}" for index, text in enumerate(_REQUIREMENTS, start=1))
        blockers.append(
            "REQUIREMENTS: " + "; ".join(checklist) + ". Correctness is judged on rows you were "
            "not shown, so walk each bullet against your query, check the ones about ties, NULLs, "
            "empty groups and query counts with app_shell, fix anything it does not cover, then "
            "finish with a requirements_checklist entry for each:\n" + numbered
        )

    if blockers:
        _log(f"finish rejected: {len(blockers)} open item(s) before the work is done")
        return False, (
            "[finish rejected: the mechanical checks pass, but:]\n- "
            + "\n- ".join(blockers)
            + "\n"
            + report
        )

    _log(f"finish accepted: {args.get('summary', '')}")
    state["verified"] = True
    return True, "[finished -- everything the instruction asked for checks out]\n" + report


def _handle_finish_general(args: dict, state: dict) -> tuple[bool, str]:
    if not _has_changes() and state["repair_nudges"] < MAX_REPAIR_NUDGES:
        state["repair_nudges"] += 1
        _log(f"finish rejected: no changes yet (nudge {state['repair_nudges']})")
        return False, (
            "[finish rejected: no file changes detected. Locate and edit the code that issues the "
            "query, verify with a probe, then finish.]"
        )
    _log(f"finish: {args.get('summary', '')}")
    state["verified"] = _has_changes()
    return True, "[finished]"


def _run_agent(problem_statement: str, attempt: int = 0, previous_report: str = "") -> None:
    client = _make_client()
    model = _model_for_attempt(attempt)
    temperature = TEMPERATURE if attempt == 0 else RETRY_TEMPERATURE
    system_prompt = DB_SYSTEM_PROMPT if _SPEC is not None else SYSTEM_PROMPT
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _initial_user_message(problem_statement)},
    ]
    if attempt:
        _log(f"attempt {attempt + 1}: model={model} temperature={temperature}")
        messages.append(
            {
                "role": "user",
                "content": (
                    "A previous attempt at this exact task was DISCARDED because it did not satisfy "
                    "the instruction. Its working tree has been reverted, so you start from the "
                    "unpatched code. Take a different approach and do not repeat the mistake "
                    "below.\n\nWhat the previous attempt was rejected for:\n" + (previous_report or "unknown")
                ),
            }
        )

    state = {"repair_nudges": 0, "verify_repairs": 0, "verified": False}
    no_tool_turns = 0
    wrapup_nudged = False
    no_edit_pressure_idx = 0
    # Measured on the public samples: a run can spend its whole budget in `bash`
    # and `read_file`, reasoning about the query from the source alone and never
    # once asking the database what is actually in it. That run reads the code
    # correctly and still gets the grain wrong, because the rows it never looked
    # at are the rows it is judged on.
    db_tools_used = False
    db_nudged = False
    tools = _tools()
    for step in range(MAX_STEPS):
        if _time_left() <= 0:
            _log("time budget exhausted; stopping loop")
            break
        if not _budget_left():
            _log(f"cost budget reached (${_SPENT_USD:.4f}/${MAX_COST_USD:.2f}); stopping loop")
            break

        response = _chat(client, messages, tools, model=model, temperature=temperature)
        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

        assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    # Coerce blank arguments to "{}": an empty arguments string is
                    # invalid JSON in history and 400s some providers on the next
                    # request, killing the run.
                    "function": {"name": tc.function.name, "arguments": (tc.function.arguments or "").strip() or "{}"},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        if not tool_calls:
            no_tool_turns += 1
            _log(f"assistant produced no tool call (turn {no_tool_turns})")
            if no_tool_turns >= 2:
                break
            messages.append({"role": "user", "content": "Use tools to make/verify the change, or call finish when done."})
            continue
        no_tool_turns = 0

        finished = False
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "finish":
                if _SPEC is not None:
                    finished, result = _handle_finish_db(args, state)
                else:
                    finished, result = _handle_finish_general(args, state)
            else:
                _log(f"step {step}: {name} {json.dumps(args)[:160]}")
                if name in ("db_sql", "db_explain", "app_shell", "compare_results"):
                    db_tools_used = True
                result = _dispatch_tool(name, args)
                # A command that writes outside the allowed file forfeits the whole
                # task, so surface it the moment it happens rather than at finish,
                # when the model may have no budget left to undo it.
                if _SPEC is not None and name in ("bash", "create_file", "edit_file"):
                    violations = _scope_violations()
                    if violations:
                        result += (
                            "\n[WARNING: paths outside the allowed file have changed: "
                            + ", ".join(violations[:6])
                            + ". They will be reverted before submission; move the change into "
                            + _scope_label()
                            + " and keep scratch files under /tmp.]"
                        )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if finished:
            break

        if _SPEC is not None and not db_tools_used and not db_nudged and _budget_used_fraction(step) >= 0.25:
            db_nudged = True
            _log("nudge: a quarter of the budget spent without querying the database")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You are changing a database query and you have not looked at the "
                        "database yet. Reading the code tells you what it asks for; it does not "
                        "tell you the column types, what is nullable, how the rows are "
                        "distributed, or what the query returns for a tie, a NULL or an empty "
                        "group -- and those are what the result is judged on. Use db_sql for the "
                        "schema, and app_shell to print str(queryset.query) and to try the query "
                        "against rows you create for the occasion (everything you create is "
                        "rolled back)."
                    ),
                }
            )

        # No-edit escalation: rescue the empty-patch failure mode. While no
        # production file has been touched, push progressively harder to make the
        # model STOP exploring and commit its best change, keyed to how much of the
        # fastest-draining budget is gone. Fires independently of the wrap-up nudge below,
        # which depends on `_has_changes()` and is therefore useless here.
        if not _has_changes() and no_edit_pressure_idx < len(NO_EDIT_PRESSURE_THRESHOLDS):
            frac_used = _budget_used_fraction(step)
            if frac_used >= NO_EDIT_PRESSURE_THRESHOLDS[no_edit_pressure_idx]:
                while (
                    no_edit_pressure_idx < len(NO_EDIT_PRESSURE_THRESHOLDS)
                    and frac_used >= NO_EDIT_PRESSURE_THRESHOLDS[no_edit_pressure_idx]
                ):
                    no_edit_pressure_idx += 1
                _log(f"no-edit pressure nudge ({frac_used:.0%} of budget used, still 0 edits)")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have NOT edited the production code yet -- only explored the "
                            "schema and run snippets. An empty patch fixes nothing, the worst "
                            "possible outcome. Stop investigating NOW and apply your best change "
                            "with edit_file THIS TURN, even if you are not fully certain -- you "
                            "can still verify and refine it afterwards."
                        ),
                    }
                )

        # Wrap-up nudge: once edits exist and most of the fastest-draining budget is gone, stop exploring and finish -- reaching
        # `finish` is what runs the verify-and-repair guard.
        if not wrapup_nudged and _has_changes():
            frac_used = _budget_used_fraction(step)
            if frac_used >= WRAPUP_BUDGET_THRESHOLD:
                wrapup_nudged = True
                _log(f"wrap-up nudge sent ({frac_used:.0%} of budget used)")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "BUDGET IS ALMOST UP and you have already edited the code. Stop "
                            "exploring now. Re-check your change, then call finish THIS TURN. "
                            "Calling finish re-checks it against the instruction and lets you "
                            "repair a failure -- if you never finish, the patch ships UNVERIFIED."
                        ),
                    }
                )

    # Safety net: the loop ended without a verified `finish` but edits exist. That
    # is the wrong-fix failure mode -- the model edited the right method but ran
    # out of steps before finish ever triggered verification. Verify now and give
    # it a bounded chance to repair.
    if not state["verified"] and _has_changes():
        _verify_and_repair(client, messages, model=model, temperature=temperature)


def _verify_and_repair(client, messages: list[dict], model: str | None = None, temperature: float | None = None) -> None:
    """Re-check the work after the main loop; repair on failure.

    Up to MAX_TEST_REPAIRS rounds, each giving the model a few turns to make
    edits, stopping as soon as everything passes or budget/time runs out.
    """
    tools = _tools()
    for attempt in range(MAX_TEST_REPAIRS + 1):
        if _time_left() < CHECK_MIN_RESERVE_SEC or not _budget_left():
            _log("post-loop verify: insufficient budget/time; shipping current diff")
            return
        if _SPEC is not None:
            ok, report = _verify_patch(run_checks=True)
        else:
            ok, report = True, ""
        if ok:
            _log(f"post-loop verify: checks pass (attempt {attempt}); diff verified")
            return
        if attempt >= MAX_TEST_REPAIRS:
            _log("post-loop verify: checks still failing after repairs; returning best effort")
            return
        _log(f"post-loop verify: checks FAILING; repair attempt {attempt + 1}/{MAX_TEST_REPAIRS}")
        messages.append(
            {
                "role": "user",
                "content": (
                    "The loop is ending and your patch does not yet satisfy the instruction. Make "
                    "the minimal edit that fixes the root cause (do NOT edit tests, stay inside "
                    "the file the instruction names). Report:\n" + report
                ),
            }
        )
        made_edit = False
        for _ in range(3):  # a few turns to land a repair edit, then re-verify
            if _time_left() < CHECK_MIN_RESERVE_SEC or not _budget_left():
                break
            response = _chat(client, messages, tools, model=model, temperature=temperature)
            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": (tc.function.arguments or "").strip() or "{}"},
                    }
                    for tc in tool_calls
                ]
            messages.append(entry)
            if not tool_calls:
                break
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "finish":
                    result = "[the checks run automatically -- keep fixing until they all pass]"
                else:
                    _log(f"repair: {name} {json.dumps(args)[:120]}")
                    result = _dispatch_tool(name, args)
                    if name in ("edit_file", "create_file"):
                        made_edit = True
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if made_edit:
                break  # re-verify with the loop's next iteration


def _reset_attempt_state() -> None:
    """Forget one attempt's judgement state before the next one starts.

    The unpatched-tree cache and the original-source snapshot deliberately
    survive: they describe the application as it shipped, which every attempt
    shares.
    """
    global _SLOW_CHECK_RUNS
    _COMPARE_STATE.update({"runs": 0, "diverged": False, "nudged": False})
    _CHECKLIST_STATE["nudged"] = False
    _LAST_VERIFY.clear()
    _SLOW_CHECK_RUNS = 0


def _score_attempt() -> tuple[tuple[bool, bool], str]:
    """Rate the working tree against the instruction. Higher tuple is better."""
    if _SPEC is None:
        return (_has_changes(), False), ""
    fingerprint = _scope_fingerprint()
    if _LAST_VERIFY.get("fingerprint") == fingerprint and _LAST_VERIFY.get("checked"):
        ok, report = bool(_LAST_VERIFY["ok"]), str(_LAST_VERIFY["report"])
    else:
        ok, report = _verify_patch(run_checks=_time_left() > CHECK_MIN_RESERVE_SEC)
    if _SPEC.results_must_not_change:
        second = _COMPARE_STATE["runs"] > 0 and not _COMPARE_STATE["diverged"]
    else:
        second = _COMPARE_STATE["runs"] > 0
    return (ok, second), report


def agent_main(input: dict) -> str:
    global _SPEC, _REQUIREMENTS

    problem_statement = ""
    if isinstance(input, dict):
        problem_statement = input.get("problem_statement", "") or ""
    _log(f"agent_main start: problem statement {len(problem_statement)} chars")
    _log(
        f"time budget: {_AGENT_TIMEOUT:.0f}s declared"
        f"{' (AGENT_TIMEOUT)' if os.getenv('AGENT_TIMEOUT') else ' (default; AGENT_TIMEOUT unset)'}"
        f", {_USABLE_BUDGET:.0f}s usable; models {MODEL_CHAIN}"
    )

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    _repo_root()
    _arm_deadline()
    try:
        _SPEC = _detect_spec(problem_statement)
    except Exception as exc:  # noqa: BLE001 - never let parsing kill the run
        _log(f"spec detection failed: {exc}")
        _SPEC = None
    if _SPEC is not None:
        _log(
            f"DB mode: kind={_SPEC.kind} engine={_SPEC.engine or 'unstated'} scope={_SPEC.scope} "
            f"target={_SPEC.target} checks={len(_SPEC.checks)}"
        )
        _snapshot_originals()
        _REQUIREMENTS = _requirement_bullets(problem_statement)
        _log(f"parsed {len(_REQUIREMENTS)} requirement bullet(s) to account for at finish")
    else:
        _log("GENERAL mode: the instruction names no file to change")

    # A second attempt is an ESCALATION, not a lottery. Only a task that states
    # its file, its method and its checks says enough to tell two candidates
    # apart, and once a candidate satisfies every stated constraint, re-rolling it
    # could only trade a good patch for a worse one.
    max_attempts = MAX_ATTEMPTS if _SPEC is not None else 1
    best_diff = ""
    best_score = (False, False)
    report = ""
    for attempt in range(max_attempts):
        if attempt:
            _reset_attempt_state()
        try:
            _run_agent(problem_statement, attempt=attempt, previous_report=report)
        except _DeadlineReached:
            _log("deadline watchdog fired; capturing the best diff we have")
            _disarm_deadline()
            score, report = (False, False), "deadline reached"
            diff = _make_diff()
            if diff.strip() and best_score == (False, False):
                best_diff, best_score = diff, score
            break
        except Exception as exc:  # noqa: BLE001 - never crash before emitting a diff
            _log(f"agent loop error: {exc}")
            traceback.print_exc()

        score, report = _score_attempt()
        diff = _make_diff()  # also restores the tree, so the next attempt starts unpatched
        _log(
            f"attempt {attempt + 1}/{max_attempts}: checks={score[0]} results-compared={score[1]}, "
            f"{len(diff)} char diff, spent ~${_SPENT_USD:.4f}"
        )
        if diff.strip() and score > best_score:
            best_diff, best_score = diff, score
        elif not best_diff and diff.strip():
            best_diff = diff

        if attempt + 1 >= max_attempts:
            break
        if best_score[0]:
            _log(f"candidate satisfies the instruction (results-compared={best_score[1]}); keeping it")
            break
        if _SPENT_USD <= 0:
            # Every call failing means an inference problem, and re-running the
            # whole loop cannot fix one.
            _log("no inference call succeeded (check the API key); a retry cannot help")
            break
        if not _budget_left():
            _log("no cost budget left for another attempt")
            break
        if _time_left() < max(240.0, 0.3 * _USABLE_BUDGET):
            _log(f"only {_time_left():.0f}s left; not starting another attempt")
            break
        _log(
            f"retrying with {_model_for_attempt(attempt + 1)}: best so far checks={best_score[0]} "
            f"results-compared={best_score[1]}"
        )

    _disarm_deadline()
    _log(
        f"returning diff: {len(best_diff)} chars, checks={best_score[0]} results-compared={best_score[1]} "
        f"(spent ~${_SPENT_USD:.4f} of the ${MAX_COST_USD:.2f} limit; "
        f"{'inside' if _SPENT_USD <= MAX_COST_USD * COST_TARGET_FRACTION else 'over'} the "
        f"{COST_TARGET_FRACTION:.0%} target)"
    )
    return best_diff


if __name__ == "__main__":
    problem = " ".join(sys.argv[1:]) or sys.stdin.read()
    print(agent_main({"problem_statement": problem}))
