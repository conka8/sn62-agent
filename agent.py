"""An agent that repairs a defect in a checked-out repository and hands in a patch.

`agent_main(input)` is the entry point. It is given a problem statement and a
checkout in the working directory, and returns a unified diff against the
commit the checkout started from. Nothing else is an answer: a change made
anywhere but in those files is not handed in, and an empty diff is no answer at
all.

The design follows one rule. The task statement is the authority, and
everything this run claims has to be traceable to the statement's own words or
to something that actually happened in the checkout. So the statement is read
for the file, the method, the commands and the limits it names; evidence is
recorded against the identity of the source it ran on; and a claim with nothing
behind it is reported as unestablished rather than asserted.

How the file is laid out, in order:

* Settings and prices. Every switch is an environment flag with a default, so
  one build serves every configuration, and a price table turns token counts
  into money.
* `Beacon`. One named slot of the run log per optional behaviour. Each one
  reports that it was reached, skipped or fired, so a run reads as a sequence
  of decisions.
* `Allowance`. What the run may still spend in time and money, and the one
  place that decides when work must stop.
* `Seat`. The model seats, in order, with the ladder that reshapes a refused
  request rather than giving up on it.
* `CheckoutSnapshot` and `Tree`. The checkout: every read and write, the copy
  taken at the start so it can be put back exactly, and the diff that becomes
  the answer.
* Patch readers. Splitting an answer by file, reading what each section does,
  and dropping what the statement puts out of bounds.
* `Shell`, `ShellPool`, `ClickHouseHttp` and `HttpProbe`. Running commands and
  database statements in the background, with lanes so two things never use the
  same resource at once.
* Statement readers. The file, the method, the commands, the requirements, the
  prohibitions and the limits, each read from the statement's literal words.
* `Warden`. Runs the project's own checks and holds the hand-in to the
  statement, quoting the clause behind every refusal.
* `Kit` and `TOOL_SCHEMAS`. The tools the model may call, and the bookkeeping
  that keeps their results honest.
* `BRIEF`. The standing instructions the model works to.
* `run_plan`, `drive` and `agent_main`. The planning seat, the working loop,
  and the assembly of the answer.
"""
from __future__ import annotations
import ast
import bisect
import collections
import decimal
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tokenize
import urllib.error
import urllib.parse
import urllib.request
FALLBACK_BASE_URL = "https://openrouter.ai/api/v1"
TS_PARSE_CHECK = (
    'const fs = require("fs"), m = require("module");'
    'if (typeof m.stripTypeScriptTypes !== "function") {'
    ' console.error("unsupported option: this node has no TypeScript parser"); process.exit(2); }'
    'try { m.stripTypeScriptTypes(fs.readFileSync(process.argv[1], "utf8"), {mode: "transform"}); }'
    'catch (e) { console.error(String((e && e.message) || e)); process.exit(1); }')
SYNTAX_CHECKS = {
    ".py": [sys.executable, "-c",
            "import sys; compile(open(sys.argv[1], 'rb').read(), sys.argv[1], 'exec')"],
    ".js": ["node", "--check"],
    ".mjs": ["node", "--check"],
    ".cjs": ["node", "--check"],
    # `node --check` does not parse TypeScript even with type stripping on, so a
    # broken file passed. Node's own TypeScript transform does parse it (enums,
    # namespaces and parameter properties included); a node without it says so
    # as an unsupported option, which reads as unavailable, not as a parse error.
    ".ts": ["node", "--no-warnings", "-e", TS_PARSE_CHECK],
    ".mts": ["node", "--no-warnings", "-e", TS_PARSE_CHECK],
    ".cts": ["node", "--no-warnings", "-e", TS_PARSE_CHECK],
    ".rb": ["ruby", "-c"],
    ".php": ["php", "-l"],
    ".pl": ["perl", "-c"],
    ".lua": ["luac", "-p"],
    ".sh": ["bash", "-n"],
    ".bash": ["bash", "-n"],
    ".go": ["gofmt", "-e"],
    # Parse-only readers where the image has them; absence stays silent.
    # rustfmt refuses --check together with --emit stdout; without --check it
    # exits 0 on any file that parses, formatted or not.
    ".rs": ["rustfmt", "--edition", "2021", "--emit", "stdout", "--color", "never"],
}
# Default inference model; the runtime may provide an explicit override.
DRIVER_MODEL = os.getenv("RIDGES_AGENT_MODEL") or "openai/gpt-5.6-luna"
# A second seat takes over when the first times out or refuses, so a run never
# ends on one endpoint's bad stretch. The relief is the model this agent ran on
# before, so its prompts and tools are known to work there.
RELIEF_MODEL = os.getenv("RIDGES_RELIEF_MODEL", "~openai/gpt-luna-latest")
# A third seat, asked only when the driver and the relief both stay unavailable;
# cheap, so a provider-side outage of the luna family does not end the run.
LAST_RESORT_MODEL = os.getenv("RIDGES_LAST_RESORT_MODEL", "deepseek/deepseek-v4-flash-0731")
PLAN_MODEL = os.getenv("RIDGES_PLAN_MODEL") or "~openai/gpt-luna-latest"
REVIEW_MODEL = os.getenv("RIDGES_REVIEW_MODEL") or "~openai/gpt-luna-latest"
# Models that refused a request carrying the reasoning field: it is left out of
# their later requests instead of ending the run.
BARE_MODELS: set = set()
# Models that refused the parallel-calls field: it is left out for them.
NO_PARALLEL_MODELS: set = set()
SEAT_CACHE_TERMS = {
    "anthropic/claude-sonnet-5.5": (0.2000e-6, 1_000_000),
    "z-ai/glm-5.3": (0.2600e-6, 1_048_576),
    "xiaomi/mimo-v2.5-pro": (0.0050e-6, 262_144),
    "xiaomi/mimo-v2.5": (0.0050e-6, 262_144),
    "minimax/minimax-m2.5": (0.0500e-6, 204_800),
    "minimax/minimax-m3": (0.0600e-6, 1_048_576),
    "deepseek/deepseek-v4-pro": (0.0036e-6, 1_048_576),
    "deepseek/deepseek-v4-pro-0813": (0.0440e-6, 1_048_576),
    "z-ai/glm-5.3-flash": (0.0300e-6, 262_144),
    "tencent/hy4-preview": (0.0420e-6, 1_048_576),
    "@preset/hy4-nothink": (0.0420e-6, 1_048_576),
    "openai/gpt-5.6-luna": (0.0200e-6, 400_000),
    "openai/gpt-6-luna": (0.0100e-6, 1_050_000),
    "~openai/gpt-luna-latest": (0.0100e-6, 1_050_000),
    "@preset/luna-high": (0.0200e-6, 400_000),
    "qwen/qwen3.8-2.4t-a95b": (0.2500e-6, 1_000_000),
    "@preset/qwen38-24t-lowthink": (0.2500e-6, 1_000_000),
    "openai/gpt-5.6-terra": (0.2000e-6, 400_000),
    "google/gemini-3.7-flash": (0.0375e-6, 1_048_576),
    "deepseek/deepseek-v4-flash-0731": (0.0280e-6, 1_048_576),
    "tencent/hy3": (0.0330e-6, 262_144),
}
UNKNOWN_CACHE_TERMS = (0.1000e-6, 131_072)
MODEL_PRICING = {
    "anthropic/claude-sonnet-5.5": (2.000e-6, 10.000e-6),
    "z-ai/glm-5.3": (1.400e-6, 4.400e-6),
    "qwen/qwen3.8-2.4t-a95b": (2.000e-6, 6.000e-6),
    "@preset/qwen38-24t-lowthink": (2.000e-6, 6.000e-6),
    "xiaomi/mimo-v2.5-pro": (0.600e-6, 1.201e-6),
    "xiaomi/mimo-v2.5": (0.140e-6, 0.280e-6),
    "minimax/minimax-m2.5": (0.150e-6, 0.900e-6),
    "minimax/minimax-m3": (0.300e-6, 1.200e-6),
    "deepseek/deepseek-v4-pro-0813": (1.320e-6, 3.960e-6),
    "z-ai/glm-5.3-flash": (0.150e-6, 0.500e-6),
    "tencent/hy4-preview": (0.834e-6, 2.501e-6),
    "@preset/hy4-nothink": (0.834e-6, 2.501e-6),
    "openai/gpt-5.6-luna": (0.200e-6, 1.200e-6),
    "openai/gpt-6-luna": (0.100e-6, 0.500e-6),
    "~openai/gpt-luna-latest": (0.100e-6, 0.500e-6),
    "@preset/luna-high": (0.200e-6, 1.200e-6),
    "openai/gpt-5.6-terra": (2.000e-6, 12.000e-6),
    "google/gemini-3.7-flash": (0.375e-6, 1.875e-6),
    "deepseek/deepseek-v4-flash-0731": (0.440e-6, 1.320e-6),
    "tencent/hy3": (0.132e-6, 0.528e-6),
}
UNKNOWN_TOKEN_PRICE = (1.0e-6, 4.0e-6)
# What a model family needs from the request beyond the OpenAI wire format every
# seat speaks: Anthropic models refuse a temperature next to reasoning, and
# cache a prompt prefix only when the request marks it.
MODEL_PROFILES = (("anthropic/", {"temperature": False, "cache": True}),)


def model_profile(model: str) -> dict:
    """The request quirks of a model family, by name prefix: an empty profile means the plain
    request shape.
    """
    for prefix, profile in MODEL_PROFILES:
        if (model or "").startswith(prefix):
            return profile
    return {}


def cache_marked(messages: list) -> list:
    """The same messages, with the leading system text marked as a cacheable prefix."""
    marked = list(messages or [])
    for index, message in enumerate(marked):
        if not isinstance(message, dict) or message.get("role") != "system":
            break
        content = message.get("content")
        if isinstance(content, str) and content:
            marked[index] = dict(message, content=[{"type": "text", "text": content,
                                                     "cache_control": {"type": "ephemeral"}}])
    return marked
TRANSCRIPT_SPEND_SHARE = 0.35
TURNS_PLANNED = 50
CHARS_PER_TOKEN = 3.5
TRANSCRIPT_FLOOR_CHARS = 40_000
DEFAULT_COST_LIMIT_USD = 0.29
DEFAULT_WALL_SEC = 1800.0
COST_SHARE = 0.88
WALL_SHARE = 1.00
WALL_RESERVE_SEC = 60.0
FINISH_BUDGET_SEC = 40.0
TURN_CEILING = 150
FIRST_EDIT_DEADLINE_TURN = 8
PLAN_READ_BUDGET = 150_000
PLAN_TURN_CAP = 40
PLAN_SPEND_SHARE = 0.75
PLAN_NOTE_CHARS = 4_000
WRAPUP_TURN = TURN_CEILING // 5
EDIT_PRESSES_MAX = 3
BLANK_REPLY_CEILING = 3
REPEAT_READ_CEILING = 2
REPLY_TOKEN_CEILING = 16000
REASONING_EFFORT = (os.getenv("RIDGES_REASONING_EFFORT") or "high").strip().lower()
REQUEST_ATTEMPTS = 64
RESHAPE_ATTEMPTS = 3
OLD_REQUEST_ATTEMPTS = 4
OLD_REFUSED_SWEEP_BENCH = 2
RETRY_TALLY = {"ladders": 0, "retried": 0, "recovered": 0,
               "old_would_bench": 0, "old_would_exhaust": 0,
               "walled": 0, "quiet": 0, "unreachable": 0, "unusable": 0,
               "limited": 0,
               "absent": 0}
SEAT_REFUSED_WAIT_SEC = 20.0
SEAT_CALL_TIMEOUT_SEC = 240.0
CALL_CLOCK_SHARE = 0.34
SEAT_RETRY_FLOOR_SEC = 20.0
SEAT_RETRY_NAP_CEILING_SEC = 15.0
SEAT_QUIET_HANDOVER = 2
LIMIT_SWEEP_BENCH = 3
PRELOCATE_BUDGET_SEC = 60.0
PRELOCATE_TERM_SEC = 20.0
SHELL_BUDGET_CEILING_SEC = 180.0
DB_FACTS_SEC = 20.0
READ_OUTPUT_CAP = 24_000
SEARCH_HEAD_LIMIT = 250
SHELL_OUTPUT_CAP = 8_000
COMMAND_CHARS_MAX = 120_000
SEARCH_OUTPUT_CAP = 8_000
TEMPERATURE = 0.0

def one_line(value: object) -> str:
    """A value as one line of text, with every run of whitespace collapsed to a single space."""
    return " ".join(str(value or "").split())

def reply_fingerprint(message: dict) -> str:
    """A short digest of what a reply said, so an identical reply can be recognised.

    Tool calls identify a reply when it made any; otherwise its text does. Two replies with
    the same fingerprint carry the same instruction, which is how a loop that is going
    nowhere is noticed.
    """
    import hashlib
    calls = (message or {}).get("tool_calls") if isinstance(message, dict) else None
    parts = []
    for call in calls if isinstance(calls, list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            function = {}
        parts.append("%s(%s)" % (function.get("name") or "", function.get("arguments") or ""))
    content = message.get("content") if isinstance(message, dict) else ""
    said = "\n".join(parts) if parts else str(content or "")
    return hashlib.sha256(said.encode("utf-8", "replace")).hexdigest()[:8]

def finite_number(value: object) -> bool:
    """Is this a real number this run can do arithmetic with? Booleans, NaN and the infinities are
    not.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf"))

def whole_number(value: object) -> int:
    """A usage field as a whole number, or -1 when the field is missing or not a number."""
    return int(value) if finite_number(value) else -1

COUNT_CEILING = 100_000_000

def counted(value: object) -> int:
    """A count from a reply, clamped to a sane range: a provider's figure is read, never trusted.
    """
    return min(COUNT_CEILING, max(0, whole_number(value)))

def reasoning_tokens(usage: dict) -> int:
    """Tokens the model spent on reasoning, from the nested usage detail, or -1 when it is absent.
    """
    details = (usage or {}).get("completion_tokens_details")
    return whole_number(details.get("reasoning_tokens")
                        if isinstance(details, dict) else None)

def prompt_split(usage: dict) -> tuple:
    """(fresh prompt tokens, cached prompt tokens) from one reply's usage.

    The cached part is subtracted from the total only when it is a sensible share of it, so
    a provider that reports the two differently cannot make the fresh count negative.
    """
    usage = usage or {}
    total = usage.get("prompt_tokens")
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    total, cached = whole_number(total), whole_number(cached)
    if total < 0:
        return -1, cached
    return (total - cached if 0 <= cached <= total else total), cached

CACHE_WRITE_FIELDS = ("cache_creation_input_tokens", "cache_creation_tokens",
                      "cached_tokens_write")

def cache_written(usage: dict) -> int:
    """Tokens written to the prompt cache, from whichever field name the provider uses, or -1.
    """
    usage = usage or {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    for field in CACHE_WRITE_FIELDS:
        for holder in (usage, details):
            if holder.get(field) is not None:
                return whole_number(holder.get(field))
    return -1

def flag(name: str, default: str = "1") -> bool:
    """An on/off switch read from the environment, on unless the value plainly says otherwise.
    """
    return (os.getenv(name) or default).strip().lower() not in ("0", "no", "off", "false", "")

PARALLEL_TOOLS = flag("RIDGES_PARALLEL_TOOLS")
# A 402 that says too many of this run's requests are in flight at once is the
# proxy's passing state, not an empty wallet: wait and ask again, as for a 429.
IN_FLIGHT_RETRY = flag("RIDGES_IN_FLIGHT_RETRY")
IN_FLIGHT_REFUSAL = re.compile(r"\bin[-_ ]?flight\b|\bconcurren(?:t|cy)\b", re.I)
# A reasoning model spends its reasoning out of the reply's token allowance. One
# that runs out before saying anything returns finish_reason "length" and an
# empty message, and asking again with the same allowance only repeats it, so
# the seat doubles its ceiling, up to REPLY_GROWTH_CEILING, for the rest of the
# run. What is left to spend still bounds every request (reply_size).
REPLY_GROWTH = flag("RIDGES_REPLY_GROWTH")
REPLY_GROWTH_CEILING = 64000
# The run must outlive what it starts. When a test runner or build exhausts the
# sandbox's memory, the kernel stops its largest process, and a run that has
# grown past each of many small workers is that process: it then returns
# nothing. Every job raises its own oom_score_adj (raising needs no privilege)
# before its login shell starts, and everything the job starts inherits it.
OOM_SHIELD = flag("RIDGES_OOM_SHIELD")
OOM_YIELD = '{ echo 1000 > /proc/self/oom_score_adj; } 2>/dev/null; exec bash -lc "$1"'


def job_argv(command: str) -> list:
    """The argv that runs one shell command, with the memory shield in front of it when it is on.
    """
    return ["bash", "-c", OOM_YIELD, "bash", command] if OOM_SHIELD else ["bash", "-lc", command]
REPLACE_ALL = flag("RIDGES_REPLACE_ALL")
ASYNC_SHELL = flag("RIDGES_ASYNC_SHELL")
PRELOCATE = flag("RIDGES_PRELOCATE")
SWEEP_WORKFLOW = flag("RIDGES_SWEEP_WORKFLOW")
TRANSCRIPT_CAP = flag("RIDGES_TRANSCRIPT_CAP")
SUBMISSION_WARDEN = flag("RIDGES_SUBMISSION_WARDEN")
WARDEN_ASK = flag("RIDGES_WARDEN_ASK")
PATCH_ENVELOPE = flag("RIDGES_PATCH_ENVELOPE")
CONTRACT_LINT = flag("RIDGES_CONTRACT_LINT")
EDIT_FENCE = flag("RIDGES_EDIT_FENCE")
# The scope and contract readers that speak at hand-in also speak once at edit
# time, so a change outside its bounds is heard when it lands.
EDIT_NOTES = flag("RIDGES_EDIT_NOTES")
# A poll waits a little for the job it asks about, so a command that is about
# to finish does not cost a turn per poll.
BASH_POLL_WAIT_SEC = 20.0
# Several PostgreSQL statements in one sql call are sent one by one, every result
# shown, inside the tool's rolled-back transaction.
SQL_SCRIPT = flag("RIDGES_SQL_SCRIPT")
HIDDEN_SELFREVIEW = flag("RIDGES_HIDDEN_SELFREVIEW", "1")
WORK_METER = flag("RIDGES_WORK_METER", "1")
SELFREVIEW_CONSULT = flag("RIDGES_SELFREVIEW_CONSULT", "0")
STATED_BASELINE = flag("RIDGES_STATED_BASELINE", "0")
LEDGER_READBACK = flag("RIDGES_LEDGER_READBACK", "0")
TELEMETRY_SLOTS = flag("RIDGES_TELEMETRY_SLOTS")
RIDGES_SCOPE_FOLLOWS_STATEMENT = flag("RIDGES_SCOPE_FOLLOWS_STATEMENT", "1")
PACK_VENV = flag("RIDGES_PACK_VENV", "0")
SUBMIT_CONFORM = flag("RIDGES_SUBMIT_CONFORM", "0")
PLAN_SEAT = flag("RIDGES_PLAN_SEAT", "0")
SUITE_SCOPE = flag("RIDGES_SUITE_SCOPE")
SUITE_IMPORTLIB = flag("RIDGES_SUITE_IMPORTLIB", "0")
SUITE_SHIM = flag("RIDGES_SUITE_SHIM", "0")
SUITE_READABLE = flag("RIDGES_SUITE_READABLE", "0")
NETWORK_FENCE = flag("RIDGES_NETWORK_FENCE")
SEARCH_LIMIT = flag("RIDGES_SEARCH_LIMIT", "0")
OUTLINE = flag("RIDGES_OUTLINE", "0")
FINDING_MAP = flag("RIDGES_FINDING_MAP")
DB_FACTS = flag("RIDGES_DB_FACTS")
DB_TOOL = flag("RIDGES_DB_TOOL")
CH_HTTP = flag("RIDGES_CH_HTTP")
MEASURE_TOOL = flag("RIDGES_MEASURE_TOOL")
NETWORK_DB_EXEMPT = flag("RIDGES_NETWORK_DB_EXEMPT")
CASE_TOOL = flag("RIDGES_CASE_TOOL")
QUERY_CHECKS = flag("RIDGES_QUERY_CHECKS")
REQUIREMENT_PAUSE = flag("RIDGES_REQUIREMENT_PAUSE")

def num_env(name: str, default: float) -> float:
    """A number read from the environment, falling back to the default when it is absent or
    unusable.
    """
    try:
        value = float((os.getenv(name) or "").strip())
    except (TypeError, ValueError):
        return default
    return value if value == value and value not in (float("inf"), float("-inf")) else default

def say(message: str) -> None:
    """Write one line to the run log, and never let a closed or broken stream end the run."""
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        pass

class Beacon:
    """One named slot of the run log.

    Every optional behaviour of this agent reports through a beacon: that it was reached,
    that it was skipped and why, or that it fired and with what detail. The slots are fixed
    and short, so a run's log can be read as a sequence of decisions rather than prose.
    """
    SLOTS = ("bgshell", "conform", "editall", "fence", "findings", "ledger",
             "plan", "prelocate", "preview", "sweep", "trim", "verbatim",
             "warden", "batch", "idnorm", "retry", "scan", "scanledger",
             "untouched", "scope", "checks",
             "lint", "bound", "envelope", "contract",
             "editfence",
             "selfreview",
             "consult",
             "steady", "meter", "scope_statement", "review", "expect", "green", "second")

    def __init__(self, slug: str) -> None:
        """Open a slot by name and start its call and spend tallies at zero."""
        self.name = slug.lower()
        self.slug = self.tag(self.name)
        self.calls = 0
        self.usd = 0.0

    @classmethod
    def tag(cls, name: str) -> str:
        """The short label this slot prints under, numbered when numbered slots are on."""
        if not TELEMETRY_SLOTS:
            return "SCOPE" if name == "scope_statement" else name.upper()
        try:
            return "M%02d" % (cls.SLOTS.index(name) + 1)
        except ValueError:
            return name.upper()

    def reached(self, step: int, spent: float, clock: float) -> None:
        """Record that the run arrived at this slot, with the step, the spend and the clock at that
        moment.
        """
        say("[%s] reached step=%d spent=$%.4f clock=%.0fs" % (self.slug, step, spent, clock))

    def skipped(self, reason: str) -> None:
        """Record that this slot did nothing, and why, so silence is never ambiguous."""
        say("[%s] skipped: %s" % (self.slug, reason))

    def fired(self, detail: str) -> None:
        """Record that this slot acted, with the detail that justified it."""
        say("[%s] fired: %s" % (self.slug, detail[:400]))

    def artefact(self, when: str, blob: str) -> str:
        """Record the digest and size of a text this slot is about to change, and return the
        digest.
        """
        import hashlib
        digest = hashlib.sha256((blob or "").encode("utf-8", "replace")).hexdigest()[:8]
        say("[%s] %s %s %dB" % (self.slug, when, digest, len(blob or "")))
        return digest

    def outcome(self, before_digest: str, after: str) -> None:
        """Record the digest and size of the text afterwards, and whether it changed at all."""
        import hashlib
        digest = hashlib.sha256((after or "").encode("utf-8", "replace")).hexdigest()[:8]
        changed = "yes" if digest != before_digest else "no"
        say("[%s] after %s %dB changed=%s" % (self.slug, digest, len(after or ""), changed))

    def bill(self) -> None:
        """Record what this slot cost: how many model calls it made and how much they came to.
        """
        say("[%s] cost calls=%d usd=%.4f" % (self.slug, self.calls, self.usd))

class Spent(Exception):
    """The run has no allowance left to finish what it was asked to do."""
    pass

class ReadExpired(Exception):
    """A read ran past the time it was given, so its result is incomplete and must not be used.
    """
    pass

class Allowance:
    """What the run may still spend: wall clock, money, and the tallies kept against both.

    Every budget decision in the agent asks this object rather than the clock or the
    provider, so one place decides when work must stop and hand in what it has.
    """
    quoted_calls = 0

    def __init__(self) -> None:
        """Read the wall clock and the money cap for this run and set the deadline a reserve short
        of both.
        """
        self.started = time.time()
        wall = num_env("AGENT_TIMEOUT", DEFAULT_WALL_SEC)
        self.ceiling_usd = num_env("RIDGES_MAX_COST_USD", DEFAULT_COST_LIMIT_USD)
        self.soft_usd = self.ceiling_usd * COST_SHARE
        self.deadline = self.started + wall * WALL_SHARE - WALL_RESERVE_SEC
        self.spent = 0.0
        self.calls = 0
        self.edits = 0
        self.tests_run = 0

    def clock_left(self) -> float:
        """Seconds left before the deadline this run set itself, which is short of the real one.
        """
        return self.deadline - time.time()

    def money_left(self) -> float:
        """Dollars left under the soft cap, which is a share of the real ceiling."""
        return self.soft_usd - self.spent

    def elapsed(self) -> float:
        """Seconds since the run started."""
        return time.time() - self.started

    def halt_reason(self) -> str:
        """Why work must stop now, in a word, or empty while it may go on."""
        if self.clock_left() <= 0:
            return "wall clock"
        if self.money_left() <= 0:
            return "budget"
        return ""

    def charge(self, model: str, usage: dict, *, prompt_estimate: int = 0,
               completion_reserve: int = 0) -> float:
        """Book one model call against the run's money, and return what it cost.

        A charge the provider reports is taken as it stands, because it has already
        happened. When no charge is reported the cost is worked out from the token
        counts and this run's price table, and an unknown model is priced at the dearest
        row so an unpriced seat cannot quietly overspend.
        """
        self.calls += 1
        usage = usage if isinstance(usage, dict) else {}
        quoted = usage.get("cost")
        # A reported charge has already happened. The planned cap is not a
        # reason to erase it; the proxy still enforces its own hard limit.
        if finite_number(quoted) and quoted >= 0:
            self.spent += float(quoted)
            self.quoted_calls += 1
            return float(quoted)

        def tokens(value):
            return (int(value) if finite_number(value) and value >= 0
                    and value == int(value) else None)

        reported_prompt = tokens(usage.get("prompt_tokens"))
        prompt = (reported_prompt if reported_prompt is not None
                  else max(0, prompt_estimate))
        completion = tokens(usage.get("completion_tokens"))
        if completion is None:
            # Reserve the requested output allowance, including invisible
            # reasoning, instead of treating missing usage as a free reply.
            completion = max(0, completion_reserve)
        details = usage.get("completion_tokens_details") or {}
        reasoning = tokens(details.get("reasoning_tokens")) if isinstance(details, dict) else None
        if reasoning is not None:
            completion = max(completion, reasoning)
        details = usage.get("prompt_tokens_details") or {}
        cached = tokens(details.get("cached_tokens")) if isinstance(details, dict) else None
        # Cache savings need a coherent reported total, not an estimated one.
        if reported_prompt is None or cached is None or cached > prompt:
            cached = 0
        fresh = prompt - cached
        in_price, out_price = MODEL_PRICING.get(model, UNKNOWN_TOKEN_PRICE)
        cache_price = SEAT_CACHE_TERMS.get(model, UNKNOWN_CACHE_TERMS)[0]
        cost = fresh * in_price + cached * cache_price + completion * out_price
        self.spent += cost
        return cost

def call_ident(call: dict, index: int) -> str:
    """The id a tool call came with, or a positional stand-in so every call can still be answered.
    """
    ident = call.get("id")
    return ident if isinstance(ident, str) and ident else "call_%d" % index

def usable_calls(raw: object, used_ids: set | None = None) -> list:
    """The tool calls of a reply, in the shape every later request must carry.

    A call is answered by its id and named in the history. One with no name can
    be neither run nor recorded, and two with one id cannot be told apart; an
    endpoint refuses a history holding either, on every request after it. So
    the first is dropped and the second renamed, once, here, and the record and
    the answers are both made from this one list.
    """
    if not isinstance(raw, list):
        return []
    kept = []
    seen = set() if used_ids is None else used_ids
    reserved = set()
    for call in raw:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            continue
        name, ident = call["function"].get("name"), call.get("id")
        if (isinstance(name, str) and name.strip() and isinstance(ident, str)
                and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", ident)):
            reserved.add(ident)
    for index, call in enumerate(raw):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        function = dict(function) if isinstance(function, dict) else {}
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        function["name"] = name.strip()
        ident = call.get("id")
        # Transport metadata has no task meaning. Keep ordinary provider IDs,
        # but bound unusual IDs and disambiguate every exchange in this run.
        valid = isinstance(ident, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", ident)
        preserve = valid and ident not in seen
        if not valid:
            ident = "call_%d" % index
        base = ident
        serial = 0
        while not preserve and (ident in seen or ident in reserved):
            serial += 1
            suffix = "_%d_%d" % (index, serial)
            ident = base[:64 - len(suffix)] + suffix
        seen.add(ident)
        kept.append({"id": ident, "type": "function", "function": function})
    return kept

def recorded_calls(calls: list) -> list:
    """The tool calls worth carrying in the transcript, with unreadable arguments marked rather
    than dropped.

    A call whose arguments are not an object cannot be run, but leaving it out of the
    transcript would make the reply that follows look unprompted.
    """
    kept = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        function = dict(function) if isinstance(function, dict) else {}
        written = function.get("arguments")
        try:
            if not isinstance(written, str) or not written.strip():
                raise ValueError("nothing was written")
            if not isinstance(json.loads(written), dict):
                raise ValueError("not an object")
        except Exception:
            function["arguments"] = "{}"
        kept.append({"id": call_ident(call, index), "type": "function",
                     "function": function})
    return kept

def transcript_cap_chars(model: str, ceiling_usd: float) -> int:
    """How much transcript this model can be re-sent each turn without the cache bill outgrowing
    the run.

    The cap is the smaller of what the budget affords over the planned turns and a safe
    share of the model's window, never below a floor that keeps the work legible.
    """
    cache_price, window = SEAT_CACHE_TERMS.get(model, UNKNOWN_CACHE_TERMS)
    affordable = (ceiling_usd * TRANSCRIPT_SPEND_SHARE) / (TURNS_PLANNED * cache_price)
    tokens = min(affordable, window * 0.6)
    return int(max(TRANSCRIPT_FLOOR_CHARS, tokens * CHARS_PER_TOKEN))

FINISH_REASONS = ("stop", "length", "tool_calls", "content_filter", "error", "")

def foreign(text: str) -> str:
    """The size of a body this run will not interpret, for the log: its bytes, or that it was
    empty.
    """
    body = "" if text is None else str(text)
    return "%dB" % len(body.encode("utf-8", "replace")) if body else "empty"

SECRET_SHAPED = re.compile(r"\b(?:sk|pk|Bearer)[-_ ][A-Za-z0-9_\-]{6,}|[A-Za-z0-9_\-]{28,}")
REFUSAL_REASON_CHARS = 200
REASON_CODES = (402, 429)
REPLY_TOKEN_FLOOR = 1024
REPLY_SHRINKS_MAX = 2
# "...requested up to 16000 tokens, but can only afford 1528."
AFFORDABLE_TOKENS = re.compile(r"can only afford\s+([0-9]+)", re.I)
CONTEXT_REFUSAL = re.compile(
    r"context length|maximum context|too many tokens|reduce the length|"
    r"token limit|too long", re.I)
CREDIT_REFUSAL = re.compile(
    r"insufficient credit|add more credits|payment required|out of credit|"
    r"quota", re.I)

def refusal_reason(detail: str) -> str:
    """The provider's own words for a refusal, unwrapped from JSON, collapsed, redacted and
    clipped.
    """
    body = "" if detail is None else str(detail)
    if not body.strip():
        return ""
    try:
        error = json.loads(body).get("error")
        said = error.get("message") if isinstance(error, dict) else error
        body = said if isinstance(said, str) and said.strip() else body
    except Exception:
        pass
    said = SECRET_SHAPED.sub("<redacted>", " ".join(body.split()))
    return said[:REFUSAL_REASON_CHARS]

def retry_after_seconds(headers) -> float:
    """The wait a response asked for in its headers, in seconds, or zero when it asked for none.
    """
    try:
        said = headers.get("Retry-After")
    except Exception:
        return 0.0
    try:
        return max(0.0, float(str(said).strip()))
    except (TypeError, ValueError):
        return 0.0

REPLY_BYTES_MAX = 32_000_000

def read_within(response, seconds: float, *, deadline: float | None = None,
                max_bytes: int = REPLY_BYTES_MAX, truncate: bool = False) -> bytes:
    """The whole body, or a timeout once this call's allowance is gone.

    A socket timeout bounds each wait for bytes, not the reply. A body that
    arrives a little at a time never trips it, and the call outlives the run.
    """
    if deadline is None:
        deadline = time.monotonic() + max(0.0, seconds)
    # HTTPError wraps HTTPResponse in one extra fp layer. Find the same real
    # socket for either response shape, so a slow error stream is bounded too.
    stream, sock = response, None
    for _ in range(4):
        raw = getattr(stream, "raw", None)
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            break
        stream = getattr(stream, "fp", None)
        if stream is None:
            break
    reader = getattr(response, "read1", None) or response.read
    parts, held = [], 0
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("the reply did not finish inside %.0fs" % seconds)
        if sock is not None:
            try:
                sock.settimeout(max(0.001, left))
            except (OSError, ValueError):
                sock = None
        if truncate and held >= max_bytes:
            return b"".join(parts)
        chunk = reader(min(65536, max_bytes - held + (0 if truncate else 1)))
        if not chunk:
            return b"".join(parts)
        parts.append(chunk)
        held += len(chunk)
        if held > max_bytes:
            if truncate:
                return b"".join(parts)[:max_bytes]
            raise ValueError("the reply is larger than %d bytes" % max_bytes)

class SeatRefused(Exception):
    """A seat would not answer this request. The run changes seats or asks again rather than
    stopping.
    """
    pass

class SeatTimedOut(SeatRefused):
    """A seat did not answer inside the time it was given."""
    pass

class SeatAbsent(SeatRefused):
    """A seat is not there to answer: the model or the route is unknown to the endpoint."""
    pass

class SeatRateLimited(SeatRefused):
    """A seat is answering other work first, so this request must wait or move."""
    pass

def base_urls() -> list[str]:
    """Where inference requests go: the sandbox proxy when one is given, otherwise the configured
    endpoint.
    """
    proxy = (os.getenv("SANDBOX_PROXY_URL") or "").strip().rstrip("/")
    if proxy:
        return [proxy + "/api/v1"]
    injected = (os.getenv("OPENROUTER_BASE_URL") or "").strip().rstrip("/")
    return [injected or FALLBACK_BASE_URL]

def request_seed(statement: str) -> int:
    """The sampler seed for a problem: the same statement, the same seed.

    Runs of the same problem then start from the same sampling state, so what
    differs between them comes from the code and the observations, not luck.
    """
    digest = hashlib.sha256((statement or "").encode("utf-8", "replace")).hexdigest()
    return int(digest[:8], 16) % (2 ** 31)


class Seat:
    """The model seats this run may ask, in order, with the retry ladder that keeps a run alive.

    One seat is the driver; the others take over when it times out, refuses or is unknown.
    Every optional part of a request (the seed, the reasoning field, the parallel-calls
    field, the temperature) is dropped in turn when an endpoint refuses it, so a request is
    reshaped rather than abandoned, and the run ends on its own budget rather than on one
    endpoint's bad stretch.
    """
    roster: list = []
    patient = True
    impatient_sec = 60.0
    effort = REASONING_EFFORT
    reply_ceiling = REPLY_TOKEN_CEILING

    def __init__(self, allowance: Allowance, models: list | None = None,
                 patient: bool = True, impatient_sec: float = 60.0,
                 effort: str | None = None,
                 reply_ceiling: int | None = None) -> None:
        """Set up a seat roster with this run's reasoning effort, reply ceiling and patience."""
        self.allowance = allowance
        self.patient = patient
        self.impatient_sec = impatient_sec
        self.effort = REASONING_EFFORT if effort is None else effort
        self.reply_ceiling = (REPLY_TOKEN_CEILING if reply_ceiling is None
                              else reply_ceiling)
        if models:
            self.models = list(models)
        else:
            self.models = [DRIVER_MODEL]
            if RELIEF_MODEL and RELIEF_MODEL != DRIVER_MODEL:
                self.models.append(RELIEF_MODEL)
            if LAST_RESORT_MODEL and LAST_RESORT_MODEL not in self.models:
                self.models.append(LAST_RESORT_MODEL)
        self.roster = list(self.models)
        self.timeouts: dict = {}
        self.seed: int | None = None
        self.reply_cap = 0
        self.reply_shrinks = 0
        self.request_prompt_estimate = 0
        self.request_completion_reserve = 0
        self.bases = base_urls()
        self.key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("RIDGES_OPENROUTER_API_KEY")
            or os.getenv("AI_PROXY_KEY")
            or ""
        )

    def current(self) -> str:
        """The model this seat is asking now."""
        return self.models[0]

    prompt_chars = 0

    def reply_size(self) -> int:
        """How large a reply to ask for.

        The endpoint refuses a request whose reply it could not pay for out of
        what is left of the per-problem cap, so asking for a fixed 16k reply
        throws away the tail of the budget. This never raises the cap or spends
        more; it only stops asking for more than is left.

        The figure is an estimate, not an accounting: the table price may not be
        the served price, prompt tokens are estimated from characters, and
        reasoning tokens are billed as output but are not separately visible
        here. It is floored at REPLY_TOKEN_FLOOR so a reply stays useful rather
        than shrinking to whatever the endpoint would merely accept.
        """
        cap = self.reply_ceiling
        if self.reply_cap:
            cap = min(cap, self.reply_cap)
        in_price, out_price = MODEL_PRICING.get(self.current(), UNKNOWN_TOKEN_PRICE)
        if out_price > 0:
            left = self.allowance.ceiling_usd - self.allowance.spent
            if self.prompt_chars and in_price > 0:
                left -= (self.prompt_chars / CHARS_PER_TOKEN) * in_price
            cap = min(cap, int(max(0.0, left) / out_price))
        # Both hard limits win outright. Asking for more than the caller's
        # ceiling, or more than what is left can pay for, only gets refused.
        # A reply too small to be useful is a reason to stop, not to ask for
        # more than there is; that decision belongs to the caller.
        return max(0, min(self.reply_ceiling, cap))

    def grow_reply(self) -> bool:
        """Double this seat's reply ceiling after a reply that ran out of tokens
        before it said anything. False when it cannot grow or the budget would
        not let a request use the larger ceiling."""
        if not REPLY_GROWTH or self.reply_cap:
            return False
        before = self.reply_size()
        grown = min(REPLY_GROWTH_CEILING, max(self.reply_ceiling * 2, REPLY_TOKEN_FLOOR))
        if grown <= self.reply_ceiling:
            return False
        was, self.reply_ceiling = self.reply_ceiling, grown
        if self.reply_size() <= before:
            self.reply_ceiling = was
            return False
        say("[SEAT] reply ran out of tokens before saying anything; ceiling %d -> %d"
            % (was, grown))
        return True

    def reply_floor(self) -> int:
        """The smallest reply worth asking for, within the caller's ceiling.

        Used when an endpoint refusal offers a smaller size: below this, taking
        the offer would buy an HTTP success and an unusable answer.
        """
        return min(REPLY_TOKEN_FLOOR, self.reply_ceiling)

    def shrink_for(self, detail: str) -> int:
        """A refusal about size can be met by asking smaller; one about
        credit cannot. Returns the new reply cap, or 0 to stop the run."""
        if self.reply_shrinks >= REPLY_SHRINKS_MAX:
            return 0
        said = detail or ""
        named = AFFORDABLE_TOKENS.search(said)
        if named:
            try:
                want = int(named.group(1))
            except ValueError:
                return 0
        elif CONTEXT_REFUSAL.search(said):
            want = max(REPLY_TOKEN_FLOOR, self.reply_size() // 2)
        elif CREDIT_REFUSAL.search(said):
            return 0
        else:
            return 0
        if want < self.reply_floor():
            return 0
        self.reply_shrinks += 1
        self.reply_cap = want if not self.reply_cap else min(self.reply_cap, want)
        return self.reply_cap

    def retire(self, model: str) -> bool:
        """Drop a model from the roster for the rest of the run, unless it is the last one left.
        """
        if model in self.models and len(self.models) > 1:
            self.models.remove(model)
            if model in self.roster:
                self.roster.remove(model)
            say("[SEAT] retired %s, now on %s" % (model, self.models[0]))
            return True
        return False

    def ask(self, messages: list[dict], tools: list[dict] | None) -> dict:
        """Ask the seats for one reply, and put aside any model a spending limit removed.

        A limit that belongs to the key rather than to the model is not a reason to lose
        that model for the whole run, so such a seat goes back on the roster afterwards.
        """
        budget = list(self._budget())
        aside: list[str] = []
        try:
            return self._ask(messages, tools, budget, aside)
        finally:
            for model in reversed(aside):
                if model not in self.models and model in self.roster:
                    self.models.insert(0, model)
            if aside:
                say("[SEAT] %s back on: a limit is the key's state, not the seat's"
                    % ", ".join(aside))

    def _ask(self, messages: list[dict], tools: list[dict] | None,
             budget, aside: list) -> dict:
        """Walk the roster until a seat answers, changing seats on a refusal and retiring one that
        keeps timing out.
        """
        budget = budget if budget is not None else list(self._budget())
        while True:
            model = self.current()
            try:
                return self._attempt(model, messages, tools, budget)
            except SeatRefused as refusal:
                say("[SEAT] %s refused: %s" % (model, str(refusal)[:200]))
                if (isinstance(refusal, SeatTimedOut)
                        and self.timeouts.get(model, 0) >= 2
                        and self.retire(model)):
                    continue
                if isinstance(refusal, SeatAbsent):
                    if model in self.roster:
                        self.roster.remove(model)
                    if len(self.models) > 1:
                        self.models.remove(model)
                        say("[SEAT] retired %s, now on %s"
                            % (model, self.models[0]))
                        continue
                if len(self.models) > 1:
                    self.models.remove(model)
                    if isinstance(refusal, SeatRateLimited):
                        if model not in aside:
                            aside.append(model)
                    elif model in aside:
                        aside.remove(model)
                    say("[SEAT] benched %s, now on %s" % (model, self.models[0]))
                    continue
                if isinstance(refusal, SeatAbsent) and not self.roster:
                    raise Spent("no seat this run can use: %s" % refusal)
                if (isinstance(refusal, SeatTimedOut)
                        and not [m for m in self.roster if m != model]):
                    raise Spent("every seat went quiet: %s" % refusal)
                if not self.patient:
                    raise
                wait = SEAT_REFUSED_WAIT_SEC
                if (self.allowance.clock_left() - wait
                        < SEAT_RETRY_FLOOR_SEC + 10.0):
                    raise Spent("no seat will serve this run")
                say("[SEAT] every seat refused; asking again in %.0fs" % wait)
                time.sleep(wait)
                self.models = list(self.roster)
                budget[:] = self._budget()

    def _budget(self):
        """How long this request may take: the whole clock before the first edit, a share of it
        afterwards.
        """
        left = self.allowance.clock_left()
        if not self.allowance.edits:
            share = left
        else:
            share = min(left, max(left * CALL_CLOCK_SHARE, SEAT_CALL_TIMEOUT_SEC))
        return share, time.monotonic(), left

    def _attempt(self, model: str, messages: list[dict], tools: list[dict] | None,
                 budget=None) -> dict:
        # Tool arguments and tool schemas are input too, so measure what will
        # actually be serialised rather than message text alone.
        """One request to one model, with the ladder that reshapes it rather than giving up.

        The body is built, sent and read inside the time this request was given. A
        refusal that names an optional field loses that field and is asked again; a
        reply that ran out of room gets a higher ceiling; a reply with no content and no
        calls is regrown. What comes back is booked against the allowance before it is
        returned.
        """
        try:
            self.prompt_chars = (len(json.dumps(messages or []))
                                 + len(json.dumps(tools or [])))
        except (TypeError, ValueError):
            self.prompt_chars = sum(len(str(message.get("content") or ""))
                                    for message in messages or [])
        body = {
            "model": model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": self.reply_size(),
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if model in BARE_MODELS:
            pass
        elif self.effort == "off":
            body["reasoning"] = {"enabled": False}
        elif self.effort:
            body["reasoning"] = {"effort": self.effort, "exclude": True}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
            if PARALLEL_TOOLS and model not in NO_PARALLEL_MODELS:
                body["parallel_tool_calls"] = True
        profile = model_profile(model)
        if profile.get("temperature") is False:
            body.pop("temperature", None)
        if profile.get("cache"):
            body["messages"] = cache_marked(messages)
        payload = json.dumps(body).encode("utf-8")

        packed = [payload]

        def repack() -> bytes:
            """The bytes to send now, re-serialised whenever the cap changed.

            The cached bytes are replaced, not just returned once: a later
            attempt must not fall back to the original oversized request.
            """
            want = self.reply_size()
            if body.get("max_tokens") != want:
                body["max_tokens"] = want
                packed[0] = json.dumps(body).encode("utf-8")
            return packed[0]
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        backoff = 3.0
        sweep: list[str] = []
        refusal = ""
        refused_sweeps = 0
        timed_out = 0
        unreachable = 0
        unusable = 0
        last_spoken = ""
        old_refused = 0
        old_bench_at = 0
        all_limits = False
        walls_seen = 0
        limits_seen = 0
        quiet_sweeps = 0
        old_dead_base = bool((os.getenv("SANDBOX_PROXY_URL") or "").strip().rstrip("/"))
        RETRY_TALLY["ladders"] += 1
        attempts = REQUEST_ATTEMPTS if self.patient else 1
        ceiling = SEAT_CALL_TIMEOUT_SEC if self.patient else self.impatient_sec
        budget = budget if budget is not None else list(self._budget())
        share, began, started_with = budget

        def spent_share() -> float:
            return max(time.monotonic() - began,
                       started_with - self.allowance.clock_left())

        def handover_reserve() -> float:
            return SEAT_RETRY_FLOOR_SEC + 5.0 if len(self.models) > 1 else 0.0

        def room_left() -> float:
            return min(share - spent_share() - handover_reserve(),
                       self.allowance.clock_left() - 10)

        def room_now() -> float:
            return min(share - spent_share(), self.allowance.clock_left() - 10)
        attempt, extra = -1, 0
        while attempt + 1 < attempts + extra:
            attempt += 1
            if self.allowance.clock_left() <= 5:
                raise Spent("clock ran out mid-request")
            if self.reply_size() <= 0:
                raise Spent("no budget left for any reply")
            asked = 0
            walled = 0
            quiet = 0
            unusable = 0
            limited = 0
            absent = 0
            advised = 0.0
            sweep = []
            granted = ceiling
            regrown = False
            reshaped = False
            for base in self.bases:
                room = room_left()
                if room < SEAT_RETRY_FLOOR_SEC:
                    break
                asked += 1
                parsed = None
                try:
                    request_grant = min(granted, room if timed_out else room_now())
                    request_deadline = time.monotonic() + request_grant
                    outgoing = repack()
                    # Bytes, including schemas and framing, deliberately err
                    # high as an input-token estimate. This is local reserve
                    # accounting, never an exact provider bill.
                    self.request_prompt_estimate = len(outgoing)
                    self.request_completion_reserve = body["max_tokens"]
                    request = urllib.request.Request(
                        base + "/chat/completions", data=outgoing, headers=headers
                    )
                    try:
                        with urllib.request.urlopen(
                                request, timeout=request_grant) as response:
                            parsed = json.loads(read_within(
                                response, request_grant, deadline=request_deadline)
                                .decode("utf-8", "replace"))
                    except urllib.error.HTTPError as error:
                        try:
                            error.request_detail = read_within(
                                error, request_grant, deadline=request_deadline,
                                max_bytes=400, truncate=True).decode("utf-8", "replace")
                        finally:
                            error.close()
                        raise
                except urllib.error.HTTPError as error:
                    detail = error.request_detail
                    said = foreign(detail)
                    if error.code in REASON_CODES:
                        reason = refusal_reason(detail)
                        said = "%s %s" % (said, reason) if reason else said
                        advised = max(advised, retry_after_seconds(error.headers))
                    sweep.append("%s %s" % (error.code, said))
                    if error.code == 403:
                        walled += 1
                        continue
                    if error.code in (401, 404):
                        absent += 1
                        continue
                    if (error.code == 402 and IN_FLIGHT_RETRY
                            and IN_FLIGHT_REFUSAL.search(detail or "")):
                        limited += 1
                        walled += 1
                        continue
                    if error.code in (400, 402, 413):
                        shrunk = self.shrink_for(detail)
                        if shrunk:
                            sweep[-1] += (" (asking for %d reply tokens now)"
                                          % shrunk)
                            walled += 1
                            continue
                        if (error.code == 400 and body.get("parallel_tool_calls")
                                and ("parallel" in detail.lower()
                                     or not any(word in detail.lower()
                                                for word in ("seed", "temperature", "reasoning")))):
                            # A refusal that names another optional field is
                            # answered below; one that names this field, or none,
                            # loses this field first: leave it out for this model
                            # and ask again.
                            body.pop("parallel_tool_calls", None)
                            NO_PARALLEL_MODELS.add(model)
                            packed[0] = json.dumps(body).encode("utf-8")
                            sweep[-1] += " (asking again without parallel tool calls)"
                            reshaped = True
                            break
                        if (error.code == 400 and "temperature" in body
                                and "temperature" in detail.lower()):
                            # This model takes no temperature next to reasoning:
                            # leave it out and ask again.
                            body.pop("temperature", None)
                            packed[0] = json.dumps(body).encode("utf-8")
                            sweep[-1] += " (asking again without the temperature)"
                            reshaped = True
                            break
                        if error.code == 400 and body.get("seed") is not None:
                            # The request itself was refused: ask again at once
                            # without the optional seed. A reshaped request is
                            # not a refusal of this seat.
                            body.pop("seed", None)
                            self.seed = None
                            packed[0] = json.dumps(body).encode("utf-8")
                            sweep[-1] += " (asking again without the seed)"
                            reshaped = True
                            break
                        if error.code == 400 and "reasoning" in body:
                            # Still refused: the reasoning field is the other
                            # optional part. Leave it out for this model.
                            body.pop("reasoning", None)
                            BARE_MODELS.add(model)
                            packed[0] = json.dumps(body).encode("utf-8")
                            sweep[-1] += " (asking again without the reasoning field)"
                            reshaped = True
                            break
                        raise Spent("endpoint refused the request: " + sweep[-1])
                    if error.code == 429 and ("budget" in detail.lower() or "cost" in detail.lower()):
                        raise Spent("allowance exhausted upstream")
                    if error.code == 429:
                        limited += 1
                        walled += 1
                        continue
                except Exception as error:
                    reason = getattr(error, "reason", None)
                    if isinstance(error, TimeoutError) or isinstance(reason, TimeoutError):
                        timed_out += 1
                        quiet += 1
                        sweep.append("timed out after %.0fs" % min(request_grant, room_now()))
                        if (timed_out == 1 and self.allowance.clock_left()
                                > SEAT_CALL_TIMEOUT_SEC + SEAT_RETRY_FLOOR_SEC):
                            share, began, started_with = self._budget()
                            if len(self.models) == 1:
                                share = min(share, ceiling + SEAT_RETRY_FLOOR_SEC)
                            budget[:] = (share, began, started_with)
                    else:
                        unreachable += 1
                        sweep.append("%s: %s" % (type(error).__name__, foreign(error)))
                if parsed is not None:
                    try:
                        booked = self._book(model, parsed)
                    except Exception as error:
                        unusable += 1
                        sweep.append("unusable reply: %s"
                                     % (foreign(error) or type(error).__name__))
                        continue
                    if booked.pop("_cut_short", False) and self.grow_reply():
                        regrown = True
                        break
                    if attempt:
                        self._tally(attempt, old_bench_at, recovered=True)
                    return booked
            if regrown or reshaped:
                # A request re-shaped after a refusal, or regrown after a reply
                # that ran out of tokens, was never answered: it is asked again
                # even on a seat that allows one attempt, a bounded number of times.
                extra = min(extra + 1, RESHAPE_ATTEMPTS)
                continue
            RETRY_TALLY["walled"] += walled
            RETRY_TALLY["quiet"] += quiet
            RETRY_TALLY["unreachable"] += unreachable
            RETRY_TALLY["unusable"] += unusable
            RETRY_TALLY["limited"] += limited
            RETRY_TALLY["absent"] += absent
            if asked and absent >= asked:
                if attempt:
                    self._tally(attempt, old_bench_at, recovered=False)
                raise SeatAbsent("%d of %d base(s) asked, all say %s is not "
                                 "one they serve: %s"
                                 % (asked, len(self.bases), model,
                                    " | ".join(sweep)))
            walls_seen += walled
            limits_seen += limited
            unreachable = 0
            if old_dead_base and asked and not quiet:
                old_refused += 1
                if old_refused >= OLD_REFUSED_SWEEP_BENCH and not old_bench_at:
                    old_bench_at = attempt + 1
            if asked and walled and not quiet:
                refused_sweeps += 1
                refusal = " | ".join(sweep)
                all_limits = limits_seen >= walls_seen
                rope = 2
                if all_limits and room_left() >= SEAT_RETRY_FLOOR_SEC * 3:
                    rope = LIMIT_SWEEP_BENCH
                if refused_sweeps >= rope:
                    self._tally(attempt, old_bench_at, recovered=False)
                    if all_limits:
                        raise SeatRateLimited(refusal)
                    raise SeatRefused(refusal)
            if quiet:
                quiet_sweeps += 1
            if (quiet_sweeps >= SEAT_QUIET_HANDOVER and len(self.models) > 1
                    and room_left() >= SEAT_RETRY_FLOOR_SEC):
                self.timeouts[model] = self.timeouts.get(model, 0) + 1
                self._tally(attempt, old_bench_at, recovered=False)
                raise SeatTimedOut("%d quiet sweep(s) in %d attempt(s): %s"
                                   % (quiet_sweeps, attempt + 1,
                                      " | ".join(sweep) or last_spoken))
            if sweep:
                last_spoken = " | ".join(sweep)
            if attempt + 1 >= attempts or room_left() < SEAT_RETRY_FLOOR_SEC:
                break
            wanted = min(advised, SEAT_RETRY_NAP_CEILING_SEC) if advised else backoff
            nap = min(wanted, max(0.0, self.allowance.clock_left() - 5),
                      max(0.0, room_left() - SEAT_RETRY_FLOOR_SEC))
            if nap <= 0:
                break
            time.sleep(nap + retry_jitter(nap, room_left() - SEAT_RETRY_FLOOR_SEC - nap))
            backoff = min(backoff * 2, SEAT_RETRY_NAP_CEILING_SEC)
        said = " | ".join(sweep) or last_spoken or "no base was asked"
        if timed_out:
            self.timeouts[model] = self.timeouts.get(model, 0) + 1
            if attempt:
                self._tally(attempt, old_bench_at, recovered=False)
            raise SeatTimedOut("%d timeout(s) in %d attempt(s): %s"
                               % (timed_out, attempt + 1, said))
        if attempt:
            self._tally(attempt, old_bench_at, recovered=False)
        if len(self.models) > 1 and (share - spent_share()) >= SEAT_RETRY_FLOOR_SEC:
            raise SeatRefused("%s (nothing left on this seat)" % said)
        if (self.patient and len(self.models) == 1
                and self.allowance.clock_left() - SEAT_REFUSED_WAIT_SEC
                >= SEAT_RETRY_FLOOR_SEC + 10.0):
            raise SeatRefused("%s (this share is spent, the run is not)" % said)
        raise Spent("no reply after %d attempts: %s" % (attempt + 1, said))

    def _tally(self, attempt: int, old_bench_at: int, recovered: bool) -> None:
        """Record what the retry ladder did, and what the older ladder would have done instead.
        """
        RETRY_TALLY["retried"] += 1
        if recovered:
            RETRY_TALLY["recovered"] += 1
        was = "carried on"
        if old_bench_at:
            RETRY_TALLY["old_would_bench"] += 1
            was = "benched the seat at sweep %d" % old_bench_at
        elif attempt + 1 > OLD_REQUEST_ATTEMPTS:
            RETRY_TALLY["old_would_exhaust"] += 1
            was = "run out of sweeps at %d" % OLD_REQUEST_ATTEMPTS
        say("[" + Beacon.tag("retry") + "] sweeps=%d recovered=%s old_ladder=%s"
            % (attempt + 1, "yes" if recovered else "no", was))

    def _book(self, model: str, parsed: dict) -> dict:
        """Read one reply, charge it to the allowance, and return the message with what it cost.
        """
        choices = parsed.get("choices") or []
        if not choices:
            raise Spent("reply carried no choices")
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise Spent("reply carried no message object")
        usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
        quoted_before = self.allowance.quoted_calls
        visible_bytes = len(json.dumps(message, ensure_ascii=False).encode("utf-8"))
        cost = self.allowance.charge(
            model, usage,
            prompt_estimate=max(self.request_prompt_estimate, self.prompt_chars),
            completion_reserve=max(self.request_completion_reserve, visible_bytes))
        served = one_line(parsed.get("provider") or parsed.get("served_by"))
        answered = one_line(parsed.get("model"))
        aside = ""
        if served:
            aside += " via=%s" % served[:40]
        if answered and answered != model:
            aside += " answered=%s" % answered[:60]
        thought = reasoning_tokens(usage)
        if thought >= 0:
            aside += " thought=%d" % thought
        fresh, cached = prompt_split(usage)
        if fresh >= 0:
            aside += " fresh=%d" % fresh
        if cached >= 0:
            aside += " cached=%d" % cached
        written = cache_written(usage)
        if written >= 0:
            aside += " written=%d" % written
        out = whole_number(usage.get("completion_tokens"))
        if out >= 0:
            aside += " out=%d" % out
        if self.allowance.quoted_calls == quoted_before:
            aside += " est"
        say(
            "[SEAT] %s call=%d $%.4f total=$%.4f left=%.0fs said=%s%s"
            % (model, self.allowance.calls, cost, self.allowance.spent,
               self.allowance.clock_left(), reply_fingerprint(message), aside)
        )
        finish = choices[0].get("finish_reason") or ""
        message["_finish"] = finish
        if (finish == "length" and not str(message.get("content") or "").strip()
                and not message.get("tool_calls")):
            message["_cut_short"] = True
        if finish not in ("stop", "tool_calls", ""):
            named = finish if finish in FINISH_REASONS else "other"
            say("[SEAT] %s reply ended on %s after %d token(s)"
                % (model, named, counted(usage.get("completion_tokens"))))
        return message

GIT_TIMED_OUT = 124
PATH_CHARS_MAX = 4096

def retry_jitter(nap: float, room: float) -> float:
    """A small, clock-derived addition to a retry nap.

    Runs of the same problem start their retries at the same moments otherwise
    and meet the same limit again together. Never more than a quarter of the
    nap, a second, or the room the caller has left.
    """
    if nap <= 0 or room <= 0:
        return 0.0
    return min(nap * 0.25, 1.0, room) * ((time.time_ns() // 1000) % 997) / 997.0


def git(args: list[str], cwd: str, timeout: float = 60.0) -> tuple[int, str]:
    """Run one git command in a checkout and return its exit code with its combined output.

    A timeout and an unexpected failure both come back as a code and a sentence rather than
    an exception, because every caller here has to carry on either way.
    """
    try:
        done = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=max(2.0, timeout),
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return GIT_TIMED_OUT, "git timed out"
    except Exception as error:
        return 1, "%s: %s" % (type(error).__name__, error)
    if done.returncode == 0:
        return 0, done.stdout or ""
    return done.returncode, (done.stdout or "") + (done.stderr or "")

def git_bytes(args: list[str], cwd: str, timeout: float = 60.0) -> tuple[int, bytes]:
    """git, with its output as the bytes it wrote.

    Text mode is not faithful: it turns a carriage return into a line feed and
    replaces any byte that is not UTF-8, and a diff altered either way no longer
    matches the file it describes.
    """
    try:
        done = subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                              timeout=max(2.0, timeout))
    except subprocess.TimeoutExpired:
        return GIT_TIMED_OUT, b"git timed out"
    except Exception as error:
        return 1, ("%s: %s" % (type(error).__name__, error)).encode("utf-8", "replace")
    if done.returncode == 0:
        return 0, done.stdout or b""
    return done.returncode, (done.stdout or b"") + (done.stderr or b"")

def fragile_text(raw: bytes) -> bool:
    """Would this diff be altered by being written and read back as text?

    The answer travels as a string: it is written to a file as UTF-8 and can be
    read back in text mode before it is applied. A carriage return does not
    survive that reading and a byte that is not UTF-8 does not survive the
    writing, and a hunk that lost either no longer matches its file.
    """
    if b"\r" in raw:
        return True
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False

def attribute_pattern(path: bytes) -> bytes:
    """A gitattributes pattern, quoted, that matches exactly this path."""
    out = bytearray(b'"')
    for value in path:
        char = bytes([value])
        if char in b"*?[]":
            out += b"\\\\" + char
        elif char == b"\\":
            out += b"\\\\\\\\"
        elif char == b'"':
            out += b'\\"'
        elif 32 <= value < 127:
            out += char
        else:
            out += b"\\%03o" % value
    return bytes(out) + b'"'

EXACT_BINARY: set = set()

class CheckoutSnapshot:
    """Disk-backed preservation of files Git cannot restore from HEAD.

    No symlink is followed. The snapshot must complete before solver commands
    are allowed to start; inability to preserve the checkout is a startup
    failure, rather than a later claim that a partial reset restored it.
    """
    def __init__(self, root: str, budget: float = PRELOCATE_BUDGET_SEC):
        """Take a copy of the checkout as it stands, so it can be put back exactly.

        Every tracked and untracked entry is recorded with its mode and digest and
        copied aside under a private directory, within the time allowed. A copy that
        cannot be completed is no copy at all, so the caller is told rather than left
        with half of one.
        """
        self.root = os.path.realpath(root)
        self.owner = None
        self.entries = {}
        self.directories = {}
        deadline = time.monotonic() + max(0.0, budget)
        try:
            code, raw = git_bytes(["ls-files", "--cached", "-z"], root, max(0.1, budget))
            if code:
                raise OSError("could not enumerate tracked files for preservation")
            self.tracked = {os.fsdecode(p) for p in raw.split(b"\0") if p}
            parent = tempfile.gettempdir()
            if os.path.commonpath([self.root, os.path.realpath(parent)]) == self.root:
                parent = os.path.dirname(self.root)
            self.owner = tempfile.TemporaryDirectory(prefix="ridges-original-", dir=parent)
            for relative, info in self.inventory(deadline):
                if stat.S_ISDIR(info.st_mode):
                    self.directories[relative] = stat.S_IMODE(info.st_mode)
                elif relative not in self.tracked:
                    path = os.path.join(self.root, relative)
                    item = {"mode": stat.S_IMODE(info.st_mode)}
                    if stat.S_ISLNK(info.st_mode):
                        item.update(kind="link", target=os.readlink(path))
                    elif stat.S_ISREG(info.st_mode):
                        item.update(kind="file", blob=str(len(self.entries)))
                        item["sha256"] = self.copy(path, os.path.join(self.owner.name, item["blob"]), deadline)
                    else:
                        raise OSError("cannot preserve special filesystem entry: %s" % relative)
                    self.entries[relative] = item
            self.check_time(deadline)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check_time(deadline):
        """Stop the work when it has run past its time allowance."""
        if time.monotonic() >= deadline:
            raise TimeoutError("checkout preservation exceeded its time allowance")

    def inventory(self, deadline):
        """Every entry under the checkout, in a stable order, with its own stat and not a symlink's
        target.
        """
        def walk(directory, relative=""):
            self.check_time(deadline)
            with os.scandir(directory) as stream:
                children = sorted(stream, key=lambda entry: entry.name)
            for entry in children:
                self.check_time(deadline)
                if not relative and entry.name == ".git":
                    continue
                name = os.path.join(relative, entry.name)
                info = entry.stat(follow_symlinks=False)
                yield name, info
                if stat.S_ISDIR(info.st_mode):
                    yield from walk(entry.path, name)
        yield from walk(self.root)

    @classmethod
    def copy(cls, source, destination, deadline):
        """Copy one regular file aside and return its digest, refusing anything that is not a
        regular file.
        """
        digest = hashlib.sha256()
        fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise OSError("preservation source is not a regular file")
            with open(destination, "wb") as target:
                while True:
                    cls.check_time(deadline)
                    data = stream.read(1024 * 1024)
                    if not data:
                        break
                    target.write(data)
                    digest.update(data)
            after = os.fstat(stream.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise OSError("file changed while its original bytes were preserved")
        return digest.hexdigest()

    @staticmethod
    def remove(path):
        """Remove one entry, a symlink or file by unlinking it and a real directory by emptying it.
        """
        if os.path.islink(path) or not os.path.isdir(path):
            os.unlink(path)
        else:
            shutil.rmtree(path)

    def restore(self, deadline):
        # Recreate original directories before touching their children. A
        # replaced parent symlink must never redirect restoration elsewhere.
        """Put the checkout back to the recorded state, parents before children.

        Directories are recreated first so a replaced parent symlink cannot redirect the
        rest of the work, then anything added is removed and anything changed is written
        back from the copy.
        """
        for relative, mode in sorted(self.directories.items(), key=lambda row: row[0].count(os.sep)):
            self.check_time(deadline)
            path = os.path.join(self.root, relative)
            if os.path.lexists(path) and (os.path.islink(path) or not os.path.isdir(path)):
                self.remove(path)
            os.makedirs(path, exist_ok=True)
            os.chmod(path, mode | stat.S_IWUSR | stat.S_IXUSR)
        current = list(self.inventory(deadline))
        for relative, info in reversed(current):
            self.check_time(deadline)
            path = os.path.join(self.root, relative)
            if stat.S_ISDIR(info.st_mode):
                if relative not in self.directories:
                    os.rmdir(path)
            elif relative not in self.tracked and relative not in self.entries:
                self.remove(path)
        for relative, item in self.entries.items():
            self.check_time(deadline)
            path = os.path.join(self.root, relative)
            if os.path.lexists(path):
                self.remove(path)
            if item["kind"] == "link":
                os.symlink(item["target"], path)
            else:
                self.copy(os.path.join(self.owner.name, item["blob"]), path, deadline)
                os.chmod(path, item["mode"])
        for relative, mode in sorted(self.directories.items(), key=lambda row: -row[0].count(os.sep)):
            os.chmod(os.path.join(self.root, relative), mode)
        observed, directories = {}, {}
        for relative, info in self.inventory(deadline):
            path = os.path.join(self.root, relative)
            if stat.S_ISDIR(info.st_mode):
                directories[relative] = stat.S_IMODE(info.st_mode)
            elif relative not in self.tracked:
                item = {"mode": stat.S_IMODE(info.st_mode)}
                if stat.S_ISLNK(info.st_mode):
                    item.update(kind="link", target=os.readlink(path))
                elif stat.S_ISREG(info.st_mode):
                    digest = hashlib.sha256()
                    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(fd, "rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            self.check_time(deadline)
                            digest.update(chunk)
                    item.update(kind="file", sha256=digest.hexdigest())
                observed[relative] = item
        expected = {name: {key: value for key, value in item.items() if key != "blob"}
                    for name, item in self.entries.items()}
        if expected != observed or self.directories != directories:
            raise OSError("original nontracked checkout state was not restored exactly")

    def close(self):
        """Release the private directory the copy was kept in."""
        if self.owner is not None:
            self.owner.cleanup()
            self.owner = None

class Tree:
    """The checkout this run works in, and the only place its answer comes from.

    Every read and write goes through here so nothing can reach outside the repository, and
    the answer is always the difference between the base commit and the tree as it stands. A
    copy of the original checkout is kept at the start so the tree can be put back byte for
    byte before the answer is handed over.
    """
    def __init__(self, root: str, statement: str = "") -> None:
        """Open a checkout: find its base commit, note what was already untracked, and copy it
        aside.
        """
        self.root = root
        self.statement = statement
        code, out = git(["rev-parse", "HEAD"], root, 30)
        self.base = out.strip() if code == 0 else ""
        self.untracked_at_start = self._untracked() or set()
        self.original_snapshot = CheckoutSnapshot(root)

    def _untracked(self, budget: float = 30.0) -> set | None:
        """The untracked paths git lists, or None when the listing itself did not finish."""
        code, out = git(["ls-files", "--others", "--exclude-standard", "-z"],
                        self.root, max(1.0, budget))
        return {p for p in out.split("\0") if p} if code == 0 else None

    def absolute(self, path: str) -> str:
        """A path inside the checkout as an absolute one, refusing anything that would escape it.
        """
        if len(path) > PATH_CHARS_MAX:
            # Said here rather than by the operating system, whose refusal
            # quotes the whole path back and would fill the transcript with it.
            raise ToolFault("that path is %d characters long, which is not a path"
                            % len(path))
        root = os.path.normpath(self.root)
        joined = os.path.normpath(os.path.join(root, path))
        if joined != root and not joined.startswith(root + os.sep):
            raise ToolFault("path escapes the repository: %s" % path)
        return joined

    def read(self, path: str) -> str:
        """One file's text, with undecodable bytes replaced so a stray byte cannot end the run.
        """
        full = self.absolute(path)
        if not os.path.isfile(full):
            raise ToolFault("no such file: %s" % path)
        with open(full, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()

    def inside(self, path: str) -> str:
        """The place a write to this path would really land, or a refusal.

        A link inside the checkout can point out of it. A change written through
        one lands where no diff of this repository will ever see it, so the
        caller is told rather than left believing the change was made.
        """
        full = self.absolute(path)
        root = os.path.realpath(self.root)
        real = os.path.realpath(full)
        if real != root and not real.startswith(root + os.sep):
            raise ToolFault(
                "%s is a link that leaves the repository, so a change written "
                "there could not be part of the answer. Edit a file inside the "
                "checkout." % path)
        return full

    def read_exact(self, path: str) -> str:
        """The file as it is on disk: no newline translation, no byte replaced.

        What is shown to a model may be tidied. What is edited and written back
        may not be, or changing one line rewrites every line ending and every
        byte that is not UTF-8 in the rest of the file.
        """
        full = self.absolute(path)
        if not os.path.isfile(full):
            raise ToolFault("no such file: %s" % path)
        with open(full, "rb") as handle:
            return handle.read().decode("utf-8", "surrogateescape")

    def write(self, path: str, text: str) -> None:
        # Validate bytes before touching a destination. Replacing a complete
        # sibling file also avoids writing through hardlinks to other files.
        """Write one file in place, preserving its mode and never writing through a link.

        The bytes are validated first and a complete sibling file is swapped in, so a
        hardlink to another file cannot be edited through and a half-written file cannot
        be left behind.
        """
        payload = text.encode("utf-8", "surrogateescape")
        full = os.path.realpath(self.inside(path))
        parent = os.path.dirname(full) or self.root
        os.makedirs(parent, exist_ok=True)
        mode = None
        try:
            original = os.stat(full, follow_symlinks=False)
        except FileNotFoundError:
            original = None
        if original is not None:
            if not stat.S_ISREG(original.st_mode):
                raise ToolFault("the write destination is not a regular file: %s" % path)
            # Preserve the existing write-permission check without truncation.
            access = os.open(full, os.O_WRONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            try:
                mode = stat.S_IMODE(os.fstat(access).st_mode)
            finally:
                os.close(access)
        temporary, descriptor = None, None
        try:
            for _ in range(10):
                candidate = os.path.join(parent, ".agent-write-" + os.urandom(12).hex())
                try:
                    descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                    temporary = candidate
                    break
                except FileExistsError:
                    continue
            else:
                raise ToolFault("could not allocate a temporary file for the edit")
            handle = os.fdopen(descriptor, "wb")
            descriptor = None  # The file object now owns this descriptor.
            with handle:
                if handle.write(payload) != len(payload):
                    raise OSError("incomplete file write")
                handle.flush()
                if mode is not None:
                    os.fchmod(handle.fileno(), mode)
            os.replace(temporary, full)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def changed_paths(self, budget: float = 15.0) -> list[str]:
        """The paths that differ from the base commit."""
        if not self.base:
            return []
        code, out = git(["-c", "core.filemode=true", "diff", "--name-only", "-z", self.base],
                        self.root, budget)
        if code != 0:
            raise ToolFault("could not list the changed files: %s" % out.strip()[:200])
        return [path for path in out.split("\0") if path]

    def at_base(self, path: str, budget: float = 15.0) -> str:
        """One file's text as the base commit has it, for comparing against the tree as it stands.
        """
        code, out = git(["show", "%s:%s" % (self.base, path)], self.root, budget)
        if code != 0:
            raise ToolFault("could not read %s as it was: %s" % (path, out.strip()[:200]))
        return out

    def generated_exclusions(self, budget: float = 15.0) -> list[str]:
        """Pathspecs that keep build by-products out of the answer unless the task asked for them.

        Anything that appeared under a vendor, dist, target or node_modules directory
        during the run is a by-product of running the project, not a change this run
        chose to make.
        """
        code, listing = git(["ls-tree", "-r", "--name-only", "-z", self.base], self.root, budget)
        if code:
            return []
        baseline = set(listing.split("\0")) | self.untracked_at_start
        code, listing = git(["ls-files", "--others", "--exclude-standard", "-z"], self.root, budget)
        if code:
            return []
        requested = requested_answer_paths(getattr(self, "statement", ""), self.root)
        return [":(top,literal,exclude)" + path for path in listing.split("\0")
                if path and path not in baseline and path not in requested and
                (set(path.split("/")[:-1]) & {"vendor", "dist", "target", "node_modules"}
                 or path.split("/")[-1] == "go.sum")]

    def base_paths(self, budget: float = 15.0) -> set:
        """The files the repository already had when this run started."""
        if (getattr(self, "_base_paths", None) is None
                or not getattr(self, "_base_paths_known", False)):
            code, listing = git(["ls-tree", "-r", "--name-only", "-z", self.base or "HEAD"],
                                self.root, max(1.0, budget))
            self._base_paths = {p for p in listing.split("\0") if p} if code == 0 else set()
            self._base_paths_known = code == 0
        return self._base_paths

    def answer_paths(self, budget: float = 15.0) -> list:
        """The files that make up the answer as the tree stands.

        Tracked files this run changed, plus new files that are not products of
        running commands. One set, used for the patch identity, for deciding
        whether an edited candidate exists, and for what a check ran against.
        """
        self._answer_paths_known = False
        try:
            known = self.base_paths(budget)
            if not self.base or not getattr(self, "_base_paths_known", False):
                return []
            changed = self.changed_paths(budget)
        except BaseException:
            return []
        requested = requested_answer_paths(getattr(self, "statement", ""), self.root)
        out = [path for path in changed if path in known]
        out += [path for path in changed
                if path not in known and (path in requested or not byproduct_path(path))]
        code, listing = git(["ls-files", "--others", "--exclude-standard", "-z", "--"],
                            self.root, max(1.0, budget))
        if code != 0:
            return []
        for path in listing.split("\0"):
            if path and path not in out and (path in requested or not byproduct_path(path)):
                out.append(path)
        # Ignored files do not appear in --others --exclude-standard. An
        # explicit answer still participates in verification before diff()
        # marks it, so identity must use the same selection as delivery.
        for path in sorted(requested):
            if path not in known and path not in out and os.path.isfile(self.absolute(path)):
                out.append(path)
        self._answer_paths_known = True
        self._answer_paths_snapshot = sorted(set(out))
        return list(self._answer_paths_snapshot)

    def source_identity(self, budget: float = 15.0, extra=(), paths=None) -> str:
        """Known Git base plus the source delta, including added helpers.

        The base anchors a pristine checkout without hashing every file. An
        unsuccessful delta listing is unknown, not a pristine checkout. Logs
        and generated caches remain outside the answer's source identity.
        """
        import hashlib
        digest = hashlib.sha256()
        try:
            if not self.base:
                return ""
            paths = list(self.answer_paths(budget) if paths is None else paths)
            if (not getattr(self, "_answer_paths_known", False)
                    or sorted(set(paths)) != self._answer_paths_snapshot):
                return ""
            known = self.base_paths(budget)
            if not getattr(self, "_base_paths_known", False):
                return ""
            paths += [p for p in (extra or ()) if p not in known and p not in paths]
        except BaseException:
            return ""
        digest.update(("git-base\0" + self.base + "\0").encode())
        for path in sorted(set(paths)):
            digest.update(json.dumps(path, ensure_ascii=True).encode() + b"\0")
            try:
                full = self.absolute(path)
            except BaseException:
                return ""
            try:
                info = os.lstat(full)
            except FileNotFoundError:
                # A deletion is a known state of the source; hash it as one.
                digest.update(b"<deleted>\0")
                continue
            except OSError:
                return ""
            try:
                digest.update(("%o\0" % info.st_mode).encode())
                if stat.S_ISLNK(info.st_mode):
                    # Read the link itself, including broken links, rather
                    # than the contents of the file it happens to point at.
                    contents = os.readlink(os.fsencode(full))
                elif stat.S_ISREG(info.st_mode):
                    with open(full, "rb") as handle:
                        contents = handle.read()
                else:
                    # A directory/submodule or special file needs its own
                    # reader; do not label an uninspected state as known.
                    return ""
                digest.update(hashlib.sha256(contents).digest())
            except OSError:
                # Content that could not be read is not known content. An
                # unknown identity must not compare equal to another one.
                return ""
        return digest.hexdigest()[:16]

    def diff(self, budget: float = 60.0, failed: list | None = None) -> str:
        # `failed`, when given, collects every step that did not complete, so a
        # caller can tell an empty diff from one git could not produce.
        """The answer: the difference between the base commit and the tree as it stands.

        Intents to add are marked first so a new file is included, by-products are
        excluded, and binary content is handled section by section. Every step that did
        not finish is collected into `failed`, so a caller can tell an empty answer from
        one git could not produce.
        """
        deadline = time.monotonic() + budget

        def left(floor: float = 1.0) -> float:
            return max(floor, deadline - time.monotonic())
        mark = ["add", "-A", "-N", "--", ":(top)",
                ":(top,exclude)*.pyc", ":(top,exclude)*.pyo"]
        if RIDGES_SCOPE_FOLLOWS_STATEMENT:
            mark += self.generated_exclusions(left())
        marked, said = git(mark, self.root, left())
        if marked != 0 and self._unlock(said):
            marked, said = git(mark, self.root, left())
        if marked != 0:
            say("[TREE] new files could not be marked and may be missing "
                "from the diff: %s" % said.strip()[:200])
            if failed is not None:
                failed.append("mark")
        # Ordinary ignore patterns and cache exclusions must not erase files
        # explicitly requested in this task or deliberately written as source.
        # Force only these paths, never the repository's incidental artifacts.
        requested = requested_answer_paths(getattr(self, "statement", ""), self.root)
        known = self.base_paths(left()) if requested else set()
        deliberate_files = [path for path in requested if path not in known
                            and os.path.isfile(self.absolute(path))
                            and not os.path.islink(self.absolute(path))]
        if deliberate_files:
            forced, _ = git(["add", "-f", "-N", "--"] + [":(top,literal)" + path
                            for path in sorted(deliberate_files)], self.root, left())
            if forced != 0 and failed is not None:
                failed.append("force")
        args = ["diff", "--binary", "--no-color"] + ([self.base] if self.base else [])
        code, out = self.exact_diff(args, left)
        if code != 0:
            if deadline - time.monotonic() < 10.0:
                say("[TREE] diff failed (%s) and there is no time to ask again: %s"
                    % (code, out.strip()[:200]))
                if failed is not None:
                    failed.append("diff")
                return ""
            say("[TREE] diff failed (%s), retrying once: %s" % (code, out.strip()[:200]))
            code, out = self.exact_diff(args, left)
            if code != 0:
                say("[TREE] diff failed again: %s" % out.strip()[:200])
                if failed is not None:
                    failed.append("diff")
                return ""
        return out

    def exact_diff(self, args: list, left) -> tuple:
        """The diff as text that still describes the files after it has travelled.

        A file whose hunks hold a carriage return or a byte that is not UTF-8 is
        sent as a binary patch instead: plain ASCII, and exact.
        """
        # Source identity includes executable bits even when the checkout's
        # configuration ignores them. The returned diff must describe the
        # same source, without changing the repository's configuration.
        code, raw = git_bytes(["-c", "core.filemode=true"] + args, self.root, left())
        if code != 0 or not fragile_text(raw):
            return code, raw.decode("utf-8", "replace")
        try:
            exact = self.binary_sections(args, raw, left)
        except BaseException as error:
            exact = None
            say("[TREE] the diff holds bytes that will not travel as text and could "
                "not be re-made exactly: %s" % type(error).__name__)
        if exact is None:
            return 0, raw.decode("utf-8", "replace")
        return 0, exact

    def binary_sections(self, args: list, raw: bytes, left) -> str | None:
        """The diff with binary files handled one section at a time, or None when the listing
        failed.
        """
        listing = ["-c", "core.filemode=true", "-c", "core.quotePath=false",
                   "diff", "--name-only", "-z"] + args[3:]
        code, names = git_bytes(listing, self.root, left())
        if code != 0:
            return None
        paths = [name for name in names.split(b"\0") if name]
        sections = [part if index == 0 else b"diff --git " + part
                    for index, part in enumerate(raw.split(b"\ndiff --git "))]
        sections = [part if part.endswith(b"\n") or index == len(sections) - 1 else part + b"\n"
                    for index, part in enumerate(sections)]
        fragile = []
        if len(sections) == len(paths):
            fragile = [path for path, part in zip(paths, sections)
                       if b"GIT binary patch" not in part and fragile_text(part)]
        else:
            # The two listings did not line up, so ask about each file by name.
            for path in paths:
                one = args + ["--", ":(top,literal)" + os.fsdecode(path)]
                code, part = git_bytes(["-c", "core.filemode=true"] + one,
                                       self.root, left())
                if code == 0 and b"GIT binary patch" not in part and fragile_text(part):
                    fragile.append(path)
        if not fragile:
            return None
        code, where = git(["rev-parse", "--git-path", "info/attributes"], self.root, left())
        if code != 0 or not where.strip():
            return None
        target = os.path.join(self.root, where.strip())
        before = None
        if os.path.isfile(target):
            with open(target, "rb") as handle:
                before = handle.read()
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "ab") as handle:
                if before and not before.endswith(b"\n"):
                    handle.write(b"\n")
                for path in fragile:
                    handle.write(attribute_pattern(path) + b" binary\n")
            code, again = git_bytes(["-c", "core.filemode=true"] + args,
                                    self.root, left())
        finally:
            try:
                if before is None:
                    os.remove(target)
                else:
                    with open(target, "wb") as handle:
                        handle.write(before)
            except OSError:
                pass
        if code != 0 or fragile_text(again):
            return None
        for path in fragile:
            EXACT_BINARY.add(os.fsdecode(path))
        say("[TREE] %d file(s) sent as exact binary patches, because their lines hold "
            "bytes a text diff would lose: %s"
            % (len(fragile), ", ".join(os.fsdecode(p) for p in fragile[:3])))
        return again.decode("utf-8")

    def applies(self, patch: str, budget: float = 30.0) -> bool | None:
        """Does this patch apply to the original checkout? None when the question could not be
        answered.
        """
        if not patch.strip():
            say("[PATCH] empty: the run finished without changing a line")
            return False
        try:
            handle, path = tempfile.mkstemp(prefix="ridges-patch-", suffix=".diff")
        except OSError as error:
            say("[PATCH] could not be written out for checking: %s" % error)
            return None
        try:
            with os.fdopen(handle, "w", encoding="utf-8", errors="surrogateescape") as fh:
                fh.write(patch)
            code, out = git(["apply", "--check", path], self.root, budget)
        except OSError as error:
            say("[PATCH] could not be filled for checking: %s" % error)
            return None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if code == 0:
            say("[PATCH] applies cleanly")
            return True
        if code == GIT_TIMED_OUT:
            say("[PATCH] could not be checked in time")
            return None
        say("[PATCH] will not apply: %s" % out.strip()[:200])
        return False

    def salvage(self, patch: str, budget: float = 45.0) -> str:
        """The largest part of a patch that still applies, when the whole of it does not.

        A patch is split by file and each section is tried on its own, so one unusable
        section costs its own file rather than the whole answer.
        """
        deadline = time.time() + max(1.0, budget)
        parts = split_by_file(patch)
        if len(parts) < 2:
            say("[PATCH] nothing to salvage: %d section(s)" % len(parts))
            return ""
        kept, dropped, unread = [], 0, 0
        for part in parts:
            left = deadline - time.time()
            answer = self.applies_quietly(part, left) if left > 0 else None
            if answer is None:
                unread = len(parts) - len(kept) - dropped
                say("[PATCH] salvage left %d section(s) unread" % unread)
                break
            if answer:
                kept.append(part)
            else:
                dropped += 1
        if not kept:
            say("[PATCH] salvage kept nothing of %d section(s)" % len(parts))
            return ""
        joined = "".join(kept)
        whole = self.applies_quietly(joined, max(1.0, deadline - time.time()))
        if whole is None:
            say("[PATCH] salvage could not re-check its %d section(s)"
                % len(kept))
            return ""
        if not whole:
            say("[PATCH] salvage kept %d section(s) that will not apply together"
                % len(kept))
            return ""
        say("[PATCH] salvaged %d of %d section(s), dropped %d, unread %d"
            % (len(kept), len(parts), dropped, unread))
        return joined

    def applies_quietly(self, patch: str, budget: float) -> bool | None:
        """Does one patch section apply, with nothing said about it either way?"""
        if not patch.strip():
            return False
        try:
            handle, path = tempfile.mkstemp(prefix="ridges-part-", suffix=".diff")
        except OSError:
            return None
        try:
            with os.fdopen(handle, "w", encoding="utf-8",
                           errors="surrogateescape") as fh:
                fh.write(patch)
            code, _ = git(["apply", "--check", path], self.root, budget)
        except OSError:
            return None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if code == GIT_TIMED_OUT:
            return None
        return code == 0

    def _unlock(self, said: str) -> bool:
        """Remove an index lock a stopped process left behind, and say whether that was the
        problem.
        """
        if "index.lock" not in (said or ""):
            return False
        try:
            os.remove(os.path.join(self.root, ".git", "index.lock"))
        except OSError:
            return False
        say("[TREE] removed a lock a stopped process left behind")
        return True

    def restore(self, budget: float = 60.0) -> bool:
        """Put the checkout back to the base commit and remove what this run added.

        True only when the tree is confirmed back as it was, because the answer is read
        against it afterwards.
        """
        deadline = time.monotonic() + budget
        args = (["reset", "--hard", self.base] if self.base
                else ["checkout", "--", "."])
        args = ["-c", "core.filemode=true"] + args
        code, out = git(args, self.root, budget)
        if code != 0 and self._unlock(out):
            code, out = git(args, self.root, max(1.0, deadline - time.monotonic()))
        if code != 0:
            say("[TREE] restore said: %s" % out.strip()[:200])
        listed = self._untracked(max(1.0, deadline - time.monotonic()))
        added = ((listed - self.untracked_at_start) if listed is not None
                 else set())
        gone = 0
        for path in sorted(added, key=len, reverse=True):
            try:
                full = self.absolute(path)
            except ToolFault:
                continue
            try:
                if os.path.isdir(full) and not os.path.islink(full):
                    shutil.rmtree(full)
                else:
                    os.remove(full)
                gone += 1
            except OSError as error:
                say("[TREE] could not remove %s: %s" % (path, error))
        preserved = False
        try:
            self.original_snapshot.restore(deadline)
            preserved = True
        except BaseException as error:
            say("[TREE] original nontracked files were not confirmed restored: %s" % error)
        vouched = code == 0 and listed is not None and gone == len(added) and preserved
        say("[TREE] restored to %s, removed %d of %d path(s) the run created%s"
            % (self.base[:8] or "?", gone, len(added),
               "" if vouched else " -- not confirmed"))
        return vouched

    def close(self):
        """Release the copy of the original checkout."""
        snapshot = getattr(self, "original_snapshot", None)
        if snapshot is not None:
            snapshot.close()

class ToolFault(Exception):
    """A tool was asked for something it will not do, with the reason the model is told."""
    pass

CLIP_NOTE = "\n... [%d characters of %s elided] ...\n"

def sig1(number: str) -> str:
    """A measured number kept to one significant figure, as text."""
    try:
        value = float(number)
    except ValueError:
        return number
    if value == 0:
        return "0"
    digits = math.floor(math.log10(abs(value)))
    rounded = round(value, -digits)
    if digits >= 0:
        return "%d" % int(rounded)
    return "%.*f" % (-digits, rounded)


def sig1_float(value: float) -> float:
    """A number rounded to one significant figure, so a timing in the log cannot be read as exact.
    """
    return float(sig1("%.6f" % value))


# How long a test runner took, and the milliseconds a query plan measured, change
# from one host to the next while the code does not. What the model reads should
# not change with them: runner durations become a marker, measured times keep one
# significant figure. Only these well-known runner lines are touched; data in
# query output is never rewritten.
STEADY_RULES = [
    (re.compile(r"(\bRan \d+ tests? in )\d+(?:\.\d+)?s\b"), r"\1[elapsed]s"),
    (re.compile(r"(\b(?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected|"
                r"no tests ran)\b[^\n]*?) in \d+(?:\.\d+)?s\b(?: \(\d+:\d+:\d+\))?"), r"\1 in [elapsed]s"),
    (re.compile(r"^((?:ok|FAIL)\s+\S+\s+)\d+(?:\.\d+)?s\b", re.M), r"\1[elapsed]s"),
    (re.compile(r"(--- (?:PASS|FAIL|SKIP|BENCH): \S+ \()\d+(?:\.\d+)?s\)"), r"\1[elapsed]s)"),
    (re.compile(r"^(\s*Time:\s+)\d+(?:\.\d+)? ?s\b", re.M), r"\1[elapsed] s"),
    (re.compile(r"(\b\d+ (?:passing|failing|pending)) \(\d+(?:\.\d+)?(?:ms|s)\)"), r"\1 ([elapsed])"),
    (re.compile(r"(\bfinished in )\d+(?:\.\d+)?s\b"), r"\1[elapsed]s"),
    (re.compile(r"^(Time: )\d+(?:\.\d+)? ms$", re.M), r"\1[elapsed] ms"),
]
STEADY_ROUNDED = [
    (re.compile(r"(actual time=)(\d+(?:\.\d+)?)\.\.(\d+(?:\.\d+)?)"),
     lambda m: m.group(1) + sig1(m.group(2)) + ".." + sig1(m.group(3))),
    (re.compile(r"((?:Planning|Execution|Trigger|Timing|Generation|Inlining|Optimization|Emission)"
                r" Time: )(\d+(?:\.\d+)?)( ms)"),
     lambda m: m.group(1) + sig1(m.group(2)) + m.group(3)),
    (re.compile(r"(\bElapsed: )(\d+(?:\.\d+)?)( sec)"),
     lambda m: m.group(1) + sig1(m.group(2)) + m.group(3)),
    (re.compile(r"(\"elapsed\":\s*)(\d+(?:\.\d+)?(?:e-?\d+)?)"),
     lambda m: m.group(1) + sig1(m.group(2))),
]


def steady(text: str) -> str:
    """Command output as the model reads it, with host timing made steady."""
    if not text:
        return text
    for pattern, replacement in STEADY_RULES:
        text = pattern.sub(replacement, text)
    for pattern, replacement in STEADY_ROUNDED:
        text = pattern.sub(replacement, text)
    return text


def clip(text: str, cap: int, label: str = "output") -> str:
    """Text cut to a cap, with a note in the middle saying how much was left out and of what."""
    if len(text) <= cap:
        return text
    keep = cap
    for _ in range(4):
        room = max(0, cap - len(CLIP_NOTE % (len(text) - keep, label)))
        if room == keep:
            break
        keep = room
    if keep <= 0:
        return (CLIP_NOTE % (len(text), label))[:cap]
    return (text[: keep // 2] + (CLIP_NOTE % (len(text) - keep, label))
            + text[len(text) - (keep - keep // 2):])

SHELL_REPORT_CAP = 120
STILL_RUNNING = "[still running]"

def report_shell(job: "Shell", out: str) -> None:
    """Log one command's duration, output size, the command itself and its last meaningful line.
    """
    target = getattr(job, "scrub_target", None) or getattr(job, "sql_target", None)
    if target is not None:
        out = scrub(out, target)
    tail = ""
    for line in reversed((out or "").splitlines()):
        if line.strip() and line.strip() != STILL_RUNNING:
            tail = line.strip()
            break
    say("[SHELL] %.1fs %dc :: %s :: %s"
        % (time.time() - job.started, len(out or ""),
           scrub(" ".join(job.command.split()), target)[:SHELL_REPORT_CAP],
           tail[:SHELL_REPORT_CAP]))

FINDING_CHECK = re.compile(r"^\s*ruff\s+check\b")
FINDING_ARROW = re.compile(r"^\s*-->\s+(\S+?):(\d+):\d+\s*$", re.M)
FINDING_CONCISE = re.compile(r"^(\S+?):(\d+):\d+:\s", re.M)
HUNK_HEAD = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,(\d+))? @@")

def path_tail(name: str, root: str = "") -> str:
    """A path as the repository names it: no root prefix, no leading dot segment, forward slashes.
    """
    text = str(name or "").replace("\\", "/")
    base = str(root or "").replace("\\", "/").rstrip("/")
    if base and text.startswith(base + "/"):
        text = text[len(base) + 1:]
    while text.startswith("./"):
        text = text[2:]
    return text

def findings_from_text(out: str, root: str = "") -> dict:
    """The file and line numbers a checker reported, read from its plain output."""
    text = str(out or "")
    rows: dict = {}
    for pattern in (FINDING_ARROW, FINDING_CONCISE):
        for name, row in pattern.findall(text):
            try:
                number = int(row)
            except ValueError:
                continue
            if number >= 1:
                rows.setdefault(path_tail(name, root), set()).add(number)
        if rows:
            return rows
    return findings_from_json(text, root)

def findings_from_json(text: str, root: str = "") -> dict:
    """The file and line numbers a checker reported, read from its JSON output."""
    try:
        items = json.loads(text or "")
    except (TypeError, ValueError):
        return {}
    if not isinstance(items, list):
        return {}
    out: dict = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        where = item.get("location")
        row = where.get("row") if isinstance(where, dict) else None
        name = str(item.get("filename") or "")
        if not name or isinstance(row, bool) or not isinstance(row, int) or row < 1:
            continue
        out.setdefault(path_tail(name, root), set()).add(row)
    return out

def split_by_file(patch: str) -> list[str]:
    """A patch split into one section per file, each starting at its own diff header."""
    out: list[str] = []
    current: list[str] = []
    for line in (patch or "").splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                out.append("".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        out.append("".join(current))
    return out

ENVELOPE_JUNK_DIRS = ("__pycache__", ".pytest_cache", ".ruff_cache",
                      ".mypy_cache", ".hypothesis", ".tox", "node_modules",
                      "htmlcov", ".idea", ".vscode")
ENVELOPE_JUNK_ENDS = (".pyc", ".pyo", ".pyd", ".orig", ".rej", ".bak", ".swp",
                      ".log", ".sqlite3", ".sqlite", ".db", ".egg-info")
ENVELOPE_JUNK_NAMES = (".DS_Store", ".coverage", "nohup.out")

def section_path(section: str) -> str:
    """The path one patch section changes, taken from its header."""
    head = section.split("\n", 1)[0]
    if not head.startswith("diff --git "):
        return ""
    body = head[len("diff --git "):].strip()
    halves = body.split(" b/", 1)
    if len(halves) == 2:
        return halves[1].strip().strip('"')
    return body.strip().strip('"')

def junk_path(path: str) -> bool:
    """Is this path a by-product of running the project rather than part of an answer?"""
    parts = [part for part in path.split("/") if part]
    if not parts:
        return False
    if parts[-1] in ENVELOPE_JUNK_NAMES:
        return True
    if any(part in ENVELOPE_JUNK_DIRS for part in parts):
        return True
    return parts[-1].endswith(ENVELOPE_JUNK_ENDS)

def operation_path(path: str, root: str = "") -> str:
    """Normalize a repository path without stripping real a/ or b/ folders."""
    path = str(path or "").replace("\\", "/")
    base = os.path.normpath(root).replace("\\", "/") if root else ""
    if base and path.startswith(base + "/"):
        path = path[len(base) + 1:]
    return os.path.normpath(path).replace("\\", "/")

def quoted_git_path(path: str) -> str:
    """Decode Git's optional C-quoted paths, including octal UTF-8 bytes."""
    if not path.startswith('"'):
        return path
    try:
        decoded = ast.literal_eval(path)
        return decoded.encode("latin-1").decode("utf-8", "surrogateescape")
    except (ValueError, SyntaxError, UnicodeError):
        return path.strip('"')

def section_operations(section: str) -> tuple[tuple, set]:
    """Read patch headers; rename touches both ends, copy only its target."""
    paths, operations = [], set()
    copy_target = None
    for line in section.splitlines():
        if line.startswith(("@@", "literal ", "delta ")):
            break
        for marker in ("rename from ", "rename to ", "copy from ", "copy to "):
            if line.startswith(marker):
                paths.append(quoted_git_path(line[len(marker):]))
                operations.add("rename" if marker.startswith("rename") else "create")
                if marker == "copy to ":
                    copy_target = quoted_git_path(line[len(marker):])
        if line.startswith(("--- ", "+++ ")):
            path = quoted_git_path(line[4:].rstrip("\t"))
            if path != "/dev/null":
                paths.append(path[2:] if path.startswith(("a/", "b/")) else path)
        if line.startswith("new file mode "):
            operations.add("create")
        elif line.startswith("deleted file mode "):
            operations.add("delete")
        elif line.startswith(("old mode ", "new mode ")):
            operations.add("mode")
        elif line.startswith(("GIT binary patch", "Binary files ")):
            operations.add("binary")
    if not paths:
        head = section.split("\n", 1)[0][len("diff --git "):]
        try:
            pieces = re.findall(r'"(?:\\.|[^"\\])*"|[^\s]+', head)
            if len(pieces) == 2:
                paths = [quoted_git_path(p)[2:] for p in pieces]
        except ValueError:
            pass
        if not paths:
            paths = [section_path(section)]
    if copy_target is not None:
        paths = [copy_target]
    return tuple(dict.fromkeys(operation_path(path) for path in paths)), operations

def operation_literals(clause: str) -> list[str]:
    """Literal path operands, including absolute paths and spaces in names."""
    return [raw for raw in re.findall(r"`([^`\n]+)`", clause)
            if not any(char in raw for char in "(){};=")
            and ("/" in raw or "." in os.path.basename(raw)
                 or re.search(r"\b(?:files?|directory|directories|under)\b", clause, re.I))]

# Words a sentence uses when it says where the change goes. A path in a
# sentence without one names a place for another reason.
EDIT_WORDS = re.compile(r"\b(?:edit(?:s|ed|able)?|change[sd]?|modif(?:y|ied|ications?)|fix(?:es|ed)?|"
                        r"repair(?:s|ed)?|implement(?:s|ed)?|rewrite|update[sd]?|touch(?:ed)?|"
                        r"method|function|body|writable)\b", re.I)
# "Editable production paths: app/, migrations/, setup.py." names the paths
# that may change without quoting them. The tokens read are directories (a
# trailing slash) and files with a code or configuration extension, nothing
# else, and only from a sentence that grants the edit.
PERMISSION_PATH = re.compile(
    r"(?<![\w`/.-])((?:[A-Za-z0-9_.-]+/)+|[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*"
    r"\.(?:py|pyi|ts|tsx|js|jsx|mjs|go|rb|rs|java|sql|cs|kt|kts|php|scala|json|ya?ml|toml|csproj|proto))(?![\w`])")
PERMISSION_CLAUSE = re.compile(
    r"\beditable\b|\bproduction\s+(?:scope|code|paths?|directories|modules?)\b|\bmay\s+(?:change|be\s+edited|be\s+implemented)\b|"
    r"\ballowed\s+(?:edits?|changes?)\b|\bwithin\s+`?[\w./-]+/`?", re.I)


def permission_paths(clause: str) -> list[str]:
    """Paths a permission sentence lists without quoting them, or []."""
    if not PERMISSION_CLAUSE.search(clause):
        return []
    found = []
    for raw in PERMISSION_PATH.findall(clause):
        token = raw.strip()
        if token and token not in found and not token.lower().startswith(("e.g", "i.e", "vs.")):
            found.append(token)
    return found


def path_boundaries(statement: str, root: str = "") -> list:
    """Explicit path boundaries, independent of which operations occur there."""
    if not RIDGES_SCOPE_FOLLOWS_STATEMENT or not scope_clause(statement):
        return []
    found = []
    for part in instruction_clauses(statement):
        clause = scope_clause(part)
        if not clause:
            continue
        raw_paths = operation_literals(clause)
        if not raw_paths and method_clause(part):
            # A method-only clause can refer back to one named source file,
            # when the statement names that file as the place of the change.
            # A file named for another reason ("constants live in `x`") is not
            # the boundary of the change, and treating it as one would drop the
            # real edit from the answer.
            named = {p for item in instruction_clauses(statement)
                     if item == part or scope_clause(item) or EDIT_WORDS.search(item)
                     for p in FENCE_PATH.findall(item)}
            raw_paths = list(named) if len(named) == 1 else []
        if not raw_paths:
            raw_paths = permission_paths(clause)
        edges = []
        for raw in raw_paths:
            directory = raw.endswith("/") or bool(re.search(
                r"\b(?:under|beneath|directory|directories)\b", clause, re.I))
            edges.append((operation_path(raw, root), directory))
        if edges:
            found.append((clause, edges))
    return found

def operation_refusal(operation: str, paths, statement: str, root: str = "") -> str:
    """Return an actual current-task prohibition applying to this operation."""
    verbs = {
        "create": r"(?:add|create|introduce)",
        "delete": r"(?:delete|remove)",
        "rename": r"(?:rename|move)",
        "mode": r"(?:chmod|change|alter|modify|make)",
        "binary": r"(?:add|create|modify|change|edit)",
    }
    verb = verbs.get(operation)
    if not verb:
        return ""
    for part in instruction_clauses(statement):
        for chunk in re.split(r"\s+(?:but|however)\s+", part, flags=re.I):
            refusal = re.search(r"\b(?:do not|don't|never|must not|should not)\s+(.+)",
                                chunk, re.I)
            if not refusal:
                continue
            tail = refusal.group(1)
            for action in re.finditer(r"\b" + verb + r"\b", tail, re.I):
                object_text = tail[action.end():].lstrip()
                coordinated = (r"^(?:(?:or|and)\s+(?:add|create|introduce|delete|remove|"
                               r"rename|move|chmod|change|alter|modify|make)\s+)+")
                object_text = re.sub(coordinated, "", object_text, flags=re.I)
                object_text = re.split(
                    r"\s+(?:or|and)\s+(?:add|create|introduce|delete|remove|rename|"
                    r"move|chmod|change|alter|modify|make)\b", object_text, flags=re.I)[0]
                direct = re.sub(
                    r"^(?:(?:the|a|an|any|all|existing|new|these|those|other)\s+)+",
                    "", object_text, flags=re.I)
                # A filename used to locate a function is not the object of
                # "do not rename the function". Bind the verb before reading
                # optional location operands, rather than treating every path
                # mentioned later in the sentence as a prohibited file action.
                literal = re.match(r"`([^`]+)`", direct)
                direct_path = bool(literal and operation_literals(literal.group(0)))
                file_object = direct_path or re.match(
                    r"(?:(?:binary|source|tracked|untracked|generated)\s+)?"
                    r"(?:files?|paths?|directories|directory)\b", direct, re.I)
                if operation == "mode":
                    applies = (re.match(r"(?:permissions?|modes?|executable\s+bits?)\b", direct, re.I)
                               or file_object and re.search(r"\b(?:permissions?|modes?|executable)\b", direct, re.I)
                               or action.group(0).lower() == "chmod" and direct_path)
                elif operation == "binary":
                    applies = re.match(r"binary\s+(?:files?|patches?|blobs?|assets?)\b", direct, re.I)
                else:
                    applies = file_object
                targets = [operation_path(raw, root) for raw in operation_literals(object_text)]
                if applies and (not targets or any(path_within(path, edge)
                                                   for path in paths for edge in targets)):
                    return chunk.strip()
    return ""

def operation_violation(paths, operations, statement: str, root: str = "") -> str:
    """Why a change to these paths is outside what the task allows, in the task's own words, or
    empty.

    The statement's own path boundaries are checked first, then each operation the change
    performs, so the reason given back always quotes the clause that forbids it.
    """
    paths = tuple(operation_path(path, root) for path in paths)
    for clause, edges in path_boundaries(statement, root):
        outside = [path for path in paths if not any(
            path_within(path, edge) if directory else path == edge
            for edge, directory in edges)]
        if outside:
            return "%s is outside the task's path boundary: %s" % (outside[0], clause)
    for operation in sorted(operations):
        clause = operation_refusal(operation, paths, statement, root)
        if clause:
            return "%s on %s is prohibited by the task: %s" % (
                operation, ", ".join(paths), clause)
    # Preserve deliberately requested artifacts; still exclude incidental
    # build output from bounded source changes, as the by-product filter does.
    bounded = path_boundaries(statement, root)
    if "create" in operations and bounded:
        named = requested_answer_paths(statement, root)
        for path in paths:
            if junk_path(path) and path not in named:
                return "adds %s, which is unrequested build or scratch output" % path
    # A statement that confines the change to one method without naming its file
    # ("find the method that ... and change only that method") still bounds the
    # files: the method is in a file that exists, so adding, deleting or renaming
    # a file is outside it.
    touched = operations & {"create", "delete", "rename"}
    if touched and not bounded and RIDGES_SCOPE_FOLLOWS_STATEMENT:
        clause = method_clause(statement)
        named = requested_answer_paths(statement, root)
        if clause and not all(path in named for path in paths):
            return "%s %s, and the task confines the change to one method: %s" % (
                {"create": "adds", "delete": "deletes", "rename": "renames"}[sorted(touched)[0]],
                ", ".join(paths), clause)
    return ""

def envelope_reason(section: str, allowed=(), bounds=(), statement: str = "", root: str = "") -> str:
    """Why one patch section lies outside what the task allows, or empty when it does not."""
    paths, operations = section_operations(section)
    # Modes can be removed without losing the otherwise permitted content.
    return operation_violation(paths, operations - {"mode"}, statement, root)

def demoded(section: str, statement: str = "", root: str = "") -> tuple[str, str]:
    """The section with a prohibited mode change removed, and what was removed.

    A mode change can be dropped without losing the content of the edit, so a file-
    permission change the task forbids does not cost the edit itself.
    """
    paths, operations = section_operations(section)
    clause = operation_refusal("mode", paths, statement, root) if "mode" in operations else ""
    if not clause:
        return section, ""
    kept, taken = [], ""
    for line in section.splitlines(keepends=True):
        head = line.rstrip("\r\n")
        if head.startswith("old mode ") or head.startswith("new mode "):
            taken = head.strip()
            continue
        kept.append(line)
    if not taken:
        return section, ""
    return "".join(kept), ("drops the mode change on %s (%s): %s"
                           % (section_path(section), taken, clause))

NEW_FILE_MARK = re.compile(r"^new file mode ", re.M)

def byproduct_trim(patch: str, beacon: "Beacon", statement: str = "",
                   root: str = "") -> str:
    """Drop files this run produced while running commands.

    Explicitly requested files are retained regardless of their artifact-like
    names. Unrequested new files matching the existing by-product profile may
    be dropped; the writing tool alone does not establish their purpose.
    """
    asked = requested_answer_paths(statement, root)
    sections = split_by_file(patch)
    kept, dropped = [], []
    for section in sections:
        paths, _ = section_operations(section)
        path = paths[-1] if paths else section_path(section)
        if (NEW_FILE_MARK.search(section) and byproduct_path(path)
                and path not in asked):
            dropped.append(path)
            continue
        kept.append(section)
    if not dropped:
        beacon.skipped("no by-product in %d section(s)" % len(sections))
        return patch
    joined = "".join(kept)
    if not joined.strip():
        beacon.fired("every section looks like a by-product; kept the answer "
                     "whole: %s" % ", ".join(dropped[:3]))
        return patch
    beacon.fired("dropped %d file(s) a command produced: %s"
                 % (len(dropped), ", ".join(dropped[:3])))
    return joined

def envelope_or_whole(patch: str, beacon: "Beacon", statement: str = "", root: str = "") -> str:
    """The envelope trim, unless it would leave nothing.

    A change that lands outside the boundary this run read from the statement
    still goes out whole: the reading may be wrong, and nothing is no answer.
    """
    if not (PATCH_ENVELOPE and patch.strip()):
        return patch
    trimmed = envelope_trim(patch, beacon, statement, root)
    if trimmed.strip():
        return trimmed
    try:
        beacon.fired("the envelope would leave nothing; the answer goes out whole")
    except BaseException as error:
        say("[ENVELOPE] the note was not written: %s" % type(error).__name__)
    return patch

def envelope_trim(patch: str, beacon: "Beacon", statement: str = "", root: str = "") -> str:
    """The answer with every section the task's own words put out of bounds removed.

    Each dropped section is reported with the clause that forbade it, so a boundary this run
    read wrongly can be seen in the log rather than guessed at.
    """
    if not RIDGES_SCOPE_FOLLOWS_STATEMENT:
        beacon.skipped("statement filtering is disabled")
        return patch
    sections = split_by_file(patch)
    kept, dropped = [], []
    for section in sections:
        why = envelope_reason(section, statement=statement, root=root)
        if why:
            dropped.append(why)
            continue
        trimmed, note = demoded(section, statement, root)
        if note:
            dropped.append(note)
        remaining_operations = section_operations(trimmed)[1]
        if note and "\n@@" not in trimmed and not remaining_operations:
            dropped[-1] = note.replace("drops the mode change on",
                                       "drops the mode-only section for")
        else:
            # Rename-only, mode-only and binary patches are complete sections.
            kept.append(trimmed)
    if not dropped:
        beacon.skipped("nothing to drop from %d section(s)" % len(sections))
        return patch
    joined = "".join(kept)
    before = beacon.artefact("before", patch)
    beacon.fired("dropped %d of %d section(s): %s"
                 % (len(dropped), len(sections), "; ".join(dropped[:3])))
    beacon.outcome(before, joined)
    return joined

# There is deliberately no list of constructs a task is assumed to forbid.
# Which ones are ruled out is read from the statement, by
# statement_refused_nodes, and a task that rules out nothing gets no such
# warning: a loop, a comprehension, a nested helper or a dunder name is
# ordinary Python, and calling it a violation because the change is bounded to
# one method states a rule the task never gave.

def contract_readable(path: str) -> bool:
    """Can this run read the contract of this file, which today means a Python source file?"""
    return path.endswith(".py")

def python_functions(text: str) -> tuple[dict, dict, dict]:
    """Which function or class each line of a Python file belongs to, with their heads and first
    lines.
    """
    import ast
    lines = text.splitlines(keepends=True)
    owner: dict = {}
    heads: dict = {}
    first: dict = {}

    def visit(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            named = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef))
            name = (prefix + "." + child.name).lstrip(".") if named else prefix
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                top = min([line.lineno for line in child.decorator_list]
                          + [child.lineno])
                body = child.body[0].lineno if child.body else child.lineno + 1
                heads[name] = "".join(lines[top - 1:body - 1])
                for line in range(top, (child.end_lineno or top) + 1):
                    owner[line] = name
                first[name] = ast.dump(child.body[0]) if child.body else ""
            visit(child, name)
    visit(ast.parse(text), "")
    return owner, heads, first

def changed_line_numbers(before: str, after: str) -> tuple[set, set]:
    """The line numbers that differ between two texts, as (lines in the old, lines in the new).
    """
    import difflib
    old, new = before.splitlines(), after.splitlines()
    was, now = set(), set()
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        was.update(range(i1 + 1, i2 + 1))
        now.update(range(j1 + 1, j2 + 1))
    return was, now

def contract_violations(path: str, before: str, after: str,
                        statement: str = "") -> tuple[list, list]:
    """What this edit breaks, and what the statement itself rules out.

    Two things, and no third. A file that stopped compiling, which is a fact
    about the edit rather than a reading of the task. And the constructs the
    statement forbids in its own words, quoted back with the clause that
    forbids them.

    Nothing is inferred from the shape of the permission. Bounding a change to
    one method says where the change goes; it says nothing about whether that
    method may hold a loop, open with a different statement, or define a
    helper inside itself. Telling the model otherwise invents a requirement
    and spends the run satisfying it.
    """
    import ast
    hard: list[str] = []
    soft: list[str] = []
    if not contract_readable(path):
        return hard, soft
    try:
        compile(before, path, "exec")
    except SyntaxError:
        return hard, soft
    try:
        compile(after, path, "exec")
    except SyntaxError as error:
        return (["%s does not compile: %s (line %s). It compiled before this "
                 "run touched it, so the edit introduced that. A file that will "
                 "not parse fails everything that imports it."
                 % (path, error.msg, error.lineno)], soft)
    refused = statement_refused_nodes(statement)
    keeps_signature = signature_clause(statement)
    effects = statement_side_effects(statement, all_clauses=True)
    if not refused and not keeps_signature and not effects:
        return hard, soft
    try:
        owner, heads, first = python_functions(after)
        was_owner, was_heads, _ = python_functions(before)
    except (SyntaxError, ValueError, RecursionError):
        return hard, soft
    _, now = changed_line_numbers(before, after)
    if keeps_signature:
        targets = refusal_targets(keeps_signature, heads)
        for name in sorted(set(heads) & set(was_heads)):
            if targets and name not in targets:
                continue
            if heads[name] == was_heads[name]:
                continue
            soft.append(
                "%s: the signature or decorators of %s are not the ones that "
                "were there, and the statement asks for that to stay as it is: "
                "\"%s\". Put the def line back and make the change in the body."
                % (path, name, keeps_signature))
    if refused:
        tree = ast.parse(after)
        seen: set = set()
        for node in ast.walk(tree):
            line = getattr(node, "lineno", 0)
            if line not in now:
                continue
            kind = type(node).__name__
            if kind not in refused or kind in seen:
                continue
            clause = refused[kind]
            # The clause may be about one function. Quoting it is not enough:
            # it has to be about the code being refused, or a task that rules
            # a loop out of one method and asks for one in another refuses the
            # work it just requested.
            targets = refusal_targets(clause, heads)
            if targets:
                holder = owner.get(line, "")
                if not holder or holder not in targets:
                    continue
            seen.add(kind)
            where = (" in %s" % owner.get(line)) if targets else ""
            hard.append(
                "%s line %d introduces the construct %s%s, and the statement "
                "rules it out: \"%s\". Express the change without it."
                % (path, line, kind, where, clause))
    if effects:
        # Compare each clause within its own scope. Counts in an unrestricted
        # function cannot cancel a newly introduced call in a restricted one.
        old = effect_calls(ast.parse(before), {kind for kind, _ in effects}, provenance=True)
        calls = effect_calls(ast.parse(after), {kind for kind, _ in effects}, provenance=True)
        told: set = set()
        for kind, clause in effects:
            scope_match = re.search(r"\b(?:in|inside|within|to|for)\s+(?:(?:the|function|method|file|module)\s+)*([`A-Za-z_].*)", clause, re.I)
            scope_text = scope_match.group(1) if scope_match else ""
            explicit_paths = FENCE_PATH.findall(scope_text)
            if explicit_paths and fence_path(path) not in {fence_path(item) for item in explicit_paths}:
                continue
            explicit_calls = set(re.findall(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(\s*\)", scope_text))
            names = set(heads) | set(was_heads)
            targets = refusal_targets(scope_text, names)
            if explicit_calls:
                targets = set()
                for target in explicit_calls:
                    targets.update({name for name in names if name.split(".")[-1] == target}
                                   if "." not in target else {target})
                    if not any(name == target or name.endswith("." + target) for name in names):
                        targets.add(target)  # An absent named target is not a file-wide ban.
            def selected(item, owners, kind=kind, targets=targets):
                return item[0] == kind and (not targets or owners.get(item[2], "") in targets)
            def identity(item, owners):
                return owners.get(item[2], ""), item[1], item[4]
            added = (collections.Counter(identity(item, owner) for item in calls if selected(item, owner))
                     - collections.Counter(identity(item, was_owner) for item in old if selected(item, was_owner)))
            for item in calls:
                _, _, line, shown, resolved = item
                key = identity(item, owner)
                if not selected(item, owner) or key not in added or (kind, key, clause) in told:
                    continue
                told.add((kind, key, clause))
                if resolved:
                    hard.append(
                        "%s line %d adds or rebinds a call to %s(), which %s, and the statement "
                        "rules out that kind of side effect: \"%s\". Express the change without it."
                        % (path, line, shown, EFFECT_WHAT[kind], clause))
                else:
                    soft.append(
                        "%s line %d calls %s(). Unverified side-effect candidate: the name alone "
                        "does not establish %s. Inspect its binding and implementation against "
                        "the current clause \"%s\"; this is not a confirmed violation."
                        % (path, line, shown, kind, clause))
    return hard, soft

def patch_touched_lines(patch: str) -> dict:
    """The line numbers each file's section of a patch changes, keyed by path."""
    out: dict = {}
    where = ""
    came_from = ""
    old = 0
    left = 0
    right = 0
    for line in (patch or "").splitlines():
        if left <= 0 and right <= 0:
            if line.startswith("diff "):
                where = ""
                came_from = ""
                old = 0
                continue
            if line.startswith("--- "):
                came_from = _diff_side(line[len("--- "):], "a/")
                continue
            if line.startswith("+++ "):
                where = _diff_side(line[len("+++ "):], "b/") or came_from
                continue
            head = HUNK_HEAD.match(line)
            if head:
                old = int(head.group(1))
                left = int(head.group(2) or 1)
                right = int(head.group(3) or 1)
            continue
        if line.startswith("\\"):
            continue
        if line.startswith("-"):
            if where and old:
                out.setdefault(where, set()).add(old)
            old += 1
            left -= 1
        elif line.startswith("+"):
            if where and old:
                out.setdefault(where, set()).add(old)
            right -= 1
        else:
            old += 1
            left -= 1
            right -= 1
    return out

def _diff_side(name: str, prefix: str) -> str:
    """One side of a diff header as a repository path, or empty when that side is absent."""
    text = name.strip()
    if text == "/dev/null" or not text:
        return ""
    if text.startswith(prefix):
        text = text[len(prefix):]
    return path_tail(text)

def finding_distances(findings: dict, touched: dict) -> list:
    """For every changed line, how far it is from the nearest line a checker reported."""
    out = []
    for where in sorted(touched):
        rows = findings.get(where)
        if not rows:
            continue
        for line in sorted(touched[where]):
            out.append(min(abs(line - row) for row in rows))
    return out

def finding_record(findings: dict, touched: dict) -> str:
    """A sentence on how the change lines up with what a checker reported, for the log.

    A change far from everything reported, or in files nothing reported, is worth seeing: it
    may be right, and it may be work the task did not ask for.
    """
    reported = sum(len(rows) for rows in findings.values())
    changed = sum(len(rows) for rows in touched.values())
    silent = sum(1 for where in touched if where not in findings)
    gaps = sorted(finding_distances(findings, touched))
    if not gaps:
        return ("reported %d line(s) over %d file(s); patch changes %d line(s) over "
                "%d file(s), none in a file the check reported"
                % (reported, len(findings), changed, len(touched)))
    return ("reported %d line(s) over %d file(s); patch changes %d line(s), nearest "
            "reported line median=%d p90=%d beyond40=%d of %d; %d file(s) unreported"
            % (reported, len(findings), changed,
               gaps[len(gaps) // 2], gaps[min(len(gaps) - 1, int(len(gaps) * 0.9))],
               sum(1 for gap in gaps if gap > 40), len(gaps), silent))

class FindingMap:
    """What the project's own checkers reported during this run, by file and line.

    The lines a checker names are read from its output as it goes, and compared at the end
    with the lines the answer changes.
    """
    def __init__(self, root: str) -> None:
        """Start an empty record of what the checkers reported in this checkout."""
        self.root = root
        self.rows: dict = {}
        self.reads = 0
        self.beacon = Beacon("findings")

    def observe(self, command: str, out: str) -> None:
        """Read one checker's output and keep the file and line numbers it reported that are new.
        """
        text = one_line(command)
        if not FINDING_CHECK.match(text):
            return
        self.reads += 1
        fresh = {where: rows for where, rows in findings_from_text(out, self.root).items()
                 if where not in self.rows}
        if not fresh:
            self.beacon.skipped("nothing new in the output of: %s" % text[:120])
            return
        self.rows.update(fresh)
        self.beacon.fired("%s -> %d line(s) over %d file(s)"
                          % (text[:120], sum(len(rows) for rows in fresh.values()), len(fresh)))

    def report(self, patch: str) -> None:
        """Log how the answer's changed lines line up with everything the checkers reported."""
        if not self.rows:
            self.beacon.skipped("no reading of the check was recorded")
            return
        self.beacon.fired(finding_record(self.rows, patch_touched_lines(patch)))

_VENV_BIN = ""

def venv_bin() -> str:
    """The project's own virtual environment bin directory, found once, or empty when it has none.
    """
    global _VENV_BIN
    if _VENV_BIN == "":
        _VENV_BIN = "-"
        for where in ("/opt/venv/bin", "/usr/local/venv/bin", ".venv/bin", "../venv/bin"):
            python = os.path.join(where, "python")
            if os.path.isfile(python) and os.access(python, os.X_OK):
                _VENV_BIN = os.path.abspath(where)
                break
    return "" if _VENV_BIN == "-" else _VENV_BIN

def repo_python() -> str:
    """The interpreter this run is running under, which is the one the project's tools expect.
    """
    return sys.executable or "python3"

PACKAGES_AT_ONCE = 1

def command_env(pack_venv: bool = True) -> dict:
    """The environment a shell command runs in: the project's tools on the path and its own build
    flags kept.
    """
    env = dict(os.environ)
    if pack_venv and PACK_VENV:
        where = venv_bin()
        if where and where not in env.get("PATH", "").split(os.pathsep):
            env["PATH"] = where + os.pathsep + env.get("PATH", "")
    flags = env.get("GOFLAGS", "")
    try:
        flag_words = shlex.split(flags)
    except ValueError:
        flag_words = flags.split()
    if not any(word == "-p" or word.startswith("-p=") for word in flag_words):
        env["GOFLAGS"] = flags + (" " if flags else "") + "-p=%d" % PACKAGES_AT_ONCE
    if "GOMAXPROCS" not in env:
        env["GOMAXPROCS"] = str(PACKAGES_AT_ONCE)
    return env

JOB_TAG_NAME = "AGENT_JOB_TAG"
JOB_TAG = "job-%d-%d" % (os.getpid(), int(time.time() * 1000) % 1_000_000_000)

def tagged_processes(tag: str) -> list:
    """Processes still carrying this run's tag in their environment.

    Killing a job's process group misses a child that moved itself into a new
    session. Every command this run starts inherits the tag, so what is left
    over can be found by it. Only this user's processes are readable, which
    is also all this run could have started.
    """
    found = []
    wanted = ("%s=%s" % (JOB_TAG_NAME, tag)).encode()
    try:
        names = os.listdir("/proc")
    except OSError:
        return found
    for name in names:
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            with open("/proc/%s/environ" % name, "rb") as handle:
                if wanted in handle.read().split(b"\0"):
                    found.append(int(name))
        except OSError:
            continue
    return found

SHELL_READ_CAP = 128_000
OUTPUT_INCOMPLETE_MARKER = "[command output incomplete:"

def bounded_shell_output(path: str) -> str:
    """Read a bounded snapshot of a spool, with omission visible at the start.

    Seeking over the middle bounds allocation and repeated polling work. An
    excerpt is diagnostic output, never a complete verification transcript.
    The spool itself remains untouched until the owning job is closed.
    """
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size <= SHELL_READ_CAP:
            raw = handle.read(size)
        else:
            notice = ("%s %d bytes produced; middle omitted; verdict unresolved]\n"
                      % (OUTPUT_INCOMPLETE_MARKER, size)).encode("ascii")
            gap = b"\n[... omitted command output ...]\n"
            room = max(0, SHELL_READ_CAP - len(notice) - len(gap))
            head_size, tail_size = room // 2, room - room // 2
            head = handle.read(head_size)
            handle.seek(size - tail_size)
            raw = notice + head + gap + handle.read(tail_size)
    # Match the previous text-mode reader's newline treatment.
    return raw.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")

class Shell:
    """One shell command, run in the background and read while it runs.

    Output goes to a file rather than a pipe so a command that writes a great deal cannot
    block, descendants are tracked so nothing is left running after the command is done, and
    credentials are passed through the environment rather than the command line where they
    would reach the process table and the log.
    """
    counter = 0

    def __init__(self, command: str, cwd: str, pack_venv: bool = True,
                 hard_timeout: float | None = None, env_extra: dict | None = None) -> None:
        """Start one command in its own process group, with its output going to a file."""
        Shell.counter += 1
        self.name = "job%d" % Shell.counter
        self.command = command
        self.started = time.time()
        self.started_monotonic = time.monotonic()
        env = command_env(pack_venv)
        if env_extra:
            # A credential belongs here and not in the command line, where it
            # would reach the process table, the log and the transcript.
            env.update(env_extra)
        where = venv_bin() if pack_venv and PACK_VENV else ""
        if where:
            command = "export PATH=%s:$PATH\n%s" % (shlex.quote(where), command)
        # Keep invocation identity separate from the short, redacted display.
        # Values (including credentials) participate only through a digest.
        self.invocation_key = command_identity(command)
        # The private ownership tag varies per process, not per execution
        # environment. Exclude it from comparable check identities.
        env.pop(JOB_TAG_NAME, None)
        self.environment_key = environment_identity(env)
        self.owner_tag = "%s-%s-%x" % (JOB_TAG, self.name, time.monotonic_ns())
        env[JOB_TAG_NAME] = self.owner_tag
        self.cwd = os.path.realpath(cwd)
        self.gate = threading.RLock()
        self.owned_pids: dict = {}
        self.sink = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", errors="replace", suffix=".out", delete=False
        )
        try:
            self.process = subprocess.Popen(
                job_argv(command),
                cwd=cwd,
                env=env,
                # Never this process's own input: a command that reads it would
                # wait on a pipe nobody is going to write to.
                stdin=subprocess.DEVNULL,
                stdout=self.sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self.sink.close()
            try:
                os.unlink(self.sink.name)
            except OSError:
                pass
            raise
        self.watchdog = None
        self.timed_out = False
        self.closed = False
        self.cached = None
        self.cached_code = None
        self.output_complete = True
        if hard_timeout is not None:
            self.watchdog = threading.Timer(max(1.0, hard_timeout), self._expire)
            self.watchdog.daemon = True
            self.watchdog.start()

    def _expire(self) -> None:
        """Mark the command as having run out of time and stop it and anything it started."""
        with self.gate:
            if not self.closed and (self.process.poll() is None or self._owned()):
                self.timed_out = True
                self._kill()

    def code(self):
        """The exit status, from the cache once the job has been closed."""
        if self.closed:
            return self.cached_code
        code = self.process.poll()
        if code is not None and not self._settle_descendants():
            # A detached child still owns the command's resources. Keep its
            # lane reserved until termination is independently observed.
            return None
        return code

    def result(self, out: str) -> str:
        """The command's status line and its output, clipped, as the model will read it."""
        code = self.code()
        status = "running" if code is None else "exit_code=%d" % code
        if self.timed_out:
            status += " timed_out=true"
        return "[%s]\n%s" % (status, clip(steady(out), SHELL_OUTPUT_CAP, "shell output") or "(no output)")

    def _owned(self) -> list:
        """The live processes this command is responsible for, by tag and by recorded start time.

        A process id can be reused, so a recorded start time has to match before a
        process is killed as this command's own.
        """
        with self.gate:
            found = []
            tagged = set(tagged_processes(self.owner_tag))
            for pid in tagged | set(self.owned_pids):
                try:
                    with open("/proc/%d/stat" % pid) as handle:
                        fields = handle.read().rsplit(")", 1)[-1].split()
                    state, started = fields[0], fields[19]
                    if pid not in tagged and self.owned_pids.get(pid) != started:
                        self.owned_pids.pop(pid, None)
                        continue
                    self.owned_pids[pid] = started
                    if state != "Z":
                        found.append(pid)
                    else:
                        # Reap adopted grandchildren when this process is their
                        # parent. Popen remains responsible for its direct child.
                        if pid != self.process.pid:
                            try:
                                os.waitpid(pid, os.WNOHANG)
                            except (ChildProcessError, OSError):
                                pass
                        self.owned_pids.pop(pid, None)
                except (OSError, IndexError):
                    self.owned_pids.pop(pid, None)
            return found

    def _kill(self) -> None:
        """Stop the command and everything it started, the process group first and strays by id.
        """
        owned = self._owned()
        group_owned = self.process.poll() is None
        if not group_owned:
            for pid in owned:
                try:
                    group_owned = os.getpgid(pid) == self.process.pid
                except OSError:
                    continue
                if group_owned:
                    break
        # An exited leader's group may still contain owned descendants. Verify
        # ownership before signaling its old identifier, which could be reused.
        if group_owned:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except OSError:
                pass
        for pid in owned:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def _settle_descendants(self, timeout: float = 1.0) -> bool:
        """Wait for every process this command started to be gone, and say whether they are."""
        with self.gate:
            until = time.monotonic() + max(0.0, timeout)
            while True:
                owned = self._owned()
                if not owned:
                    return True
                for pid in owned:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                if time.monotonic() >= until:
                    return False
                time.sleep(0.01)

    front: str = ""
    back: str = ""

    def attach(self, text: str, front: bool = False) -> None:
        """Add a report collected after the command to what it printed.

        A database window read once the command has finished belongs with the
        command's own output, ahead of it so the reading's units are the
        report's, not those of any figures the application happened to print.
        A closed job's cache holds its output between the reports attached so
        far, so a late report goes in beside them.
        """
        earlier = self.front
        if front:
            self.front += text
        else:
            self.back += text
        if getattr(self, "closed", False):
            cached = self.cached or ""
            cut = len(earlier) if front else len(cached)
            self.cached = cached[:cut] + text + cached[cut:]

    def _text(self) -> str:
        """Everything the command has written so far, with a marker when the output could not all
        be read.
        """
        if self.closed:
            return self.cached or ""
        try:
            self.sink.flush()
            text = bounded_shell_output(self.sink.name)
            self.output_complete = OUTPUT_INCOMPLETE_MARKER not in text
            return getattr(self, "front", "") + text + getattr(self, "back", "")
        except OSError:
            self.output_complete = False
            return (getattr(self, "front", "") + OUTPUT_INCOMPLETE_MARKER
                    + " captured output could not be read]" + getattr(self, "back", ""))

    def _tail(self, cap: int = SHELL_OUTPUT_CAP) -> str:
        """The last of the command's output, read from the end of its file."""
        if self.closed:
            return (self.cached or "")[-cap:]
        try:
            self.sink.flush()
            with open(self.sink.name, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                end = handle.tell()
                handle.seek(max(0, end - cap))
                return handle.read(cap).decode("utf-8", errors="replace")
        except OSError:
            return ""

    def wait(self, timeout: float) -> tuple:
        """Wait for the command, then return whether it settled and what it wrote."""
        try:
            self.process.wait(timeout=max(1.0, timeout))
            return self._settle_descendants(), self._text()
        except subprocess.TimeoutExpired:
            return False, self._text()

    def finished(self) -> bool:
        """Has the command exited?"""
        return self.code() is not None

    def drain(self) -> str:
        """The command's output, with a note appended while it is still running."""
        if self.code() is None:
            return self._text() + "\n" + STILL_RUNNING
        return self._text()

    def stop(self) -> None:
        """Finish this job once. Safe to call again, and from either holder.

        A job can have two observers: the tool that started it and whatever
        else holds a reference, such as the Warden's baseline. The final output
        and status are cached before the sink goes, so the second observer
        reads a real result instead of a closed file.
        """
        if self.closed:
            return
        if self.watchdog is not None:
            self.watchdog.cancel()
        self._kill()
        try:
            self.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        self._settle_descendants()
        self.cached = self._text()
        self.cached_code = self.process.poll()
        self.closed = True
        try:
            self.sink.close()
            os.unlink(self.sink.name)
        except OSError:
            pass

SUITE_LANE = "suite"
# What the checks in this run actually established, each tied to the exact code
# it ran against. Observational only: nothing here selects a patch.
CHECKS: list = []
MEASUREMENT_REPORT_CAP = 256
MEASUREMENT_BYTES_CAP = 4 * 1024 * 1024
MEASUREMENTS: dict = {}
MEASUREMENT_BYTES_USED = 0
MEASUREMENT_COUNTER = 0
MEASUREMENT_LOCK = threading.RLock()
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.I)
CHECK_EMPTY = re.compile(
    r"^\s*(?:=+\s*)?(?:no tests (?:ran|collected)\b|collected 0 items\b|"
    r"Ran 0 tests?\b)", re.I | re.M)
RAN_COUNT = re.compile(r"^Ran (\d+) tests?\b", re.M)

# Paths the agent deliberately created this run. Kept for the log and for the
# identity; it is not a permission list, because a file's origin says nothing
# about its purpose. A heredoc, a script and an editor all write source.
CREATED: set = set()
# What a file IS, not how it got there: caches, editor leftovers and captured
# output. A run that writes source under one of these names loses it, which is
# the cost of not deleting real answers on a guess about the writing tool.
BYPRODUCT_ENDS = (".log", ".tmp", ".out")

def requested_answer_paths(statement: str = "", root: str = "") -> set:
    """Paths deliberately written or named by the current task.

    Selection is not permission: current-task operation and scope restrictions
    still apply separately. Artifact-shaped names alone do not override intent.
    """
    requested = {operation_path(path, root) for path in CREATED}
    for part in instruction_clauses(statement or ""):
        requested.update(operation_path(path, root) for path in FENCE_PATH.findall(part))
        requested.update(operation_path(path, root) for path in operation_literals(part))
    return {path for path in requested if path not in ("", ".", "..")
            and not path.startswith("../") and not os.path.isabs(path)}

def byproduct_path(path: str) -> bool:
    """Is this new file a by-product of running commands rather than an answer?

    Decided from the path itself: the junk profile this run already carried, or
    a log-shaped name. Deliberately NOT decided from which tool wrote it or
    from shell syntax seen in a command: a redirection is where a file came
    from, and source is written that way routinely. Where that leaves a stray
    file in the answer, the statement's own path scope is what refuses it.
    """
    return bool(path) and (junk_path(path) or path.endswith(BYPRODUCT_ENDS))

# ---------------------------------------------------------------------------
# One concrete case, recorded by the run and kept honest by execution.
#
# The model states what it thinks is wrong in a form that can be checked: a
# requirement quoted from the statement, a quotation from the code it blames,
# the inputs, what happens now, what the statement requires, and one command
# that would tell the two apart. Quoting is verified against the real sources,
# which establishes the anchors and nothing else: only running the observation
# moves the case from proposed to observed, and this run decides that from the
# command's own result rather than from an assertion about it.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# One check on the lines an edit actually added: a name the file neither
# imports nor defines. There is deliberately no check that recognises a shape
# of expression and returns the repair such code usually needs. Whatever that
# shape suggests is a guess about a task this has not read, and handing it to
# the model before it has looked costs the run the reading it would otherwise
# have done.
# ---------------------------------------------------------------------------
QUERY_NOTES_SEEN: set = set()
def added_lines(before: str, after: str) -> list:
    """The lines this edit introduced, with their new line numbers."""
    import difflib
    old, new = before.splitlines(), after.splitlines()
    out = []
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(
            None, old, new, autojunk=False).get_opcodes():
        if tag in ("replace", "insert"):
            out.extend((number + 1, new[number]) for number in range(j1, j2))
    return out

# The note below reaches further when this is on: names a local star import
# provides, names an edit's deletion left unbound anywhere in the file, and the
# hand-in review. It stays once per run; what the module binds is read the same
# either way.
UNBOUND_READING = flag("RIDGES_UNBOUND_READING")
UNBOUND_STAR_DEPTH = 4
MODULE_NAMES = {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__",
                "__builtins__", "__path__", "__all__", "__class__", "__debug__"}
RUN_TIME_BINDERS = {"globals", "vars", "locals", "exec", "eval", "__import__"}
# Python 3.12 syntax (PEP 695), which older interpreters' ast does not have: a type
# parameter list binds its names (def first[T](), class Box[T], type Pair[K, V] = ...),
# and a type alias statement binds the alias at module level.
TYPE_PARAMETERS = tuple(getattr(ast, kind) for kind in ("TypeVar", "ParamSpec", "TypeVarTuple") if hasattr(ast, kind))
TYPE_ALIASES = tuple(getattr(ast, kind) for kind in ("TypeAlias",) if hasattr(ast, kind))


def module_bindings(tree) -> set:
    """Every name the module's own code binds: imports, definitions, assignments,
    arguments, handlers, global and nonlocal declarations, match captures and
    type parameters."""
    bound = set()
    captures = tuple(getattr(ast, kind) for kind in ("MatchAs", "MatchStar") if hasattr(ast, kind))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names if alias.name != "*")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            bound.update(node.names)
        elif captures and isinstance(node, captures) and node.name:
            bound.add(node.name)
        elif hasattr(ast, "MatchMapping") and isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif isinstance(node, TYPE_PARAMETERS):
            bound.add(node.name)
    return bound


def binds_at_run_time(tree) -> bool:
    """Whether the module can bind names no reading of its text lists: a module
    __getattr__, or a call to globals(), vars(), locals(), exec, eval or __import__."""
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__"
           for node in tree.body):
        return True
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in RUN_TIME_BINDERS
               for node in ast.walk(tree))


def star_module_path(root: str, path: str, level: int, module: str) -> str:
    """The checkout file a `from <module> import *` in path reads, or ""."""
    parts = [part for part in (module or "").split(".") if part]
    if level:
        base = os.path.dirname(path)
        for _ in range(level - 1):
            base = os.path.dirname(base)
        bases = [base]
    else:
        # An absolute import resolves from the checkout root or from a source
        # directory the importing file sits under (src/, or the project folder).
        bases, folder = [""], os.path.dirname(path)
        while folder:
            bases.append(folder)
            folder = os.path.dirname(folder)
    real_root = os.path.realpath(root)
    for base in bases:
        stem = os.path.join(base, *parts) if parts else base
        for candidate in (stem + ".py", os.path.join(stem, "__init__.py")):
            full = os.path.realpath(os.path.join(root, candidate))
            if full.startswith(real_root + os.sep) and os.path.isfile(full):
                return os.path.relpath(full, real_root)
    return ""


def star_import_names(root: str, path: str, seen: set | None = None) -> set | None:
    """The names `from <path's module> import *` binds, read from its source, or None
    when that cannot be read: no file, a non-literal __all__, run-time binders, a
    star import of its own that cannot be read, or too deep a chain."""
    seen = set() if seen is None else seen
    if path in seen:
        return set()
    if len(seen) >= UNBOUND_STAR_DEPTH:
        return None
    seen.add(path)
    try:
        with open(os.path.join(root, path), encoding="utf-8", errors="replace") as handle:
            tree = ast.parse(handle.read())
    except (OSError, SyntaxError, ValueError):
        return None
    if binds_at_run_time(tree):
        return None
    listed = None
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, (ast.AugAssign, ast.AnnAssign)) else []
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = getattr(node, "value", None)
        if not isinstance(value, (ast.List, ast.Tuple)) or not all(
                isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts):
            return None
        names = {item.value for item in value.elts}
        listed = names if listed is None or isinstance(node, ast.Assign) else listed | names
    if listed is not None:
        return listed
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names if alias.name != "*")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.If, ast.Try, ast.With, *TYPE_ALIASES)):
            names.update(sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store))
            names.update(sub.name for sub in ast.walk(node)
                         if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            source = star_module_path(root, path, node.level, node.module or "")
            more = star_import_names(root, source, seen) if source else None
            if more is None:
                return None
            names |= more
    return {name for name in names if not name.startswith("_")}


def unbound_findings(path: str, after: str, added: list, before: str = "", root: str = "") -> list:
    """(name, line, bound_before) for names the edit reads that nothing in the module binds.

    Reads the added lines, and, when the reach is on, every line reading a name
    the file bound before the edit and no longer does. Says nothing when the
    module binds names at run time, or has a star import this cannot read.
    """
    try:
        tree = ast.parse(after)
    except SyntaxError:
        return []
    if binds_at_run_time(tree):
        return []
    import builtins as _builtins
    bound = module_bindings(tree) | set(dir(_builtins)) | MODULE_NAMES
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            if not (UNBOUND_READING and root):
                return []
            source = star_module_path(root, path, node.level, node.module or "")
            names = star_import_names(root, source) if source else None
            if names is None:
                return []
            bound |= names
    lost = set()
    if UNBOUND_READING and before:
        try:
            lost = module_bindings(ast.parse(before)) - bound
        except SyntaxError:
            lost = set()
    numbers = {number for number, _ in added}
    found: dict = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound
                and (getattr(node, "lineno", 0) in numbers or node.id in lost)):
            if node.id not in found or node.lineno < found[node.id]:
                found[node.id] = node.lineno
    return sorted(((name, line, name in lost) for name, line in found.items()), key=lambda row: row[1])


def unbound_note(path: str, findings: list, stars: bool = False) -> str:
    """Names the edit uses that this file does not define or import.

    Only that, and deliberately not when it would fail. Position in the file
    does not settle that: a name under `if False` at module level never runs,
    and a name in a default argument runs at definition rather than at call,
    so the two obvious readings are both wrong in an ordinary case. Deciding
    it properly needs reachability and evaluation-order analysis this does not
    do, and a warning is not worth that, so it says what it found and leaves
    when to a run of the file's own check.
    """
    listed = ", ".join("%s (line %d%s)" % (name, line, ", bound before this edit" if lost else "")
                       for name, line, lost in findings[:4])
    return ("%s uses %s. This run found no import or definition for %s in this "
            "file%s. It has not worked out whether that line is reached, nor when "
            "it would run, so this is a name to check rather than a fault: the "
            "file's own check settles it."
            % (path, listed, "them" if len(findings) > 1 else "it",
               ", nor among the names its star imports provide" if stars else ""))


def unbound_names(path: str, after: str, added: list, before: str = "", root: str = "") -> str:
    """The note for every name the edit reads that this file does not bind, or ""."""
    findings = unbound_findings(path, after, added, before, root)
    return unbound_note(path, findings, stars="import *" in after) if findings else ""


def query_notes(path: str, before: str, after: str, root: str = "") -> str:
    """What this edit can be shown to have introduced, at most once per run.

    A name the file neither imports nor defines, and nothing else. A diagnosis
    worth acting on comes from the task, the surrounding code and a command
    that was actually run; one handed over because an expression looked
    familiar is a guess about a task this has not read. The hand-in review
    names what the change as it stands still reads unbound.
    """
    if not QUERY_CHECKS or not path.endswith(".py") or "unbound" in QUERY_NOTES_SEEN:
        return ""
    added = added_lines(before, after)
    # A deletion adds no line and can still leave a name unbound.
    if not added and not UNBOUND_READING:
        return ""
    try:
        note = unbound_names(path, after, added, before, root)
    except BaseException:
        note = ""
    if not note:
        return ""
    QUERY_NOTES_SEEN.add("unbound")
    return "\n\n" + note


def unbound_at_hand_in(tree: "Tree", limit: int = 3) -> str:
    """A paragraph naming what the change as it stands reads unbound, or ""."""
    if not UNBOUND_READING or not QUERY_CHECKS:
        return ""
    notes = []
    for path in [p for p in tree.changed_paths(10.0) if p.endswith(".py")][:12]:
        try:
            after = tree.read(path)
            before = tree.at_base(path, 10.0)
        except (ToolFault, OSError):
            continue
        note = unbound_names(path, after, added_lines(before, after), before, tree.root)
        if note:
            notes.append(note)
        if len(notes) >= limit:
            break
    return ("\n\nOne reading of the change as it stands, to settle first: " + " ".join(notes)) if notes else ""

CASE: dict = {}
NAMED_CASES: dict = {}
CASE_LIMIT = 8
CASE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,31}\Z")
CASE_QUOTE_MIN = 12

def declared_cases() -> list[tuple[str, dict]]:
    """The cases this run has declared, the unnamed one first when there is one."""
    return ([("", CASE)] if CASE else []) + list(NAMED_CASES.items())

def case_bindings(command: str) -> dict:
    """Snapshot declarations before launching an invocation, never at collection."""
    key = command_identity(command)
    return {name: record.get("revision", 1) for name, record in declared_cases()
            if (record.get("observation_key") or command_identity(record.get("observation", ""))) == key}

def case_scopes(command: str) -> list:
    """Claims captured with the declaration; they are never observed fixtures."""
    names = case_bindings(command)
    return [{"case_id": name, "revision": case.get("revision", 1),
             "requirement_ids": list(case.get("requirement_ids", [])),
             "scope": dict(case.get("scope", {}))}
            for name, case in declared_cases() if name in names]

def workload_context(record: dict):
    """The caller, workload and dataset a record's claims are scoped to, or None when any is
    missing.
    """
    scopes = record.get("case_scopes", [])
    if not scopes:
        return None
    values = []
    for claim in scopes:
        scope = claim.get("scope", {})
        if not all(scope.get(key) for key in ("caller", "workload", "dataset")):
            return None
        values.append(tuple(scope[key] for key in ("caller", "workload", "dataset")))
    return tuple(sorted(set(values)))

def case_status(record: dict, current_identity: str) -> str:
    """The case's standing against the source as it is now."""
    if not record:
        return "none"
    status = record.get("status", "proposed")
    if status != "proposed" and record.get("observed_identity", record.get("identity")) != current_identity:
        return status + " (superseded: the source changed after that run)"
    return status

def case_block(record: dict, current_identity: str) -> str:
    """One case as the model reads it: its status, what it asserts, its evidence and where it came
    from.
    """
    if not record:
        return ""
    lines = [
        "  case: %s; revision: %s" % (record.get("case_id") or "unnamed",
                                     record.get("revision", 1)),
        "  status: %s; assertion role: %s; relationship: %s" % (
            case_status(record, current_identity), record.get("asserts", "unspecified"),
            record.get("relationship", "not established")),
        "  evidence: %s" % (record.get("evidence") or "not yet run"),
        "  quotation source: %s at %s%s" % (
            record.get("declared_identity") or record.get("identity") or "unknown",
            record.get("declared_at", record.get("at", "unknown")),
            " (different or unknown current source)" if not current_identity or
            record.get("declared_identity", record.get("identity")) != current_identity else ""),
        "  requirement: %s" % record.get("requirement", ""),
        "  requirement IDs (claimed coverage): %s" % (", ".join(record.get("requirement_ids", [])) or "none"),
        "  scope (model claims, not observed facts): %s" % json.dumps(record.get("scope", {}), ensure_ascii=False, sort_keys=True),
        "  code: %s -- %s" % (record.get("code_path", ""), record.get("code_quote", "")),
        "  inputs: %s" % record.get("inputs", ""),
        "  now: %s" % record.get("current_result", ""),
        "  required: %s" % record.get("required_result", ""),
        "  observation: %s" % record.get("observation", ""),
        "  must not change: %s" % record.get("unchanged", ""),
        "  observed output: %s" % record.get("observed_output", "not recorded"),
    ]
    return "\n".join(lines)

def reset_checks() -> None:
    """Evidence belongs to one task. A second call must not inherit the first."""
    global _VENV_BIN, MEASUREMENT_BYTES_USED, MEASUREMENT_COUNTER
    global CH_QUERY_COUNTER, MEASURE_COUNTER, MEASURE_WINDOW_COUNTER
    _VENV_BIN = ""
    CH_QUERY_COUNTER = MEASURE_COUNTER = MEASURE_WINDOW_COUNTER = 0
    HttpProbe.counter = 0
    BARE_MODELS.clear()
    for key in RETRY_TALLY:
        RETRY_TALLY[key] = 0
    del CHECKS[:]
    del EXPECTATIONS[:]
    NO_PARALLEL_MODELS.clear()
    with MEASUREMENT_LOCK:
        MEASUREMENTS.clear()
        MEASUREMENT_BYTES_USED = 0
        MEASUREMENT_COUNTER = 0
    CREATED.clear()
    CASE.clear()
    NAMED_CASES.clear()
    QUERY_NOTES_SEEN.clear()
    PLAN_NOTES_SEEN.clear()
    CH_NAME_STATE.clear()
    EXACT_BINARY.clear()
    REFERENCED_READ.clear()

def patch_digest(patch: str) -> str:
    """A short digest of an answer, so the same answer can be recognised later."""
    import hashlib
    return hashlib.sha256((patch or "").encode("utf-8", "replace")).hexdigest()[:12]

def pristine_identity(base: str) -> str:
    """The identity Tree.source_identity gives an unchanged checkout of base.

    A separate checkout of the base commit carries no answer paths, so its
    identity is the base prefix alone, exactly as the live tree's was before
    the first edit. Computed here so a reading taken in that checkout is
    recorded against a known source rather than the tree it did not run in.
    """
    if not base:
        return ""
    return hashlib.sha256(("git-base\0" + base + "\0").encode()).hexdigest()[:16]

def worktree_clean(where: str, budget: float = 15.0) -> bool:
    """Whether a checkout's tracked files are as committed. Unknown is False."""
    code, out = git(["status", "--porcelain", "--untracked-files=no"], where, max(1.0, budget))
    return code == 0 and not out.strip()

def command_identity(command: str) -> str:
    """Exact invocation bytes; whitespace and text beyond the display matter."""
    return hashlib.sha256((command or "").encode("utf-8", "replace")).hexdigest()

def environment_identity(environment: dict) -> str:
    """Compare effective environments without recording their secret values."""
    values = sorted((str(key), str(value)) for key, value in environment.items())
    return hashlib.sha256(json.dumps(values, ensure_ascii=True).encode()).hexdigest()

def record_command_key(record: dict) -> str:
    """The full identity of the command a record is about, or empty when only a clipped display
    survives.
    """
    if record.get("command_key"):
        return record["command_key"]
    # Older in-memory records did not save a full key. A display at its cap
    # could conceal a different suffix and cannot establish command identity.
    command = record.get("command", "")
    return command_identity(command) if command and len(command) < 200 else ""

def record_context(record: dict) -> tuple:
    """Comparable invocation, directory and environment, or unknown."""
    command = record_command_key(record)
    cwd = record.get("cwd")
    environment = record.get("environment_key") or record.get("env")
    if not command or not cwd or not environment:
        return ()
    context = (command, record.get("invocation_key", ""), os.path.realpath(cwd), environment)
    if record.get("case_scopes"):
        # Incomplete declarations cannot establish measurement comparability,
        # but their known differences must still keep observations separate.
        scopes = tuple(sorted(set(tuple(claim.get("scope", {}).get(key, "")
                                         for key in ("caller", "workload", "dataset"))
                                  for claim in record["case_scopes"])))
        context += (scopes,)
    return context

def record_exit(record: dict) -> int | None:
    """The exit code a command record carries, from the field or from its status line, or None.
    """
    value = record.get("exit_code")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    found = re.fullmatch(r"exit_code=(-?\d+)", record.get("status", ""))
    return int(found.group(1)) if found else None

def stable_record(record: dict) -> bool:
    """A completed reading of known, unchanged source, including failures."""
    return bool(record.get("identity")
                and record.get("identity_end") == record["identity"]
                and not record.get("stale") and not record.get("timed_out")
                and record_exit(record) is not None)

def usable_record(record: dict) -> bool:
    """Is this record evidence? It must be settled, complete, exit clean and have actually run
    something.
    """
    return bool(stable_record(record) and record.get("evidence")
                and completed_record_time(record) is not None
                and record.get("output_complete", True)
                and record_exit(record) == 0
                and record.get("outcome") in ("ran", "measured"))

def earlier_record(before: dict, after: dict) -> bool:
    """The earlier reading must finish before the later invocation starts."""
    start, finish = before.get("at"), before.get("completed_at", before.get("at"))
    later = after.get("at")
    return bool(isinstance(start, (int, float))
                and isinstance(finish, (int, float))
                and isinstance(later, (int, float))
                and start <= finish <= later and start < later)

def completed_record_time(record: dict) -> float | None:
    """An ordered completed reading, including an unsuccessful recheck."""
    if (not record.get("identity")
            or record.get("identity_end") != record["identity"]
            or record.get("stale") or record_exit(record) is None):
        return None
    start = record.get("at")
    finish = record.get("completed_at", start)
    if (not isinstance(start, (int, float)) or isinstance(start, bool)
            or not isinstance(finish, (int, float)) or isinstance(finish, bool)
            or not math.isfinite(start) or not math.isfinite(finish)
            or finish < start):
        return None
    return finish

def current_records(records: list, current: str) -> list:
    """Unsuperseded readings per full invocation/context on current source.

    Completion is recorded when output is collected, which may be much later
    than process exit. Only a check started after another result was collected
    establishes their order. Overlapping intervals remain unordered; collecting
    an old background success late cannot erase a subsequent failure. History
    and different commands, directories, and environments remain independent.
    """
    if not current:
        return []
    readings = [record for record in records if record.get("identity") == current]
    latest: dict = {}
    for record in readings:
        context, finish = record_context(record), completed_record_time(record)
        if context and finish is not None:
            if context not in latest or record["at"] > latest[context]["at"]:
                latest[context] = record
    selected = []
    for record in readings:
        context, finish = record_context(record), completed_record_time(record)
        if (not context or finish is None
                or not earlier_record(record, latest[context])):
            selected.append(record)
    return selected

def current_usable_records(records: list, current: str) -> list:
    """The records that vouch for the source as it stands now, with ambiguous invocations left out.

    Two readings of the same invocation that disagree, including ones with equal timestamps,
    make that invocation's outcome unclear, so neither is treated as evidence.
    """
    readings = current_records(records, current)
    # Unordered conflicting readings, including equal timestamps and delayed
    # collection, make success ambiguous for that invocation.
    uncertain = {record_context(record) for record in readings
                 if record_context(record)
                 and (completed_record_time(record) is not None or stable_record(record))
                 and not usable_record(record)}
    return [record for record in readings if usable_record(record)
            and record_context(record) not in uncertain]

def assertion_failure(out: str) -> bool:
    """Positive runner failure evidence, distinct from collection/setup errors."""
    text = out or ""
    if OUTPUT_INCOMPLETE_MARKER in text:
        return False
    # Mixed setup errors leave the observation ambiguous, even if another
    # test also fails. A nonzero process status by itself proves no assertion.
    if re.search(r"(?im)^ERROR(?: collecting| at (?:setup|teardown))\b|"
                 r"\b[1-9]\d* errors?\b|\berrors=[1-9]\d*|\[build failed\]", text):
        return False
    if re.search(r"(?m)^FAIL(?:\s|$)|^--- FAIL:", text):
        return bool(re.search(r"(?m)^--- FAIL: \S+", text))
    return bool(re.search(r"(?m)^--- FAIL: \S+|^FAIL: \S+|"
                          r"(?<![\w.])[1-9]\d* failed\b|"
                          r"FAILED \(failures=[1-9]\d*(?:\)|, errors=0\))", text))

def meaningful_failure(record: dict) -> bool:
    """Did this command fail because the code is wrong, rather than because it could not run?"""
    if (not stable_record(record) or record.get("outcome") != "failed"
            or (record_exit(record) or 0) <= 0):
        return False
    if "failure_type" in record:
        return record["failure_type"] == "assertion"
    # Compatibility with older caller-classified observations. Every
    # live v14 record carries failure_type; a diagnostic without that explicit
    # classification remains unknown rather than being called an assertion.
    return "detail" not in record and record_exit(record) == 1

def observed_change_records(records: list, current: str) -> list:
    """One fail-to-pass interpretation shared by review and final reporting."""
    if not current:
        return []
    current_successes = current_usable_records(records, current)
    groups: dict = {}
    for record in records:
        context = record_context(record)
        if context and record.get("kind") not in ("other command", "database query"):
            groups.setdefault(context, []).append(record)
    moved = []
    for group in groups.values():
        passing = [r for r in group if r.get("after_edit")
                   and r.get("identity") == current and r.get("outcome") == "ran"
                   and r in current_successes]
        failed = [r for r in group if meaningful_failure(r)
                  and r.get("identity") != current]
        for success in passing:
            if any(earlier_record(failure, success) for failure in failed):
                moved.append(success["command"])
                break
    return moved

def env_identity(command: str) -> str:
    """Which variables a command sets, and a digest of what it set them to.

    Names are useful for reading the log; values can carry credentials, so they
    are only ever hashed.
    """
    import hashlib
    names, values = [], []
    for line in logical_lines(command or ""):
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        for word in words:
            if not ASSIGNMENT.match(word):
                break
            name = word.split("=", 1)[0]
            names.append(name)
            values.append(word if not SECRET_NAME.search(name) else name + "=<withheld>")
    if not names:
        return "none"
    digest = hashlib.sha256("\0".join(values).encode("utf-8", "replace")).hexdigest()[:8]
    return "%s (values %s)" % (",".join(names), digest)

ASSIGNED_VALUE = re.compile(
    r"\b([A-Za-z_][A-Za-z_0-9]*)=('[^']*'?|\"[^\"]*\"?|\S*)")
# A connection string carries its password in the same word as its host, so no
# assignment is there to recognise. This is the other shape a credential takes
# in a command, and it reaches the log and the record by the same path.
DSN_CREDENTIAL = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^\s:/@]+):([^\s/@]*)@")

def redact(command: str) -> str:
    """Keep a command readable in the log without copying credentials into it.

    Quote-aware: a quoted value is withheld whole, and a value whose quote is
    never closed withholds the rest of the line rather than leaking its tail.
    """
    def hide(found):
        name, value = found.group(1), found.group(2)
        if not SECRET_NAME.search(name):
            return found.group(0)
        opened = value[:1] in ("'", '"')
        closed = len(value) > 1 and value[-1] == value[0]
        if opened and not closed:
            return name + "=<withheld to end of line>"
        return name + "=<withheld>"
    return DSN_CREDENTIAL.sub(r"\1:<withheld>@", ASSIGNED_VALUE.sub(hide, command or ""))

def check_outcome(out: str, command: str = "", exit_code=...) -> tuple:
    """Interpret supported runner summaries, retaining the real process result.

    Omitted status permits offline interpretation of a transcript. Callers that
    have a process must pass its status: an unknown or nonzero exit cannot make
    a successful check. Cached results and skipped tests are not fresh execution.
    """
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", out or "")

    def result(outcome: str, detail: str) -> tuple:
        if exit_code is None and outcome in ("ran", "cached"):
            return "unrecognized", "the process has not reported an exit status"
        if exit_code is not ... and exit_code is not None and exit_code != 0:
            # pytest exits 5 when it collected no tests. Keep that distinct
            # from an assertion failure while still refusing success evidence.
            if outcome not in ("none_ran", "skipped_only"):
                return "failed", "%s; command exited with status %s" % (detail, exit_code)
        if OUTPUT_INCOMPLETE_MARKER in text:
            if outcome == "failed":
                return outcome, detail + "; additional command output omitted"
            return "incomplete", "command output omitted; the complete check verdict is unresolved"
        return outcome, detail

    # A Go package verdict owns its nested output. In particular a helper's
    # Python-looking summary cannot turn a failed Go package into a pass.
    go_packages = list(re.finditer(
        r"^(ok|\?)[ \t]+(\S+)[ \t]+([^\n]+)$", text, re.M))
    go_failure = re.search(r"^\s*--- FAIL: .+|^FAIL(?:[ \t].*)?$", text, re.M)
    go_tests = re.search(r"^\s*--- (?:PASS|SKIP|FAIL): .+", text, re.M)
    is_go = (re.search(r"(?:^|[\s/])go\s+test\b", command or "")
             or re.search(r"^FAIL[ \t]+\S+[ \t]+(?:\d|\[)", text, re.M)
             or go_tests or any(
                 re.match(r"(?:\d+(?:\.\d+)?s\b|\(cached\)|\[no test files\])", m[3])
                 for m in go_packages))
    if is_go:
        if go_failure:
            return result("failed", "the Go runner reported %s" % go_failure[0].strip())
        fresh = cached = skipped = empty = 0
        previous = 0
        for package in go_packages:
            body, summary = text[previous:package.start()], package[3]
            previous = package.end()
            if package[1] == "?" or "[no tests to run]" in summary:
                empty += 1
            elif summary.startswith("(cached)"):
                cached += 1
            elif re.match(r"\d+(?:\.\d+)?s\b", summary):
                if (re.search(r"^\s*--- SKIP:", body, re.M)
                        and not re.search(r"^\s*--- PASS:", body, re.M)):
                    skipped += 1
                else:
                    fresh += 1
        if fresh:
            return result("ran", "Go reported %d successful package(s), %d cached; "
                          "individual test counts are not available" % (fresh, cached))
        if cached:
            return result("cached", "Go reused %d cached package result(s); no fresh "
                          "test execution was reported" % cached)
        if skipped:
            return result("skipped_only", "every reported Go test was skipped")
        if empty:
            return result("none_ran", "Go reported no tests in the selected packages")
        # An explicitly invoked test binary can report no package summary.
        if (re.search(r"^\s*--- PASS: .+", text, re.M)
                and re.search(r"^PASS\s*$", text, re.M)):
            return result("ran", "the Go test binary reported passing tests")
        return result("unrecognized", "no completed Go test result recognised in the output")

    # pytest summaries are whole summary lines, not arbitrary log substrings.
    count_word = r"(?:passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)"
    summaries = list(re.finditer(
        r"^[= \t]*(\d+ " + count_word + r"(?:, \d+ " + count_word + r")*)"
        r"(?: in \d+(?:\.\d+)?s(?: \([^\n]*\))?)?[= \t]*$", text, re.M))
    for summary in summaries:
        counts = {name: int(count) for count, name in re.findall(
            r"(\d+) (" + count_word + r")", summary[1])}
        if any(counts.get(name, 0) for name in ("failed", "error", "errors")):
            return result("failed", "the runner reported %s" % summary[1])
    # Retain explicit runner failure banners even if a pipeline exits zero.
    failed = re.search(r"^FAILED\b[^\n]*|^(?:FAIL|ERROR): \S+ \([^\n]*\)|"
                       r"^Found [1-9]\d* errors?\b[^\n]*", text, re.M)
    if failed:
        return result("failed", "the runner reported %s" % failed[0])
    # unittest / Django: a count alone says nothing about the final verdict.
    # Interpret positive banners only after checking explicit failure summaries;
    # a nested unittest result can appear in a failed pytest capture.
    ran = list(RAN_COUNT.finditer(text))
    verdicts = list(re.finditer(r"^(OK(?: \([^\n]*\))?|FAILED \([^\n]*\))\s*$",
                               text, re.M))
    if ran and verdicts:
        count = int(ran[-1][1])
        skipped = re.search(r"\bskipped=(\d+)", verdicts[-1][1])
        skipped_count = int(skipped[1]) if skipped else 0
        if not count:
            return result("none_ran", "the runner reported Ran 0 tests")
        if skipped_count >= count:
            return result("skipped_only", "every test was skipped (%d of %d)"
                          % (skipped_count, count))
        if verdicts[-1].start() > ran[-1].start():
            return result("ran", "%s; %s" % (ran[-1][0], verdicts[-1][1]))
    if summaries:
        last = summaries[-1]
        counts = {name: int(count) for count, name in re.findall(
            r"(\d+) (" + count_word + r")", last[1])}
        if counts.get("passed", 0) > 0 or counts.get("xpassed", 0) > 0:
            return result("ran", last[1])
        if counts.get("skipped", 0) > 0 and not counts.get("xfailed", 0):
            return result("skipped_only", last[1] + "; no executed passing tests")
        if not any(counts.values()) or (set(counts) <= {"passed", "deselected"}):
            return result("none_ran", last[1] + "; no tests executed")
    if CHECK_EMPTY.search(text):
        return result("none_ran", "the command ran no checks")
    if re.search(r"^All checks passed!\s*$", text, re.M) and not suite_shaped(command):
        return result("ran", "all checks passed")
    return result("unrecognized", "no check result recognised in the output")

def check_kind(command: str, out: str, runner: str = "") -> str:
    """Regression check, static check and measurement are different evidence."""
    normal = " ".join((command or "").split())
    if runner and " ".join(runner.split()) in normal:
        return "named regression check"
    if database_evidence(out or ""):
        return "database measurement"
    if normal.startswith("sql "):
        # An observation of what the database returns. Recorded and carried,
        # but not evidence that the application produces that query.
        return "database query"
    if suite_shaped(command or ""):
        return "test run"
    for line in logical_lines(command or ""):
        try:
            if check_command(shlex.split(line, comments=True)):
                return "static check"
        except ValueError:
            continue
    return "other command"
# A query or test run waits for the shared database lane rather than reporting
# "not started": whether the baseline suite is still on the lane depends on the
# host's speed, and the reading of a result should not. The tool's own wait
# budget still bounds it.
LANE_GRACE_SEC = 300.0

def suite_shaped(command: str) -> bool:
    """Does this command run a project test suite?

    A suite usually owns a database, a cache and a set of fixtures, so two
    of them at once corrupt each other rather than measuring anything.
    """
    for line in logical_lines(command or ""):
        if runner_segments(line) is not None:
            return True
    # Locking is deliberately more willing than execution: a command this run
    # would not reproduce itself can still be a suite the model typed by hand,
    # and holding the lane for it is cheap next to two suites at once.
    return bool(RUNNER_SHAPED.search(command or ""))

class LaneBusy(Exception):
    """Raised when a lane a command needs is held by a job still running."""

    def __init__(self, lane: str) -> None:
        """Say which lane is busy, so the caller can wait on that one rather than guess."""
        super().__init__("the %s lane is busy" % lane)
        self.lane = lane

class ShellPool:
    """The background commands this run has started, and the lanes that keep them from colliding.

    Commands that use the same resource, such as the project's database, share a lane so
    only one runs at a time, and everything is stopped and reaped when the pool closes.
    """
    def __init__(self, cwd: str) -> None:
        """Open an empty pool of background commands for this checkout."""
        self.cwd = cwd
        self.jobs = {}
        self.lanes: dict = {}
        self.gate = threading.RLock()
        self.cleanups: list = []

    def own_cleanup(self, cleanup) -> None:
        """Remove only resources created by this pool, after jobs have stopped."""
        self.cleanups.append(cleanup)

    def lane_busy(self, lane: str):
        """The job holding this lane, if one still is."""
        job = self.lanes.get(lane)
        if job is None or job.finished():
            return None
        return job

    def await_lane(self, lane: str, budget: float) -> float | None:
        """Wait for a lane to clear. Seconds waited, or None if it did not."""
        began = time.monotonic()
        while True:
            if self.lane_busy(lane) is None:
                return time.monotonic() - began
            if time.monotonic() - began >= max(0.0, budget):
                return None
            time.sleep(0.2)

    def start(self, command: str, pack_venv: bool = True,
              hard_timeout: float | None = None,
              lane: str | None = None, lane_wait: float = 0.0,
              env_extra: dict | None = None, cwd: str | None = None) -> Shell:
        """Launch a command, taking its lane first if it asked for one.

        Acquisition lives here so that every caller is serialised, not only the
        one that remembers to ask: the Warden's baseline, its recheck and its
        confirmation all reach this function too.
        """
        where = cwd or self.cwd
        if lane:
            with self.gate:
                if self.lane_busy(lane) is not None:
                    if self.await_lane(lane, lane_wait) is None:
                        raise LaneBusy(lane)
                job = Shell(command, where, pack_venv, hard_timeout, env_extra)
                self.jobs[job.name] = job
                self.lanes[lane] = job
                return job
        job = Shell(command, where, pack_venv, hard_timeout, env_extra)
        self.jobs[job.name] = job
        return job

    def reserve(self, lane: str, job, lane_wait: float = 0.0) -> None:
        """Register an in-process job the way start() registers a shell.

        A database request that runs inside this process still shares the
        database with a suite run, so it waits for the same lane and holds it
        until it reports finished.
        """
        with self.gate:
            if self.lane_busy(lane) is not None:
                if self.await_lane(lane, lane_wait) is None:
                    raise LaneBusy(lane)
            self.jobs[job.name] = job
            self.lanes[lane] = job

    def get(self, name: str) -> Shell:
        """One background command by name, or a refusal naming what was asked for."""
        job = self.jobs.get(name)
        if job is None:
            raise ToolFault("no background job named %s" % name)
        return job

    def close(self) -> None:
        """Stop and reap every command, log its last output, and run the pool's own cleanups."""
        for job in list(self.jobs.values()):
            try:
                job._kill()
                report_shell(job, job._tail())
            except Exception:
                pass
            finally:
                # Diagnostic failure must not skip reaping or capture cleanup.
                try:
                    job.stop()
                except Exception:
                    pass
        self.jobs.clear()
        self.lanes.clear()
        for cleanup in reversed(self.cleanups):
            try:
                cleanup()
            except Exception:
                pass
        self.cleanups.clear()

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file, optionally a line range. Prefer a range once you know where you are looking. A read that does not fit stops early and the header gives a continuation cursor, including a column for an oversized line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repository root."},
                    "start": {"type": "integer", "description": "First line, 1-based."},
                    "count": {"type": "integer", "description": "How many lines to return."},
                    "column": {"type": "integer", "description": "Zero-based character offset on the starting line; use the continuation cursor for an oversized line."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Extended-regex search across tracked files. Use mode=files first to see where the matches are, then read narrowly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory or file to search under. Defaults to the whole repository."},
                    "mode": {
                        "type": "string",
                        "enum": ["content", "files", "count"],
                        "description": "content returns matching lines, files returns paths only, count returns per-file totals.",
                    },
                    "include": {"type": "string", "description": "Only search paths matching this glob, e.g. *.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "List tracked files whose path matches a glob, e.g. src/**/*.py",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0,
                               "description": "Start at this zero-based matching path; use the next offset returned for an incomplete page."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace an exact span of text in a file. Set replace_all when the same span occurs at several places and every one of them needs the same change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "Exact text to find, including indentation."},
                    "new": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of requiring a unique one.",
                    },
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Write a whole file, creating it or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the repository root. Set background for anything slow, such as a test suite, and collect it later with bash_poll instead of waiting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "background": {"type": "boolean"},
                    "timeout": {"type": "integer", "description": "Seconds to wait when not backgrounded."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash_poll",
            "description": "Collect output from a background command started with bash.",
            "parameters": {
                "type": "object",
                "properties": {"job": {"type": "string"}},
                "required": ["job"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Finish the run. Call this only once the change is complete and you have re-run the command that found the problem to confirm nothing is left.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]

def _schema(name: str) -> dict:
    """One tool's schema by name, for a caller that needs to read or extend it."""
    return next(t["function"] for t in TOOL_SCHEMAS if t["function"]["name"] == name)

if SEARCH_LIMIT:
    _search = _schema("search_text")
    _search["parameters"]["properties"].update({
        "context": {"type": "integer",
                    "description": "Lines of surrounding code to return with each "
                                   "match, up to 20. Enough of them and the match "
                                   "is the read."},
        "head_limit": {"type": "integer",
                       "description": "Stop after this many matching lines "
                                      "(default %d)." % SEARCH_HEAD_LIMIT},
    })
    _search["description"] += (" Ask for context lines rather than following the "
                               "match with a read of the whole file.")
if CASE_TOOL:
    TOOL_SCHEMAS.append({
        "type": "function",
        "function": {
            "name": "case",
            "description": (
                "Record the concrete case you are working from, once you have "
                "one: which requirement, which code, what inputs, what happens "
                "now, what the statement requires, and one command that tells "
                "the two apart. The quotations are checked against the statement "
                "and the file. Declare what the command actually asserts; an "
                "exit status alone does not establish the claimed result."),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string", "description": "Optional name: letter first, then letters, digits, dots, underscores or hyphens; at most 32 characters. Eight named cases plus the unnamed case; reuse a name to replace its declaration."},
                    "requirement_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional existing R-number IDs this assertion is claimed to cover. Coverage remains your claim."},
                    "scope": {"type": "object", "properties": {key: {"type": "string"} for key in ("caller", "workload", "dataset")}, "additionalProperties": False, "description": "Optional bounded descriptions of caller, workload and dataset. These are claims, not automatically verified facts."},
                    "requirement": {"type": "string",
                                    "description": "Quoted from the problem statement, word for word."},
                    "code_path": {"type": "string", "description": "The production file this is about."},
                    "code_quote": {"type": "string",
                                   "description": "Quoted from that file, word for word."},
                    "inputs": {"type": "string",
                               "description": "The rows or objects that show it, where the task has such a shape."},
                    "current_result": {"type": "string", "description": "What the code produces now."},
                    "required_result": {"type": "string", "description": "What the statement requires instead."},
                    "observation": {"type": "string",
                                    "description": "Exact command (or SQL text for sql) that distinguishes them."},
                    "asserts": {"type": "string",
                                "enum": ["current_result", "required_result", "unspecified"],
                                "description": "Which result this command explicitly asserts. Leave unspecified for inspection, printed output, or an unrelated suite."},
                    "unchanged": {"type": "string",
                                  "description": "A neighbouring behaviour that must stay as it is."},
                },
                "required": ["requirement", "code_path", "code_quote",
                             "current_result", "required_result", "observation"],
            },
        },
    })
    TOOL_SCHEMAS.append({
        "type": "function", "function": {
            "name": "read_case", "description": "Read a case declaration and current observation, including omitted detail. Character pagination; empty case_id reads the legacy unnamed case.",
            "parameters": {"type": "object", "properties": {
                "case_id": {"type": "string"},
                "offset": {"type": "integer", "description": "Nonnegative character offset; default 0."}},
                "required": ["case_id"]}}})
TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "read_requirement",
        "description": "Read the complete text of requirements from this task's extracted index: one id such as R7, a range such as R3-R12, or all (at most 20 items per call). Character pagination with offset preserves text omitted from the displayed index; the original statement remains authoritative.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "One ID such as R7, a range such as R3-R12, or all."},
                "offset": {"type": "integer", "description": "Nonnegative character offset, default 0."},
            },
            "required": ["id"],
        },
    },
})
TOOL_SCHEMAS.append({
    "type": "function", "function": {
        "name": "read_measurement",
        "description": "Read retained literal database reports and invocation provenance by check_id. Character pagination; omission and unknown counts remain explicit. Query IDs do not prove application coverage.",
        "parameters": {"type": "object", "properties": {
            "check_id": {"type": "string"},
            "offset": {"type": "integer", "description": "Nonnegative character offset; default 0."}},
            "required": ["check_id"]}}})
if DB_TOOL:
    TOOL_SCHEMAS.append({
        "type": "function",
        "function": {
            "name": "sql",
            "description": (
                "Run one SQL statement against the database this repository "
                "configures, through its own connection settings. PostgreSQL: sent "
                "through psql inside a transaction that is rolled back; the result "
                "reports whether that completed. " + (
                    "ClickHouse: sent over HTTP read-only "
                    "unless write=true; nothing is rolled back; the result carries the "
                    "server's work summary for the statement and, when readable, its "
                    "system.query_log row. " if CH_HTTP else "") +
                "External and nontransactional effects are "
                "not covered. Respect execution restrictions in the statement. Use "
                "it to establish what a query returns or how much work it does, "
                "then confirm the application itself produces that query."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "One statement, written as the engine expects it. PostgreSQL also "
                                             "takes several, separated by semicolons: every result prints in "
                                             "order, all inside the one rolled-back transaction."},
                    "target": {"type": "string",
                               "description": "Which configured database, when more than one exists."},
                    "explain": {"type": "boolean",
                                "description": "Return the plan without running the statement "
                                               "(PostgreSQL EXPLAIN%s)." % (
                                                   "; ClickHouse EXPLAIN PLAN with actions" if CH_HTTP else "")},
                    "analyze": {"type": "boolean",
                                "description": "PostgreSQL: run EXPLAIN (ANALYZE, BUFFERS) and return the plan "
                                               "with real row counts, timing and buffers, as text (format "
                                               "\"json\" for the tree). This EXECUTES the statement.%s" % (
                                                   " ClickHouse statements always report real counts."
                                                   if CH_HTTP else "")},
                    "format": {"type": "string",
                               "description": "PostgreSQL plans: \"json\" for a machine-readable "
                                              "plan tree instead of the text form."},
                    "timeout": {"type": "integer", "description": "Seconds to wait."},
                },
                "required": ["query"],
            },
        },
    })
    if CH_HTTP:
        _schema("sql")["parameters"]["properties"]["write"] = {
            "type": "boolean",
            "description": "ClickHouse only: allow a statement that changes data "
                           "or settings. It persists; there is no rollback."}
if DB_TOOL and MEASURE_TOOL:
    TOOL_SCHEMAS.append({
        "type": "function",
        "function": {
            "name": "measure",
            "description": (
                "Run one command twice against the same database: in a separate checkout "
                "of the original code (the base commit, with installed dependencies shared) "
                "and in the working tree; report statements, rows read, bytes, memory and "
                "wall time side by side from " + ("system.query_log (ClickHouse) or " if CH_HTTP else "") +
                "pg_stat_database (PostgreSQL). Both runs are recorded as measurements that "
                "pair for the hand-in reading. Do not obtain a baseline by copying or "
                "checking out files in the working tree. Optional scales run the pair at "
                "several sizes after a setup command and report how the work grew against "
                "the size; nothing here is a verdict."),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string",
                                "description": "The application command to run, as for bash."},
                    "caller": {"type": "string",
                               "description": "Which code path or entry point this exercises."},
                    "workload": {"type": "string",
                                 "description": "Which request, selection or workload it runs."},
                    "dataset": {"type": "string",
                                "description": "Which fixture or data the database holds for it."},
                    "target": {"type": "string",
                               "description": "Which configured database, when more than one exists."},
                    "timeout": {"type": "integer", "description": "Seconds per run (default 120)."},
                    "baseline": {"type": "boolean",
                                 "description": "Also run in the separate base checkout (default true)."},
                    "scales": {"type": "array",
                               "description": "Up to 6 {label, size, setup} entries. setup is a shell command run in the working tree before each pair to prepare data at that size; it runs in its own shell, so it cannot set variables for the measured command.",
                               "items": {"type": "object", "properties": {
                                   "label": {"type": "string"}, "size": {"type": "number"},
                                   "setup": {"type": "string"}}, "required": ["label", "size"]}},
                },
                "required": ["command", "caller", "workload", "dataset"],
            },
        },
    })
if OUTLINE:
    TOOL_SCHEMAS.append({
        "type": "function",
        "function": {
            "name": "outline",
            "description": "Index one Python file: every class and function in it "
                           "with the lines it spans, and nothing of what they say. "
                           "Call it before reading a long file, then read the range "
                           "it names.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    })
HISTORY_GIT = re.compile(
    # Global options may carry a value (git -C . stash, git -c k=v commit); step over both.
    r"\bgit\s+(?:(?:-C|-c|--git-dir|--work-tree|--namespace|--exec-path|--config-env)\s+\S+\s+|-[^\s]+\s+)*"
    r"(commit|stash|checkout|switch|restore|reset|clean|revert|rebase|merge|cherry-pick|push)\b"
)
URL_IN_COMMAND = re.compile(r"(?i)\b(https?)://([A-Za-z0-9_.-]+|\[[0-9a-f:]+\])(?::(\d{2,5}))?")
ENV_REFERENCE = re.compile(r"\$\{?([A-Z][A-Z0-9_]*)\}?")
NETWORK_COMMAND = re.compile(
    r"(?:^|[|&;]|\$\(|`)\s*(?:sudo\s+)?"
    r"(curl|wget|nc|ncat|telnet|ssh|scp|rsync|ftp|"
    r"git\s+(?:fetch|pull|clone|remote|ls-remote|submodule))(?![\w-])"
)
NETWORK_TOOL_WORD = re.compile(r"(?<![\w.-])(curl|wget|nc|ncat|telnet|ssh|scp|rsync|ftp)(?![\w-])")
NETWORK_WORD = re.compile(
    r"(?<![\w.-])(?:curl|wget|nc|ncat|telnet|ssh|scp|rsync|ftp)(?![\w-])|\bgit\b")
NETWORK_TOOLS = ("curl", "wget", "nc", "ncat", "telnet", "ssh", "scp", "rsync", "ftp")
GIT_NETWORK = ("fetch", "pull", "clone", "remote", "ls-remote", "submodule")
# git's global options whose value is the next word: `git -C repo fetch` fetches.
GIT_VALUE_OPTIONS = ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--config-env")
COMMAND_WRAPPERS = ("sudo", "nohup", "time", "command", "exec", "nice", "timeout", "env", "stdbuf",
                    "xargs", "ionice", "setsid", "unbuffer", "chronic")
SHELLS = ("bash", "sh", "zsh", "dash", "ksh")
# Wrapper options whose value is a separate word: stepping over the option alone would
# take its value for the command word and hide the program that actually runs.
WRAPPER_VALUE_OPTIONS = {
    "sudo": ("-u", "-g", "-C", "-h", "-p", "-D", "-r", "-t", "-U", "--user", "--group", "--chdir"),
    "env": ("-u", "-C", "-S", "--unset", "--chdir", "--split-string"),
    "timeout": ("-s", "-k", "--signal", "--kill-after"),
    "nice": ("-n", "--adjustment"), "ionice": ("-c", "-n", "-p"), "stdbuf": ("-i", "-o", "-e"),
    "xargs": ("-I", "-n", "-P", "-L", "-d", "-a", "-E", "-s")}
SHELL_SEPARATORS = ("&&", "||", ";", "|", "&", "|&", ";;", "(", ")", "\n")
SUBSTITUTION = re.compile(r"`|\$\(|<\(|>\(")
SHELL_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

def shell_segments(command: str, depth: int = 0) -> list | None:
    """Each simple command as (assignments, argv) with redirections removed, or None.

    A shell -c script is read as commands of its own, and wrappers such as
    timeout, env or xargs are stepped over, so the command word is the program
    that actually runs. None means the line could not be read as shell words.
    """
    try:
        lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    segments, current = [], []
    for token in [*tokens, ";"]:
        if token in SHELL_SEPARATORS:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    out = []
    for words in segments:
        argv, skip = [], False
        for word in words:
            if skip:
                skip = False
                continue
            if word and set(word) <= set("<>&|"):
                skip = True  # a redirection and its target
                if argv and argv[-1].isdigit():
                    argv.pop()  # the descriptor number in 2>&1
                continue
            argv.append(word)
        assignments = []
        while argv and SHELL_ASSIGNMENT.match(argv[0]):
            assignments.append(argv.pop(0))
        while argv and argv[0] in COMMAND_WRAPPERS:
            takes_value = WRAPPER_VALUE_OPTIONS.get(argv.pop(0), ())
            while argv and (argv[0].startswith("-") or SHELL_ASSIGNMENT.match(argv[0])
                            or re.fullmatch(r"\d+(?:\.\d+)?[smhd]?|\{\}", argv[0])):
                option = argv.pop(0)
                if SHELL_ASSIGNMENT.match(option):
                    assignments.append(option)
                elif option in takes_value and argv:
                    argv.pop(0)  # the option's own value (sudo -u postgres, timeout -s KILL)
        script_flag = next((i for i, word in enumerate(argv[1:], 1)
                            if word == "-c" or (re.fullmatch(r"-[A-Za-z]+", word) and "c" in word)), None)
        if argv and argv[0].rsplit("/", 1)[-1] in SHELLS and script_flag is not None:
            if depth >= 3:
                return None  # nested past what is read: the line cannot be checked
            script = argv[script_flag + 1:script_flag + 2]
            inner = shell_segments(script[0], depth + 1) if script else []
            if inner is None:
                return None
            out.extend((assignments + a, v) for a, v in inner)
            continue
        if argv:
            out.append((assignments, argv))
    return out

def network_invocations(command: str) -> list | None:
    """The network commands a line would run: [{"tool", "args", "assignments"}], or None."""
    segments = shell_segments(command)
    if segments is None:
        return None
    found = []
    for assignments, argv in segments:
        program = argv[0].rsplit("/", 1)[-1]
        if program in NETWORK_TOOLS:
            found.append({"tool": program, "args": argv[1:], "assignments": assignments})
        elif program == "git":
            rest, index = argv[1:], 0
            while index < len(rest) and rest[index].startswith("-"):
                index += 2 if rest[index] in GIT_VALUE_OPTIONS else 1
            if index < len(rest) and rest[index] in GIT_NETWORK:
                found.append({"tool": "git " + rest[index], "args": argv[1:], "assignments": assignments})
    return found

# Statements that change a database do not go out through bash: the sql tool is
# the path there, where a PostgreSQL statement runs inside a transaction that is
# rolled back and a lasting ClickHouse change asks for write=true. A database
# client carrying such a statement inline, or curl to a configured ClickHouse
# endpoint carrying one, is refused before it runs. What cannot be read from
# the line (a file of statements, a script) is not guessed at.
DB_WRITE_FENCE = flag("RIDGES_DB_WRITE_FENCE")
CHANGING_VERBS = frozenset(("INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "REPLACE", "TRUNCATE",
                            "CREATE", "ALTER", "DROP", "RENAME", "COMMENT", "GRANT", "REVOKE", "REFRESH",
                            "CLUSTER", "OPTIMIZE", "ATTACH", "DETACH", "EXCHANGE", "UNDROP"))
CLIENT_SQL_OPTIONS = {"psql": ("-c", "--command"), "clickhouse-client": ("-q", "--query")}
# Each client's short options that take no value, which may therefore share one word with the
# query option after them: psql -tAc "...", clickhouse-client -nq "..." (as each --help lists them).
CLIENT_FLAG_LETTERS = {"psql": "lVX1?abeEnqsSAHtxz0wW", "clickhouse-client": "VnmAtEs"}
HEREDOC_TO_CLIENT = re.compile(
    r"(?:^|[\s;&|(])(?:\S*/)?(psql|clickhouse-client|clickhouse\s+client)\b[^\n]*?<<-?\s*(['\"]?)([A-Za-z_]\w*)\2"
    r"[^\n]*\n([\s\S]*?)\n[ \t]*\3[ \t]*(?:\n|$)")
HERESTRING_TO_CLIENT = re.compile(
    r"(?:^|[\s;&|(])(?:\S*/)?(psql|clickhouse-client|clickhouse\s+client)\b[^\n]*?<<<\s*"
    r"(?:'([^']*)'|\"((?:[^\"\\]|\\.)*)\")")
PIPE_TO_CLIENT = re.compile(
    r"(?:^|[\s;&|(])(?:echo(?:\s+-[neE]+)*|printf)\s+(?:'([^']*)'|\"((?:[^\"\\]|\\.)*)\")\s*\|\s*"
    r"(?:\S*/)?(psql|clickhouse-client|clickhouse\s+client)\b")
TEMP_TABLE = re.compile(r"^CREATE\s+(?:(?:LOCAL|GLOBAL)\s+)?TEMP(?:ORARY)?\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.]+)", re.I)
CHANGED_TABLE = re.compile(
    r"^(?:INSERT\s+(?:OR\s+\w+\s+)?INTO|UPDATE|DELETE\s+FROM|TRUNCATE(?:\s+TABLE)?|DROP\s+TABLE(?:\s+IF\s+EXISTS)?)"
    r"\s+(?:ONLY\s+)?([\w.]+)", re.I)

def statement_write(statement: str, temporary: set) -> str:
    """The verb when one statement changes the database, "" when it does not.

    Strings and comments are not read. A temporary table this run's command
    created is its own scratch space, so writing to it changes nothing lasting;
    EXPLAIN ANALYZE carries out the statement it explains.
    """
    code = " ".join(sql_code_only(statement or "").split())
    words = code.upper().split()
    if not words:
        return ""
    verb = words[0]
    if verb == "EXPLAIN":
        analyzed = re.match(r"EXPLAIN\s+(?:\(([^)]*)\)|(ANALY[SZ]E)(?:\s+VERBOSE)?)\s*(.*)$", code, re.I)
        if analyzed and (analyzed.group(2) or re.search(r"\bANALY[SZ]E\b(?!\s+(?:FALSE|OFF|0)\b)",
                                                        analyzed.group(1) or "", re.I)):
            return statement_write(analyzed.group(3), temporary)
        return ""
    if verb == "WITH":
        # A changing statement in a WITH query heads a part of it: a CTE's body, or the
        # statement after the CTE list. FOR UPDATE only locks rows, and a column or table
        # function called merge or update heads nothing.
        found = re.search(r"[()]\s*(INSERT|UPDATE|DELETE|MERGE)\b", code, re.I)
        return found.group(1).upper() if found else ""
    if verb == "SELECT":
        into = re.search(r"\bINTO\s+(?!OUTFILE\b)", outside_parentheses(code), re.I)
        return "SELECT INTO" if into else ""
    if verb == "COPY":
        # COPY table FROM loads rows; COPY (query) TO only copies a query's rows out.
        return "COPY" if re.search(r"\bFROM\b", outside_parentheses(code), re.I) else ""
    if verb == "SYSTEM":
        return "" if re.match(r"SYSTEM\s+(?:FLUSH\s+LOGS|DROP\s+[\w\s]*CACHE)\b", code, re.I) else "SYSTEM"
    created = TEMP_TABLE.match(code)
    if created:
        temporary.add(created.group(1).lower())
        return ""
    if verb not in CHANGING_VERBS:
        return ""
    table = CHANGED_TABLE.match(code)
    if table and table.group(1).lower() in temporary:
        return ""
    return verb

def statements_write(sql_parts: list) -> str:
    """The first changing verb among statements run in order, "" when none changes anything lasting.

    Statements that open a transaction and close it with ROLLBACK, and nothing
    in between that commits, change nothing that lasts.
    """
    statements = [part for text in sql_parts for part in sql_code_only(text or "").split(";") if part.strip()]
    heads = [" ".join(part.upper().split()[:2]) for part in statements]
    if heads and heads[0].split()[0] in ("BEGIN", "START") and heads[-1] in ("ROLLBACK", "ABORT"):
        if not any(head.split()[0] in ("COMMIT", "END", "PREPARE") or head in ("ROLLBACK", "ABORT")
                   for head in heads[1:-1]):
            return ""
    temporary: set = set()
    for statement in statements:
        verb = statement_write(statement, temporary)
        if verb:
            return verb
    return ""

def client_invocations(command: str) -> list:
    """(engine, argv, inline statements) for each database client a line runs, statements in order."""
    found = []
    for _, argv in shell_segments(command) or []:
        program, words = argv[0].rsplit("/", 1)[-1], list(argv)
        if program == "clickhouse" and words[1:2] == ["client"]:
            program, words = "clickhouse-client", words[:1] + words[2:]
        options = CLIENT_SQL_OPTIONS.get(program)
        if not options:
            continue
        parts, index = [], 1
        while index < len(words):
            word = words[index]
            index += 1
            if word in options and index < len(words):
                parts.append(words[index])
                index += 1
            elif any(word.startswith(option + "=") for option in options if option.startswith("--")):
                parts.append(word.split("=", 1)[1])
            elif word[:2] in options and len(word) > 2 and not word.startswith("--"):
                parts.append(word[2:])
            elif word.startswith("-") and not word.startswith("--"):
                # Flags that take no value, then the query option: its value is the rest of
                # the word, or the next word. Any other letter ends the reading.
                letters = word[1:]
                flags = len(letters) - len(letters.lstrip(CLIENT_FLAG_LETTERS[program]))
                if flags and "-" + letters[flags:flags + 1] in options:
                    if letters[flags + 1:]:
                        parts.append(letters[flags + 1:])
                    elif index < len(words):
                        parts.append(words[index])
                        index += 1
        whole = "-1" in words or "--single-transaction" in words
        found.append(("postgresql" if program == "psql" else "clickhouse", words, parts, whole))
    for match in HEREDOC_TO_CLIENT.finditer(command or ""):
        engine = "postgresql" if match.group(1) == "psql" else "clickhouse"
        found.append((engine, [match.group(1)], [match.group(4)], False))
    for match in PIPE_TO_CLIENT.finditer(command or ""):
        engine = "postgresql" if match.group(3) == "psql" else "clickhouse"
        text = match.group(1) if match.group(1) is not None else match.group(2)
        found.append((engine, [match.group(3)], [text], False))
    for match in HERESTRING_TO_CLIENT.finditer(command or ""):
        engine = "postgresql" if match.group(1) == "psql" else "clickhouse"
        text = match.group(2) if match.group(2) is not None else match.group(3)
        found.append((engine, [match.group(1)], [text], False))
    return found

def network_payloads(invocation: dict) -> list:
    """The statement texts a curl or wget invocation would send: its body and any query= in its URL."""
    args, parts = invocation["args"], []
    body_options = ("-d", "--data", "--data-ascii", "--data-binary", "--data-raw", "--data-urlencode",
                    "--post-data", "--body-data")
    for index, word in enumerate(args):
        name, _, attached = word.partition("=") if word.startswith("--") else (word, "", "")
        value = attached or (args[index + 1] if name in body_options and index + 1 < len(args) else "")
        if name in body_options and value and not value.startswith("@"):
            named = urllib.parse.parse_qs(value).get("query") if value.startswith("query=") else None
            parts.extend(named or [value])
        if "?" in word and "query=" in word:  # a URL, or a URL variable with a query string after it
            parts.extend(urllib.parse.parse_qs(urllib.parse.urlsplit(word).query).get("query") or [])
    return parts

# Options that can send a request somewhere other than the URLs on the line, or
# read more of the request from a file: a destination they add cannot be checked.
CURL_UNVERIFIABLE = ("-K", "--config", "-x", "--proxy", "--preproxy", "--socks4", "--socks4a", "--socks5",
                     "--socks5-hostname", "--resolve", "--connect-to", "--unix-socket", "--abstract-unix-socket",
                     "--dns-servers", "--doh-url", "-:", "--next")
CURL_VALUE = ("-d", "--data", "--data-ascii", "--data-binary", "--data-raw", "--data-urlencode", "--json", "-H",
              "--header", "-X", "--request", "-o", "--output", "-u", "--user", "-A", "--user-agent", "-b",
              "--cookie", "-c", "--cookie-jar", "-e", "--referer", "-F", "--form", "--form-string", "-T",
              "--upload-file", "-w", "--write-out", "-m", "--max-time", "--connect-timeout", "--retry",
              "--retry-delay", "--retry-max-time", "-r", "--range", "-Y", "--speed-limit", "-y", "--speed-time",
              "-z", "--time-cond", "-C", "--continue-at", "--limit-rate", "--max-filesize", "--max-redirs",
              "--url-query", "--oauth2-bearer", "--cacert", "--capath", "--cert", "--key", "-E", "--cert-type",
              "--key-type", "-D", "--dump-header", "--stderr", "--trace", "--trace-ascii", "--output-dir",
              "--expect100-timeout", "--keepalive-time", "--variable", "--etag-save", "--etag-compare",
              "--interface", "--local-port")
CURL_SHORT_VALUE = "dHXouAbceFTwmrYyzCED"
WGET_UNVERIFIABLE = ("-i", "--input-file", "-e", "--execute", "--config", "-B", "--base", "--use-askpass")
WGET_VALUE = ("-O", "--output-document", "-o", "--output-file", "-a", "--append-output", "-P",
              "--directory-prefix", "--header", "--post-data", "--post-file", "-U", "--user-agent", "-T",
              "--timeout", "-t", "--tries", "--user", "--password", "--http-user", "--http-password", "--method",
              "--body-data", "--body-file", "-w", "--wait", "--limit-rate", "-Q", "--quota", "--bind-address",
              "--referer", "--load-cookies", "--save-cookies")
WGET_SHORT_VALUE = "OoaPUTtwQ"

def network_destinations(tool: str, args: list) -> list | None:
    """The destinations a curl or wget invocation names, or None when one is unverifiable."""
    unverifiable, value, short = ((CURL_UNVERIFIABLE, CURL_VALUE, CURL_SHORT_VALUE) if tool == "curl"
                                  else (WGET_UNVERIFIABLE, WGET_VALUE, WGET_SHORT_VALUE))
    refused_short = "".join(o[1] for o in unverifiable if len(o) == 2)
    destinations, index, positional_only = [], 0, False
    while index < len(args):
        word = args[index]
        index += 1
        if positional_only or not word.startswith("-") or word == "-":
            destinations.append(word)
            continue
        if word == "--":
            positional_only = True
            continue
        if word.startswith("--"):
            name, _, attached = word.partition("=")
            if name in unverifiable or name.startswith("--proxy"):
                return None
            if name == "--url":
                if attached:
                    destinations.append(attached)
                elif index < len(args):
                    destinations.append(args[index])
                    index += 1
                continue
            if name in value and not attached:
                index += 1
            continue
        for position, flag in enumerate(word[1:], 1):
            if flag in refused_short:
                return None
            if flag in short:
                if position == len(word) - 1:
                    index += 1  # the value is the next word
                break  # otherwise the rest of the word is the value
    return destinations

def verified_destination(word: str, hosts: set) -> str:
    """host:port when word is an http(s) URL or a URL variable of a configured endpoint, else ""."""
    variable = re.match(r"^\$\{?([A-Z][A-Z0-9_]*)\}?", word)
    if variable:
        name = variable.group(1)
        if env_variable_family(name)[1] != "URL":
            return ""
        endpoint = endpoint_from_url(os.environ.get(name, ""), name)
        if not endpoint or (str(endpoint["host"]).lower(), endpoint["port"]) not in hosts:
            return ""
        return "$%s=%s:%d" % (name, endpoint["host"], endpoint["port"])
    try:
        parts = urllib.parse.urlsplit(word)
        port = parts.port
    except ValueError:
        return ""
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        return ""
    port = port or (443 if parts.scheme.lower() == "https" else 80)
    if (parts.hostname.lower(), port) not in hosts:
        return ""
    return "%s:%d" % (parts.hostname, port)

TEST_PATH = re.compile(
    r"(^|/)conftest\.py$|(^|/)tests?(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")
CHECK_TEST_PATH = TEST_PATH
if RIDGES_SCOPE_FOLLOWS_STATEMENT:
    TEST_PATH = re.compile(TEST_PATH.pattern + r"|_test\.go$|\.(?:test|spec)\.(?:js|ts)$")
NOQA_DIRECTIVE = re.compile(r"#\s*(?:(?:ruff|flake8)\s*:\s*)?noqa\b", re.I)
FAILED_TEST = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.M)
PASSED_COUNT = re.compile(r"(\d+) passed")
RAN_TESTS = re.compile(r"^Ran (\d+) tests?\b", re.M)
FENCED_BLOCK = re.compile(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)```", re.S)
CODE_SPAN = re.compile(r"`([^`\n]+)`")
COMMAND_SEPARATOR = frozenset(("&&", "||", "|", "|&", "&", ";", ";;"))
REDIRECT_IN_TOKEN = re.compile(r"\d*(?:>>|>|<<|<)&?")

def split_redirect(token: str) -> tuple:
    """A shell token split from a redirection stuck to it, and whether the redirection was all of
    it.
    """
    found = REDIRECT_IN_TOKEN.search(token)
    if not found:
        return token, False
    return token[:found.start()], not token[found.end():]

LINE_CONTINUATION = re.compile(r"\\\n[ \t]*")
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*=")

def logical_lines(block: str) -> list:
    """One entry per command, with backslash continuations joined.

    A fenced block holds one command per line, and a statement is free to
    wrap a long command over several lines. Splitting the block with shlex
    drops the newlines, which merges every command in the block into one.
    """
    joined = LINE_CONTINUATION.sub(" ", block or "")
    return [line for line in joined.split("\n") if line.strip()]

def strip_assignments(argv: list) -> list:
    """Drop leading VAR=value words, which name the environment, not the tool."""
    rest = list(argv)
    while rest and ASSIGNMENT.match(rest[0]):
        rest = rest[1:]
    return rest

def statement_markdown_blocks(text: str, *, prose_lines: set | None = None) -> tuple[list[str], list[str]]:
    """Separate closed code fences from bounded inline Markdown blocks."""
    fenced: list[str] = []
    paragraphs: list[str] = []
    lines: list[str] = []
    quote_depth = 0
    list_indent = 0
    list_columns: list[int] = []
    fence = None
    html_end = None
    html_quote_depth = 0
    html_list_indent = 0

    def flush() -> None:
        if lines:
            paragraphs.append("\n".join(lines))
            lines.clear()

    for line_number, raw in enumerate(text.split("\n")):
        if fence is not None:
            line = raw
            contained = True
            for _ in range(fence["quote_depth"]):
                marker = re.match(r" {0,3}>[ \t]?", line)
                if marker is None:
                    contained = False
                    break
                line = line[marker.end():]
            indent = fence["list_indent"]
            if indent and line.strip():
                if line.startswith(" " * indent):
                    line = line[indent:]
                else:
                    contained = False
            if contained:
                closing = r" {0,3}" + re.escape(fence["marker"][0]) + "{" + str(len(fence["marker"])) + r",}[ \t]*"
                if re.fullmatch(closing, line):
                    fenced.append("\n".join(fence["body"]))
                    fence = None
                else:
                    removed = min(fence["indent"], len(line) - len(line.lstrip(" ")))
                    fence["body"].append(line[removed:])
                continue
            # An interrupted, unclosed fence is not a source of inferred checks.
            fence = None
            quote_depth = 0
            list_indent = 0
            list_columns.clear()

        line = raw
        depth = 0
        while True:
            if html_end is not None and depth >= html_quote_depth:
                break
            marker = re.match(r" {0,3}>[ \t]?", line)
            if marker is None:
                break
            depth += 1
            line = line[marker.end():]
        if html_end is not None and (depth < html_quote_depth or (
                html_list_indent and line.strip() and not line.startswith(" " * html_list_indent))):
            html_end = None
            quote_depth = depth
            list_indent = 0
            list_columns.clear()
        if html_end is not None:
            if html_end:
                if re.search(html_end, line, re.I):
                    html_end = None
                continue
            if line.strip():
                continue
            html_end = None
        if not line.strip():
            flush()
            # Blank lines also separate paragraphs within the same list item.
            continue
        if not lines:
            if depth != quote_depth:
                list_columns.clear()
            while list_columns and not line.startswith(" " * list_columns[-1]):
                list_columns.pop()
            list_indent = list_columns[-1] if list_columns else 0
            quote_depth = depth
        # A new explicit container interrupts an inline span. Unmarked text can
        # be a lazy continuation of the current paragraph in a container.
        if depth and depth != quote_depth:
            flush()
            list_indent = 0
            list_columns.clear()
        if depth:
            quote_depth = depth

        marker = re.match(r" {0,3}(?:[-+*]|[0-9]{1,9}[.)])(?:[ \t]+|$)", line)
        in_list = False
        if marker is not None:
            flush()
            list_indent = marker.end()
            while list_columns and list_columns[-1] >= list_indent:
                list_columns.pop()
            list_columns.append(list_indent)
            line = line[list_indent:]
            in_list = True
        elif list_indent and line.startswith(" " * list_indent):
            line = line[list_indent:]
            in_list = True
            nested = re.match(r" {0,3}(?:[-+*]|[0-9]{1,9}[.)])(?:[ \t]+|$)", line)
            if nested is not None:
                flush()
                list_indent += nested.end()
                list_columns.append(list_indent)
                line = line[nested.end():]

        # Raw HTML blocks and comments are not inline command declarations.
        tag_line = line.lstrip(" ") if len(line) - len(line.lstrip(" ")) <= 3 else ""
        html_stop = None
        if re.match(r"<(?:pre|script|style|textarea)(?:[ \t>]|$)", tag_line, re.I):
            html_stop = r"</(?:pre|script|style|textarea)>"
        elif tag_line.startswith("<!--"):
            html_stop = "-->"
        elif tag_line.startswith("<?"):
            html_stop = r"\?>"
        elif tag_line.startswith("<![CDATA["):
            html_stop = r"\]\]>"
        elif re.match(r"<![A-Za-z]", tag_line):
            html_stop = ">"
        elif re.match(
                r"</?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?:[ \t>]|/>|$)",
                tag_line, re.I):
            html_stop = ""
        elif not lines and re.fullmatch(r"</?[A-Za-z][A-Za-z0-9-]*(?:[ \t][^<>]*)?/?>[ \t]*", tag_line):
            html_stop = ""
        if html_stop is not None:
            flush()
            if not in_list:
                list_indent = 0
                list_columns.clear()
            if not html_stop or not re.search(html_stop, line, re.I):
                html_end = html_stop
                html_quote_depth = depth
                html_list_indent = list_indent
            continue

        opening = re.fullmatch(r"( {0,3})(`{3,}|~{3,})(.*)", line)
        if opening and not (opening[2][0] == "`" and "`" in opening[3]):
            flush()
            if not in_list:
                # A fence at the margin is not a continuation of the list item
                # above it; it ends that item, as a blank line would.
                list_indent = 0
                list_columns.clear()
            fence = {"marker": opening[2], "indent": len(opening[1]),
                     "quote_depth": depth, "list_indent": list_indent, "body": []}
            continue
        if re.match(r" {0,3}#{1,6}(?:[ \t]+|$)", line):
            flush()
            if not in_list:
                list_indent = 0
                list_columns.clear()
            paragraphs.append(line)
            if prose_lines is not None:
                prose_lines.add(line_number)
            continue
        if (re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", line)
                or re.fullmatch(r" {0,3}(?:(?:\*[ \t]*){3,}|(?:_[ \t]*){3,}|(?:-[ \t]*){3,})", line)):
            flush()
            if not in_list:
                list_indent = 0
                list_columns.clear()
            continue
        # Indented code has no inline Markdown spans. Indentation inside an
        # already open paragraph remains ordinary inline content.
        if not lines and (line.startswith("    ") or line.startswith("\t")):
            continue
        if prose_lines is not None:
            prose_lines.add(line_number)
        lines.append(line)
    flush()
    return fenced, paragraphs

def statement_command_blocks(statement: str) -> list[str]:
    """Keep fenced command boundaries; unwrap inline code within a paragraph."""
    text = (statement or "").replace("\r\n", "\n").replace("\r", "\n")
    blocks, paragraphs = statement_markdown_blocks(text)
    for paragraph in paragraphs:
        runs = list(re.finditer(r"`+", paragraph))
        following, last = {}, {}
        for index in range(len(runs) - 1, -1, -1):
            width = len(runs[index].group())
            following[index] = last.get(width)
            last[width] = index
        index = 0
        cursor = 0
        while index < len(runs):
            start = runs[index].start()
            comment = paragraph.find("<!--", cursor, start)
            if comment >= 0:
                close = paragraph.find("-->", comment + 4)
                cursor = len(paragraph) if close < 0 else close + 3
                while index < len(runs) and runs[index].start() < cursor:
                    index += 1
                continue
            cursor = runs[index].end()
            slash = start - 1
            while slash >= 0 and paragraph[slash] == "\\":
                slash -= 1
            end = following[index]
            if (start - slash - 1) % 2 or end is None:
                index += 1
                continue
            value = paragraph[runs[index].end():runs[end].start()].replace("\n", " ")
            if value.startswith(" ") and value.endswith(" ") and value.strip(" "):
                value = value[1:-1]
            blocks.append(value)
            cursor = runs[end].end()
            index = end + 1
    return blocks

def command_lines(statement: str) -> list:
    """Every command the statement spells out, one argv per command."""
    found = []
    text = statement or ""
    for block in statement_command_blocks(text):
        for line in logical_lines(block):
            try:
                tokens = shlex.split(line, comments=True)
            except ValueError:
                continue
            current = []
            skip = False
            for token in tokens + [";"]:
                if skip:
                    skip = False
                elif token in COMMAND_SEPARATOR:
                    if current:
                        found.append(current)
                    current = []
                else:
                    head, skip = split_redirect(token)
                    if head:
                        current.append(head)
    for command in bare_commands(text):
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        if tokens and tokens not in found:
            found.append(tokens)
    return found

SUITE_TALLY = re.compile(r"^=*\s*(\d+ (?:failed|passed|error|errors)[^=]*?)\s*=*$", re.M)
SUITE_REASON = re.compile(r"^\S+\.py:\d+:\s*(\S.*)$", re.M)
SUITE_FAULT = re.compile(r"^E[ \t]+([\w.]*(?:Error|Exception)\w*)\b(.*)$", re.M)
SUITE_FAILURES_HEAD = re.compile(r"^=+ FAILURES =+\s*$", re.M)
SUITE_TIERS = ("", " --noconftest", " --noconftest --import-mode=importlib")
MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z_][A-Za-z_0-9.]*)'")
MISSING_DIST = re.compile(
    r"^(?:E[ \t]+)?(?:[\w.]+\.)?PackageNotFoundError: "
    r"No package metadata was found for "
    r"([A-Za-z0-9](?:[\w.-]*[A-Za-z0-9])?)[ \t]*$", re.M)
SUITE_SHIM_LIMIT = 6
SHIM_SOURCE = '''"""Stands in for a distribution this image does not carry.

This permissive stand-in can allow diagnostic collection to continue. Its
attributes and calls do not implement the missing dependency's behavior, and
can change test results in either direction. Results obtained with it cannot
establish correctness or a behavioral baseline. Warden.collect excludes any
reading that used dependency or metadata stand-ins from baseline admission.
"""
import functools as _functools
import importlib.abc as _abc
import importlib.machinery as _machinery
import inspect as _inspect
import sys as _sys


class _Null:
    __slots__ = ("_name",)

    def __init__(self, name="?"):
        object.__setattr__(self, "_name", name)

    def __call__(self, *args, **kwargs):
        # Two shapes arrive here and they want opposite answers.  Used bare, as
        # `@thing`, the one argument IS the test being decorated and handing it
        # back unchanged is the only answer that leaves the test as the project
        # wrote it.  Used as a factory, as `@thing(spec)`, the argument is the
        # spec and handing it back would put the spec where the test belongs.
        #
        # Narrow on purpose.  Only something that is itself a definition is
        # taken for a decoration; any other callable -- a strategy object, a
        # registry, anything of this module's own -- is not.  `callable` alone
        # made `register(handler)` hand `handler` back as though it had been
        # decorated, which is a claim about the project rather than an absence
        # of one.
        #
        # But "a definition" is wider than `def`.  Stacked decorators hand this
        # one whatever the decorator below it returned, and that is routinely a
        # `classmethod`, a `staticmethod`, a `functools.partial` or a builtin;
        # dropping those replaces the test with nothing, which is the same loss
        # by a different route.  So the test is `isroutine`, plus the two
        # descriptors and `partial` by name.
        #
        # The residue, written down rather than papered over: a factory handed a
        # plain function still gets it back.  `@thing` and `thing(fn)` are the
        # same call and nothing at this end can separate them.
        if len(args) == 1 and not kwargs and (
                _inspect.isroutine(args[0]) or _inspect.isclass(args[0])
                or isinstance(args[0], (classmethod, staticmethod,
                                        _functools.partial))):
            return args[0]
        return _Null(self._name + "()")

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Null(self._name + "." + name)

    def __mro_entries__(self, bases):
        # `class Thing(absent.Base):` at import time is one of the commonest
        # ways a suite touches an optional dependency, and without this it is a
        # TypeError -- the very failure this module exists to prevent,
        # reintroduced one line lower down.  The interpreter asks a non-class
        # used as a base what to put there instead, and the answer is NOTHING:
        # an empty tuple drops this base and leaves the others alone.  Naming
        # `object` instead looks equivalent and is not -- in
        # `class Thing(absent.Base, Real)` it produces `(object, Real)`, whose
        # linearisation does not exist, so the class raises at definition time
        # and the collection is lost exactly as before.  With no bases left
        # Python supplies `object` by itself.
        return ()

    def __getitem__(self, item):
        # `absent.Type[int]` in an annotation evaluated at runtime.
        return self

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False

    def __setitem__(self, item, value):
        # `absent.REGISTRY[name] = handler`, run by a test package's own
        # `__init__` as it registers itself.  Reading an item was answered from
        # the first version and writing one was not, which is the wrong place
        # to draw the line: a registry that is read is a registry that is
        # written, and the write comes first.  Nothing is stored, for the same
        # reason nothing else here is: what comes back out is `__getitem__`'s
        # stand-in either way, so remembering the value would only make the
        # answer look more like the absent package than it is.
        return None

    def __delitem__(self, item):
        return None

    def __repr__(self):
        return "<stand-in %s>" % self._name


def _combine(kind):
    """One answer for every binary operator, rather than the met ones.

    A module being imported combines what it imported with something of its
    own, and which operator it reaches for belongs to that module rather than
    to this one: `PREFIX + absent.NAME` while a message is built,
    `absent.Markup | None` in an annotation that is evaluated at runtime,
    `flags & absent.MASK` in a default.  Adding them one at a time as they are
    met means the next absent package still loses a whole reading to the next
    operator -- which is exactly how `|` came to be missing after `+` and `%`
    had been written down.  Answered rather than raised for the reason the rest
    of this module is answered: nothing that comes back is a claim about the
    project, and the guard is the admission check on the recovered reading, not
    the poverty of what a stand-in can do.
    """

    def operator(self, other, *rest):
        # `*rest` is three-argument `pow`, which hands the modulus here as a
        # third positional and would otherwise be the one arithmetic call that
        # still raises.
        return _Null(self._name + kind)

    return operator


for _op in ("add sub mul matmul truediv floordiv mod divmod pow lshift rshift"
            " and xor or").split():
    for _form in ("__%s__", "__r%s__"):
        setattr(_Null, _form % _op, _combine("<" + _op + ">"))

# Ordering, for the same reason and with one more thing to say.  `if limit >
# absent.THRESHOLD:` raises without these, and the answer is a stand-in, which
# is false -- so the branch that would have run over the real package does not
# run, deterministically, in both readings.  The four are reflections of each
# other, so `int > stand-in` is answered by `__lt__` and needs nothing more.
#
# Equality and hashing are deliberately not here.  Those two are what a dict
# and a set are built out of; answering them with something that is neither
# true nor false does not make a reading inert, it makes lookups wrong in a way
# that has nothing to do with the absent package.  Identity is the right answer
# for a stand-in and it is already the default.
for _op in ("lt", "le", "gt", "ge"):
    setattr(_Null, "__%s__" % _op, _combine("<" + _op + ">"))

# The loop variables, which are otherwise two ordinary strings sitting in a
# module whose whole contract is that every name in it answers with a stand-in.
del _op, _form


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _Null(__name__ + "." + name)


class _Finder(_abc.MetaPathFinder, _abc.Loader):
    """Answers for any submodule of this stand-in, however deep.

    A single module file cannot: `from x.y import z` needs `x` to be a package
    with a `y` inside it, and the interpreter reports the missing name one level
    at a time, so the name written down is the top one.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == __name__ or fullname.startswith(__name__ + "."):
            return _machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__

        def _attr(attr, _n=name):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            return _Null(_n + "." + attr)

        module.__getattr__ = _attr
        module.__path__ = []


__path__ = []
_sys.meta_path.append(_Finder())
'''

def resolvable(name: str) -> bool:
    """Can this module be imported here? An unanswerable question is treated as yes, not as
    missing.
    """
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except BaseException:
        return True

def missing_modules(out: str) -> list:
    """The top-level modules an output says are missing and that really are not importable here.
    """
    seen = []
    for name in MISSING_MODULE.findall(out or ""):
        if "." in name:
            continue
        if name not in seen and name.isidentifier() and not resolvable(name):
            seen.append(name)
    return seen

def missing_dists(out: str) -> list:
    """The distributions an output says are missing, by name."""
    seen = []
    for name in MISSING_DIST.findall(out or ""):
        if name not in seen and re.fullmatch(r"[A-Za-z0-9._-]+", name):
            seen.append(name)
    return seen

def inside(path: str, root: str) -> bool:
    """Is this path inside this root? A question that cannot be answered is treated as yes, not as
    an escape.
    """
    try:
        path, root = os.path.realpath(path), os.path.realpath(root)
        if (os.path.splitdrive(path)[0].lower()
                != os.path.splitdrive(root)[0].lower()):
            return False
        return os.path.commonpath([path, root]) == root
    except (ValueError, OSError, TypeError):
        return True

def write_dist_records(names: list, where: str) -> list:
    """Write minimal metadata for named distributions so a version check can find them."""
    made = []
    for name in names:
        try:
            room = os.path.join(where, "%s-0.0.0.dist-info"
                                % re.sub(r"[-_.]+", "_", name))
            os.makedirs(room, exist_ok=True)
            with open(os.path.join(room, "METADATA"), "w", encoding="utf-8") as handle:
                handle.write("Metadata-Version: 2.1\nName: %s\nVersion: 0.0.0\n" % name)
        except OSError:
            continue
        made.append(name)
    return made

def write_shims(names: list, where: str) -> list:
    """Write importable stand-ins for named modules, and return the ones written."""
    made = []
    for name in names:
        try:
            room = os.path.join(where, name)
            os.makedirs(room, exist_ok=True)
            with open(os.path.join(room, "__init__.py"), "w", encoding="utf-8") as handle:
                handle.write(SHIM_SOURCE)
        except OSError:
            continue
        made.append(name)
    return made

SUITE_BASELINE_SEC = 300.0
SUITE_MODULE_CAP = 40
SUITE_RECHECK_SEC = 240.0
SUITE_SETTLE_SEC = 90.0
WARDEN_REFUSALS_MAX = 3
CONFIRM_MAX = 12
WARDEN_RELEASE_SEC = 150.0
RUNG_MIN_WALL_SEC = WARDEN_RELEASE_SEC + SUITE_SETTLE_SEC
WARDEN_QUESTION = (
    " Before changing anything, compare the failing invocation and its inputs "
    "with the baseline. Cite any supported behaviour difference and the "
    "relevant source line. If none is identified, the cause remains "
    "unresolved; inspect the observed failure without assuming the test is wrong."
)
CONFORM_MIN_WALL_SEC = 300.0
SELFREVIEW_MAX_SHARE = 0.20
SELFREVIEW_MIN_WALL_SEC = 120.0
SELFREVIEW_DIFF_SEC = 4.0
SELFREVIEW_CLOSE_SEC = 10.0
CONSULT_HISTORY_LINES = 60
CONSULT_SECTION_CHARS = 12_000
CONSULT_REPLY_CHARS = 6_000
CONSULT_ANSWER_SHORT = 400
CONSULT_ANSWER_LONG = 2_000
CONSULT_CALL_SEC = 180.0
CONSULT_MIN_WALL_SEC = 480.0
CONSULT_MODEL = "openai/gpt-5.6-luna"
CONSULT_EFFORT = "high"
CONSULT_MAX_TOKENS = 16000


SUITE_BASELINE_SEC = 300.0
SUITE_MODULE_CAP = 40
SUITE_RECHECK_SEC = 240.0
SUITE_SETTLE_SEC = 90.0
WARDEN_REFUSALS_MAX = 3
CONFIRM_MAX = 12
WARDEN_RELEASE_SEC = 150.0
RUNG_MIN_WALL_SEC = WARDEN_RELEASE_SEC + SUITE_SETTLE_SEC
WARDEN_QUESTION = (
    " Before changing anything, compare the failing invocation and its inputs "
    "with the baseline. Cite any supported behaviour difference and the "
    "relevant source line. If none is identified, the cause remains "
    "unresolved; inspect the observed failure without assuming the test is wrong."
)
CONFORM_MIN_WALL_SEC = 300.0
SELFREVIEW_MAX_SHARE = 0.20
SELFREVIEW_DIFF_SEC = 4.0
SELFREVIEW_CLOSE_SEC = 10.0
CONSULT_HISTORY_LINES = 60
CONSULT_SECTION_CHARS = 12_000
CONSULT_REPLY_CHARS = 6_000
CONSULT_ANSWER_SHORT = 400
CONSULT_ANSWER_LONG = 2_000
CONSULT_CALL_SEC = 180.0
CONSULT_MIN_WALL_SEC = 480.0
CONSULT_MODEL = "openai/gpt-5.6-luna"
CONSULT_EFFORT = "high"
CONSULT_MAX_TOKENS = 16000

# A second reader with a fresh view of the repository goes over the change once,
# at the first clean hand-in. It can read and search, not edit or run. Every
# doubt it raises has to quote the task's words and the current code, or it is
# sent back; what it anchors reaches the driver as doubts to settle, not facts.
REVIEW_SEAT = flag("RIDGES_REVIEW_SEAT")
REVIEW_TURNS = 8
REVIEW_READ_CHARS = 64_000
REVIEW_SPEND_USD = 0.12
REVIEW_WALL_SEC = 300.0
REVIEW_MIN_WALL_SEC = 480.0
REVIEW_MIN_USD = 0.14
REVIEW_DIFF_CHARS = 24_000
REVIEW_FINDINGS_MAX = 3
REVIEW_REPLY_TOKENS = 8000
REVIEW_TOOL_NAMES = ("read_file", "search_text", "find_files", "outline")
# The reader may correct the changed definition itself, on a file the change
# already touches; the task's stated check then decides whether the correction
# stays. Off, the reader only reports.
# Disabled by default; available through an explicit runtime switch.
CLOSER_SEAT = flag("RIDGES_CLOSER_SEAT", "0")
CLOSER_EDITS_MAX = 2
CLOSER_CHECK_SEC = 150.0
CLOSER_KEPT = ("\n\nA second reader corrected the change itself in %s, and the task's stated check "
               "passed on the corrected code (%s). Read the diff as it stands now before handing in; "
               "it is the answer unless you find it wrong.")
CLOSER_REVERTED = ("\n\nA second reader tried a correction in %s, but the stated check failed on it "
                   "(%s: %s), so the files were put back exactly as you left them.")
CLOSER_UNCHECKED = ("\n\nA second reader corrected the change itself in %s. The task names no check "
                    "to run on it, so verify the corrected code yourself before handing in.")
REVIEW_FIELDS = ("requirement_quote", "path", "code_quote", "failing_input", "expected_result", "check")
REVIEW_EXTRA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "report_review",
            "description": "End the review. Give at most %d anchored doubts, or an empty list when "
                           "you could not anchor one." % REVIEW_FINDINGS_MAX,
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "requirement_quote": {"type": "string", "description": "The task's own words, copied exactly."},
                                "path": {"type": "string"},
                                "code_quote": {"type": "string", "description": "The current code, copied exactly from the file."},
                                "failing_input": {"type": "string", "description": "The input or case where the code as written falls short."},
                                "expected_result": {"type": "string", "description": "What the task's words ask for in that case."},
                                "check": {"type": "string", "description": "One command or query that would settle it."},
                            },
                            "required": list(REVIEW_FIELDS),
                        },
                    }
                },
                "required": ["findings"],
            },
        },
    },
]
REVIEW_POWERS = (
    "You can read files and search them. When a doubt you can anchor has a contained fix, you may make "
    "the correction yourself with edit, on a file the change already touches, at most %d times; the "
    "task's stated check then runs on the corrected code and the correction stays only if it passes. "
    "You cannot run commands or hand anything in." % CLOSER_EDITS_MAX
    if CLOSER_SEAT else
    "You can read files and search them; you cannot edit, run commands or hand anything in.")
REVIEW_BRIEF = """You are a second reader of a change to a repository, with a fresh view of it.
%s
Read the changed definition and, as needed, what calls it or tests it.

Look for one concrete way the change falls short of what the task's own words ask: a
case where the code as written gives a result the task does not want (empty input, a
NULL, a relationship that repeats a row, a row that does not belong, the ordering or
the count or the columns asked for, transaction or laziness behaviour), or a limit the
task states that the diff crosses. The change may well be right. Do not propose style
changes and do not widen the work.

When you are done, call report_review with at most %d findings. Each must quote the
task's words exactly, quote the current code exactly, name the input that falls short
and the result the task asks for there, and give one check that would settle it. A
quote anchors a doubt; it does not prove it. Report an empty list when you cannot
anchor one, and after a correction of your own. You have %d turns and %d characters of reading.""" % (
    REVIEW_POWERS, REVIEW_FINDINGS_MAX, REVIEW_TURNS, REVIEW_READ_CHARS)
REVIEW_REQUEST = """The task:

%s

The change as it stands, as a diff against the original:

%s

What has been executed in this run so far, oldest first, with the last line each
printed (a command that passed shows what it covered, nothing more):

%s"""
REVIEW_SPLICE = ("\n\nA second reader went over the change with a fresh view of the repository "
                 "and anchored these doubts in the task's words and the current code. Each is a "
                 "doubt to settle with a check, not a finding of fact; keep code that is right:\n\n%s")
REVIEW_EMPTY = ("\n\nA second reader went over the change with a fresh view of the repository and "
                "anchored no doubt in the task's words or the current code.")
REVIEW_REJECTED = ("finding %d is not anchored: %s. Quote the task's words and the file's current "
                   "text exactly, or leave the finding out.")


# Expectations: the model pins what one sentence of the task requires a probe
# to return; the probe runs and the difference, if any, is reported row by row.
# One that fails on the code as it stands holds the hand-in, a bounded number
# of times. The sentence must be the task's own words, so every expectation is
# anchored to a sentence the task stated; what that sentence requires of the
# rows is the model's reading of it, and the tool says so.
EXPECT_TOOL = flag("RIDGES_EXPECT_TOOL")
EXPECT_PAUSES_MAX = 2
EXPECT_ROWS_MAX = 400
EXPECT_QUOTE_MIN = 12
EXPECT_PROBE_SEC = 120.0
EXPECTATIONS: list = []
# The last tree that passed the task's named check is kept as a patch, so a
# later edit that breaks it can be undone in one call.
GREEN_MEMORY = flag("RIDGES_GREEN_MEMORY")
GREEN_PATCH_CHARS = 400_000
GREEN_RESTORE_MIN_SEC = 90.0
# --- A second derivation and a pick (v52) ---
# When the first answer is in with time and money to spare, the tree is put
# back and the problem is worked a second time from a clean transcript. The
# two answers are then run against the same checks and expectations; the one
# that passes more goes out; without a demonstrated difference the first stays.
SECOND_DERIVATION = flag("RIDGES_SECOND_DERIVATION", "0")
SECOND_MIN_SEC = 700.0
SECOND_MIN_USD = 0.08
SECOND_MAX_SPENT_USD = 0.06
PICK_SEC = 300.0
PICK_CHECK_SEC = 120.0
PICK_CHECKS_MAX = 3
PICK_EXPECTATIONS_MAX = 6
# Below this much room the two answers are not compared: the first stands.
PICK_MIN_SEC = 60.0
# A third derivation settles two answers that differ but tie on every check.
THIRD_DERIVATION = flag("RIDGES_THIRD_DERIVATION", "0")
THIRD_MAX_SPENT_USD = 0.12
# --- A required expectation (v53) ---
# On a task that lists what the rows must be and configures a database, the
# hand-in waits once until at least one expectation has been checked on the
# application's data with the code as it stands.
EXPECT_REQUIRED = flag("RIDGES_EXPECT_REQUIRED")
EXPECT_REQUIRED_MAX = 1
EXPECT_REQUIRED_MIN_SEC = 240.0
EXPECT_REQUIRED_TEXT = (
    "Not handed in yet: the task lists what the rows must be, and no expectation has been "
    "checked on the code as it stands. Pin at least one with expect: quote the sentence that "
    "decides the rows, give a probe that reads the application's data (sql, or a command that "
    "runs the application's own code against it), and the rows that sentence requires of rows "
    "you have read. When a sentence gives a threshold, pin rows on both sides of it in the "
    "statement's unit. Then submit again.")
EXPECT_UNMET = ("Not handed in yet: an expectation you pinned to the task's own words fails "
                "against the code as it stands:\n%s\nFix the code, or if the expectation misread "
                "the task, pin it again from the task's words; then submit.")
EXPECT_STALE = ("Not handed in yet: %d expectation(s) you pinned failed and have not been run "
                "again since your last edit: %s. Run expect again on the code as it stands, then submit.")
GREEN_LOST = ("Not handed in yet: the tree as it stood after edit %d passed %s, and the tree as it "
              "stands fails it. restore_green puts the files back to that passing state; or fix "
              "forward and run the check again.")
PSQL_RULE = re.compile(r"^\s*-+(?:\+-+)*\s*$")
SHELL_STATUS_LINE = re.compile(r"^\[(?:exit_code=|running|still running|.*verification unresolved\])")
PROBE_NOT_RUN = re.compile(r"^not started\.|started in the background as|\[still running", re.M)
PROBE_CUT = re.compile(r"characters of [^\]]* elided\]|\[rows truncated:")
CH_ROWS_MARK = re.compile(r"^-- (?:rows|plan) \(FORMAT ([^,]+),")
LIST_ITEM = re.compile(r"[\[(]([^\[\]()]*)[\])]")
# A command probe that brings its own rows, or swaps the data access for a stand-in,
# can only agree with the reading that wrote it.
ROWS_SUPPLIED = re.compile(
    r"=\s*\[\s*[\[(]|\.\w+\s*=\s*(?:fake|stub|mock|lambda)\w*|\bdef\s+(?:fake|stub|mock)\w*\s*\(|"
    r"\bmonkeypatch\b|\bMagicMock\b|\bunittest\.mock\b|\bmock\.patch\b", re.I)
ROW_COUNT = re.compile(r"^\(\d+ rows?\)$")


def quoted_in(statement: str, quote: str) -> bool:
    """Is the quote the task's own words: a dozen characters at least, present up to whitespace and case?"""
    wanted = " ".join((quote or "").split()).casefold()
    return len(wanted) >= EXPECT_QUOTE_MIN and wanted in " ".join((statement or "").split()).casefold()


def table_rows(text: str, sql: bool = False) -> list:
    """Rows read from a probe's output.

    For sql probes: the psql table when the output holds one (any number of
    columns, zero rows included), or the ClickHouse row block after its
    "-- rows (FORMAT ...)" line (JSONEachRow objects or the statement's own
    format). For command probes: printed lists or tuples, tab-separated lines,
    or plain lines, after the status line.
    """
    lines = (text or "").splitlines()
    if sql:
        for index, line in enumerate(lines):
            mark = CH_ROWS_MARK.match(line.strip())
            if mark:
                rows = []
                for row in lines[index + 1:]:
                    body = row.strip()
                    if not body or body.startswith(("--", "[")) or body == "(no rows)":
                        break
                    if body.startswith("{") and body.endswith("}"):
                        try:
                            record = json.loads(body)
                        except ValueError:
                            rows.append([body])
                            continue
                        rows.append([value for value in record.values()] if isinstance(record, dict) else [body])
                    else:
                        rows.append([cell.strip() for cell in body.split("\t")])
                return rows
        for index, line in enumerate(lines):
            if index and PSQL_RULE.match(line) and lines[index - 1].strip() and not lines[index - 1].startswith("["):
                rows = []
                for row in lines[index + 1:]:
                    if not row.strip() or ROW_COUNT.match(row.strip()):
                        break
                    rows.append([cell.strip() for cell in row.split("|")])
                return rows
        return []
    rows = []
    for line in lines:
        text_line = line.strip()
        if not text_line or SHELL_STATUS_LINE.match(text_line) or ROW_COUNT.match(text_line):
            continue
        if text_line.startswith(("[[", "[(")) and text_line.endswith(("]]", ")]")):
            for inner in LIST_ITEM.findall(text_line):
                rows.append([cell.strip().strip("'\"") for cell in inner.split(",") if cell.strip()])
            continue
        if (text_line.startswith("[") and text_line.endswith("]")) or (text_line.startswith("(") and text_line.endswith(")")):
            rows.append([cell.strip().strip("'\"") for cell in text_line[1:-1].split(",") if cell.strip()])
            continue
        rows.append([cell.strip() for cell in text_line.split("\t")] if "\t" in text_line else [text_line])
    return rows


def cell_key(cell) -> str:
    """One cell as it compares: whitespace collapsed, numbers exact by value, booleans and nulls by meaning."""
    if cell is None:
        return ""
    if isinstance(cell, bool):
        return "true" if cell else "false"
    text = " ".join(str(cell).split())
    lowered = text.casefold()
    if lowered in ("t", "true", "yes"):
        return "true"
    if lowered in ("f", "false", "no"):
        return "false"
    if lowered in ("null", "none", "\\n"):
        return ""
    try:
        number = decimal.Decimal(text)
    except (decimal.InvalidOperation, ValueError):
        return text
    if not number.is_finite():
        return text
    normal = number.normalize()
    return "0" if normal == 0 else format(normal, "f")


def show_row(row: list) -> str:
    """One row of values as the model reads it."""
    return "[" + ", ".join(str(cell) for cell in row) + "]"


def rows_differ(returned: list, expected: list, ordered: bool) -> str:
    """'' when the rows match; otherwise the first difference, in words."""
    got = [tuple(cell_key(c) for c in row) for row in returned]
    want = [tuple(cell_key(c) for c in row) for row in expected]
    if ordered:
        for index, (a, b) in enumerate(zip(got, want, strict=False)):
            if a != b:
                return "row %d differs: returned %s, required %s" % (index + 1, show_row(returned[index]), show_row(expected[index]))
        if len(got) != len(want):
            return "%d row(s) returned, %d required" % (len(got), len(want))
        return ""
    counts: dict = {}
    for row in want:
        counts[row] = counts.get(row, 0) + 1
    for row in got:
        counts[row] = counts.get(row, 0) - 1
    missing = [row for row, n in counts.items() if n > 0 for _ in range(n)]
    extra = [row for row, n in counts.items() if n < 0 for _ in range(-n)]
    if not missing and not extra:
        return ""
    parts = []
    if missing:
        parts.append("required but not returned: " + ", ".join(show_row(list(row)) for row in missing[:3]))
    if extra:
        parts.append("returned but not required: " + ", ".join(show_row(list(row)) for row in extra[:3]))
    return "; ".join(parts) + " (%d returned, %d required)" % (len(got), len(want))


if EXPECT_TOOL:
    TOOL_SCHEMAS.append({
        "type": "function",
        "function": {
            "name": "expect",
            "description": (
                "Pin what one sentence of the task requires and check it against the real thing: "
                "quote the sentence, give one probe (sql, run as the sql tool runs it, or command, "
                "run as bash runs it) and the rows that sentence requires the probe to return. The "
                "probe runs and any difference is reported row by row. The rows you pin come from "
                "the sentence applied to rows you have read from the application's data, and the "
                "probe reads that data too: a probe on rows you invent proves nothing. Run it again "
                "after your last edit: an expectation that fails on the code as it stands holds the "
                "hand-in."),
            "parameters": {
                "type": "object",
                "properties": {
                    "requirement_quote": {"type": "string", "description": "The task's own words, copied exactly."},
                    "sql": {"type": "string", "description": "One SQL statement; its result rows are compared."},
                    "command": {"type": "string", "description": "One shell command; its output lines are the rows (tab-separated cells)."},
                    "target": {"type": "string", "description": "Which configured database, when more than one exists."},
                    "expected": {"type": "array", "description": "The rows the sentence requires, in the probe's own columns.",
                                 "items": {"type": "array", "items": {"type": ["string", "number", "boolean", "null"]}}},
                    "ordered": {"type": "boolean", "description": "Whether row order matters (default false)."},
                    "label": {"type": "string", "description": "A short name for this expectation."},
                },
                "required": ["requirement_quote", "expected"],
            },
        },
    })
if GREEN_MEMORY:
    TOOL_SCHEMAS.append({
        "type": "function",
        "function": {
            "name": "restore_green",
            "description": ("Put the files back to the last state in this run that passed the task's "
                            "named check, when a later edit broke it."),
            "parameters": {"type": "object", "properties": {
                "why": {"type": "string", "description": "Optional: what the later edit broke."}}},
        },
    })


def review_tools() -> list[dict]:
    """The tools the second reader is offered: read-only, plus the editor when it may correct the
    change.
    """
    names = set(REVIEW_TOOL_NAMES) | ({"edit"} if CLOSER_SEAT else set())
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in names] + REVIEW_EXTRA_TOOLS


def diff_paths(diff: str) -> set:
    """The files a unified diff touches."""
    return {fence_path(path) for path in re.findall(r"^diff --git a/(\S+) b/", diff or "", re.M)}


def closing_edit(kit, args: dict, touched: set, closed: list) -> str:
    """The reader's own correction: only where the change already is, only so many."""
    path = fence_path(str(args.get("path") or ""), kit.tree.root)
    if path not in touched:
        return ("edit is allowed only on a file the change already touches (%s); report anything "
                "else as a finding" % ", ".join(sorted(touched)))
    if len(closed) >= CLOSER_EDITS_MAX:
        return "no further correction in this pass; report what remains as a finding"
    result = read_only_call(kit, "edit", args)
    if result.startswith("edited "):
        closed.append(path)
    return result


def grounded_findings(statement: str, tree, findings: object) -> tuple[list, list]:
    """(kept, rejected): findings whose quotes are in the task and the file, and why not."""
    kept: list[dict] = []
    rejected: list[str] = []
    task = " ".join((statement or "").split())
    if not isinstance(findings, list):
        return kept, ["findings must be a list"]
    for number, finding in enumerate(findings[:REVIEW_FINDINGS_MAX], 1):
        if not isinstance(finding, dict):
            rejected.append(REVIEW_REJECTED % (number, "it is not an object"))
            continue
        values = {name: " ".join(str(finding.get(name) or "").split()) for name in REVIEW_FIELDS}
        missing = [name for name in REVIEW_FIELDS if not values[name]]
        if missing:
            rejected.append(REVIEW_REJECTED % (number, "it leaves out " + ", ".join(missing)))
            continue
        if len(values["requirement_quote"]) < CASE_QUOTE_MIN or values["requirement_quote"] not in task:
            rejected.append(REVIEW_REJECTED % (number, "the task does not say %r" % values["requirement_quote"][:120]))
            continue
        try:
            source = " ".join(tree.read(values["path"]).split())
        except Exception:
            rejected.append(REVIEW_REJECTED % (number, "no such file in this checkout: %s" % values["path"]))
            continue
        if len(values["code_quote"]) < CASE_QUOTE_MIN or values["code_quote"] not in source:
            rejected.append(REVIEW_REJECTED % (number, "%s does not contain %r" % (values["path"], values["code_quote"][:120])))
            continue
        kept.append(values)
    return kept, rejected


def review_findings_text(findings: list) -> str:
    """The second reader's anchored doubts as the driver reads them, each quoting the task and the
    code.
    """
    parts = []
    for number, finding in enumerate(findings, 1):
        parts.append("%d. The task says: \"%s\"\n   In %s: \"%s\"\n   Falls short for: %s\n"
                     "   The task asks for: %s\n   A check that settles it: %s"
                     % (number, finding["requirement_quote"][:300], finding["path"], finding["code_quote"][:300],
                        finding["failing_input"][:300], finding["expected_result"][:300], finding["check"][:300]))
    return "\n\n".join(parts)


def spoken_turn(reply: dict, calls: list) -> dict:
    """The assistant's turn as the transcript keeps it: its text, and its calls if any."""
    turn = {"role": "assistant", "content": str(reply.get("content") or "")}
    if calls:
        turn["tool_calls"] = recorded_calls(calls)
    return turn


def call_arguments(call: object) -> tuple[str, dict, str]:
    """A tool call's name and arguments, or the reason they could not be read."""
    function = call.get("function") if isinstance(call, dict) else None
    function = function if isinstance(function, dict) else {}
    name = str(function.get("name") or "")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except Exception as error:
        return name, {}, "could not read the arguments: %s" % error
    if not isinstance(arguments, dict):
        return name, {}, "could not read the arguments: they were not an object"
    return name, arguments, ""


def read_only_call(kit, name: str, args: dict) -> str:
    """One of the reader's tools, run; a fault comes back as text, never as an exception."""
    try:
        return str(kit.run(name, args))
    except (ToolFault, Finished) as fault:
        return "error: %s" % fault
    except Exception as error:
        return "error: %s: %s" % (type(error).__name__, error)


def run_review(statement: str, tree, pool, allowance, warden, beacon: "Beacon",
               seed, diff: str, ran: str) -> tuple[str, list, list]:
    """A second reader's bounded pass over the change: (how it stopped, anchored findings,
    files it corrected itself)."""
    spent_at_entry, calls_at_entry = allowance.spent, allowance.calls
    ceiling = spent_at_entry + REVIEW_SPEND_USD
    deadline = time.monotonic() + REVIEW_WALL_SEC
    kit = Kit(tree, pool, allowance, warden, label="REVIEW")
    seat = Seat(allowance, models=[REVIEW_MODEL], patient=False, impatient_sec=60.0,
                reply_ceiling=REVIEW_REPLY_TOKENS)
    seat.seed = seed
    messages = [{"role": "system", "content": REVIEW_BRIEF},
                {"role": "user", "content": REVIEW_REQUEST % (
                    statement.strip(), clip(diff, REVIEW_DIFF_CHARS, "diff"), ran or "nothing yet")}]
    offered = set(REVIEW_TOOL_NAMES) | {"report_review"} | ({"edit"} if CLOSER_SEAT else set())
    touched = diff_paths(diff)
    closed: list = []
    findings: list = []
    read, stop, turns = 0, "turns", 0
    used_call_ids: set = set()
    try:
        while turns < REVIEW_TURNS:
            if read >= REVIEW_READ_CHARS:
                stop = "budget"
                break
            if allowance.spent >= ceiling or allowance.money_left() <= 0:
                stop = "spend"
                break
            if time.monotonic() >= deadline:
                stop = "time"
                break
            turns += 1
            reply = seat.ask(messages, review_tools())
            calls = usable_calls(reply.get("tool_calls"), used_call_ids)
            messages.append(spoken_turn(reply, calls))
            if not calls:
                stop = "silent"
                break
            done = strayed = False
            for index, call in enumerate(calls):
                name, args, unreadable = call_arguments(call)
                if unreadable:
                    result = unreadable
                else:
                    if name not in offered:
                        result = "%s is not offered to this reader; it can read and search only" % (name or "that")
                        strayed = True
                    elif name == "edit":
                        result = closing_edit(kit, args, touched, closed)
                    elif name == "report_review":
                        kept, rejected = grounded_findings(statement, tree, args.get("findings"))
                        if rejected:
                            result = "\n".join(rejected)
                        else:
                            findings, done = kept, True
                            result = "review recorded, %d finding(s)" % len(kept)
                    else:
                        result = read_only_call(kit, name, args)
                served = clip(str(result), READ_OUTPUT_CAP)
                read += len(served)
                messages.append({"role": "tool", "tool_call_id": call_ident(call, index), "content": served})
            if done:
                stop = "done"
                break
            if strayed:
                stop = "strayed"
                break
            messages.append({"role": "user", "content": "Review turns left: %d; read characters left: %d."
                             % (REVIEW_TURNS - turns, max(0, REVIEW_READ_CHARS - read))})
    except Exception as error:
        stop = "error"
        say("[%s] gave up: %s: %s" % (beacon.slug, type(error).__name__, str(error)[:200]))
    beacon.calls = allowance.calls - calls_at_entry
    beacon.usd = allowance.spent - spent_at_entry
    beacon.fired("stopped=%s turns=%d read=%dc findings=%d corrected=%d"
                 % (stop, turns, read, len(findings), len(closed)))
    beacon.bill()
    return stop, findings, closed


def length_class(text: str) -> str:
    """How long an answer is, in words rather than numbers, for a sentence about it."""
    size = len(text or "")
    if size < CONSULT_ANSWER_SHORT:
        return "a short"
    return "a middling" if size < CONSULT_ANSWER_LONG else "a long"

def statement_of(warden) -> str:
    """The task statement this run is working to, or empty when there is none."""
    return (warden.statement if warden is not None else "") or ""

CONSULT_BRIEF = """Review only the supplied task, patch and observed command results. You cannot edit files or run tools. Identify a concrete mismatch with an explicit task requirement or an assumption visible in the changed code. Cite the relevant requirement or changed line for each concern; do not supply a predefined checklist or infer requirements from the task's topic.

Judge completion against what the task asked for. Treat supported but untested concerns as risks worth ruling out rather than as things the task demands. Respect the task's execution restrictions. If a permitted check could resolve a supported concern, describe that check and what observation would distinguish the outcomes. Do not ask to start services or execute database work that the task prohibits. Distinguish an observed failure from an untested possibility. If the supplied evidence supports no concrete concern, say so briefly."""
CONSULT_REQUEST = """What the work was asked to do:

%s

What has been changed so far, as a patch:

%s

What has been run, oldest first -- the command, then the last line it printed:

%s

Name the properties that could apply to this change, and for each the command that settles it against the real thing. Nothing else."""
CONSULT_SPLICE = """

Another reader was shown what you changed and what you have run, and was asked that same question. It cannot change anything, it has no tools, and it did not see this note. What it said:

---
%s
---

It may be wrong, and you hold the pen. Where it names a check you have not made and can make from here, make it before you submit."""
STANDIN_MIN_WALL_SEC = CONFORM_MIN_WALL_SEC + SUITE_SETTLE_SEC
LEDGER_BULLET = re.compile(r"^[-*+][ \t]+(.*)$")
LEDGER_FENCE = re.compile(r"^[ \t]*(?:```|~~~)")
LEDGER_MIN = 3
LEDGER_MAX_ASKS = 2
FENCE_MAX_REFUSALS = 2
HANDIN_PAUSES = (("requirement_state", "requirement_note"),
                 ("selfreview_state", "selfreview_note"),
                 ("conform_state", "conform_note"))
HANDIN_PAUSES_MAX = 1
# A command the model started and has not read holds the hand-in once while the
# run still has time and money to act on it.
JOB_HANDIN = flag("RIDGES_JOB_HANDIN")
HANDIN_HOLD_MIN_SEC = 60.0
# An empty patch is no answer. While the patch agent_main would hand in is
# empty, a submit is refused and the run goes on. The patch is worked out the
# way agent_main does it, so the check and the hand-in cannot disagree.
EMPTY_HANDIN_REFUSAL = flag("RIDGES_EMPTY_HANDIN_REFUSAL")
EMPTY_ANSWER_DIFF_SEC = 10.0
EMPTY_ANSWER_MIN_CLOCK_SEC = 30.0
EMPTY_ANSWER_NO_CHANGE = ("the diff against the commit this run started from holds no "
                          "change (bytecode, ignored files and new vendor/, dist/, "
                          "target/, node_modules/ or go.sum files stay out of it unless "
                          "the statement names them)")
EMPTY_ANSWER_ALL_LEFT_OUT = "every change in the repository is left out of the answer: %s"


class QuietBeacon(Beacon):
    """A Beacon that keeps its lines rather than printing them."""

    def __init__(self, slug: str) -> None:
        """Open a beacon that keeps its lines instead of printing them."""
        super().__init__(slug)
        self.lines: list[str] = []
        self.fired_details: list[str] = []

    def skipped(self, reason: str) -> None:
        """Keep the reason this slot did nothing, for a caller that will decide whether to say it.
        """
        self.lines.append("skipped: " + reason)

    def fired(self, detail: str) -> None:
        """Keep what this slot did, and its detail, without printing either."""
        self.lines.append(detail[:400])
        self.fired_details.append(detail[:400])

    def artefact(self, when: str, blob: str) -> str:
        """Keep the digest and size of a text about to change, and return the digest."""
        digest = hashlib.sha256((blob or "").encode("utf-8", "replace")).hexdigest()[:8]
        self.lines.append("%s %s %dB" % (when, digest, len(blob or "")))
        return digest

    def outcome(self, before_digest: str, after: str) -> None:
        """Keep the digest and size afterwards, and whether anything changed."""
        digest = hashlib.sha256((after or "").encode("utf-8", "replace")).hexdigest()[:8]
        self.lines.append("after %s %dB changed=%s" % (digest, len(after or ""),
                                                      "yes" if digest != before_digest else "no"))


def empty_answer(tree, statement: str, allowance) -> str:
    """Why the patch handed in now would be empty, or "".

    The patch is the tree's diff followed by the by-product and envelope trims,
    in agent_main's order. Anything unclear, a failed or late diff included, is
    "", so a hand-in is never held up by the check itself.
    """
    try:
        if not EMPTY_HANDIN_REFUSAL or tree is None or not getattr(tree, "base", ""):
            return ""
        left = allowance.clock_left() if allowance is not None else 0.0
        if left < EMPTY_ANSWER_MIN_CLOCK_SEC:
            return ""
        failed: list = []
        patch = tree.diff(min(EMPTY_ANSWER_DIFF_SEC, left - 20.0), failed=failed)
        if failed:
            return ""
        if not patch.strip():
            return EMPTY_ANSWER_NO_CHANGE
        root = getattr(tree, "root", "")
        patch = byproduct_trim(patch, QuietBeacon("byproduct"), statement, root)
        if not patch.strip():
            return EMPTY_ANSWER_NO_CHANGE
        # The envelope never empties an answer: what it would leave out goes out whole.
        patch = envelope_or_whole(patch, QuietBeacon("envelope"), statement, root)
        return "" if patch.strip() else EMPTY_ANSWER_NO_CHANGE
    except Exception:
        return ""


def empty_answer_note(kit) -> str:
    """The reply to a submit while the answer would be an empty patch, or ""."""
    try:
        warden = getattr(kit, "warden", None)
        statement = statement_of(warden) if warden is not None else ""
    except Exception:
        statement = ""
    why = empty_answer(getattr(kit, "tree", None), statement, getattr(kit, "allowance", None))
    if not why:
        return ""
    try:
        hint = kit.open_requirement_hint()
    except Exception:
        hint = ""
    return ("Not handed in: the answer would be an empty patch, because %s.%s "
            "Call submit again once the patch holds your change." % (why, hint))

def bullet_runs(text: str) -> list[list[str]]:
    """The bulleted lists in a text, each as its own run of items, with fenced blocks left out.

    A list inside a fenced block is example text rather than a requirement, and an indented
    line continues the item above it.
    """
    runs: list[list[str]] = []
    current: list[str] = []
    fenced = False
    for line in (text or "").splitlines():
        if LEDGER_FENCE.match(line):
            if current and not fenced and not line[:1].isspace():
                runs.append(current)
                current = []
            fenced = not fenced
            continue
        if fenced:
            continue
        bullet = LEDGER_BULLET.match(line)
        if bullet:
            current.append(bullet.group(1).strip())
        elif line.strip() and current:
            if line.startswith((" ", "\t")):
                current[-1] += " " + line.strip()
            else:
                runs.append(current)
                current = []
    if current:
        runs.append(current)
    return [[item for item in run if item] for run in runs]

def stated_requirements(text: str) -> list[str]:
    """Every bulleted requirement the statement lists, in order."""
    return [item for run in bullet_runs(text) for item in run]

LEDGER_GUARD = re.compile(
    r"\b(preserve|preserves|keep|keeps|retain|retains|remain|remains|"
    r"continue|continues|unchanged|intact|still|do not|don't|must not|"
    r"never|leave|leaves|untouched)\b", re.I)

def wants_an_edit(item: str) -> bool:
    """Does this requirement ask for a change, rather than asking that something stay as it is?
    """
    return not LEDGER_GUARD.search(item or "")

def read_back(items: list[str]) -> str:
    """The request to point each extracted requirement at the code that satisfies it before handing
    in.
    """
    lines = "\n".join("  %d. %s" % (n + 1, item) for n, item in enumerate(items))
    return ("Before this goes in, compare these extracted clauses with the "
            "current task, which remains authoritative:\n%s\n"
            "For each, identify supporting code or a recorded check, or state "
            "what remains unverified. Existing code may already satisfy a "
            "requirement without a diff. Unchanged code alone does not prove "
            "preservation when its callers or dependencies changed. Do not "
            "invent an unmet requirement from a missing edit. Hand in again "
            "after this review." % lines)

def importable(path: str) -> bool:
    """Is this path a Python source file this run could import?"""
    return any(path.endswith(suffix) for suffix in importlib.machinery.SOURCE_SUFFIXES)

NESTED_SOURCE = "src"

def package_roots(root: str) -> list[str]:
    """Where this project's packages live: the checkout, and a nested source directory when it has
    one.
    """
    nested = os.path.join(root, NESTED_SOURCE)
    return [root, nested] if os.path.isdir(nested) else [root]

EXCLUDING_OPTION = frozenset(("--exclude", "--extend-exclude", "--force-exclude"))

CHECK_TOOLS = frozenset((
    "ruff", "flake8", "pylint", "pyflakes", "mypy", "pyright", "black",
    "isort", "eslint", "rubocop", "phpcs", "gofmt", "clang-format"))
CHECK_SUBCOMMANDS = frozenset(("check", "format", "lint"))

def check_head(argv: list) -> tuple:
    """Split a check command into its tool name and the words after it."""
    rest = strip_assignments(argv)
    if not rest:
        return "", []
    tool = os.path.basename(rest[0])
    rest = rest[1:]
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", tool) and rest[:1] == ["-m"]:
        tool = os.path.basename(rest[1]) if len(rest) > 1 else ""
        rest = rest[2:]
    if rest[:1] and rest[0] in CHECK_SUBCOMMANDS:
        rest = rest[1:]
    return tool, rest

def check_command(argv: list) -> bool:
    """Is this a linter or type checker, whose arguments are paths to files?

    A test runner is not one: its arguments are test labels, not files.
    """
    rest = strip_assignments(argv)
    tool, _ = check_head(argv)
    if tool not in CHECK_TOOLS:
        return False
    if tool == "ruff":
        after = [word for word in rest[1:] if not ASSIGNMENT.match(word)]
        if rest[:1] and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?",
                                     os.path.basename(rest[0])):
            after = rest[3:]
        return bool(after) and after[0] in CHECK_SUBCOMMANDS
    return True

def check_commands(statement: str) -> list:
    """The commands the statement names that report findings rather than running the project's
    tests.
    """
    return [argv for argv in command_lines(statement) if check_command(argv)]

def check_targets(argv: list) -> list:
    """The paths a check command names. Empty for anything that is not one."""
    if not check_command(argv):
        return []
    targets, skip = [], False
    for token in check_head(argv)[1]:
        if skip:
            skip = False
        elif token.startswith("-"):
            skip = token in EXCLUDING_OPTION
        else:
            targets.append(token)
    return targets

# The linters a statement names are run again when the answer is handed in,
# exactly as the statement wrote them. Only a form that reports can run there:
# a fixing form would change the answer where the model cannot see it, and a
# watching form never returns.
NAMED_CHECKS = flag("RIDGES_NAMED_CHECKS")
NAMED_CHECK_SEC = 90.0
NAMED_CHECK_LIMIT = 4
NAMED_CHECK_QUOTED = 8
CHECK_OPTIONS_NOT_RUN = frozenset((
    "--fix", "--fix-only", "--unsafe-fixes", "--write", "-w", "-i", "--in-place",
    "-a", "-A", "--autocorrect", "--auto-correct", "--autocorrect-all",
    "--auto-correct-all", "--apply", "--watch"))
# Formatters rewrite files unless an option says to report instead.
REPORTING_OPTIONS = {"black": ("--check", "--diff"),
                     "isort": ("--check", "--check-only", "-c", "--diff"),
                     "gofmt": ("-l", "-d"),
                     "clang-format": ("--dry-run", "-n")}

def reports_only(argv: list) -> bool:
    """Whether a check command only reports: it rewrites no file and it ends."""
    rest = strip_assignments(argv)
    if not rest:
        return False
    tool, words = os.path.basename(rest[0]), rest[1:]
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", tool) and words[:1] == ["-m"]:
        tool, words = (os.path.basename(words[1]) if len(words) > 1 else ""), words[2:]
    if any(word.split("=", 1)[0] in CHECK_OPTIONS_NOT_RUN for word in words):
        return False
    if tool == "ruff" and words[:1] == ["format"]:
        return any(word in ("--check", "--diff") for word in words)
    wanted = REPORTING_OPTIONS.get(tool)
    return not wanted or any(word in wanted for word in words)

def named_checks(statement: str) -> list:
    """The reporting linter commands the statement spells out, in its order, once each."""
    found: list = []
    for argv in check_commands(statement):
        if reports_only(argv):
            line = shlex.join(argv)
            if line not in found:
                found.append(line)
    return found

# "path:line[:col]: message", the form most checkers print, and the form that
# names the rule first and points at the place on the next line or two.
FINDING_PLACED = re.compile(r"^(\S+?):\d+(?::\d+)?:\s*(\S.*)$")
FINDING_RULE_FIRST = re.compile(r"^[A-Z]{1,4}\d{1,5}\b")
FINDING_POINTER = re.compile(r"^\s*-->\s+(\S+?):\d+(?::\d+)?\s*$")

def finding_keys(output: str) -> list:
    """What a checker reported, each as "path: message", with no line number.

    An edit moves lines, so a finding the unchanged tree already had keeps its
    key after the change; a message that names a line has the number blanked.
    """
    lines = (output or "").splitlines()
    keys = []
    for index, line in enumerate(lines):
        placed = FINDING_PLACED.match(line)
        if placed:
            keys.append("%s: %s" % (placed.group(1),
                                    re.sub(r"\bline \d+\b", "line #", placed.group(2).strip())))
        elif FINDING_RULE_FIRST.match(line):
            for following in lines[index + 1:index + 3]:
                pointer = FINDING_POINTER.match(following)
                if pointer:
                    keys.append("%s: %s" % (pointer.group(1), line.strip()))
                    break
    return keys

def new_findings(now: str, before: str) -> list:
    """What `now` reports that `before` does not, counted, in `now`'s order.

    Output in which neither reading has a recognizable finding is compared
    line by line, with digits blanked, so a count in a summary does not differ.
    """
    current, earlier = finding_keys(now), finding_keys(before)
    if not current and not earlier:
        current, earlier = ([re.sub(r"\d+", "#", " ".join(line.split()))
                             for line in (text or "").splitlines() if line.strip()]
                            for text in (now, before))
    remaining = collections.Counter(earlier)
    fresh = []
    for key in current:
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            fresh.append(key)
    return fresh

def fresh_lines(now: str, before: str) -> list:
    """The lines of `now` that carry a finding `before` does not have, as printed."""
    remaining = collections.Counter(new_findings(now, before))
    lines = []
    for line in (now or "").splitlines():
        keys = finding_keys(line)
        if keys and remaining[keys[0]] > 0:
            remaining[keys[0]] -= 1
            lines.append(line.rstrip())
    return lines

# Go reports an unused import or variable, and a type error, only when it
# compiles; gofmt reads syntax. An edited package is compiled together with its
# tests and nothing is run, with no module download and no toolchain switch.
GO_COMPILE = flag("RIDGES_GO_COMPILE")
GO_COMPILE_EDIT_SEC = 60.0
GO_COMPILE_HANDIN_SEC = 150.0
GO_COMPILE_QUOTED = 10
GO_PACKAGE_LIMIT = 6
GO_DIAGNOSTIC = re.compile(r"^\S+\.go:\d+:\d+: ", re.M)
GO_INSTALL_DIRS = ("/usr/local/go/bin",)

def go_tool(name: str) -> str:
    """A Go tool on PATH, else where the standard install puts it, else ""."""
    found = shutil.which(name)
    if found:
        return found
    for folder in GO_INSTALL_DIRS:
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""

def go_packages(root: str, paths: list) -> list:
    """(module folder, package folder within it) for each Go package the paths are in."""
    base = os.path.realpath(root)
    found: list = []
    for path in paths:
        if not path.endswith(".go"):
            continue
        folder = os.path.realpath(os.path.dirname(os.path.join(base, path)))
        if not os.path.isdir(folder) or not any(name.endswith(".go") for name in os.listdir(folder)):
            continue
        module = folder
        while not os.path.isfile(os.path.join(module, "go.mod")):
            parent = os.path.dirname(module)
            if module == base or parent == module:
                module = ""
                break
            module = parent
        if not module or os.path.commonpath([module, base]) != base:
            continue
        entry = (os.path.relpath(module, base), os.path.relpath(folder, module))
        if entry not in found:
            found.append(entry)
    return found

def go_compile_output(where: str, module: str, package: str, budget: float) -> tuple:
    """Compile one package and its tests in `where`, running no test.

    Returns ("clean" | "errors" | "unverified" | "timeout", output). Only the
    compiler's file:line:col diagnostics make "errors"; a module that cannot be
    fetched, or a toolchain that cannot run, is "unverified".
    """
    go = go_tool("go")
    if not go:
        return "unverified", "no Go toolchain here"
    target = "./" + package if package != "." else "."
    command = shlex.join([go, "test", "-c", "-o", os.devnull, "-vet=off", target])
    try:
        job = Shell(command, os.path.join(where, module), pack_venv=False, hard_timeout=budget + 5.0,
                    env_extra={"GOPROXY": "off", "GOTOOLCHAIN": "local"})
    except OSError as error:
        return "unverified", str(error)
    try:
        done, out = job.wait(budget)
        code = job.process.poll()
    finally:
        job.stop()
    if not done:
        return "timeout", ""
    if code == 0:
        return "clean", out
    return ("errors" if GO_DIAGNOSTIC.search(out) else "unverified"), out

# Files that are machinery rather than the subject of a change.
RUNNER_SCRIPT_NAMES = frozenset((
    "manage.py", "setup.py", "conftest.py", "noxfile.py", "tasks.py",
    "fabfile.py", "wsgi.py", "asgi.py"))
CONFIG_SUFFIXES = (".cfg", ".ini", ".toml", ".yaml", ".yml", ".json", ".txt",
                   ".lock", ".md")

def production_named(statement: str, root: str) -> list:
    """Source files the statement names outright and the checkout really has.

    A linter invocation is one way a task points at a file; naming it in the
    text is another, and plenty of tasks do only the latter. Test files, runner
    scripts and configuration are excluded: they are how a repository is driven,
    not what a change is about. Only backticked paths count, so prose that
    merely mentions a module is not mistaken for a target.
    """
    found = set()
    for raw in FENCE_PATH.findall(statement or ""):
        token = fence_path(raw, root)
        if not token or CHECK_TEST_PATH.search(token):
            continue
        if os.path.basename(token) in RUNNER_SCRIPT_NAMES:
            continue
        if token.endswith(CONFIG_SUFFIXES):
            continue
        where = os.path.join(root, token)
        if inside(where, root) and os.path.isfile(where):
            found.add(token)
    return sorted(found)

def declared_file(statement: str, root: str) -> str | None:
    """The one production file this task is about, or None if that is not clear.

    A localization hint only. Whether the change is confined to it is a separate
    question, and one the statement has to answer (see statement_bounded).
    """
    found = set()
    for argv in check_commands(statement):
        for token in check_targets(argv):
            if not RIDGES_SCOPE_FOLLOWS_STATEMENT and not token.endswith(".py"):
                continue
            if CHECK_TEST_PATH.search(token):
                continue
            candidate = os.path.join(root, token)
            if inside(candidate, root) and os.path.isfile(candidate):
                found.add(token)
    if found:
        return found.pop() if len(found) == 1 else None
    named = production_named(statement, root)
    return named[0] if len(named) == 1 else None

def named_literals(statement: str) -> list[str]:
    """The short backticked names the statement mentions, which are identifiers rather than
    commands.
    """
    out = []
    for raw in dict.fromkeys(re.findall(r"`([^`\n]{1,40})`", statement or "")):
        if (" " in raw or "/" in raw or "\\" in raw
                or raw.startswith(("ruff", "--", "# ", "PLR"))):
            continue
        if "<" in raw or ">" in raw:
            continue
        if not re.search(r"[A-Za-z0-9]", raw):
            continue
        if (raw not in ("False", "None", "True")
                and all(part.isidentifier() for part in raw.split("."))):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*[\[(=]", raw):
            continue
        out.append(raw)
    return out

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+", re.M)
ROW_LINE_RE = re.compile(r"^line (\d+):")

def touched_lines(diff: str) -> list[tuple[int, int]]:
    """The line spans a diff changes, as (first, last) pairs read from its hunk headers."""
    spans = []
    for start, length in HUNK_RE.findall(diff or ""):
        first = int(start)
        spans.append((first, first + (int(length) if length else 1) - 1))
    return spans

MUTATION_SCAN_MAX_CHARS = 1_000_000

TEST_MODULE_RE = re.compile(r"^(?:test_.*|.*_test)\.py$")

def _module_level_imports(path: str) -> set[str]:
    """The top-level module names a Python file imports at module level."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read(MUTATION_SCAN_MAX_CHARS)
        tree = ast.parse(source)
    except (MemoryError, OSError, RecursionError, SyntaxError, ValueError):
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module.split(".")[0])
    return names

def _provides(base: str, name: str) -> bool:
    """Does this directory provide that module, as a file, a package or a directory of sources?
    """
    if os.path.isfile(os.path.join(base, name + ".py")):
        return True
    here = os.path.join(base, name)
    if os.path.isdir(here):
        if os.path.isfile(os.path.join(here, "__init__.py")):
            return True
        try:
            return any(entry.endswith(".py") for entry in os.listdir(here))
        except OSError:
            return False
    try:
        entries = os.listdir(base)
    except OSError:
        return False
    return any(_is_extension_of(base, entry, name) for entry in entries)

def _is_extension_of(base: str, entry: str, name: str) -> bool:
    """Is this directory entry a compiled extension for that module name?"""
    for suffix in (".so", ".pyd", ".dylib"):
        if not entry.endswith(suffix) or not entry.startswith(name):
            continue
        tag = entry[len(name):len(entry) - len(suffix)]
        if tag and not (tag.startswith(".") and any(c.isdigit() for c in tag[1:])
                        and "." not in tag[1:]):
            continue
        return os.path.isfile(os.path.join(base, entry))
    return False

def _absent(root: str, names: set[str]) -> set[str]:
    """Which of these module names the project does not provide and cannot be imported here."""
    missing = set()
    roots = package_roots(root)
    for name in names:
        if not name or name in sys.builtin_module_names:
            continue
        if any(_provides(base, name) for base in roots):
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except Exception:
            continue
        missing.add(name)
    return missing

def readable_scope(root: str, scope: list[str]) -> tuple[list[str], str]:
    """The part of a test scope this run can actually read, and a sentence on what it had to leave
    out.

    A file that will not parse, one that imports something absent, and one too large to scan
    are each dropped for a stated reason, so the scope that remains is one the run can
    reason about.
    """
    if not scope:
        return scope, ""
    kept: list[str] = []
    dropped: dict[str, tuple[set[str], set[str]]] = {}
    blocked: dict[str, set[str]] = {}
    truncated: dict[str, int] = {}
    for entry in scope:
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            kept.append(entry)
            continue
        conftests = []
        here = full
        while inside(here, root):
            conftest = os.path.join(here, "conftest.py")
            if os.path.exists(conftest):
                conftests.append(conftest)
            if os.path.normpath(here) == os.path.normpath(root):
                break
            here = os.path.dirname(here)
        if conftests:
            gone = _absent(root, set().union(*map(_module_level_imports, conftests)))
            if gone:
                blocked[entry] = gone
                kept.append(entry)
                continue
        modules = []
        for here, directories, names in os.walk(full):
            directories[:] = [name for name in directories
                              if name != "__pycache__" and not name.startswith(".")]
            modules.extend(os.path.relpath(os.path.join(here, name), full)
                           for name in names
                           if TEST_MODULE_RE.match(name)
                           and os.path.isfile(os.path.join(here, name)))
        modules.sort()
        if not modules:
            kept.append(entry)
            continue
        good, bad = [], {}
        for name in modules:
            path = os.path.join(full, name)
            imports = _module_level_imports(path)
            here = os.path.dirname(path)
            while os.path.normpath(here) != os.path.normpath(full):
                conftest = os.path.join(here, "conftest.py")
                if os.path.exists(conftest):
                    imports.update(_module_level_imports(conftest))
                here = os.path.dirname(here)
            gone = _absent(root, imports)
            (bad.__setitem__(name, gone) if gone else good.append(name))
        if not bad:
            kept.append(entry)
            continue
        if not good:
            blocked[entry] = set().union(*bad.values())
            kept.append(entry)
            continue
        dropped[entry] = (set(bad), set().union(*bad.values()))
        good.sort(key=lambda name: -os.path.getsize(os.path.join(full, name)))
        if len(good) > SUITE_MODULE_CAP:
            truncated[entry] = len(good) - SUITE_MODULE_CAP
        kept.extend(os.path.join(entry, name) for name in good[:SUITE_MODULE_CAP])
    parts: list[str] = []
    if blocked:
        parts.append("cannot be read: %s" % "; ".join(
            "%s needs %s" % (k, ", ".join(sorted(v))) for k, v in sorted(blocked.items())))
    if truncated:
        parts.append("held back %d readable module(s) at the %d cap"
                     % (sum(truncated.values()), SUITE_MODULE_CAP))
    note = ""
    if dropped:
        count = sum(len(m) for m, _ in dropped.values())
        needs = set().union(*(n for _, n in dropped.values()))
        parts.append("dropped %d module(s) the image cannot collect, needing %s" % (
            count, ", ".join(sorted(needs))))
    note = "; ".join(parts)
    return kept, note

def declared_trace(statement: str, root: str) -> str:
    """A sentence on which of the statement's named files were kept in scope and which were
    dropped.
    """
    argvs = check_commands(statement)
    seen = dropped = existed = 0
    for argv in argvs:
        for token in check_targets(argv):
            if not token.endswith(".py"):
                continue
            seen += 1
            if CHECK_TEST_PATH.search(token):
                dropped += 1
                continue
            candidate = os.path.join(root, token)
            if inside(candidate, root) and os.path.isfile(candidate):
                existed += 1
    return ("%d check command(s), %d path(s) on them, %d dropped as tests, "
            "%d that exist here" % (len(argvs), seen, dropped, existed))

TEST_RUNNER_HEADS = (("pytest",), ("python", "-m", "pytest"),
                     ("python3", "-m", "pytest"))
RUNNER_CHAIN = re.compile(r";|&&|\|\||\||&")
RUNNER_VALUE_OPTIONS = frozenset((
    "-k", "-m", "-p", "-o", "-c", "-n", "-W", "-r",
    "--tb", "--rootdir", "--basetemp", "--deselect", "--ignore",
    "--ignore-glob", "--maxfail", "--junitxml", "--junit-xml",
    "--override-ini", "--import-mode", "--confcutdir", "--log-level",
    "--log-cli-level", "--log-file", "--capture", "--assert", "--dist",
    "--numprocesses", "--durations", "--color",
    "--cov", "--cov-report", "--cov-config", "--cov-fail-under",
    "--ds", "--dc", "--settings", "--reruns", "--reruns-delay",
    "--timeout", "--timeout-method", "--html", "--parallel",
    "--tag", "--exclude-tag"))
RUNNER_FLAG_OPTIONS = frozenset((
    "-q", "-v", "-vv", "-x", "-s", "-l", "-h",
    "--quiet", "--verbose", "--exitfirst", "--no-header", "--no-summary",
    "--collect-only", "--co", "--last-failed", "--lf", "--failed-first",
    "--ff", "--new-first", "--nf", "--pdb", "--trace", "--full-trace",
    "--showlocals", "--disable-warnings", "--help", "--version",
    "--strict-markers", "--strict-config", "--no-cov", "--noconftest",
    "--continue-on-collection-errors", "--keepdb", "--failfast", "--buffer",
    "--locals", "--no-input", "--noinput", "--create-db", "--reuse-db"))
RUNNER_SHORT_FLAGS = frozenset("qvxslh")
RUNNER_UNMODELLED = re.compile(r"[$`(){}<>*?\[\]!~]")

def stated_test_scope(statement: str, root: str) -> list[str]:
    """The paths the statement's own test command runs, as this run reads that command."""
    found: list[str] = []
    for block in statement_command_blocks(statement):
        for line in logical_lines(block or ""):
            chain = []
            for piece in RUNNER_CHAIN.split(line):
                if not piece.strip():
                    continue
                try:
                    chain.append(shlex.split(piece, comments=True))
                except ValueError:
                    chain = None
                    break
            if not chain:
                continue
            if any(argv and argv[0] == "cd" for argv in chain):
                continue
            for argv in chain:
                found.extend(_stated_targets(argv, root))
    return sorted(set(found))

def _stated_path(token: str, root: str) -> tuple[str, str]:
    """One token of a test command as a path, or a refusal saying why it is not one."""
    token = token.split("::", 1)[0]
    if not token:
        return ("refuse", "a selector with no file in front of it")
    if RUNNER_UNMODELLED.search(token):
        return ("refuse", "shell this does not model: %s" % token[:40])
    here = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(here, token))
    if not os.path.exists(candidate):
        return ("none", "")
    if candidate == here:
        return ("refuse", "the project itself")
    if not inside(candidate, root):
        return ("refuse", "a path outside the project: %s" % token[:40])
    return ("target", os.path.normpath(os.path.relpath(candidate, here)))

def _stated_targets(argv: list, root: str) -> list[str]:
    """The file arguments of a recognised test runner command, with its options skipped."""
    for head in TEST_RUNNER_HEADS:
        if len(argv) > len(head) and tuple(argv[:len(head)]) == head:
            rest = argv[len(head):]
            break
    else:
        return []
    out, takes_value, positional_only = [], False, False
    for index, token in enumerate(rest):
        if takes_value:
            takes_value = False
            continue
        if not positional_only and token == "--":
            positional_only = True
            continue
        if not positional_only and token.startswith("-") and token != "-":
            if "=" in token or token in RUNNER_FLAG_OPTIONS:
                continue
            if token in RUNNER_VALUE_OPTIONS:
                takes_value = True
                continue
            if (not token.startswith("--")
                    and set(token[1:]) <= RUNNER_SHORT_FLAGS):
                continue
            nxt = rest[index + 1] if index + 1 < len(rest) else ""
            if nxt and not nxt.startswith("-"):
                if _stated_path(nxt, root)[0] != "none":
                    return []
                takes_value = True
            continue
        kind, value = _stated_path(token, root)
        if kind == "refuse":
            return []
        if kind == "target":
            out.append(value)
    return [] if takes_value else out

def suite_scope(root: str, declared: str | None, specific: bool = False) -> list[str]:
    """The test files that cover the named file, by the project's own naming conventions."""
    if not declared:
        return []
    parts = declared.split("/")
    if parts and parts[0] == NESTED_SOURCE:
        parts = parts[1:]
    inner = parts[1:-1]
    base = os.path.splitext(os.path.basename(declared))[0]
    names = [base]
    if base.startswith("_"):
        names.append(base.strip("_") or base)
    names.extend(reversed(inner))
    ancestors, here = [], os.path.dirname(declared)
    while here:
        ancestors.append(here)
        here = os.path.dirname(here)
    ancestors.append("")
    tries = []

    def every_top(make):
        for stem in ancestors:
            for top in ("tests", "test"):
                root_dir = os.path.join(stem, top) if stem else top
                got = make(root_dir)
                if got:
                    tries.append(got)
    if inner:
        every_top(lambda d: (os.path.join(d, *inner, "test_%s.py" % base),
                             os.path.isfile))
    for name in names:
        every_top(lambda d, n=name: (os.path.join(d, "test_%s.py" % n), os.path.isfile))
    if not specific:
        for cut in range(len(inner)):
            every_top(lambda d, c=cut: (os.path.join(d, *inner[c:]), os.path.isdir))
    every_top(lambda d: (os.path.join(d, "test_%s" % base), os.path.isdir))
    if not specific:
        every_top(lambda d: (d, os.path.isdir))
    for rel, is_right in tries:
        if is_right(os.path.join(root, rel)):
            return [rel]
    return []

DIFF_TRIVIAL = re.compile(r"^[\s)\]}:,]*$")

def patch_shape(patch: str) -> str:
    """A sentence describing an answer by shape: how many files, hunks and lines it changes."""
    files = hunks = 0
    added: list = []
    removed: list = []
    inside = False
    for line in (patch or "").split("\n"):
        if line.startswith("diff --git "):
            files += 1
            inside = False
        elif line.startswith("@@ "):
            hunks += 1
            inside = True
        elif inside and line.startswith("+"):
            added.append(line[1:].strip())
        elif inside and line.startswith("-"):
            removed.append(line[1:].strip())
    pool: dict = {}
    for line in added:
        pool[line] = pool.get(line, 0) + 1
    carried = dropped = 0
    for line in removed:
        if not line or DIFF_TRIVIAL.match(line):
            continue
        if pool.get(line):
            pool[line] -= 1
            carried += 1
        else:
            dropped += 1
    return ("files=%d hunks=%d +%d -%d carried=%d dropped=%d"
            % (files, hunks, len(added), len(removed), carried, dropped))

def fold_report(patch: str) -> str:
    """How much the answer changes, counted. Not what it is presumed to be.

    This used to classify each hunk against a list of shapes a defect is said
    to take and name the one it matched. Counting lines is an observation;
    calling a line a folded membership test is a diagnosis of a task this has
    not read, and it does not become one by being written to a log.
    """
    hunks = files = added = removed = 0
    for line in (patch or "").split("\n"):
        if line.startswith("diff --git "):
            files += 1
        elif line.startswith("@@ "):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return "files=%d hunks=%d +%d -%d" % (files, hunks, added, removed)

def definition_preservation_clause(statement: str, definition: str) -> str:
    """Find a stated name-preservation rule, not an assumed refactor contract."""
    parsed = re.fullmatch(r"(async(?: def)?|def|class)\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)",
                          definition or "")
    if not parsed:
        return ""
    kind, name = parsed.groups()
    public = all(not part.startswith("_") or (part.startswith("__") and part.endswith("__"))
                 for part in name.split("."))
    identifier = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    operand = r"(?:`" + identifier + r"(?:\(\))?`|" + identifier + r"(?:\(\))?)"
    listed = re.compile(operand + r"(?:\s*(?:,\s*(?:and\s+)?|\band\b\s*)" + operand + r")*")
    category = r"definitions?|functions?|methods?|classes|class|names?|APIs?"
    modifiers = r"(?:(?:the|all|any|existing|current|public|private|exported|top-level|module-level)\s+)*"
    generic = re.compile(r"^(?:the\s+)?(?:names?\s+of\s+)?(" + modifiers + r")(" + category + r")$", re.I)
    qualified = re.compile(r"^(" + modifiers + r")(" + category + r")\s+(.+)$", re.I)

    def matches_category(label: str, qualifiers: str) -> bool:
        label = label.lower()
        if re.search(r"\b(?:public|exported)\b", qualifiers, re.I) and not public:
            return False
        if re.search(r"\bprivate\b", qualifiers, re.I) and public:
            return False
        if re.search(r"\b(?:top-level|module-level)\b", qualifiers, re.I) and "." in name:
            return False
        if label in ("class", "classes"):
            return kind == "class"
        if label in ("method", "methods"):
            return kind != "class" and "." in name
        if label in ("function", "functions"):
            return kind != "class" and "." not in name
        if label.lower() in ("api", "apis"):
            return public
        return True

    def object_rule(target: str) -> tuple[bool, bool]:
        """Whether the object is understood, and whether it names this definition."""
        target = target.strip().rstrip(".;!?").strip()
        match = generic.fullmatch(target)
        if match:
            return True, matches_category(match[2], match[1])
        matches_kind = True
        match = qualified.fullmatch(target)
        if match:
            matches_kind = matches_category(match[2], match[1])
            target = match[3].strip()
        else:
            target = re.sub(r"^(?:the|existing|current)\s+", "", target, flags=re.I)
        if not listed.fullmatch(target):
            # Behavior, signatures, return values, file qualifiers, and other
            # complex objects are not unambiguous name-preservation rules.
            return False, False
        names = [item.strip("` ").removesuffix("()")
                 for item in re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", target)]
        return True, matches_kind and (name in names or name.rsplit(".", 1)[-1] in names)

    for clause in instruction_clauses(statement):
        for match in re.finditer(
                r"\b(?:(?:do not|don't|must not|never)\s+(?:remove|delete|drop|rename)|"
                r"preserve|retain|keep)\s+(.+)", clause, re.I):
            target = re.split(
                r"\b(?:but|while|and then)\b|\band\s+(?:remove|delete|drop|rename|replace|add|create|change|update)\b",
                match[1], 1, flags=re.I)[0]
            parts = re.split(r"\b(?:except(?:\s+for)?|excluding|other than|apart from)\b",
                             target, 1, flags=re.I)
            if len(parts) > 1:
                exception = parts[1].strip().rstrip(".;!?").strip()
                understood, excluded = object_rule(exception)
                if excluded:
                    continue
                # Only an understood list/category can narrow a universal rule.
                if not understood:
                    continue
            if object_rule(parts[0])[1]:
                return clause
    return ""


def suppression_prohibition_clause(statement: str, path: str = "", root: str = "") -> str:
    """The clause forbidding suppression comments, when the statement has one for this path."""
    for clause in instruction_clauses(statement):
        paths = operation_literals(clause)
        if path and paths and not any(path_within(operation_path(path, root),
                                                 operation_path(target, root))
                                      for target in paths):
            continue
        if re.search(r"\b(?:do not|don't|must not|never)\s+(?:add|use|introduce)\s+"
                     r"(?:any\s+|new\s+)?(?:suppression(?:\s+comments?)?|(?:#\s*)?noqa)\b|"
                     r"\bno\s+(?:new\s+)?(?:suppression\s+comments?|noqa)\b", clause, re.I):
            return clause
    return ""


def suppression_comments(source: str) -> collections.Counter:
    """Count actual Python comments, excluding strings that quote a directive."""
    found = collections.Counter()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT and NOQA_DIRECTIVE.search(token.string):
                found[token.string.strip()] += 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return found


def visible_definitions(source: str) -> list[str]:
    """The names a module defines for its callers, so a change that removes one can be noticed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[str] = []

    def walk(node: ast.AST, prefix: str, inside: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async" if isinstance(child, ast.AsyncFunctionDef) else "def"
                if not inside:
                    found.append("%s %s%s" % (kind, prefix, child.name))
                walk(child, prefix + child.name + ".", True)
            elif isinstance(child, ast.ClassDef):
                if not inside:
                    found.append("class %s%s" % (prefix, child.name))
                walk(child, prefix + child.name + ".", inside)
            else:
                walk(child, prefix, inside)
    walk(tree, "", False)
    return sorted(found)

def outline_source(source: str) -> list[str]:
    """A file's classes and functions as an indented outline, for reading without the whole text.
    """
    rows: list[tuple[int, int, int, str]] = []

    def walk(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = ("class" if isinstance(child, ast.ClassDef)
                        else "async def" if isinstance(child, ast.AsyncFunctionDef)
                        else "def")
                rows.append((child.lineno, getattr(child, "end_lineno", child.lineno),
                             depth, "%s %s" % (kind, child.name)))
                walk(child, depth + 1)
            else:
                walk(child, depth)
    walk(ast.parse(source), 0)
    rows.sort()
    return ["%5d-%-5d %s%s" % (start, end, "  " * depth, name)
            for start, end, depth, name in rows]

FENCE_METHOD = re.compile(r"`([A-Za-z_]\w*)\.([A-Za-z_]\w*)\(\)`")
FENCE_METHOD_OF = re.compile(r"`([A-Za-z_]\w*)`\s+method\s+of\s+`([A-Za-z_]\w*)`")
FENCE_CALL = re.compile(r"`([A-Za-z_]\w*)\(\)`")
FENCE_BODY_OF = re.compile(r"\bbody\s+of\s+`([A-Za-z_]\w*)`", re.I)
FENCE_PATH = re.compile(
    r"`((?:/[A-Za-z0-9_./-]*|[A-Za-z0-9_.][A-Za-z0-9_./-]*)"
    r"\.(?:py|pyi|ts|tsx|js|jsx|go|rb|rs|java|sql|cs|kt|kts|php|scala))`")
FENCE_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)
FENCE_IMPORT_LINE = re.compile(r"^\s*(?:import|from)\s")
FENCE_FORBIDS = re.compile(
    r"\b(?:do not|don't|never|must not|should not|without)\b", re.I)
FENCE_SENTENCE = re.compile(r"(?<=[.;!?])\s+")
# A Markdown paragraph, list item or quote wrapped over several source lines is
# one run of prose: "Limit production changes to" / "`app/models.py`, specifically"
# is a single clause, and a line break is not a sentence boundary. Fenced and
# indented code, headings, tables, rules, hard breaks and blank lines keep their
# own lines, and a new list item or quote line starts a new one.
MARKDOWN_ITEM = re.compile(r"(?:[-*+]|\d+[.)]|[A-Z]{1,2}\d{2,3}:)\s")
MARKDOWN_RULE = re.compile(r"[-=*_\s]{3,}")

def reflowed_lines(statement: str) -> list:
    """The statement's lines, with each soft-wrapped run of prose joined into one."""
    out: list = []
    fence = ""
    joinable = False  # whether the next plain line continues the last one
    for raw in (statement or "").splitlines():
        line = raw.strip()
        if fence:
            out.append(raw)
            if re.fullmatch(re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*", line):
                fence = ""
            continue
        opening = re.fullmatch(r"(`{3,}|~{3,})(.*)", line)
        code = raw.startswith(("    ", "\t")) and (not out or not out[-1].strip()
                                                    or out[-1].startswith(("    ", "\t")))
        if opening and not (opening[1][0] == "`" and "`" in opening[2]):
            fence = opening[1]
        if (fence or code or not line or line.startswith(("#", "|")) or MARKDOWN_RULE.fullmatch(line)):
            out.append(raw)
            joinable = False
            continue
        quote = line.startswith(">")
        inner = re.sub(r"^(?:>\s*)+", "", line) if quote else line
        continues = joinable and not MARKDOWN_ITEM.match(inner) and quote == out[-1].lstrip().startswith(">")
        if continues and inner:
            out[-1] = out[-1].rstrip() + " " + inner
        else:
            out.append(raw)
        joinable = bool(inner) and not raw.endswith(("  ", "\\"))
    return out

def instruction_clauses(statement: str, *, with_lines: bool = False, reflow: bool = True) -> list:
    """Current prose instructions, with quoted task data kept out of guards.

    This bounded prose/Markdown reader is not a general permissions solver.
    Inline identifiers and paths stay attached to their instruction. Examples,
    old rules and negated rules do not supply new restrictions. Explicitly
    adopted quotations (including Requirements blockquotes) remain instructions.
    The model still receives the original statement intact.
    """
    adopt = re.compile(
        r"\b(?:follow|obey|apply|enforce|respect)\s+(?:(?:this|the|these|following|current)\s+)*"
        r"(?:requirements?|instructions?|rules?|constraints?)\b|"
        r"\b(?:current\s+)?(?:requirements?|instructions?|rules?|constraints?)"
        r"\s*(?:for (?:this|the current) (?:task|work))?\s*:\s*$", re.I)
    data = re.compile(
        r"\b(?:historical|obsolete|superseded|withdrawn|outdated)\s+"
        r"(?:\w+\s+){0,3}(?:instructions?|requirements?|rules?|constraints?)\b|"
        r"\b(?:instructions?|requirements?|rules?|constraints?)\b[^.!?;]*"
        r"\b(?:obsolete|superseded|withdrawn|outdated)\b|"
        r"\b(?:old|previous|former)\s+(?:instructions?|requirements?|rules?|constraints?)"
        r"\s+(?:said|says|was|were|stated|asked)\b|"
        r"^\s*(?:[-*+]\s*)?(?:example|sample|counterexample)\s*:|"
        r"\b(?:ignore|disregard)\s+(?:the\s+)?(?:old|previous|quoted|following)\s+"
        r"(?:instructions?|requirements?|rules?|constraints?)\b", re.I)
    withdrawn = re.compile(
        r"\b(?:there is|there's)\s+no\s+(?:requirement|need|obligation)\b|"
        r"\b(?:it is|it's)\s+(?:not\s+(?:true|the case)|false)\s+that\b|"
        r"\b(?:you|we)\s+are\s+not\s+(?:required|obliged)\b|"
        r"\b(?:need not|do not have to|don't have to|no longer (?:need|have) to)\b|"
        r"\b(?:no|neither)\s+(?:rule|requirement|instruction)\s+requires?\b|"
        r"\b(?:is|are)\s+not\s+(?:a\s+)?(?:requirement|instruction|constraint|rule)"
        r"\b|"
        r"\b(?:do not|don't|never|must not|should not)\s+(?:\w+\s+){0,2}?"
        r"(?:restrict|limit|confine|bound|require)\b|"
        r"\b(?:do not|don't)\s+(?:keep|preserve)\s+(?:its|the|their)\s+signature\b", re.I)
    # A quoted path/name or annotation is an operand, not a separate rule.
    operand = re.compile(r"[A-Za-z0-9_./:$-]+(?:\(\))?/?|#\s*[^\n]+|(?:type|pyright):\s*ignore(?:\[[^\]]+\])?$")
    rule_text = re.compile(
        r"\b(?:only\s+(?:edit|modify|change|repair|touch|use)|"
        r"(?:edit|modify|change|repair|touch)\s+only|"
        r"(?:limit|restrict|confine)\s+(?:the\s+)?(?:changes?|edits?|work)|"
        r"do not|don't|must not|no\s+(?:imports?|loops?|queries)|"
        r"keep\s+(?:its|the)\s+signature|use\s+only\s+names)\b", re.I)
    quoted = re.compile(r'"([^"\n]*)"|“([^”\n]*)”|(?<!\w)\'([^\'\n]*)\'(?!\w)')
    parts = []
    fence = None
    quoted_requirements = False
    prose_lines = set()
    statement_markdown_blocks(statement or "", prose_lines=prose_lines)
    original_lines = (statement or "").splitlines()
    # Apply the same Markdown source eligibility to guards and quotations.
    # Blank placeholders retain code/HTML boundaries before prose is reflowed.
    eligible = [raw if number in prose_lines else "" for number, raw in enumerate(original_lines)]
    lines = (original_lines if with_lines else
             reflowed_lines("\n".join(eligible)) if reflow else eligible)
    for line_number, raw in enumerate(lines):
        if with_lines and line_number not in prose_lines:
            continue
        line = raw.strip()
        if fence is not None:
            if re.fullmatch(re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*", line):
                fence = None
            continue
        opening = re.fullmatch(r"(`{3,}|~{3,})(.*)", line)
        if opening and not (opening[1][0] == "`" and "`" in opening[2]):
            fence = opening[1]
            continue
        if not line:
            continue
        if line.startswith(">"):
            if not quoted_requirements:
                continue
            line = re.sub(r"^(?:>\s*)+", "", line)
        else:
            quoted_requirements = bool(adopt.search(line)) and not data.search(line)

        def unquote(match):
            value = next(group for group in match.groups() if group is not None)
            prefix = re.split(r"(?<=[.!?;])[\"'”`]*\s+", line[:match.start()])[-1]
            if operand.fullmatch(value.strip()):
                return match.group(0)
            if with_lines and not rule_text.search(value):
                # Requirement quotations can specify literal input/output
                # grammar. Keep those bytes; they are not scope instructions.
                return match.group(0)
            if adopt.search(prefix) and not data.search(prefix) and not withdrawn.search(prefix):
                return value
            # Preserve the boundary of a quoted sentence so an old quoted
            # rule cannot absorb the current instruction that follows it.
            return value[-1] + " " if value.endswith((".", "!", "?", ";")) else " "

        line = quoted.sub(unquote, line)
        def inline_code(match):
            if not rule_text.search(match[1]):
                return match[0]
            prefix = re.split(r"(?<=[.!?;])[\"'”`]*\s+", line[:match.start()])[-1]
            if adopt.search(prefix) and not data.search(prefix) and not withdrawn.search(prefix):
                return match[1]
            return match[1][-1] + " " if match[1].endswith((".", "!", "?", ";")) else " "

        line = re.sub(r"`([^`]+)`", inline_code, line)
        # Explicit contrast introduces a new clause: withdrawing one rule
        # must not erase a separate current instruction after "but".
        clauses = [part for sentence in FENCE_SENTENCE.split(line)
                   for part in re.split(r",?\s+(?:but|however)\s+", sentence, flags=re.I)]
        accepted = []
        for part in clauses:
            part = " ".join(part.split())
            withdrawn_match = withdrawn.search(part)
            if with_lines and withdrawn_match and re.match(
                    r"(?:need not|do not have to|don't have to|no longer (?:need|have) to)\b",
                    withdrawn_match[0], re.I):
                # A domain fact such as "values need not be ordered" is a
                # current input constraint, not withdrawal of an instruction
                # addressed to the solver. Permission guards keep their old
                # conservative behavior; the requirement reader keeps facts.
                prefix = part[:withdrawn_match.start()].strip()
                if prefix and not re.search(r"\b(?:you|we)\s*$", prefix, re.I):
                    withdrawn_match = None
            if not part or data.search(part) or withdrawn_match:
                continue
            accepted.append(part)
        if with_lines:
            # Keep contrast words and punctuation when the entire line is
            # current. Rejoining split clauses would invent an unquotable
            # requirement, even if the fragments retain similar meaning.
            if accepted and len(accepted) == len([c for c in clauses if c.strip()]):
                parts.append((line_number, line))
            else:
                parts.extend((line_number, part) for part in accepted)
        else:
            parts.extend(accepted)
    return parts

def stated_methods(statement: str) -> list[tuple[str, str]]:
    """The (class, method) pairs the statement confines the change to, as it names them."""
    text = " ".join(part for part in instruction_clauses(statement)
                    if not FENCE_FORBIDS.search(part))
    qualified = [(holder, name) for holder, name in FENCE_METHOD.findall(text)]
    qualified += [(holder, name) for name, holder in FENCE_METHOD_OF.findall(text)]
    if qualified:
        return sorted(set(qualified))
    return sorted({("", name) for name in FENCE_CALL.findall(text) + FENCE_BODY_OF.findall(text)})

# FENCE_PATH needs a file extension, so it cannot see a directory boundary.
# "Only edit files under `lib/`" names a place, and the place is the point.
FENCE_ANY_PATH = re.compile(r"`([A-Za-z0-9_.][A-Za-z0-9_./-]*/?)`")
CREATION_FORBIDDEN = re.compile(
    r"\b(?:do not|don't|never|must not|should not)\s+(?:\w+\s+){0,2}?"
    r"(?:add|create|introduce)\b", re.I)

def path_within(path: str, boundary: str) -> bool:
    """Is this path the boundary, or inside it?"""
    boundary = (boundary or "").rstrip("/")
    if not boundary:
        return False
    return path == boundary or path.startswith(boundary + "/")

def refusal_targets(clause: str, names) -> set:
    """The functions an explicit prohibition is about, if it names any.

    A prohibition carries a target as often as not: "implement quick without
    loops" rules out a loop in `quick` and says nothing about the rest of the
    file. A name counts as the target when the clause uses it as a word, and
    for a name too short to survive ordinary prose only when the clause quotes
    it. No target found means the clause is read as applying throughout, which
    is what "No loops." means.
    """
    # An explicit qualified method names that owner, not every matching basename.
    qualified = set()
    for match in re.finditer(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*\(\s*\)", clause or ""):
        prefix = clause[:match.start()].rstrip(" `")
        located = re.search(r"\b(?:in|inside|within|to|for|of|method|function)\s*$", prefix, re.I)
        example = re.search(r"\b(?:call|calling|use|using|prefer|via|like|as)\s*$", prefix, re.I)
        if located or (match.group(1) in (names or ()) and not example):
            qualified.add(match.group(1))
    if qualified:
        return qualified
    found = set()
    for name in names or ():
        bare = name.split(".")[-1]
        if not bare:
            continue
        word = r"(?<![\w.])%s(?![\w.])" % re.escape(bare)
        if re.search(r"`[^`]*" + word + r"[^`]*`", clause or ""):
            found.add(name)
        elif len(bare) >= 3 and re.search(word, clause or ""):
            found.add(name)
    return found

def fence_path(path: str, root: str = "") -> str:
    """A path as the statement and the diff both name it: no root, no leading dot, no a/ or b/
    prefix.
    """
    path = path.replace("\\", "/")
    root = os.path.normpath(root).replace("\\", "/") if root else ""
    if root and path.startswith(root + "/"):
        path = path[len(root) + 1:]
    while path.startswith("./"):
        path = path[2:]
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return os.path.normpath(path).replace("\\", "/")

def stated_file(statement: str, root: str = "") -> str | None:
    """The one file the statement names as the place of the change, or None.

    A path named for another reason, where constants live or which module
    holds the entrypoint, is not that file: only sentences about the change,
    its bounds or its method are read.
    """
    paths = set()
    for part in instruction_clauses(statement):
        if scope_clause(part) or EDIT_WORDS.search(part):
            paths.update(fence_path(path, root) for path in FENCE_PATH.findall(part))
    return paths.pop() if len(paths) == 1 else None

def method_bounds(source: str, holder: str, name: str) -> tuple[int, int] | None:
    """The first and last line of a named method, but only when it resolves exactly once."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    found: list[tuple[int, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name == name and (not holder or prefix == holder):
                    found.append((child.lineno,
                                  getattr(child, "end_lineno", child.lineno)))
                walk(child, prefix)
            elif isinstance(child, ast.ClassDef):
                walk(child, child.name)
            else:
                walk(child, prefix)
    walk(tree, "")
    return found[0] if len(found) == 1 else None

def fenced_region(source: str, wanted: list) -> tuple[str, int, int] | None:
    """The one method the statement's words resolve to, with its line span, or None when it is
    ambiguous.
    """
    hits = []
    for holder, name in wanted:
        found = method_bounds(source, holder, name)
        if found:
            hits.append(("%s.%s" % (holder, name) if holder else name,
                         found[0], found[1]))
    return hits[0] if len(hits) == 1 else None

# A restriction is a clause that tells this run not to change something. A
# sentence that merely describes where a defect lives ("the bug is in one
# function") grants no permission and withdraws none, and a sentence that
# refuses a restriction ("do not restrict edits to one method") is the opposite
# of one. Both were read as restrictions before, which let this run tell the
# model the task forbade things the task never mentioned.
# "Edit only ..." closes the file when what follows names a file, a method or
# the code; "Repair only rejected messages" bounds the domain, not the change.
SCOPE_OBJECT = (r"(?=[^.;]*(?:`|\b(?:files?|method|function|body|module|package|directory|directories|paths?|"
                r"lines?|code|source|implementation|what|inside|within|production)\b|"
                r"\.(?:py|pyi|ts|tsx|js|go|rb|rs|java|sql|cs|kt|php|scala)\b|\w+/))")
SCOPE_RESTRICTION = re.compile(
    r"(?<!-)\bonly\s+(?:edit|modify|change|repair|touch)\b|"
    r"\b(?:edit|modify|change|repair|touch)\s+only\b" + SCOPE_OBJECT + r"|"
    r"\b(?:edit|modify|change)\s+`[^`]+`\s+only\b|"
    r"\b(?:limit|restrict|confine)\s+(?:the\s+)?(?:production\s+)?"
    r"(?:changes?|edits?|work|modifications?)\s+to\b|"
    r"\b(?:do not|don't|never|must not|should not)\s+"
    r"(?:change|edit|modify|touch|alter)\s+(?:anything|any\s+\w+|any)?\s*"
    r"(?:else|outside|other than)\b|"
    r"\b(?:leave|keep)\s+(?:the\s+)?rest\s+of\s+(?:the|its|this)\s+file\s+"
    r"(?:as it (?:is|was)|unchanged)\b|"
    r"\brest of (?:the|its|this) file (?:unchanged|(?:must|should) (?:stay|remain) unchanged)\b|"
    r"\b(?:you\s+)?may\s+edit\s+(?:that|the|this)\s+(?:complete|whole|entire)\s+file\b|"
    r"\beditable(?:\s+production)?(?:\s+(?:paths?|directories|files|code|scope))?\s*(?::|\s+(?:is|are)\b)|"
    r"\bproduction\s+(?:scope|code)\s+(?:is|in)\b|"
    r"\bedit\s+[\w./-]+/\s+production\b|"
    r"\bonly\s+`?[\w./-]+/?`?\s+is\s+editable\b|"
    r"\beditable\s+production\s+(?:files|paths|directories|code)\s+[\w./-]+/?\b|"
    r"\ballowed\s+(?:edits?|changes?|paths?|files?)\s*:|"
    r"\b(?:within|inside|under)\s+`?[\w-][\w./-]*/`?(?=[\s,.;:]|$)|"
    r"\bproduction\s+(?:code\s+)?(?:in\s+)?(?:`[^`]+`(?:\s*,\s*|\s+and\s+)?)+\s+may\s+change\b",
    re.I)
# The same clause, but one whose target is a method body rather than a file.
METHOD_RESTRICTION = re.compile(
    r"\b(?:only\s+(?:edit|modify|change|repair|touch)|"
    r"(?:edit|modify|change|repair|touch)\s+only)\s+"
    r"(?:(?:that|this|the|one|single)\s+)?(?:`[^`]+`\s+)?(?:method|function)\b|"
    r"\b(?:limit|restrict|confine)\s+(?:the\s+)?(?:changes?|edits?|work)\s+to\s+"
    r"(?:(?:that|this|the|one|single)\s+)?(?:`[^`]+`\s+)?(?:method|function)\b|"
    r"\b(?:do not|don't|never|must not|should not)\s+"
    r"(?:change|edit|modify|touch|alter)\s+(?:anything\s+)?outside\s+"
    r"(?:that|this|the)\s+(?:method|function)\b|"
    r"\b(?:keep|confine|restrict|limit)\s+(?:(?:all|the)\s+)?(?:changes?|edits?|work)\s+"
    r"within (?:that|this|the) (?:method|function)\b|"
    r"\brest of (?:the|its|this) file (?:unchanged|(?:must|should) (?:stay|remain) unchanged)\b|"
    r"\b(?:edit|modify|change)\s+only\s+the\s+(?:implementation\s+)?body\s+of\s+`[^`]+`",
    re.I)
# "Do not restrict edits to one method" negates the restricting verb itself, so
# the sentence withdraws a restriction rather than making one. A negation of an
# ordinary verb ("do not change anything outside this method") does the
# opposite and is matched by the patterns above, so this is not a rule that
# every sentence containing "do not" is ignored.
SCOPE_CANCELLED = re.compile(
    r"\b(?:do not|don't|never|must not|should not)\s+(?:\w+\s+){0,2}?"
    r"(?:restrict|limit|confine|bound)\b", re.I)
SCOPE_WIDENED = re.compile(
    r"\b(?:fix|update|change|edit|repair|modify)\s+(?:all|whichever|any)\s+"
    r"(?:affected\s+|relevant\s+|necessary\s+)?files?\b", re.I)

def scope_clause(statement: str, pattern: "re.Pattern" = None) -> str:
    """The sentence that restricts what this change may touch, or "".

    The clause itself is returned, not a yes or no, so whatever acts on it can
    quote the words that authorise it instead of asserting a rule of its own.
    """
    pattern = pattern or SCOPE_RESTRICTION
    parts = instruction_clauses(statement)
    for part in parts:
        if SCOPE_WIDENED.search(part):
            return ""
    for part in parts:
        if SCOPE_CANCELLED.search(part):
            continue
        if pattern.search(part):
            return " ".join(part.split())[:REQUIREMENT_CHARS]
    return ""

def method_clause(statement: str) -> str:
    """The sentence that confines the change to one method body, or ""."""
    return scope_clause(statement, METHOD_RESTRICTION)

def statement_bounded(statement: str) -> bool:
    """Does the statement confine the change to something, rather than leaving it open?"""
    return bool(scope_clause(statement))

def statement_method_bounded(statement: str) -> bool:
    """Does the statement confine the change to one method body?

    Naming a file to edit is not the same permission as naming a method: a
    task that says "only edit this file" allows an import, a helper and a new
    line anywhere in it. Only a statement that bounds the method body earns the
    stricter reading. Naming a method to fix is not that either -- it says
    which method is wrong, not that the rest of the file is closed.
    """
    return bool(method_clause(statement))

# "Keep the method signature" and "the function's signature" name the same thing
# as "keep its signature".
SIGNATURE_OWNER = r"(?:(?:method|function)(?:'s|\u2019s)?\s+)?"
SIGNATURE_KEPT = re.compile(
    r"\bkeep\s+(?:its|the|their)\s+" + SIGNATURE_OWNER + r"signatures?\b|"
    r"\b(?:same|unchanged|identical)\s+signature\b|"
    r"\b(?:do not|don't|never|must not|should not)\s+change\s+"
    r"(?:its|the)\s+" + SIGNATURE_OWNER + r"signature\b|"
    r"\bsignature\s+must\s+(?:stay|remain)\b", re.I)

def signature_clause(statement: str) -> str:
    """The sentence asking for the signature to stay as it is, or ""."""
    for part in instruction_clauses(statement):
        if SIGNATURE_KEPT.search(part):
            return " ".join(part.split())[:REQUIREMENT_CHARS]
    return ""

def statement_refused_nodes(statement: str) -> dict:
    """Constructs the statement forbids in so many words, each with its clause.

    The clause travels with the refusal so that whatever reports it can quote
    the words that forbid the construct rather than assert a rule of its own.
    """
    parts = instruction_clauses(statement)
    text = "; ".join(parts)
    refused: dict = {}
    found = re.search(r"[^.!?;:]*\bonly names the file already imports\b[^.!?;:]*",
                      text, re.I)
    if found:
        clause = " ".join(found.group(0).split())[:REQUIREMENT_CHARS]
        refused.update({"Import": clause, "ImportFrom": clause})
    for part in parts:
        if re.search(r"\b(?:do not|don't|never|must not|should not)\s+"
                     r"(?:add|introduce|use)\s+(?:any\s+|new\s+)?imports?\b|"
                     r"\bno\s+(?:new\s+)?imports?\s*(?:[.!?;:]|$)", part, re.I):
            refused.update({"Import": part[:REQUIREMENT_CHARS],
                            "ImportFrom": part[:REQUIREMENT_CHARS]})
    categories = (
        (r"loops?", ("For", "AsyncFor", "While")),
        (r"comprehensions?", ("ListComp", "SetComp", "DictComp", "GeneratorExp")),
        (r"lambdas?", ("Lambda",)),
        (r"exception handling", ("Try", "TryStar", "Raise")),
        (r"context managers?", ("With", "AsyncWith")),
    )
    keyword_kinds = {"for": ("For", "AsyncFor"), "while": ("While",), "lambda": ("Lambda",),
                     "try": ("Try", "TryStar"), "with": ("With", "AsyncWith")}
    for part in parts:
        # The construct named in so many words: "must contain no `for` or `while` statement".
        if re.search(r"\b(?:contain|use|have|include|write|add)\s+no\s+`|\bno\s+`|\bwithout\s+(?:an?\s+|any\s+)?`|"
                     r"\bmust not\s+(?:use|contain|include|have)\s+(?:an?\s+|any\s+)?`", part, re.I):
            for word in re.findall(r"`(for|while|lambda|try|with)`", part, re.I):
                for kind in keyword_kinds[word.lower()]:
                    refused.setdefault(kind, " ".join(part.split())[:REQUIREMENT_CHARS])
    item = r"(?:Python\s+)?(?:" + "|".join(label for label, _ in categories) + ")"
    separator = r"(?:\s*,\s*(?:(?:or|and)\s+)?|\s+(?:or|and)\s+)"
    lead = (r"(?:(?:^|[.!?;:])\s*(?:[-*+]\s+|\d+[.)]\s+)?|\b(?:write|implement)\b[^.!?;:]*?\s+)"
            r"(?:no\s+|without\s+|(?:(?:it|the method|the function|you)\s+)?"
            r"must not\s+(?:use|contain|include|have)\s+)")
    tail = r"(?:\s+(?:inside (?:it|the method|the function)|in (?:it|the method|the function)))?\s*(?=[.!?;:]|$)"
    for match in re.finditer(lead + "(" + item + "(?:" + separator + item + r")*)\b" + tail,
                             text, re.I):
        clause = " ".join(match.group(0).split()).strip()[:REQUIREMENT_CHARS]
        for label, kinds in categories:
            if re.search(r"\b(?:" + label + r")\b", match[1], re.I):
                refused.update({kind: clause for kind in kinds})
    return refused

# "Do not add database writes, process, filesystem, network, repository, or
# dynamic-code side effects": the kinds of effect the statement rules out, and
# the Python calls that are that kind of effect. Each kind is read only when the
# statement names it, and only calls the change adds are counted, so a task
# that asks for writes and rules out raw SQL keeps its writes.
SIDE_EFFECT_CLAUSE = re.compile(
    r"\b(?:do not|don't|never|must not|should not)\s+(?:add|introduce|perform|cause)\s+"
    r"(?P<kinds>[^.;:]*?)\s+(?:side[- ]effects?|behaviou?rs?)\b[^.;]*", re.I)
SIDE_EFFECT_KINDS = (
    ("database writes", r"database\s+writes?"),
    ("raw SQL", r"raw\s+sql"),
    ("process", r"(?:sub)?process(?:es)?"),
    ("filesystem", r"file(?:\s*system|s)?"),
    ("network", r"network(?:ing)?"),
    ("repository", r"repository|repositories"),
    ("dynamic code", r"dynamic[- ]code"),
)
EFFECT_ATTRIBUTES = {
    "database writes": {"save", "update", "delete", "create", "bulk_create", "bulk_update",
                        "get_or_create", "update_or_create", "execute", "executemany"},
    "raw SQL": {"raw", "extra", "execute", "executemany", "cursor", "RawSQL"},
    "filesystem": {"write_text", "write_bytes", "read_text", "read_bytes", "unlink", "rmtree", "touch"},
    "dynamic code": {"import_module"},
}
EFFECT_NAMES = {
    "raw SQL": {"RawSQL"},
    "process": {"Popen"},
    "filesystem": {"open"},
    "network": {"urlopen"},
    "dynamic code": {"eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
                     "globals", "locals", "vars", "breakpoint"},
}
EFFECT_ROOTS = {
    "process": {"subprocess"},
    "filesystem": {"shutil"},
    "network": {"requests", "urllib", "urllib3", "httpx", "socket", "http", "aiohttp", "smtplib", "ftplib"},
    "repository": {"git", "pygit2", "dulwich"},
}
EFFECT_OS = {
    "process": {"system", "popen", "fork", "forkpty", "spawnl", "spawnv", "spawnlp", "spawnvp",
                "execl", "execv", "execlp", "execvp", "kill"},
    "filesystem": {"remove", "unlink", "rmdir", "removedirs", "makedirs", "mkdir", "rename",
                   "renames", "replace", "chmod", "chown", "truncate"},
}
EFFECT_WHAT = {
    "database writes": "writes to the database when it is called on a model or queryset",
    "raw SQL": "sends raw SQL", "process": "starts or controls a process",
    "filesystem": "reads or changes files", "network": "uses the network",
    "repository": "works on a repository", "dynamic code": "is dynamic code",
}

def statement_side_effect_clauses(statement: str) -> list:
    """Keep every current clause; the same category can have several targets."""
    found = []
    for part in instruction_clauses(statement):
        for match in SIDE_EFFECT_CLAUSE.finditer(part):
            # Interpret the complete clause before clipping display text.
            clause = " ".join(part[match.start():].split())
            kinds = match.group("kinds")
            for kind, words in SIDE_EFFECT_KINDS:
                if re.search(r"(?:^|[\s,])(?:%s)(?=\s*(?:,|\bor\b|\band\b|$))" % words, kinds, re.I):
                    if (kind, clause) not in found:
                        found.append((kind, clause))
    return found


def statement_side_effects(statement: str, *, all_clauses: bool = False) -> dict | list:
    """Return the legacy summary or each clause with its independent scope."""
    clauses = statement_side_effect_clauses(statement)
    if all_clauses:
        return clauses
    found: dict = {}
    for kind, clause in clauses:
        found.setdefault(kind, clause[:REQUIREMENT_CHARS])
    return found

# Qualified API names identify operations only when imports/builtins resolve
# without a conflicting local binding. A method name alone is not that evidence.
EFFECT_API_CALLS = {
    "process": {"subprocess." + name for name in
                ("Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput")}
               | {"os." + name for name in EFFECT_OS["process"]}
               | {"asyncio.create_subprocess_exec", "asyncio.create_subprocess_shell"},
    "filesystem": {"builtins.open", "io.open"}
                  | {"os." + name for name in EFFECT_OS["filesystem"]}
                  | {"shutil." + name for name in
                     ("copy", "copy2", "copyfile", "copytree", "copymode", "copystat", "move", "rmtree", "chown")},
    "network": {"urllib.request.urlopen", "urllib.request.urlretrieve", "socket.create_connection"}
               | {module + "." + name for module in ("requests", "httpx")
                  for name in ("request", "get", "post", "put", "patch", "delete", "head", "options")},
    "dynamic code": {"builtins.eval", "builtins.exec", "builtins.compile", "builtins.__import__",
                     "importlib.import_module"},
}


def effect_observations(tree, kinds) -> list:
    """Return (kind, callable, line, display, resolved) without executing source.

    Unresolved receiver types and reflection names are advisory candidates.
    Binding collection is conservative: any competing assignment/declaration
    invalidates import provenance, including one on another control-flow path.
    This is a bounded static check, not proof of arbitrary Python behavior.
    """
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    scopes = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
              ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    cached: dict = {}

    def scope_of(node):
        trail = {node}
        child, parent = node, parents.get(node)
        while parent is not None:
            if isinstance(parent, scopes):
                outer = (isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                         and child not in parent.body)
                if isinstance(parent, ast.Lambda):
                    outer = child is not parent.body
                if isinstance(parent, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    outer = parent.generators[0].iter in trail
                if not outer:
                    return parent
            trail.add(parent)
            child, parent = parent, parents.get(parent)
        return None

    def bindings(scope):
        if scope in cached:
            return cached[scope]
        found: dict = {}
        direct = getattr(scope, "body", [])
        direct = direct if isinstance(direct, list) else []

        def bind(name, value=""):
            found[name] = value if name not in found else ""

        def outside_expressions(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                expressions = list(node.args.defaults) + [x for x in node.args.kw_defaults if x is not None]
                if not isinstance(node, ast.Lambda):
                    expressions += list(node.decorator_list)
                    expressions += [x.annotation for x in ast.walk(node.args)
                                    if isinstance(x, ast.arg) and x.annotation is not None]
                    if node.returns is not None:
                        expressions.append(node.returns)
                return expressions
            if isinstance(node, ast.ClassDef):
                return list(node.bases) + list(node.decorator_list) + [x.value for x in node.keywords]
            return []

        def walrus_bindings(node):
            # Comprehension targets have their own scope, but named expressions
            # bind in the enclosing non-comprehension scope.
            if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for expr in outside_expressions(node):
                    walrus_bindings(expr)
                return
            if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                bind(node.target.id)
            for child in ast.iter_child_nodes(node):
                walrus_bindings(child)

        def visit(node):
            if node is not scope and isinstance(node, scopes):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bind(node.name)
                for expr in outside_expressions(node):
                    visit(expr)
                if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    walrus_bindings(node)
                return
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if isinstance(node, ast.Import):
                        name = alias.asname or alias.name.split(".")[0]
                        value = alias.name if alias.asname else name
                    else:
                        name = alias.asname or alias.name
                        value = ((node.module + "." + alias.name)
                                 if node.module and not node.level and alias.name != "*" else "")
                    bind(name, value if node in direct else "")
                return
            if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                root = node
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    bind(root.id)
            elif isinstance(node, ast.arg):
                bind(node.arg)
                return
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                for name in node.names:
                    bind(name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bind(node.name)
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                bind(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                bind(node.rest)
            for child in ast.iter_child_nodes(node):
                visit(child)
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for arg in scope.args.posonlyargs + scope.args.args + scope.args.kwonlyargs:
                bind(arg.arg)
            for arg in (scope.args.vararg, scope.args.kwarg):
                if arg is not None:
                    bind(arg.arg)
            for node in scope.body if isinstance(scope.body, list) else [scope.body]:
                visit(node)
        elif isinstance(scope, ast.ClassDef):
            for node in scope.body:
                visit(node)
        else:
            visit(scope)
        cached[scope] = found
        return found

    def resolve(node, scope):
        if isinstance(node, ast.Attribute):
            base = resolve(node.value, scope)
            return base + "." + node.attr if base else ""
        if not isinstance(node, ast.Name):
            return ""
        current = scope
        while current is not None:
            found = bindings(current)
            if "*" in found:
                # Unknown exports may replace an explicit import as well as a
                # builtin. Do not assume an imported alias survived that scope.
                return ""
            if node.id in found:
                return found[node.id]
            current = scope_of(current)
            # Methods do not close over names in their class namespace.
            while isinstance(current, ast.ClassDef):
                current = scope_of(current)
        return "builtins." + node.id

    receiver_cache: dict = {}

    def receiver_info(scope):
        if scope in receiver_cache:
            return receiver_cache[scope]
        result = ("", "", set())
        owner = parents.get(scope)
        if (isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
                and isinstance(owner, ast.ClassDef) and not scope.decorator_list):
            args = scope.args.posonlyargs + scope.args.args
            if args:
                name = args[0].arg
                own = {item.name for item in owner.body
                       if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                own.update(n.id for item in owner.body if isinstance(item, (ast.Assign, ast.AnnAssign))
                           for n in ast.walk(item) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
                rebound = False
                for node in ast.walk(scope):
                    if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
                        rebound = True
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                         ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and getattr(node, "name", None) == name:
                        rebound = True
                    if isinstance(node, ast.MatchMapping) and node.rest == name:
                        rebound = True
                    if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                            (alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)) == name
                            for alias in node.names):
                        rebound = True
                    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                            and node.value.id == name and isinstance(node.ctx, (ast.Store, ast.Del))):
                        own.add(node.attr)
                known = {"django.db.models.QuerySet": "queryset", "django.db.models.query.QuerySet": "queryset",
                         "django.db.models.Manager": "queryset", "django.db.models.manager.Manager": "queryset",
                         "django.db.models.Model": "model", "django.db.models.base.Model": "model"}
                bases = [known.get(resolve(base, scope_of(owner)), "") for base in owner.bases]
                if not rebound and len(bases) == 1 and bases[0]:
                    result = (name, bases[0], own)
        receiver_cache[scope] = result
        return result

    def receiver_kind(node, scope):
        # Cache function/class facts once; inspect each receiver chain iteratively.
        name, kind, own = receiver_info(scope)
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if kind != "queryset" or node.func.attr in own or node.func.attr not in ("all", "none", "filter", "exclude", "using"):
                return "", own
            node = node.func.value
        if isinstance(node, ast.Name) and node.id == name:
            return kind, own
        return "", own

    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        scope = scope_of(node)
        canonical = resolve(node.func, scope)
        root = node.func
        while isinstance(root, ast.Attribute):
            root = root.value
        lexical = root.id if isinstance(root, ast.Name) else ""
        member = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for kind in kinds:
            resolved = canonical in EFFECT_API_CALLS.get(kind, ())
            if kind == "filesystem" and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
                resolved = (node.func.attr in EFFECT_ATTRIBUTES["filesystem"]
                            and resolve(node.func.value.func, scope) in ("pathlib.Path", "pathlib.PosixPath", "pathlib.WindowsPath"))
            if kind in ("database writes", "raw SQL") and isinstance(node.func, ast.Attribute):
                receiver, overridden = receiver_kind(node.func.value, scope)
                methods = ({"save", "delete"} if receiver == "model" else
                           {"update", "delete", "create", "bulk_create", "bulk_update", "get_or_create", "update_or_create"}
                           if receiver == "queryset" else set())
                if kind == "raw SQL":
                    methods = {"raw", "extra"} if receiver == "queryset" else set()
                resolved = node.func.attr in methods and node.func.attr not in overridden
            hinted = ((isinstance(node.func, ast.Name) and member in EFFECT_NAMES.get(kind, ()))
                      or (isinstance(node.func, ast.Attribute)
                          and (member in EFFECT_ATTRIBUTES.get(kind, ())
                               or lexical in EFFECT_ROOTS.get(kind, ())
                               or (lexical == "os" and member in EFFECT_OS.get(kind, ())))))
            if not resolved and not hinted:
                continue
            if resolved:
                display = canonical.removeprefix("builtins.") if canonical else "." + member
            else:
                display = lexical + "." + member if isinstance(node.func, ast.Attribute) and lexical else member
            identity = canonical if resolved and canonical else member
            out.append((kind, identity, node.lineno, display, resolved))
    return out


def effect_calls(tree, kinds, *, provenance: bool = False) -> list:
    """Return candidate calls, optionally including resolved-binding evidence."""
    observations = effect_observations(tree, kinds)
    return observations if provenance else [item[:4] for item in observations]


def project_contract(root: str, statement: str = "", path: str = "") -> tuple[str, str]:
    """What language this project is and the command it uses for its own checks.

    Read from the statement's own commands first and from the files in the checkout second,
    so a project this run does not recognise still gets its own runner rather than a guess.
    """
    declared_languages = set()
    for block in statement_command_blocks(statement):
        try:
            argv = shlex.split(block)
        except ValueError:
            continue
        if not argv:
            continue
        tool = os.path.basename(argv[0])
        language = {"psql": "sql", "sqlite3": "sql", "mysql": "sql",
                    "go": "go", "node": "node", "npm": "node", "npx": "node", "tsx": "node",
                    "cargo": "rust", "mvn": "java", "javac": "java", "gradle": "java",
                    "gradlew": "java", "dotnet": "csharp", "kotlinc": "kotlin",
                    "ruby": "ruby", "bundle": "ruby", "rspec": "ruby",
                    "python": "python", "python3": "python", "pytest": "python"}.get(tool)
        if language:
            declared_languages.add(language)
            if path and path in argv:
                return language, ""
    directory = os.path.dirname(os.path.join(root, fence_path(path, root))) if path else root
    while inside(directory, root):
        for marker, language, runner in (
                ("go.mod", "go", "go test ./..."),
                ("Cargo.toml", "rust", "cargo test"),
                ("pom.xml", "java", "mvn -q test"),
                ("build.gradle.kts", "kotlin", "gradle test"),
                ("build.gradle", "java", "gradle test"),
                ("Gemfile", "ruby", ""),
                ("pyproject.toml", "python", ""),
                ("setup.py", "python", ""), ("setup.cfg", "python", ""),
                ("package.json", "node", "npm test")):
            filename = os.path.join(directory, marker)
            if not os.path.isfile(filename):
                continue
            if marker == "package.json":
                try:
                    with open(filename) as stream:
                        data = json.load(stream)
                    scripts = data.get("scripts", {})
                    if not isinstance(scripts, dict) or not isinstance(scripts.get("test"), str):
                        runner = ""
                except (OSError, ValueError, AttributeError):
                    runner = ""
            return language, runner
        try:
            if any(entry.endswith((".csproj", ".sln")) for entry in os.listdir(directory)):
                return "csharp", ""
        except OSError:
            pass
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return (declared_languages.pop() if len(declared_languages) == 1 else "unknown"), ""

STATED_RUNNER_TOOLS = frozenset((
    "go", "cargo", "mvn", "gradle", "gradlew", "npm", "yarn", "pnpm",
    "make", "dotnet", "ctest", "rake", "bundle", "rspec", "phpunit"))
STATED_RUNNER_WORDS = frozenset(("test", "tests", "rspec", "phpunit", "ctest"))

def stated_command_words(line: str) -> list:
    """One command line split into words the way a shell would, keeping quoted text together."""
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = ""
            elif quote == '"' and (char == "`" or
                    (char == "$" and line[index + 1:index + 2] != '"')):
                return []
        elif char in "\"'":
            quote = char
        elif char == "#":
            break
        elif RUNNER_UNMODELLED.search(char) or RUNNER_CHAIN.search(char):
            return []
    try:
        return shlex.split(line, comments=True)
    except ValueError:
        return []

def stated_check(argv: list) -> bool:
    """Is this argv a check this run may run on the project's behalf?

    An environment prefix and a wrapper such as a package runner are looked through to the
    real runner, because what matters is the program that ends up running.
    """
    if not argv:
        return False
    tool = os.path.basename(argv[0])
    # Preserve an explicitly supplied Python runner, including its environment.
    if tool in ("uv", "poetry") and len(argv) > 2 and argv[1] == "run":
        return stated_check(argv[2:])
    if ASSIGNMENT.match(argv[0]):
        # An environment prefix names the environment, not the runner.
        return stated_check(strip_assignments(argv))
    if tool in ("pytest", "py.test"):
        return True
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", tool):
        rest = argv[1:]
        while rest and rest[0] in ("-B", "-u", "-I", "-E", "-s"):
            rest = rest[1:]
        return bool((len(rest) >= 2 and rest[0] == "-m"
                     and rest[1] in ("pytest", "unittest"))
                    or (len(rest) >= 2 and os.path.basename(rest[0]) == "manage.py"
                        and rest[1] == "test")
                    # A script the statement names as the check: "python3 smoke.py".
                    or (rest and rest[0].endswith(".py")
                        and re.search(r"smoke|test", os.path.basename(rest[0]), re.I)))
    if tool == "node":
        # A test runner the statement names through node: a test file run by
        # tsx, or node's own --test.
        rest = argv[1:]
        return bool(rest and (any(word.endswith(("tsx/dist/cli.mjs", "/tsx", "/vitest", "/jest", "/mocha")) for word in rest)
                              or "--test" in rest)
                    and any("test" in os.path.basename(word).lower() for word in rest))
    if tool not in STATED_RUNNER_TOOLS:
        return False
    if tool in ("ctest", "rspec", "phpunit"):
        return True
    operands = {"-C", "--directory", "-f", "--file", "--manifest-path",
                "--project", "-p", "--prefix", "--cwd", "-c", "--configuration",
                "-s", "--settings", "-I", "--init-script"}
    actions = []
    skip = False
    for word in argv[1:]:
        if skip:
            skip = False
        elif word in operands:
            skip = True
        elif not word.startswith("-"):
            actions.append(word)
    if tool in ("go", "cargo", "dotnet"):
        return bool(actions and actions[0] == "test")
    if tool in ("npm", "pnpm", "yarn"):
        if actions and actions[0] == "run":
            actions = actions[1:]
        return bool(actions and actions[0] in STATED_RUNNER_WORDS)
    if tool == "bundle":
        return bool(len(actions) > 1 and actions[0] == "exec"
                    and actions[1] in STATED_RUNNER_WORDS)
    return bool(set(actions) & STATED_RUNNER_WORDS)

# The shell this run is willing to reproduce from a statement, and no more:
# an optional `cd <relative dir>` followed by one recognised runner. Anything
# else is reported as not understood rather than reinterpreted.
RUNNER_SHAPED = re.compile(
    r"manage\.py\s+test\b|\bpytest\b|\bpy\.test\b|-m\s+unittest\b|"
    r"\bgo\s+test\b|\bcargo\s+test\b|\brspec\b|\bphpunit\b|\bctest\b")

def runner_segments(line: str, root: str = "") -> list | None:
    """The supported subset, as segments to run, or None if unsupported."""
    # Find control operators before removing quotes. shlex.split removes the
    # distinction between a literal quoted '&&' argument and a real operator.
    line = line or ""
    pieces, start, quote, escaped, index = [], 0, "", False, 0
    while index < len(line):
        char = line[index]
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "#":
            # The existing supported subset uses shell comments. Let its word
            # parser handle them; operators in comments are never executable.
            break
        elif line[index:index + 2] == "&&":
            pieces.append(line[start:index].strip())
            index += 1
            start = index + 1
        index += 1
    pieces.append(line[start:].strip())
    if len(pieces) > 2 or not pieces[-1]:
        return None
    runner = stated_command_words(pieces[-1])
    if not stated_check(runner):
        return None
    if len(pieces) == 1:
        return [runner]
    prefix = stated_command_words(pieces[0])
    if len(prefix) != 2 or prefix[0] != "cd":
        return None
    target = prefix[1]
    if os.path.isabs(target) or ".." in target.replace("\\", "/").split("/"):
        return None
    if root:
        where = os.path.join(root, target)
        if not inside(where, root) or not os.path.isdir(where):
            return None
    return [prefix, runner]

def render_word(word: str) -> str:
    """Quote a word for the shell, keeping VAR=value an assignment.

    Quoting the whole word would make `VAR=two words` a command name.
    """
    if ASSIGNMENT.match(word):
        name, _, value = word.partition("=")
        return "%s=%s" % (name, shlex.quote(value))
    return shlex.quote(word)

def render_segments(segments: list) -> str:
    """Command segments written back as one line, joined the way the statement joined them."""
    return " && ".join(" ".join(render_word(word) for word in segment)
                       for segment in segments)

BARE_RUN = re.compile(r"(?:\b(?:Run|Execute|Use)\s+|\b(?:(?:public\s+|visible\s+)?tests?|typecheck|compile|build|lint|setup|smoke|check):\s*)"
                      r"((?:python[\d.]*|pytest|py\.test|npm|yarn|pnpm|go|cargo|dotnet|node|make)\b"
                      r"[^`;:!?\n]*?)(?=\s*(?:[.;:!?](?:\s|$)|$))", re.I)


# Where the sentence goes on in prose after the command: "Run python -m unittest
# discover -s tests before you finish."
BARE_TAIL = re.compile(r"\s+(?:before|after|for|to|and|then|so|which|when|while|until|in|on|with|from|as|"
                       r"because|if|once|the|an?|is|are)(?=\s|$)", re.I)


def bare_commands(statement: str) -> list[str]:
    """Commands a sentence spells out without code marks: "Run python -m unittest discover -s tests." """
    found = []
    for part in instruction_clauses(statement):
        for match in BARE_RUN.finditer(part):
            command = " ".join(BARE_TAIL.split(match.group(1), 1)[0].split())
            if command and command not in found:
                found.append(command)
    return found


def stated_runner(statement: str, root: str = "") -> str:
    """The command the statement says to run before finishing, as this run will run it.

    A command this run cannot model is reported in the log rather than run, so nothing is
    executed on a guess at its meaning.
    """
    unsupported = []
    for block in statement_command_blocks(statement) + bare_commands(statement):
        for line in logical_lines(block or ""):
            segments = runner_segments(line, root)
            if segments:
                return render_segments(segments)
            if RUNNER_SHAPED.search(line or ""):
                unsupported.append(one_line(line)[:160])
    for line in unsupported[:2]:
        say("[RUNNER] not understood, so it will not be run as the baseline "
            "check: %s" % line)
    return ""

def native_import_lines(source: str, language: str) -> set:
    """The import lines of a file in a language other than Python, by that language's own syntax.
    """
    starts = {
        "go": r'^\s*import\s+(?:\(|(?:[\w.]+\s+)?["`])',
        "node": r'^\s*import\s+(?!\()[\w*{\'\"]',
        "java": r'^\s*import\s+(?:static\s+)?[\w.]',
        "rust": r'^\s*(?:pub(?:\([^)]*\))?\s+)?use\s+[\w:{]',
        "csharp": r'^\s*(?:global\s+)?using\s+(?:static\s+)?[\w.]+(?:\s*=\s*[\w.<>,\s]+)?\s*;',
        "kotlin": r'^\s*import\s+[\w.]+(?:\.\*)?(?:\s+as\s+\w+)?\s*$',
    }
    pattern = starts.get(language)
    if not pattern:
        return set()
    masked = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`[^`]*`',
                    lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]),
                    source, flags=re.S)
    lines, code = source.splitlines(), masked.splitlines()
    found, active, depth = set(), False, 0
    for number, (line, clean) in enumerate(zip(lines, code), 1):
        if not active:
            if not re.match(pattern, line) or not re.search(r'\b(?:import|use|using)\b', clean):
                continue
            active = True
        found.add(number)
        if language in ("csharp", "kotlin"):
            # One statement per line in both languages' import forms.
            active, depth = False, 0
            continue
        depth += sum(clean.count(c) for c in '({[') - sum(clean.count(c) for c in ')}]')
        if depth <= 0 and (language == "go" or ';' in clean or
                           (language == "node" and re.search(r'[\'\"][^\'\"]+[\'\"]\s*;?\s*$', line))):
            active, depth = False, 0
    return found

def import_lines(source: str, language: str | None = None) -> set:
    """The line numbers of a file's imports, so a change that touches them can be noticed."""
    if RIDGES_SCOPE_FOLLOWS_STATEMENT and language not in (None, "python"):
        return native_import_lines(source, language)
    try:
        tree = ast.parse(source or "")
    except (SyntaxError, ValueError):
        if RIDGES_SCOPE_FOLLOWS_STATEMENT and language is not None:
            return set()
        return {number for number, text in enumerate((source or "").splitlines(), 1)
                if FENCE_IMPORT_LINE.match(text)}
    lines: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.update(range(node.lineno,
                               getattr(node, "end_lineno", node.lineno) + 1))
    return lines

def hunk_spans(patch: str) -> list[tuple[int, int, int, int]]:
    """Every hunk of a patch as (old start, old count, new start, new count)."""
    spans = []
    for old, old_count, new, new_count in FENCE_HUNK.findall(patch or ""):
        spans.append((int(old), 1 if old_count == "" else int(old_count),
                      int(new), 1 if new_count == "" else int(new_count)))
    return spans

def touches(span: tuple[int, int], numbers: set) -> bool:
    """Does this hunk span cover any of these line numbers?"""
    start, count = span
    return any(number in numbers for number in range(start, start + count))

class Warden:
    """The part of the run that reads the task's own words and holds the answer to them.

    It finds the file and the method the statement names, runs the project's own check
    command, reads what changed against what the statement allows, and refuses a hand-in
    that falls outside it. Every refusal quotes the clause it came from, so the run is held
    to the task's words rather than to a rule of its own.
    """
    def __init__(self, tree: Tree, pool: ShellPool, allowance: Allowance,
                 statement: str = "") -> None:
        """Read the statement: the file it names, the runner it names and the scope it allows.
        """
        self.tree = tree
        self.tree.statement = statement
        self.pool = pool
        self.allowance = allowance
        self.declared = declared_file(statement, tree.root)
        self.statement = statement
        self.native_runner = project_contract(tree.root)[1] if RIDGES_SCOPE_FOLLOWS_STATEMENT else ""
        self.runner = (stated_runner(statement, tree.root)
                       if RIDGES_SCOPE_FOLLOWS_STATEMENT else "")
        self.no_baseline_reported = False
        if RIDGES_SCOPE_FOLLOWS_STATEMENT:
            scope = Beacon("scope_statement")
            scope.reached(0, 0.0, 0.0)
            original = scope.artefact("before", "imports=all shape=all paths=closed runner=pytest")
            scope.skipped("no model call needed to read scope and manifests")
            rules = "bounded=%s runner=%s" % (
                statement_bounded(statement),
                self.runner or ("none" if self.native_runner else "pytest"))
            scope.fired(rules)
            scope.outcome(original, rules)
            scope.bill()
        self.ledger = Beacon("ledger")
        self.read_back_calls: int | None = None
        self.read_backs = 0
        self.records: list = []
        self.beacon = Beacon("warden")
        self.named = Beacon("namedchecks")
        self.contract = Beacon("contract")
        self.fence = Beacon("editfence")
        self.fence_asks = 0
        self.fence_read = False
        self.contract_did_read = False
        self.read_until: float | None = None
        self.job: Shell | None = None
        self.started = time.time()
        self.before: tuple[set, int] | None = None
        self.refusals = 0
        self.stood_down = False
        self.where: str | None = None
        self.tier = 0
        self.scope: list[str] = []
        self.armed_after: float | None = None
        self.counted = True
        self.shims: list = []
        self.shim_dir = ""
        self.stood_in = 0

    def lane_start(self, command: str, hard_timeout: float | None = None,
                   wait: float | None = None) -> "Shell | None":
        """Start a suite reading, waiting for the database rather than racing.

        Returns None when the lane stays busy. Not being able to take a reading
        is a missing reading; it is not a reason to run a second suite against
        the same database.
        """
        forbidden = execution_prohibited(getattr(self, "statement", ""))
        if forbidden:
            self.beacon.skipped("automatic suite execution withheld: %s" % forbidden)
            return None
        room = self.allowance.clock_left() - WARDEN_RELEASE_SEC
        patience = room if wait is None else wait
        patience = max(0.0, min(patience, SUITE_RECHECK_SEC))
        try:
            return self.pool.start(command, pack_venv=False,
                                   hard_timeout=hard_timeout,
                                   lane=SUITE_LANE, lane_wait=patience)
        except LaneBusy:
            self.beacon.skipped("a suite is still running against the same "
                                "database; no reading taken")
            return None

    def suite_command(self, root: str | None = None) -> str:
        """The command that runs the project's own checks, in its own runner or through Python.
        """
        root = root or self.tree.root
        runner = getattr(self, "runner", "") if RIDGES_SCOPE_FOLLOWS_STATEMENT else ""
        if runner:
            return ("cd %s && ( %s; scope_status=$?; "
                    "printf '\nRIDGES_NATIVE_STATUS=%%s\n' \"$scope_status\"; exit \"$scope_status\" )"
                    % (shlex.quote(root), runner))
        path = os.pathsep.join(package_roots(root) + ([self.shim_dir] if self.shim_dir else []))
        where = " ".join(shlex.quote(rel) for rel in self.scope)
        pyc = tempfile.mkdtemp(prefix="pyc")
        if inside(pyc, self.tree.root):
            shutil.rmtree(pyc, ignore_errors=True)
            cache = "PYTHONDONTWRITEBYTECODE=1"
        else:
            cache = "PYTHONPYCACHEPREFIX=%s" % shlex.quote(pyc)
            self.pool.own_cleanup(lambda path=pyc: shutil.rmtree(path, ignore_errors=True))
        return (
            "cd %s && PYTHONPATH=%s PYTHONHASHSEED=0 %s %s -m pytest -q "
            "--no-header --tb=line -rfE -p no:cacheprovider -o addopts= "
            "--continue-on-collection-errors -W ignore::DeprecationWarning%s%s"
            % (shlex.quote(root), shlex.quote(path), cache,
               shlex.quote(repo_python()), SUITE_TIERS[self.tier],
               (" " + where) if where else "")
        )

    def ensure_pristine(self) -> tuple:
        """A separate checkout of the base commit, for readings of the original code.

        Reuses the checkout the baseline suite ran in when there is one; makes
        one the same way otherwise. The live tree is never the source of a
        baseline reading.
        """
        for where in (self.where, getattr(self, "measure_where", None)):
            if where and os.path.isdir(where):
                return where, "separate checkout of the base commit (reused)"
        if not self.tree.base:
            return None, "the checkout has no base commit to check out separately"
        try:
            where = self.pristine()
        except BaseException as error:
            return None, "no separate checkout: %s" % type(error).__name__
        if where is None:
            return None, "no separate checkout could be created (see the warden log)"
        self.measure_where = where
        return where, "separate checkout of the base commit"

    def pristine(self) -> str | None:
        """A separate checkout of the base commit, for comparing against the code as it was."""
        owner = tempfile.mkdtemp(prefix="start")
        where = os.path.join(owner, "tree")
        if inside(where, self.tree.root):
            shutil.rmtree(os.path.dirname(where), ignore_errors=True)
            self.beacon.skipped("the only place for a separate checkout is "
                                "inside the tree being handed in")
            return None
        code, out = git(["worktree", "add", "--detach", where, self.tree.base or "HEAD"],
                        self.tree.root, 60)
        if code != 0:
            self.beacon.skipped("no separate checkout to read: %s" % out.strip()[:120])
            shutil.rmtree(owner, ignore_errors=True)
            return None
        # A caller may already have its own worktrees. The exact path created
        # here is the only worktree this callback may remove.
        def remove_owned_checkout():
            code, _ = git(["worktree", "remove", "--force", where], self.tree.root, 10)
            if code == 0 or not os.path.lexists(where):
                shutil.rmtree(owner, ignore_errors=True)
        self.pool.own_cleanup(remove_owned_checkout)
        return where

    def arm(self) -> None:
        """Start the project's own check in the background, so its result is ready when it is
        needed.
        """
        if not SUBMISSION_WARDEN:
            self.beacon.skipped("not switched on for this run")
            return
        self.beacon.reached(0, self.allowance.spent, self.allowance.clock_left())
        # A documented command is not authorization to run it. The automatic
        # suite cannot establish that arbitrary project tests avoid the database;
        # leave independent static work and explicitly chosen checks available.
        forbidden = execution_prohibited(self.statement)
        if forbidden:
            self.beacon.skipped("automatic baseline withheld: %s" % forbidden)
            self.report_no_baseline()
            return
        if self.native_runner and not self.runner:
            self.report_no_baseline()
            return
        try:
            # A whole test directory is not evidence that its tests concern
            # this task. Do not occupy the shared lane with that guess before
            # the solver has chosen a useful check.
            self.scope = (suite_scope(self.tree.root, self.declared, specific=True)
                          if SUITE_SCOPE else [])
            if SUITE_READABLE:
                self.scope, note = readable_scope(self.tree.root, self.scope)
                if note:
                    say("[" + Beacon.tag("warden") + "] " + note)
            if STATED_BASELINE:
                named = stated_test_scope(self.statement, self.tree.root)
                if named:
                    self.scope = named
                    say("[" + Beacon.tag("warden") + "] the statement names a "
                        "runnable scope: %s" % ", ".join(named[:4]))
            if not self.runner and not self.scope:
                self.beacon.skipped("automatic baseline has no explicit runner "
                                    "or specifically localized test scope")
                self.report_no_baseline()
                return
            room = self.allowance.clock_left() - WARDEN_RELEASE_SEC
            if room < 5.0 or self.allowance.money_left() <= 0:
                self.beacon.skipped("too little allowance for an automatic baseline")
                self.report_no_baseline()
                return
            self.where = self.pristine()
            if self.where is None:
                self.report_no_baseline()
                return
            say("[" + Beacon.tag("warden") + "] baseline scope: %s"
                % (self.runner or ", ".join(self.scope)))
            room = self.allowance.clock_left() - WARDEN_RELEASE_SEC
            if room < 5.0:
                self.beacon.skipped("too little time after creating the baseline checkout")
                self.report_no_baseline()
                return
            self.job = self.lane_start(self.suite_command(self.where),
                                       hard_timeout=min(SUITE_BASELINE_SEC, room),
                                       wait=min(LANE_GRACE_SEC, room))
        except Exception as error:
            self.beacon.skipped("could not start the baseline reading: %s" % error)

    def escalate(self) -> bool:
        """Move up to a wider way of running the project's tests when the narrower one found
        nothing.
        """
        if RIDGES_SCOPE_FOLLOWS_STATEMENT and (self.runner or self.native_runner):
            return False
        if self.where is None:
            return False
        top = len(SUITE_TIERS) - 1 if SUITE_IMPORTLIB else len(SUITE_TIERS) - 2
        if self.tier >= top:
            if self.tier < len(SUITE_TIERS) - 1:
                self.beacon.skipped("this rung is not switched on for this run")
            return False
        if self.allowance.clock_left() < RUNG_MIN_WALL_SEC:
            self.beacon.skipped("too little of the run left for a second reading")
            return False
        self.tier += 1
        self.beacon.fired("nothing in the project's suite passed; asking again, "
                          "rung %d of %d"
                          % (self.tier + 1, len(SUITE_TIERS)))
        try:
            self.job = self.lane_start(self.suite_command(self.where),
                                       hard_timeout=SUITE_BASELINE_SEC,
                                       wait=LANE_GRACE_SEC)
        except Exception as error:
            self.beacon.skipped("could not start the second reading: %s" % error)
            return False
        return self.job is not None

    def stand_in(self, out: str) -> bool:
        """Write importable stand-ins for modules the project's own tests need and the host lacks.

        Without them a whole test file fails to import, which says nothing about the
        change; with them the tests that do not need the absent module still run.
        """
        if self.runner:
            return False
        if not SUITE_SHIM or self.where is None:
            if not SUITE_SHIM:
                self.beacon.skipped("standing in is not switched on for this run")
            return False
        wanted = [n for n in missing_modules(out) if n not in self.shims]
        records = [n for n in missing_dists(out) if n not in self.records]
        room_left = max(0, SUITE_SHIM_LIMIT - len(self.shims) - len(self.records))
        if (wanted or records) and not room_left:
            self.beacon.skipped("%d stand-in(s) over %d attempt(s) and the "
                                "reading still names more"
                                % (len(self.shims + self.records), self.stood_in))
            return False
        records_first = len(self.records) <= len(self.shims)
        first, second = ((records, wanted) if records_first
                         else (wanted, records))
        share: list = []
        for step in range(max(len(first), len(second))):
            if step < len(first):
                share.append((records_first, first[step]))
            if step < len(second):
                share.append((not records_first, second[step]))
        share = share[:room_left]
        records = [name for is_record, name in share if is_record]
        wanted = [name for is_record, name in share if not is_record]
        if not wanted and not records:
            return False
        if self.allowance.clock_left() < STANDIN_MIN_WALL_SEC:
            self.beacon.skipped("too little of the run left to read the suite again")
            return False
        if not self.shim_dir:
            try:
                room = tempfile.mkdtemp(prefix="standin")
            except OSError as error:
                self.beacon.skipped("nowhere to write a stand-in: %s" % error)
                self.shim_dir = ""
                return False
            if inside(room, self.tree.root):
                shutil.rmtree(room, ignore_errors=True)
                self.beacon.skipped("the only place to write a stand-in is inside "
                                    "the tree being handed in")
                self.shim_dir = ""
                return False
            self.shim_dir = room
            self.pool.own_cleanup(lambda path=room: shutil.rmtree(path, ignore_errors=True))
        made = write_shims(wanted, self.shim_dir)
        kept = write_dist_records(records, self.shim_dir)
        if not made and not kept:
            self.beacon.skipped("could not write a stand-in for %s"
                                % one_line(", ".join(wanted + records))[:80])
            return False
        told = []
        if made:
            told.append("does not carry " + one_line(", ".join(made))[:90])
        if kept:
            told.append("has no installed record of " + one_line(", ".join(kept))[:90])
        self.beacon.fired("the image %s; standing in for it and reading again"
                          % " and ".join(told))
        self.tier = 0
        try:
            self.job = self.lane_start(self.suite_command(self.where),
                                       hard_timeout=SUITE_BASELINE_SEC,
                                       wait=LANE_GRACE_SEC)
        except Exception as error:
            self.beacon.skipped("could not start the reading again: %s" % error)
            return False
        if self.job is None:
            return False
        self.stood_in += 1
        self.shims.extend(made)
        self.records.extend(kept)
        return True

    def settle(self) -> None:
        # The project's own suite is waited for until it finishes or runs out
        # of its allowance, not for one window: whether it is still running
        # when the answer is ready depends on the host's speed, and the checks
        # a hand-in gets should not.
        """Wait for the project's own check to finish, so a hand-in is judged against a real
        result.

        It waits for the check's own allowance rather than one window, because whether
        it is still running when the answer is ready depends on the host's speed and
        should not change what the answer is held to.
        """
        for _ in range((1 + SUITE_SHIM_LIMIT) * len(SUITE_TIERS) + 8):
            self.collect()
            if self.before is not None or self.job is None:
                return
            room = self.allowance.clock_left() - WARDEN_RELEASE_SEC
            window = min(SUITE_SETTLE_SEC, room)
            if window < 5.0:
                break
            self.job.wait(window)
            if (not self.job.finished()
                    and time.time() - self.job.started > SUITE_BASELINE_SEC):
                break
        self.collect()

    def collect(self) -> None:
        """Read the project's check if it has finished, and record what passed and what failed.
        """
        if self.job is None or self.before is not None:
            if self.before is None:
                self.report_no_baseline()
            return
        if not self.job.finished():
            if time.time() - self.job.started > SUITE_BASELINE_SEC:
                self.job.stop()
                self.pool.jobs.pop(self.job.name, None)
                self.job = None
                self.beacon.skipped("the project's tests did not finish in the "
                                    "time this rung allows")
            return
        job = self.job
        done, out = job.wait(0.5)
        self.pool.jobs.pop(job.name, None)
        code = job.process.poll()
        job.stop()
        self.job = None
        reading = self.read_suite(out, code)
        if not reading[1] and self.escalate():
            return
        if (not reading[1] and
                not (RIDGES_SCOPE_FOLLOWS_STATEMENT and getattr(self, "native_runner", ""))
                and self.stand_in(out)):
            return
        if not reading[1]:
            self.report_no_baseline()
            self.beacon.skipped("the project's tests produced no usable "
                                "baseline at tier %d: %s -- %s"
                                % (self.tier + 1, self.tally(out), self.reason(out)))
            return
        propped = len(self.shims) + len(self.records)
        self.counted = True
        if propped:
            self.counted = False
            self.beacon.skipped("dependency stand-ins are diagnostic only; no baseline recorded")
            return
        self.before = reading
        self.armed_after = time.time() - self.started
        say("[" + Beacon.tag("warden") + "] the project's tests at the start: %d failing, %d passing "
            "(tier %d, scope %s, %.0fs)"
            % (len(self.before[0]), self.before[1], self.tier + 1,
               ",".join(self.scope) or "repository", self.armed_after))

    @staticmethod
    def tally(out: str) -> str:
        """The last count line a check printed, or a note that it printed none."""
        found = SUITE_TALLY.findall(out or "")
        return found[-1].strip() if found else "no tally"

    @staticmethod
    def reason(out: str) -> str:
        """The first reason a check gave for failing, and whether there were other kinds."""
        faults = SUITE_FAULT.findall(out or "")
        if faults:
            kinds = len({name for name, _ in faults})
            said = ("%s%s" % faults[0]).strip()[:140]
            return said if kinds == 1 else "%s (+%d other kinds)" % (said, kinds - 1)
        found = SUITE_REASON.findall(out or "")
        return found[0].strip()[:160] if found else "no reason given"

    @staticmethod
    def read_suite(out: str, returncode: int | None = None) -> tuple[set, int]:
        """The failing test names and the passing count from one check's output."""
        failures = set(FAILED_TEST.findall(out or ""))
        outcome, detail = check_outcome(out, exit_code=(... if returncode is None
                                                       else returncode))
        passing = 0
        if outcome in ("ran", "failed") and OUTPUT_INCOMPLETE_MARKER not in (out or ""):
            # Share the result parser's verdict before interpreting counts.
            # A package result has no test count; only verbose Go test records
            # can provide one. Cached/no-test summaries cannot be a baseline.
            go = re.search(r"^\s*--- (?:PASS|FAIL|SKIP):|^ok[ \t]+\S+[ \t]+|"
                           r"^FAIL[ \t]+\S+[ \t]+(?:\d|\[)", out or "", re.M)
            if go:
                passing = len(re.findall(r"^--- PASS: \S+", out or "", re.M))
                failures.update(re.findall(r"^--- FAIL: (\S+)", out or "", re.M))
            elif PASSED_COUNT.search(detail):
                passing = int(PASSED_COUNT.findall(detail)[-1])
            elif RAN_TESTS.search(out or ""):
                passing = int(RAN_TESTS.findall(out)[-1])
                verdicts = re.findall(r"^(?:OK|FAILED) \(([^\n]*)\)\s*$", out, re.M)
                if verdicts:
                    excluded = sum(int(value) for value in re.findall(
                        r"(?:failures|errors|skipped)=(\d+)", verdicts[-1]))
                    passing = max(0, passing - excluded)
        if returncode not in (None, 0):
            failures.add("suite-exit-%d" % returncode)
        if RIDGES_SCOPE_FOLLOWS_STATEMENT:
            status = re.findall(r"^RIDGES_NATIVE_STATUS=(\d+)$", out or "", re.M)
            if status and status[-1] != "0":
                failures.add("native-suite")
        return failures, passing

    def report_no_baseline(self) -> None:
        """Say once that there is no baseline result to compare a hand-in against."""
        if (RIDGES_SCOPE_FOLLOWS_STATEMENT and getattr(self, "before", None) is None
                and not getattr(self, "no_baseline_reported", False)):
            self.beacon.skipped("no baseline: watched=0")
            self.no_baseline_reported = True

    def suite_faults(self) -> list[str]:
        """Tests that pass on the base commit and fail on the code as it stands, which is a
        regression.
        """
        if self.before is None:
            self.report_no_baseline()
            return []
        room = self.allowance.clock_left() - WARDEN_RELEASE_SEC
        if room < 30.0:
            return []
        job = self.lane_start(self.suite_command())
        if job is None:
            return []
        done, out = job.wait(min(SUITE_RECHECK_SEC, room))
        self.pool.jobs.pop(job.name, None)
        if not done:
            job.stop()
            self.beacon.skipped("the project's tests did not finish in the time left")
            return []
        broke, passing = self.read_suite(out, job.process.poll())
        job.stop()
        fresh = self.confirm(sorted(broke - self.before[0]))
        if fresh:
            say("[" + Beacon.tag("warden") + "] the refusal %s the question"
                % ("carries" if WARDEN_ASK else "does not carry"))
            # The comparison identifies new failure reports, not causation.
            # External state, order, and flaky tests can also change outcomes.
            return ["The project's recheck reports failures not listed among "
                    "the baseline failures: %s. These observations alone do "
                    "not establish whether the edit or another condition "
                    "caused the difference. Investigate the failures. Unless "
                    "the task asked for that behaviour "
                    "to change, fix the behaviour rather than the test; if it "
                    "did ask, say which clause and carry on.%s"
                    % (", ".join(fresh[:6]), WARDEN_QUESTION if WARDEN_ASK else "")]
        if not passing:
            outcome, detail = check_outcome(out, self.suite_command(), job.process.poll())
            if self.counted and self.before[1] and outcome in ("none_ran", "skipped_only"):
                return ["The project's suite ran %d test(s) when this run "
                        "started and runs none now (%s). A suite that selects "
                        "no test proves nothing about behaviour: run the tests "
                        "the task names, and leave them selectable."
                        % (self.before[1], detail)]
            if (not broke and SUITE_FAULT.search(out or "")
                    and not SUITE_FAILURES_HEAD.search(out or "")):
                return ["The project's suite produced a reading at the start "
                        "of this run (%d passing) and produces none now: %s. "
                        "A suite that no longer even starts proves nothing "
                        "about behaviour; make the project import cleanly "
                        "again." % (self.before[1], self.reason(out))]
            self.beacon.skipped("the re-reading produced no usable count: %s -- %s"
                                % (self.tally(out), self.reason(out)))
            return []
        if self.counted and passing < self.before[1]:
            return ["The project's suite reported %d passing tests at the start "
                    "of this run and %d now. A suite that got smaller is not "
                    "evidence that behaviour was kept: a skipped or deselected "
                    "test proves nothing." % (self.before[1], passing)]
        return []

    def named_check_faults(self) -> list[str]:
        """What a linter the statement names reports on the change and not on the unchanged tree.

        Each check runs as the statement wrote it, in the tree being handed in;
        only when it reports something does it run again in a separate checkout
        of the base commit, so a finding the tree already had is not the
        change's. A check that cannot run here, or does not finish, gives no
        reading rather than a verdict.
        """
        statement = getattr(self, "statement", "")
        if not NAMED_CHECKS or not statement:
            return []
        checks = named_checks(statement)
        if not checks:
            return []
        beacon = getattr(self, "named", None) or Beacon("namedchecks")
        forbidden = execution_prohibited(statement)
        if forbidden:
            beacon.skipped("the named checks are withheld: %s" % forbidden)
            return []
        faults = []
        for command in checks[:NAMED_CHECK_LIMIT]:
            now = self.named_reading(command, self.tree.root, beacon)
            if now is None or (now[0] == 0 and not now[1].strip()):
                continue
            where, why = self.ensure_pristine()
            if not where:
                beacon.skipped("no reading of the unchanged tree for %s: %s" % (command, why))
                continue
            before = self.named_reading(command, where, beacon)
            if before is None:
                continue
            fresh = new_findings(now[1], before[1])
            if fresh and not finding_keys(now[1]) and now[0] == 0:
                fresh = []  # a clean exit with a different summary is not a finding
            if not fresh and before[0] == 0 and now[0] != 0:
                fresh = [clip(redact(now[1].strip()), 1200, "check output")]
            if not fresh:
                continue
            beacon.fired("%s: %d finding(s) the unchanged tree does not have" % (command, len(fresh)))
            listed = "\n".join("  - " + item for item in fresh[:NAMED_CHECK_QUOTED])
            more = len(fresh) - NAMED_CHECK_QUOTED
            faults.append(
                "The task names this check: `%s`. On the change it reports %s the unchanged "
                "checkout does not:\n%s%s\nFix what the change introduced, and run the check "
                "again before handing in." % (
                    command, "this, which" if len(fresh) == 1 else "these, which", listed,
                    "\n  (%d more)" % more if more > 0 else ""))
        return faults

    def go_reading(self, paths: list, budget: float) -> tuple[list, str]:
        """Comparable new Go diagnostics and unresolved compiler observations.

        Current diagnostics are not classified as new without a usable base
        reading. The base commit is compiled only when the change's package
        fails, in a separate checkout, once per package per run.
        """
        deadline = time.monotonic() + budget
        fresh: list = []
        unread = ""
        for module, package in go_packages(self.tree.root, paths)[:GO_PACKAGE_LIMIT]:
            room = deadline - time.monotonic()
            if room < 3.0:
                unread = unread or "no time was left to compile %s" % package
                break
            status, out = go_compile_output(self.tree.root, module, package, room)
            if status == "timeout":
                unread = unread or "compiling %s did not finish in %ds" % (package, int(room))
                continue
            if status == "unverified":
                unread = unread or "compiling %s gave no reading (%s)" % (
                    package, one_line(out)[:160] or "no output")
                continue
            if status != "errors":
                continue
            before = self.go_base_output(module, package, deadline - time.monotonic())
            if before is None:
                # Current errors do not establish a regression when the
                # unchanged package has no usable compiler observation.
                diagnostics = [line for line in out.splitlines() if GO_DIAGNOSTIC.match(line)]
                note = ("the unchanged package %s could not be checked; current compiler "
                        "diagnostics (comparison unverified):\n%s" % (
                            package, clip(redact("\n".join(diagnostics[:GO_COMPILE_QUOTED])),
                                          1200, "compiler diagnostics")))
                unread = clip("\n".join(part for part in (unread, note) if part),
                              2400, "unresolved compile checks")
            else:
                fresh += fresh_lines(out, before)
        return fresh, unread

    def go_base_output(self, module: str, package: str, budget: float) -> str | None:
        """What compiling this package printed at the base commit, or None if unknown."""
        cache = getattr(self, "go_base", None)
        if cache is None:
            cache = self.go_base = {}
        if (module, package) in cache:
            return cache[(module, package)]
        where, _ = self.ensure_pristine()
        if not where or budget < 3.0:
            return None
        status, out = go_compile_output(where, module, package, budget)
        cache[(module, package)] = out if status in ("clean", "errors") else None
        return cache[(module, package)]

    def go_compile_faults(self) -> list[str]:
        """The change's Go packages compile with errors the unchanged packages do not have."""
        if not GO_COMPILE or execution_prohibited(getattr(self, "statement", "")):
            return []
        paths = [path for path in self.changed_paths() if path.endswith(".go")]
        room = min(GO_COMPILE_HANDIN_SEC, self.allowance.clock_left() - WARDEN_RELEASE_SEC)
        if not paths or room < 10.0:
            return []
        fresh, unread = self.go_reading(paths, room)
        if unread:
            self.beacon.skipped("Go compile: %s" % unread)
        if not fresh:
            return []
        self.beacon.fired("Go compile: %d diagnostic(s) the unchanged packages do not have" % len(fresh))
        return ["Additional Go compiler diagnostics compared with the checked baseline "
                "(compiled with tests; no test was run):\n%s\n"
                "These observations do not establish the cause. Inspect the diagnostics "
                "against the current task.%s" % (
                    "\n".join("  " + redact(line) for line in fresh[:GO_COMPILE_QUOTED]),
                    "\nOther compile comparisons remain unverified: " + unread if unread else "")]

    def named_reading(self, command: str, where: str, beacon: "Beacon") -> tuple | None:
        """The exit status and output of one named check run in `where`, or None."""
        room = self.allowance.clock_left() - WARDEN_RELEASE_SEC
        if room < 10.0:
            beacon.skipped("too little of the run left to run %s" % command)
            return None
        limit = min(NAMED_CHECK_SEC, room)
        job = self.pool.start(command, hard_timeout=limit + 5.0, cwd=where)
        try:
            done, out = job.wait(limit)
            code = job.process.poll()
        finally:
            job.stop()
            self.pool.jobs.pop(job.name, None)
        if not done:
            beacon.skipped("%s did not finish in %ds" % (command, int(limit)))
            return None
        failure = execution_failure(code, out or "", checker=True)
        words = shlex.split(command)
        module = words[2] if len(words) > 2 and words[1] == "-m" else ""
        if failure in ("missing_or_unusable_executable", "unsupported_checker_configuration") or (
                code and module and re.search(r"No module named '?%s\b" % re.escape(module), out or "")):
            beacon.skipped("%s cannot run here (%s)" % (command, failure))
            return None
        # A checker that prints absolute paths names the tree it ran in; the two
        # readings are compared relative to their own checkout.
        text = out or ""
        for prefix in dict.fromkeys((os.path.realpath(where), os.path.abspath(where))):
            text = text.replace(prefix.rstrip(os.sep) + os.sep, "")
        return code, text

    def confirm(self, names: list[str]) -> list[str]:
        """Re-run named failures on their own, so a failure caused by another test is not blamed on
        the change.
        """
        if not names:
            return []
        room = self.allowance.clock_left() - WARDEN_RELEASE_SEC
        if room < 20.0:
            return names
        asked = names[:CONFIRM_MAX]
        cut = {n: n.split("[")[0] for n in asked
               if "[" in n and not n.endswith("]")}
        scope = [cut.get(n, n) for n in asked]
        command = self.suite_command()
        if not (RIDGES_SCOPE_FOLLOWS_STATEMENT and (self.runner or self.native_runner)):
            # Exit failures describe the whole command, not pytest node IDs.
            scope = [n for n in scope if not n.startswith("suite-exit-")]
            command = "%s %s" % (command, " ".join(shlex.quote(n) for n in scope))
        job = self.lane_start(command)
        if job is None:
            return names
        done, out = job.wait(min(SUITE_RECHECK_SEC, room))
        self.pool.jobs.pop(job.name, None)
        if not done:
            job.stop()
            return asked
        again, count = self.read_suite(out, job.process.poll())
        job.stop()
        if OUTPUT_INCOMPLETE_MARKER in out:
            self.beacon.skipped("confirmation output incomplete; earlier failures remain unresolved")
            return asked
        if not again and not count:
            return asked
        settled = [n for n in asked if n in again]
        if len(settled) != len(asked):
            self.beacon.fired("%d of %d only failed once and were let go"
                              % (len(asked) - len(settled), len(asked)))
        return settled

    def changed_paths(self) -> list[str]:
        """Every path that differs from the base commit, including files this run added."""
        code, out = git(["diff", "--name-only", self.tree.base or "HEAD"],
                        self.tree.root, self.read_room(30.0))
        paths = [p for p in out.splitlines() if p.strip()] if code == 0 else []
        return paths + sorted((self.tree._untracked() or set())
                              - self.tree.untracked_at_start)

    def read_room(self, want: float) -> float:
        """How long one read may take inside the shared reading window, or an expiry when it is
        spent.
        """
        until = getattr(self, "read_until", None)
        if until is None:
            return want
        left = until - time.monotonic()
        if left <= 0:
            raise ReadExpired("the shared reading window is spent")
        return min(want, left)

    def original(self, path: str) -> str | None:
        """One file as the base commit has it, or None when it cannot be read."""
        code, out = git(["show", "%s:%s" % (self.tree.base, path)],
                        self.tree.root, self.read_room(30.0))
        return out if code == 0 else None

    def whitespace_advisory(self) -> str:
        """What `git diff --check` says about the change, as an observation.

        Trailing blanks and conflict markers are what it reports. They are not
        a refusal: the statement did not rule them out, so the note only says
        they are there.
        """
        if not self.tree.base:
            return ""
        try:
            code, out = git(["diff", "--check", self.tree.base, "--"], self.tree.root, self.read_room(15.0))
        except Exception:
            return ""
        if code != 2 or not out.strip():
            return ""
        lines = [line for line in out.splitlines() if line.strip()][:6]
        return ("Whitespace observations from `git diff --check` on the change (an observation, "
                "not a requirement):\n" + redact("\n".join(lines)))

    def contract_read(self) -> tuple[list, list]:
        """Read the changed files for contract breaks, returning the refusals and the advisory
        notes.
        """
        self.contract_did_read = False
        self.contract_advisories = []
        if not CONTRACT_LINT:
            self.contract.skipped("not switched on for this run")
            return [], []
        try:
            changed = (self.tree.changed_paths() if getattr(self, "read_until", None) is None
                       else self.tree.changed_paths(self.read_room(15.0)))
        except BaseException as error:
            self.contract.skipped("the changed files could not be listed: %s"
                                  % type(error).__name__)
            return [], []
        if not changed:
            self.contract.skipped("nothing differs from the base")
            return [], []
        hard: list[str] = []
        soft: list[str] = []
        read = 0
        for path in changed[:8]:
            read += self.contract_one(path, hard, soft)
        self.contract_did_read = read > 0
        self.contract_advisories = soft[:3]
        if not hard and not soft:
            self.contract.skipped("%d changed file(s), %d read, nothing to say"
                                  % (len(changed), read))
            return [], []
        self.contract.fired("%d changed, %d read: %d refusing, %d listed: %s"
                            % (len(changed), read, len(hard), len(soft),
                               (hard or soft)[0][:160]))
        return hard[:2], soft[:3]

    def contract_one(self, path: str, hard: list, soft: list) -> int:
        """Read one changed file both ways and collect what the change broke, hard and soft."""
        try:
            before = (self.tree.at_base(path) if getattr(self, "read_until", None) is None
                      else self.tree.at_base(path, self.read_room(15.0)))
            after = self.tree.read(path)
        except BaseException as error:
            self.contract.skipped("%s could not be read both ways: %s"
                                  % (path, type(error).__name__))
            return 0
        try:
            found, listed = contract_violations(path, before, after, getattr(self, "statement", ""))
        except BaseException as error:
            self.contract.skipped("%s could not be checked: %s"
                                  % (path, type(error).__name__))
            return 0
        hard.extend(found)
        soft.extend(listed)
        return 1 if contract_readable(path) else 0

    def change_faults(self) -> list[str]:
        """Changes the statement's own words forbid: edited tests, dropped definitions, added
        suppressions.
        """
        faults: list[str] = []
        test_clause = next((part.strip() for part in instruction_clauses(self.statement)
                            if re.search(r"\b(?:do not|don't|never|must not)\s+"
                                         r"(?:edit|change|modify|touch)\s+(?:the\s+)?"
                                         r"tests?\b", part, re.I)), "")
        for path in self.changed_paths():
            if TEST_PATH.search(path) and test_clause:
                faults.append(
                    "This run edited %s, but the task explicitly says: %s. "
                    "Restore that test file and keep the change within the "
                    "stated permission." % (path, test_clause))
                continue
            faults.extend(self.file_faults(path))
        return faults

    def file_faults(self, path: str) -> list[str]:
        """What one changed file breaks against the statement's words, by name and by clause."""
        before = self.original(path)
        if before is None:
            return []
        if not importable(path):
            return []
        carried = visible_definitions(before)
        try:
            after = self.tree.read(path)
        except ToolFault:
            after = ""
        out: list[str] = []
        gone = sorted((collections.Counter(carried)
                       - collections.Counter(visible_definitions(after))).elements())
        statement = getattr(self, "statement", "")
        for definition in gone:
            clause = definition_preservation_clause(statement, definition)
            if clause:
                out.append("%s no longer defines %s, but the task says: %s. "
                           "Resolve this mismatch with the stated requirement."
                           % (path, definition, clause))
        clause = suppression_prohibition_clause(statement, path, self.tree.root)
        added = sum((suppression_comments(after) - suppression_comments(before)).values())
        if added and clause:
            out.append(
                "This run added %d suppression comment(s) to %s, but the task "
                "says: %s." % (added, path, clause))
        return out

    def change_shapes(self) -> list[tuple[str, str, str, str]]:
        """Every change by shape: its status, its modes and its path, with renames resolved."""
        rows: list[tuple[str, str, str, str]] = []
        self.shape_sources = {}
        code, out = git(["diff", "--raw", "-z", "-M", self.tree.base or "HEAD"],
                        self.tree.root, self.read_room(30.0))
        if code == 0:
            fields = iter(out.split("\0"))
            for meta in fields:
                if not meta.startswith(":"):
                    continue
                parts = meta[1:].split()
                if len(parts) < 5:
                    continue
                source = next(fields, "")
                status = parts[4][:1]
                path = next(fields, "") if status in ("R", "C") else source
                if not path:
                    continue
                if status in ("R", "C"):
                    self.shape_sources[path] = source
                rows.append((status, parts[0], parts[1], path))
        room = self.read_room(30.0)
        if room < 1.0:
            raise ReadExpired("the shared reading window is spent")
        for path in sorted((self.tree._untracked(room) or set())
                           - self.tree.untracked_at_start):
            rows.append(("A", "000000", "100644", path))
        return rows

    def fenced_diff(self, path: str) -> str:
        """One file's diff with no context lines, so the changed lines can be located exactly.
        """
        code, out = git(["diff", "-U0", self.tree.base or "HEAD", "--", path],
                        self.tree.root, self.read_room(30.0))
        return out if code == 0 else ""

    def scope_faults(self, budgeted: bool = True) -> list[str]:
        """Changes that reach outside the file, the method or the lines the statement allows."""
        self.fence_read = False
        if not EDIT_FENCE:
            self.fence.skipped("not switched on for this run")
            return []
        if budgeted and self.fence_asks >= FENCE_MAX_REFUSALS:
            self.fence.skipped("sent back %d time(s) already" % self.fence_asks)
            return []
        try:
            shapes = self.change_shapes()
        except BaseException as error:
            self.fence.skipped("the change did not list: %s" % type(error).__name__)
            return []
        if not shapes:
            self.fence.skipped("nothing differs from the base")
            return []
        named = stated_file(self.statement, self.tree.root)
        wanted = stated_methods(self.statement)
        marks = self.shape_faults(shapes, named)
        checked = 0
        for _, _, _, path in shapes:
            try:
                before = self.original(path)
                after = self.tree.read(path)
            except BaseException:
                continue
            if before is None:
                continue
            spans = hunk_spans(self.fenced_diff(path))
            checked += len(spans)
            marks.extend(self.line_faults(path, before, after, spans, wanted))
        self.fence_read = True
        if marks:
            if budgeted:
                self.fence_asks += 1
            self.fence.fired("%s; %d hunk(s) read"
                             % ("; ".join(tag for tag, _ in marks[:3]), checked))
            return [text for _, text in marks[:3]]
        self.fence.fired("reached %d hunk(s) over %d file(s), none outside"
                         % (checked, len(shapes)))
        return []

    def shape_faults(self, shapes: list, named: str | None) -> list[tuple[str, str]]:
        """Creations, deletions, renames and mode changes the statement's own words forbid."""
        if not RIDGES_SCOPE_FOLLOWS_STATEMENT:
            return []
        out: list[tuple[str, str]] = []
        for status, old, new, path in shapes:
            operations = ({"create"} if status in ("A", "C") else
                          {"delete"} if status == "D" else
                          {"rename"} if status == "R" else set())
            if old != new and "000000" not in (old, new):
                operations.add("mode")
            source = getattr(self, "shape_sources", {}).get(path) if status == "R" else None
            paths = (source, path) if source else (path,)
            reason = operation_violation(paths, operations, self.statement, self.tree.root)
            if not reason and operation_refusal("binary", paths, self.statement, self.tree.root):
                diff = self.fenced_diff(path)
                if re.search(r"^(?:GIT binary patch|Binary files )", diff, re.M):
                    reason = operation_violation(paths, {"binary"}, self.statement, self.tree.root)
            if reason:
                out.append(("tree %s" % path, reason + ". Restore only the prohibited change."))
        # Under a method bound that names no file, the answer is the one file that
        # holds the method, so two changed files that already existed cannot both be it.
        clause = method_clause(self.statement)
        edited = [path for status, _, _, path in shapes if status in ("M", "T")]
        if clause and len(edited) > 1 and not path_boundaries(self.statement, self.tree.root):
            out.append(("tree %s" % edited[0],
                        "This run changed %d files that were already there (%s), and the task "
                        "confines the change to one method: %s. Keep the change in the file that "
                        "holds that method and restore the others." % (len(edited), ", ".join(edited[:5]), clause)))
        return out

    def line_faults(self, path: str, before: str, after: str,
                    spans: list, wanted: list) -> list[tuple[str, str]]:
        """Changed lines that fall outside the method or the region the statement confines them to.
        """
        if not RIDGES_SCOPE_FOLLOWS_STATEMENT:
            return []
        out: list[tuple[str, str]] = []
        language, _ = project_contract(self.tree.root, self.statement, path)
        bound = method_clause(self.statement)
        method_bound = bool(bound)
        region = fenced_region(before, wanted)
        forbids_imports = statement_refused_nodes(self.statement).get("Import", "")
        if forbids_imports:
            # Compare import contents, not line numbers: adding an import on
            # the same line as a replaced import is still an added import;
            # moving an existing import does not invent a new one.
            if language in ("python", "unknown"):
                try:
                    previous = {ast.dump(node, include_attributes=False)
                                for node in ast.walk(ast.parse(before))
                                if isinstance(node, (ast.Import, ast.ImportFrom))}
                    additions = [node for node in ast.walk(ast.parse(after))
                                 if isinstance(node, (ast.Import, ast.ImportFrom))
                                 and ast.dump(node, include_attributes=False) not in previous]
                    added_line = additions[0].lineno if additions else None
                except (SyntaxError, ValueError):
                    added_line = None
            else:
                old_imports = {before.splitlines()[number - 1].strip()
                               for number in import_lines(before, language)}
                added_line = next((number for number in import_lines(after, language)
                                   if after.splitlines()[number - 1].strip() not in old_imports), None)
            if added_line is not None:
                out.append((
                    "import %s:%d" % (path, added_line),
                    "The change to %s adds an import at line %d, and the "
                    "statement rules that out: \"%s\". Follow that explicit "
                    "restriction or explain why it conflicts with the requested change."
                    % (path, added_line, forbids_imports)))
        if not region or not method_bound:
            # Naming a method says which one is wrong. Only a clause that
            # confines the change to its body closes the rest of the file.
            return out
        name, low, high = region
        reach, where = None, low
        try:
            ast.parse(after)
        except (SyntaxError, ValueError) as error:
            line = getattr(error, "lineno", None) or low
            return out + [(
                "parse %s:%d" % (path, line),
                "The candidate %s does not parse at line %d. Until it parses, "
                "%s cannot be located in it, so this answer is wrong. Repair "
                "the syntax and keep the change inside the "
                "method body." % (path, line, name))]
        landed = fenced_region(after, wanted)
        if not landed or landed[0] != name:
            return out + [(
                "missing %s:%d" % (path, low),
                "The candidate method is missing or no longer resolves "
                "exactly once as %s in %s, and the statement confines the "
                "change to it: \"%s\". While it cannot be located, that clause "
                "cannot be checked against the answer. Restore its name, its "
                "`def` and its containing class, or say which clause permits "
                "moving it." % (name, path, bound))]
        if landed[0] == name:
            old_lines, new_lines = before.splitlines(True), after.splitlines(True)
            head, tail = old_lines[:low - 1], old_lines[high:]
            if head != new_lines[:landed[1] - 1]:
                reach = "the text before its `def` line"
                where = next((n for n, pair in enumerate(
                    zip(head, new_lines), 1) if pair[0] != pair[1]), low)
            elif tail != new_lines[landed[2]:]:
                reach = "the text after its last line"
                where = high + next((n for n, pair in enumerate(
                    zip(tail, new_lines[landed[2]:]), 1)
                    if pair[0] != pair[1]), len(tail) + 1)
        if reach:
            out.append((
                "outside %s:%d" % (path, where),
                "The statement limits the change to %s, which is lines %d-%d "
                "of %s as this run found it, and your diff reaches %s. It asks "
                "for that in these words: \"%s\". Restore changes outside "
                "that region, or identify the current instruction that permits "
                "a wider edit." % (name, low, high, path, reach, bound)))
        return out

    def ledger_faults(self) -> list[str]:
        """The request to read the statement's requirements back, when it lists enough of them.
        """
        if not LEDGER_READBACK:
            self.ledger.skipped("not switched on for this run")
            return []
        items = [item.get("full_text", item["text"]) for item in
                 requirement_catalog(self.statement, getattr(self.tree, "root", ""))]
        if len(items) < LEDGER_MIN:
            self.ledger.skipped("%d item(s), below the floor" % len(items))
            return []
        if self.read_back_calls is not None:
            if self.allowance.calls > self.read_back_calls:
                return []
        if self.allowance.clock_left() < CONFORM_MIN_WALL_SEC:
            self.ledger.skipped("too little of the run left to act on it")
            return []
        if self.read_backs >= LEDGER_MAX_ASKS:
            self.ledger.skipped("asked %d time(s) already" % self.read_backs)
            return []
        held = [read_back(items)]
        if self.read_back_calls is not None:
            self.read_backs += 1
            self.ledger.fired("again; nothing was answered in between")
            return held
        self.read_backs += 1
        self.read_back_calls = self.allowance.calls
        self.ledger.fired("%d item(s), %d of the kind that asks for an edit"
                          % (len(items),
                             sum(1 for i in items if wants_an_edit(i))))
        return held

    def guarded_reader(self, reader, empty, beacon):
        """Run one reader and, if it fails, say so in its own slot and carry on with nothing."""
        try:
            return reader()
        except BaseException as error:
            beacon.skipped("the answer could not be checked: %s"
                           % type(error).__name__)
            return empty

    def final_faults(self, window: float | None = None) -> tuple[int, list[str]]:
        """The closing readings of the change, inside one shared time window, with how many
        finished.
        """
        if RIDGES_SCOPE_FOLLOWS_STATEMENT:
            Warden.report_no_baseline(self)
        done, faults = 0, []
        previous = getattr(self, "read_until", None)
        self.read_until = (None if window is None
                           else time.monotonic() + window)
        try:
            for reader, flag in ((lambda: self.scope_faults(budgeted=False),
                                  "fence_read"),
                                 (lambda: self.contract_read()[0],
                                  "contract_did_read")):
                try:
                    found = reader()
                except BaseException as error:
                    self.beacon.skipped("a closing reading did not finish: %s"
                                        % type(error).__name__)
                    continue
                if getattr(self, flag, False):
                    done += 1
                if found:
                    faults.extend(found)
                    break
        finally:
            self.read_until = previous
        return done, faults

    def verdict(self) -> list[str]:
        """Whether to hold this hand-in, and the reasons, each quoting the clause it came from.

        The project's check is waited for first. A reader that has already sent the run
        back enough times stands down, and so does one with too little of the run left
        to act on a refusal, because a refusal nobody can answer is worse than none.
        """
        if not SUBMISSION_WARDEN:
            return []
        self.settle()
        if self.before is None and self.job is not None:
            self.beacon.skipped("the project's tests were still running when "
                                "the answer was ready")
        if self.refusals >= WARDEN_REFUSALS_MAX:
            self.stood_down = True
            self.beacon.skipped("already sent the run back %d times" % self.refusals)
            return []
        if self.allowance.clock_left() < WARDEN_RELEASE_SEC:
            self.stood_down = True
            self.beacon.skipped("too little of the run left to act on a refusal")
            return []
        refusing, listed = self.guarded_reader(
            self.contract_read, ([], []), self.contract)
        faults = (self.guarded_reader(self.scope_faults, [], self.fence)
                  or self.guarded_reader(self.change_faults, [], self.beacon)
                  or refusing
                  or self.guarded_reader(self.go_compile_faults, [], self.beacon)
                  or self.suite_faults()
                  or self.guarded_reader(self.named_check_faults, [], self.beacon))
        if faults and listed:
            faults = faults + listed
        said, mine = (faults[0][:120] if faults else ""), True
        if not faults:
            faults = self.ledger_faults()
            if faults:
                said, mine = ("the list, %d item(s)" % len(
                    stated_requirements(self.statement)), False)
        if faults and mine:
            self.refusals += 1
            self.beacon.fired("refused hand-in #%d: %s" % (self.refusals, said))
        elif faults:
            self.beacon.fired("held hand-in: %s" % said)
        return faults

class Finished(Exception):
    """The answer has been handed in and the run is over."""
    pass

WORK_METER_WORDS = (
    "scale", "scales", "scaled", "scaling", "scalability", "grow", "grows",
    "growing", "growth", "row", "rows", "query", "queries", "statement",
    "statements", "memory", "index", "indexes", "indices", "indexed",
    "indexing", "scan", "scans", "scanning", "scanned", "prune", "prunes",
    "pruned", "pruning", "slow", "slower", "slowly", "slowdown", "n+1",
    "performance", "performant", "optimize", "optimized", "optimization",
    "optimise", "optimised", "optimisation",
    "bytes", "read", "reads", "gate", "gates", "workload", "workloads",
    "latency", "duration", "durations", "throughput",
)
MEASUREMENT_PROHIBITED = re.compile(
    r"\b(?:do not|don't|never|must not|should not|without)\s+(?:\w+\s+){0,3}?"
    r"(?:execut\w+|run\w*|issu\w+|send\w*|touch\w*)\s+"
    r"(?:any\s+|a\s+|the\s+)?(?:database\s+|live\s+)?"
    r"(?:quer(?:y|ies)|statements?|sql|database)\b", re.I)
MEASUREMENT_PASSIVE_PROHIBITED = re.compile(
    r"\bno\s+(?:database\s+)?(?:queries|sql(?:\s+statements)?|statements)\s+"
    r"(?:(?:may|must|should|can)\s+be\s+)?(?:executed|run|issued|sent)\b|"
    r"\b(?:database\s+)?(?:queries|sql(?:\s+statements)?|statements)\s+"
    r"(?:must|may|should|can)\s+not\s+be\s+(?:executed|run|issued|sent)\b", re.I)
# A measurement is worth the run's remaining budget when the statement asks for
# something measurable about database work. The word "query" appearing in a
# sentence is not that: a docstring repair in a file called query.py contains
# it and asks for nothing of the kind.
PERFORMANCE_REQUIREMENT = re.compile(
    r"\b(?:reduce|lower|cut|avoid|eliminate|minimi[sz]e|fewer|no more than|at most)\b"
    r"[^.!?;:]{0,60}\b(?:quer(?:y|ies)|statements?|round\s?trips?|scans?|joins?|"
    r"rows?\s+(?:read|scanned|examined|fetched|returned)|(?:read|scanned|examined|fetched)\s+rows?)\b|"
    r"\b(?:quer(?:y|ies)|statements?|round\s?trips?)\b[^.!?;:]{0,40}"
    r"\b(?:must not|should not|may not|must stay|should stay)\b|"
    # The bound can come before the noun as easily as after it: "must not
    # issue more than one query" says the same as "one query at most".
    r"\b(?:must not|should not|may not|cannot\s+(?:issue|run|execute|send|perform|make|use|exceed|require)|"
    r"no more than|at most)\b"
    r"[^.!?;:]{0,40}\b(?:quer(?:y|ies)|statements?|round\s?trips?)\b|"
    r"\bn\s?\+\s?1\b|"
    r"\b(?:must|should)\s+not\s+(?:grow|scale|increase)\b|"
    r"\b(?:optimi[sz]e|optimi[sz]ation|optimi[sz]ing|speed\s+up)\b|"
    r"\b(?:constant|fixed|bounded|single|exactly one)\s+(?:number\s+of\s+)?(?:sql\s+|database\s+)?"
    r"(?:quer(?:y|ies)|statements?)\b", re.I)

# A clause that names what is measured, with the bound or the report the
# statement attaches to it. The words are the statement's; the clause comes
# back verbatim, never a threshold of this run's own.
# Words that name a measurement of database work outright. Weaker words
# (latency, duration, memory, scans) also occur in domain rules, so they count
# only next to a verb or label that says something is being measured.
METRIC_WORDS = (
    r"(?:rows?\s+(?:read|examined|scanned|returned)|read[_ ]rows|rows?[_ ]read|scanned\s+rows?|"
    r"read[_ ]bytes|bytes\s+(?:read|returned|transferred|sent|received)|"
    r"(?:response|result|transfer(?:red)?|payload|output)[- ]?(?:body[- ]?)?bytes|response[- ]size|"
    r"peak\s+(?:application\s+)?memory|memory\s+usage|"
    r"wall(?:[- ]clock)?\s+(?:time|latency)|round[- ]?trips?|statement\s+counts?|"
    r"quer(?:y|ies)\s+counts?|number\s+of\s+(?:quer(?:y|ies)|statements|round[- ]?trips))")
# "allocation" left out: in these statements it names domain data (item and
# lot allocations), not memory allocations.
WEAK_METRIC_WORDS = (
    r"(?:" + METRIC_WORDS[3:-1] + r"|memory|latenc(?:y|ies)|durations?|elapsed|scans?|"
    r"throughput|p\d{2}\b)")
BOUND_WORDS = (
    r"(?:at\s+most|no\s+more\s+than|(?:must|may|should)\s+not\s+exceed|not\s+exceed|under|below|"
    r"within|less\s+than|fewer\s+than|limit(?:s|ed)?\s+to|bounded\s+by|bound(?:s|ed)?|"
    r"gates?\s+(?:are|is|on)|gated\s+(?:on|by)|budgets?|allowance|thresholds?|headroom|\u2264|<=|twofold|2x)")
METRIC_REQUIREMENT = re.compile(
    r"\b" + METRIC_WORDS + r"[^.!?;]{0,80}?\b" + BOUND_WORDS + r"\b|"
    r"\b" + BOUND_WORDS + r"[^.!?;]{0,80}?\b" + METRIC_WORDS + r"|"
    r"\bgates?\s+(?:are|is|include|cover)\b[^.!?;]{0,80}?\b" + WEAK_METRIC_WORDS + r"|"
    # A report verb counts only next to a word that names database work outright:
    # "report response latency" or "report paid covered duration" is the task's
    # own output, not a request to measure the application.
    r"\b(?:record|report|measure|collect|capture)\b[^.!?;]{0,60}?\b" + METRIC_WORDS + r"|"
    # A read budget stated as a share or a count of rows: "must read at most 12%
    # of the events table's rows", "must scan no more than 1800 database rows".
    r"\b(?:read|reads|reading|scan|scans|scanning|examine|examines|touch|touches|fetch|fetches)\s+"
    r"(?:at\s+most|no\s+more\s+than|fewer\s+than|less\s+than|under|below|within|only)\b"
    r"[^.!?;]{0,60}?\brows?\b|"
    r"\bread[- ]budget\b|"
    r"\bmetrics?\b[^.!?;]{0,40}?\b" + WEAK_METRIC_WORDS, re.I)
# Digit runs are bounded (25 characters covers 10,000,000,000,000): an unbounded
# run backtracked over every shorter length at every start, which was quadratic
# on a long "1,1,1,..." sequence.
WORKLOAD_CLAUSE = re.compile(
    r"\b(?:workloads?|datasets?|fixtures?|scales?|sizes?)\b[^.!?;]{0,100}?\b\d[\d,._]{0,24}\s*[kKmM]?\b\s*"
    r"(?:logical\s+)?(?:rows?|records?|entries|events|items?|entities|identities|graphs?|nodes?)\b|"
    r"\b\d[\d,._]{0,24}\s*[kKmM]?\s*(?:logical\s+)?(?:rows?|records?|entities)\b[^.!?;]{0,80}?"
    r"\b(?:workloads?|scale|sizes?|fixtures?)\b|"
    r"\b(?:small|medium|large)\b[^.!?;]{0,40}?\bworkloads?\b|\b\d+\s?k?/\d+\s?k?/\d+\s?[kKmM]?\b", re.I)

def execution_prohibited(statement: str) -> str:
    """The clause forbidding this run from executing queries, or "".

    Read once, here, so that every path that would otherwise ask for a live
    measurement asks the same question of the same words. A prohibition on
    running a query does not cancel the change the task wants; it decides how
    that change can be shown, which is a different thing and has to be said
    as one.
    """
    for part in instruction_clauses(statement):
        match = (MEASUREMENT_PROHIBITED.search(part)
                 or MEASUREMENT_PASSIVE_PROHIBITED.search(part))
        if match:
            # A requirement about the repaired program's behavior does not
            # prohibit this run from testing that behavior. Recognize explicit
            # program subjects without treating every unknown sentence as an
            # instruction to the tool operator.
            subject = part[:match.start()].strip()
            if re.fullmatch(
                    r"(?:(?:the|this|that|our|your)\s+)?"
                    r"(?:(?:repaired|updated|resulting|new|existing|generated|fixed|target|returned)\s+)*"
                    r"(?:method|function|implementation|application|code|query|endpoint|api|loop|helper|class|routine|operation|result)"
                    r"(?:\s+(?:`[^`]+`|[A-Za-z_][\w.]*\(\)))?", subject, re.I):
                continue
            if re.fullmatch(r"(?:`[^`]+`|[A-Za-z_][\w.]*\([^)]*\))", subject):
                continue
            return " ".join(part.split())[:REQUIREMENT_CHARS]
    return ""

def static_python_command(command: str, tree: Tree) -> str | None:
    """Translate a small syntax-only subset into a trusted isolated Python.

    Arbitrary shell/application execution cannot be proven database-free.
    Under a query prohibition, accept only parsing source files, never execute
    repository code, imports, shell expansions or a repository interpreter.
    """
    try:
        argv = shlex.split(command)
        if not argv or not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?",
                                       os.path.basename(argv[0])):
            return None
        paths, messages = [], []
        if len(argv) > 3 and argv[1:3] == ["-m", "py_compile"]:
            paths = argv[3:]
            if any(path.startswith("-") for path in paths):
                return None
        elif len(argv) == 3 and argv[1] == "-c":
            code = ast.parse(argv[2])
            for node in code.body:
                if isinstance(node, ast.Import) and all(
                        alias.name == "ast" and alias.asname is None
                        for alias in node.names):
                    continue
                if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                    return None
                call = node.value
                if (isinstance(call.func, ast.Name) and call.func.id == "print"
                        and not call.keywords and all(isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str) for arg in call.args)):
                    messages.append(" ".join(arg.value for arg in call.args))
                    continue
                if not (isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "ast" and call.func.attr == "parse"
                        and len(call.args) == 1 and not call.keywords):
                    return None
                read = call.args[0]
                if not (isinstance(read, ast.Call) and not read.args
                        and not read.keywords and isinstance(read.func, ast.Attribute)
                        and read.func.attr == "read"):
                    return None
                opened = read.func.value
                if not (isinstance(opened, ast.Call)
                        and isinstance(opened.func, ast.Name) and opened.func.id == "open"
                        and len(opened.args) == 1 and not opened.keywords
                        and isinstance(opened.args[0], ast.Constant)
                        and isinstance(opened.args[0].value, str)):
                    return None
                paths.append(opened.args[0].value)
        else:
            return None
        if not paths:
            return None
        absolute = [tree.inside(path) for path in paths]
        if any(not os.path.isfile(path) for path in absolute):
            return None
        source = ("import sys\n"
                  "for name in sys.argv[1:]:\n"
                  "    with open(name, 'rb') as stream: data = stream.read()\n"
                  "    compile(data, name, 'exec')\n")
        source += "\n".join("print(%r)" % message for message in messages)
        return shlex.join([sys.executable, "-I", "-S", "-c", source, *absolute])
    except (ValueError, SyntaxError, OSError, ToolFault):
        return None

DB_CONTEXT_WORDS = re.compile(
    r"\b(?:databases?|sql|postgres(?:ql)?|clickhouse|mysql|mariadb|sqlite|"
    r"queries|querysets?|orm|relational|round[ -]?trips?)\b|\bn\s?\+\s?1\b|"
    r"\bquery\b(?!\.\w)", re.I)
DATABASE_NOUN_END = (
    r"(?=\s*(?:[.,;:!?)]|$)|\s+(?:is|was|exists?|involved|needed|required|used|"
    r"here|at\s+all|for|in|of)\b)")
NON_DATABASE_CONTEXT = re.compile(
    r"\b(?:no|without(?:\s+(?:a|any))?)\s+(?:relational\s+)?database\b"
    + DATABASE_NOUN_END + r"|"
    r"\b(?:non[ -]database|database[ -]free)\b|"
    r"\b(?:does not|doesn't|do not|don't)\s+(?:use|involve|access)\s+"
    r"(?:a\s+|any\s+|the\s+)?database\b" + DATABASE_NOUN_END, re.I)
DB_CLIENT_MODULES = (
    "django.db", "sqlalchemy", "sqlite3", "psycopg", "psycopg2", "asyncpg",
    "pymysql", "MySQLdb", "mysql.connector", "clickhouse_driver",
    "clickhouse_connect", "peewee", "tortoise",
)

def database_context(statement: str, root: str = "", *, observed_database: bool = False) -> bool:
    """Whether current task text, named source or usable observations supply DB context.

    Generic performance words and a database elsewhere in the repository do
    not make this a database task. Only inspect a named source file, without
    importing it, and treat that evidence as context rather than an obligation.
    An explicit non-database declaration takes precedence over keyword hits
    and observations. Callers may supply observed context only from usable
    current-source database measurements, never from arbitrary old checks.
    """
    clauses = instruction_clauses(statement)
    text = " ".join(clauses)
    if NON_DATABASE_CONTEXT.search(text):
        return False
    if observed_database:
        return True
    # A filename or an example expression alone does not identify the work.
    prose = re.sub(r"`[^`]+`", " ", text)
    if DB_CONTEXT_WORDS.search(prose):
        return True
    if not root:
        return False
    try:
        path = stated_file(text, root)
        if not path:
            return False
        base = os.path.realpath(root)
        target = os.path.realpath(os.path.join(base, path))
        if os.path.commonpath([base, target]) != base or not os.path.isfile(target):
            return False
        if target.lower().endswith(".sql"):
            return True
        if not target.lower().endswith(".py"):
            return False
        with open(target, encoding="utf-8", errors="replace") as stream:
            source = stream.read(128_001)
        if len(source) > 128_000:
            return False
        for node in ast.walk(ast.parse(source)):
            names = ([alias.name for alias in node.names]
                     if isinstance(node, ast.Import) else
                     [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(name == module or name.startswith(module + ".")
                   for name in names for module in DB_CLIENT_MODULES):
                return True
    except (OSError, ValueError, SyntaxError, RecursionError):
        pass
    return False

def measurement_obligation(statement: str, root: str = "", *, observed_database: bool = False) -> str:
    """The statement's own words asking for something measurable, or "".

    Returned as the clause rather than a flag, so the request below can quote
    what it is asking on behalf of instead of announcing a requirement of its
    own. A statement that forbids running queries gets no obligation at all,
    whatever else it says.
    """
    if (execution_prohibited(statement)
            or not database_context(statement, root, observed_database=observed_database)):
        return ""
    bound = query_obligation(statement, root, observed_database=observed_database)
    if bound:
        return bound
    # A clause naming the metric outranks one naming a workload, which
    # outranks generic optimisation words: the most specific words the
    # statement itself uses are the ones a measurement has to answer.
    clauses = list(instruction_clauses(statement))
    for pattern in (METRIC_REQUIREMENT, WORKLOAD_CLAUSE, PERFORMANCE_REQUIREMENT):
        for part in clauses:
            found = pattern.search(part)
            if found and pattern is PERFORMANCE_REQUIREMENT and domain_bound(found.group(0), part):
                continue
            if found:
                return " ".join(part.split())[:REQUIREMENT_CHARS]
    return ""

WORK_METER_REQUEST = """

The statement asks for something about the work this does: "%s". Choose how
to measure that requirement from the current application, its documentation,
available fixtures, configured database client and observed environment.
When execution is permitted, exercise the application path and collect the
statements and parameters it actually sends using instrumentation supported
by that application. Identify which observations belong to that invocation;
incomplete or unavailable observations do not establish complete coverage.

Compare the original code and your change with compatible commands, settings,
fixtures and requests. Preserve your patch while obtaining a baseline, and
restore it afterwards%s
Keep setup and source changes separate from validation
commands so each result can be attributed to the code it checked. Record the
caller, dataset, request scope and exact commands for each measurement.
Choose representative workloads from the current task and public fixtures,
including different request sizes when the required behavior depends on size.

Record application completion and elapsed time separately from database work,
memory and transferred bytes, retaining the reported units. Fewer scanned rows
alone do not establish faster application completion. Compare results against
the current task's requirement; missing or incompatible measurements remain
unresolved. Preserve correctness while checking for performance regressions.

Respect execution restrictions and the remaining time and model budget. If a
measurement cannot be obtained, report the command and its actual outcome, or
why execution was not attempted, and leave that behavior unverified. A missing
utility or a mocked dependency does not establish database availability or
application behavior. If further measurement is not practical within the
remaining budget, state the limitation and hand in the work.
"""

WORK_METER_REQUEST = WORK_METER_REQUEST.replace("restore it afterwards%s", "restore it afterwards" + (
    ": the measure tool does this in a separate checkout of the\nbase commit, so your patch stays in place; "
    "sql reports each statement's own work." if DB_TOOL and MEASURE_TOOL else "."), 1)

def work_meter_hits(statement: str, root: str = "") -> list[str]:
    """The words in the statement that ask for something measurable about database work."""
    if not database_context(statement, root):
        return []
    text = " ".join(instruction_clauses(statement))
    return [word for word in WORK_METER_WORDS
            if re.search(r"(?<![\w])" + re.escape(word) + r"(?![\w])",
                         text, re.I)]

class WorkMeter:
    """What the run observed about database work, and what it may therefore claim.

    An observation of what the database did is recorded as an observation. It is never
    turned into a claim that the change improved anything, because this run does not compare
    before with after unless it measured both.
    """
    def __init__(self):
        """Open the meter armed, with nothing observed yet."""
        self.beacon = Beacon("meter")
        self.state = "armed"
        self.command = ""
        self.evidence = ""
        self.before = None
        self.calls = self.spent = 0

    def extend(self, kit, note):
        """Add what the meter observed to a hand-in note, once, when the statement asked for it.
        """
        if self.state != "armed":
            return note
        self.state = "done"
        try:
            self.beacon.reached(kit.allowance.calls, kit.allowance.spent,
                                kit.allowance.clock_left())
            reason = ""
            if not WORK_METER:
                reason = "flag off"
            else:
                statement = statement_of(kit.warden)
                root = getattr(kit.tree, "root", "")
                hits = work_meter_hits(statement, root)
                obligation = measurement_obligation(statement, root)
                if not hits:
                    reason = "no vocabulary hit"
                elif not obligation:
                    # The words are there, the obligation is not. Asking for a
                    # before-and-after measurement anyway spends the run on
                    # something the task never asked for, and on a task that
                    # forbids running queries it asks for the opposite.
                    reason = "the statement asks nothing measurable about database work"
                elif not kit.allowance.edits:
                    reason = "no edits"
                elif kit.allowance.clock_left() < SELFREVIEW_MIN_WALL_SEC:
                    reason = "insufficient time for verification"
                elif kit.allowance.money_left() <= 0:
                    reason = "no model budget left"
            if reason:
                self.beacon.skipped(reason)
                return note
            self.before = kit.tree.diff(SELFREVIEW_DIFF_SEC)
            self.calls, self.spent = kit.allowance.calls, kit.allowance.spent
            result = ((note or "Not handed in yet: check the database behavior "
                       "once before submitting.") + WORK_METER_REQUEST % obligation)
            self.beacon.fired(", ".join(hits))
            self.state = "asked"
            return result
        except BaseException as error:
            self.skip_error(error)
            return note

    def skip_error(self, error):
        """Record that a reading ended on an error rather than a result."""
        try:
            self.beacon.skipped("reading ended on %s" % type(error).__name__)
        except BaseException:
            pass

    def record(self, command, out="", returncode=None):
        """Keep the output of a command that actually showed what the database did."""
        try:
            if returncode != 0:
                return
            evidence = database_evidence(out)
            if evidence:
                self.command = redact(" ".join(command.split()))
                self.evidence = evidence
        except BaseException as error:
            self.skip_error(error)

    def close(self, kit):
        """Say what was observed and that an improvement was not certified, then stand down."""
        if self.state != "asked":
            return
        self.state = "done"
        try:
            import hashlib
            say("[%s] database output observed: %s; before/after improvement is not certified :: %s" % (
                self.beacon.slug, self.evidence or "none", self.command or "(no successful measurement output)"))
            after = kit.tree.diff(SELFREVIEW_DIFF_SEC)
            sha = lambda value: hashlib.sha256(value.encode(
                "utf-8", "replace")).hexdigest()[:8]
            say("[%s] before %s / after %s changed=%s" % (
                self.beacon.slug, sha(self.before), sha(after),
                "yes" if self.before != after else "no"))
        except BaseException as error:
            self.skip_error(error)
        finally:
            try:
                self.beacon.calls = kit.allowance.calls - self.calls
                self.beacon.usd = kit.allowance.spent - self.spent
                self.beacon.bill()
            except BaseException as error:
                self.skip_error(error)

# A node line: the cost clause is optional (EXPLAIN ... COSTS OFF omits it) and
# every number is bounded, so an absurd counter fails to match instead of
# raising while it is converted.
TEXT_PLAN_NODE = re.compile(
    r"^(\s*)(->\s+)?(.+?)(?:\s+\(cost=[^)]*\))?\s+\(actual time=(\d{1,15}(?:\.\d{1,9})?)\.\.(\d{1,15}(?:\.\d{1,9})?)"
    r" rows=(\d{1,30}) loops=(\d{1,30})\)\s*$")
TEXT_PLAN_NEVER = re.compile(r"^\s*(?:->\s+)?.+\(never executed\)\s*$")
TEXT_PLAN_SHAPED = re.compile(r"\(actual |\(cost=|^\s*->\s")
# psql's table around a text plan: the header, its rule and the row-count footer.
TEXT_PLAN_TRANSPORT = re.compile(r"\s*(?:QUERY PLAN|-{3,}|\(\d{1,20} rows?\))\s*")
# Plan-level lines PostgreSQL prints at the root's indentation after the tree.
TEXT_PLAN_SUMMARY = re.compile(r"\s*(?:JIT|Settings|Query Identifier|Serialization|Trigger [^:]{1,200}):")
TEXT_PLAN_BUFFERS = {"hit": "Hit Blocks", "read": "Read Blocks", "dirtied": "Dirtied Blocks", "written": "Written Blocks"}
TEXT_PLAN_JOIN = re.compile(
    r"^(Hash|Merge) (?:(Left|Right|Full|Semi|Anti|Right Semi|Right Anti) )?Join$|"
    r"^Nested Loop(?: (Left|Right|Full|Semi|Anti|Right Semi|Right Anti) Join)?$")
TEXT_PLAN_AGGREGATE = {"Aggregate": "Plain", "GroupAggregate": "Sorted", "HashAggregate": "Hashed",
                       "MixedAggregate": "Mixed"}

def text_plan_node_type(label: str) -> dict:
    """The JSON form's Node Type and qualifiers for a text plan label."""
    kind = re.split(r"\s+(?:on|using)\s+", label, 1)[0].strip()
    fields: dict = {}
    if kind.startswith("Parallel "):
        kind = kind[len("Parallel "):]
        fields["Parallel Aware"] = True
    mode = re.match(r"(Partial|Finalize) ", kind)
    if mode:
        kind = kind[mode.end():]
        fields["Partial Mode"] = mode.group(1)
    if kind.endswith(" Backward"):
        kind = kind[:-len(" Backward")]
        fields["Scan Direction"] = "Backward"
    if kind in TEXT_PLAN_AGGREGATE:
        fields["Strategy"] = TEXT_PLAN_AGGREGATE[kind]
        kind = "Aggregate"
    join = TEXT_PLAN_JOIN.match(kind)
    if join:
        if join.group(1):
            kind, fields["Join Type"] = "%s Join" % join.group(1), join.group(2) or "Inner"
        else:
            kind, fields["Join Type"] = "Nested Loop", join.group(3) or "Inner"
    setop = re.match(r"(Hash)?SetOp (Except|Intersect)( All)?$", kind)
    if setop:
        fields["Strategy"] = "Hashed" if setop.group(1) else "Sorted"
        fields["Command"] = setop.group(2) + (setop.group(3) or "")
        kind = "SetOp"
    if kind in ("Insert", "Update", "Delete", "Merge"):
        fields["Operation"] = kind
        kind = "ModifyTable"
    if kind.startswith("Custom Scan"):
        kind = "Custom Scan"
    fields["Node Type"] = kind
    return fields

def text_plan_documents(lines: list) -> tuple:
    """PostgreSQL text plans as the node trees their JSON form would carry.

    Only what the text states: each node's actual time, rows and loops, rows
    removed by a filter, buffer counts and the summary times. Nothing is summed
    or inferred; a node without an actual reading is not a report. A root node
    (the only kind without an arrow) starts a document of its own, so several
    plans in one output keep their own trees and times. Returns (documents or
    None, unread) where unread counts node lines that could not be decoded and
    every line outside a plan block: psql's command tags, the application's
    own output. A plan's property and summary lines are read, not unknown.
    """
    documents: list = []
    stack: list = []
    unread = 0
    planning_indent = None
    base = None
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        indent_now = len(line) - len(line.lstrip())
        if planning_indent is not None and indent_now <= planning_indent:
            planning_indent = None
        found = TEXT_PLAN_NODE.match(line)
        if found:
            indent = len(found.group(1)) + (2 if found.group(2) else 0)
            node = text_plan_node_type(found.group(3).strip())
            node.update({"Actual Startup Time": float(found.group(4)), "Actual Total Time": float(found.group(5)),
                         "Actual Rows": int(found.group(6)), "Actual Loops": int(found.group(7)), "Plans": []})
            if not found.group(2) or not documents:
                # A root: the text form gives no other node a line without "->".
                documents.append({"Plan": node})
                stack = [(indent, node)]
                base = indent_now
                continue
            while stack and stack[-1][0] >= indent:
                stack.pop()
            if stack:
                stack[-1][1]["Plans"].append(node)
            stack.append((indent, node))
            continue
        if TEXT_PLAN_NEVER.match(line):
            continue
        if TEXT_PLAN_SHAPED.search(line):
            unread += 1
            continue
        if TEXT_PLAN_TRANSPORT.fullmatch(line):
            if line.strip().startswith("("):
                base = None
            continue
        if base is None:
            unread += 1
            continue
        match = re.match(r"^\s*(Planning|Execution) Time:\s*(\d{1,15}(?:\.\d{1,9})?) ms", line)
        if match:
            if documents:
                documents[-1][match.group(1) + " Time"] = float(match.group(2))
            continue
        if re.match(r"^\s*Planning:\s*$", line):
            # The planner's own buffer use follows, indented under this line;
            # it belongs to the plan as a whole, not to its last node.
            planning_indent = indent_now
            continue
        match = re.match(r"^\s*Buffers:\s*(.+)$", line)
        if match:
            if planning_indent is not None:
                target = documents[-1].setdefault("Planning", {}) if documents else None
            else:
                target = stack[-1][1] if stack else None
            if target is not None:
                for scope, values in re.findall(r"(shared|local|temp)((?:\s+\w+=\d{1,30})+)", match.group(1)):
                    for name, number in re.findall(r"(\w+)=(\d{1,30})", values):
                        field = TEXT_PLAN_BUFFERS.get(name)
                        if field:
                            target["%s %s" % (scope.capitalize(), field)] = int(number)
            continue
        if TEXT_PLAN_SUMMARY.match(line):
            continue
        if indent_now <= base:
            # Back at or left of the root without a plan line: the block ended.
            unread += 1
            base = None
            continue
        match = re.match(r"^\s*Rows Removed by (Filter|Join Filter|Index Recheck):\s*(\d{1,30})\s*$", line)
        if match:
            stack[-1][1]["Rows Removed by " + match.group(1)] = int(match.group(2))
    if not documents:
        return None, unread
    return documents, unread

def measurement_documents(out: str) -> tuple[list, int]:
    """Decode complete JSON reports, including psql's single-column table.

    Table margins and continuation markers belong to the client transport.
    Require a complete one-row table and valid JSON; never repair JSON or
    infer missing metrics. Unparsed wrapper output remains explicitly unknown.
    A PostgreSQL text plan is decoded into the node tree its JSON form carries.
    """
    try:
        return [("$", json.loads(out))], 0
    except (ValueError, TypeError, RecursionError):
        pass
    lines = (out or "").splitlines()
    if re.search(r"\(actual time=[\d.]+\.\.[\d.]+ rows=\d+ loops=\d+\)", out or ""):
        documents, unread = text_plan_documents(lines)
        if documents is not None:
            return [("text-plan:$", documents)], unread
    roots, unreadable, index = [], 0, 0
    while index < len(lines):
        line = lines[index]
        if (line.strip() == "QUERY PLAN" and index + 1 < len(lines)
                and re.fullmatch(r"\s*-{3,}\s*", lines[index + 1])):
            end = index + 2
            while (end < len(lines) and lines[end].strip()
                   and not re.fullmatch(r"\(\d+ rows?\)", lines[end].strip())):
                end += 1
            body = lines[index + 2:end]
            decoded = None
            if (end < len(lines) and lines[end].strip() == "(1 row)" and body
                    and all(row.startswith(" ") for row in body)
                    and all(row.rstrip().endswith("+") for row in body[:-1])):
                parts = [row[1:].rstrip()[:-1].rstrip() for row in body[:-1]]
                parts.append(body[-1][1:].rstrip())
                try:
                    value = json.loads("\n".join(parts))
                    if (isinstance(value, list) and len(value) == 1
                            and isinstance(value[0], dict)
                            and isinstance(value[0].get("Plan"), dict)):
                        decoded = value
                except (ValueError, TypeError, RecursionError):
                    pass
            if decoded is not None:
                roots.append(("line:%d:psql:$" % (index + 1), decoded))
            else:
                unreadable += sum(bool(row.strip()) for row in lines[index:end + 1])
            index = end + 1
            continue
        if line.strip():
            try:
                roots.append(("line:%d:$" % (index + 1), json.loads(line)))
            except (ValueError, TypeError, RecursionError):
                unreadable += 1
        index += 1
    return roots, unreadable

# Literal report shapes a database or its client prints. Each entry names the
# fields retained and their units; nothing here is a target or a threshold.
CH_REPORT_FIELDS = {
    "QueryFinish": {"read_rows": "rows", "read_bytes": "bytes", "memory_usage": "bytes",
                    "result_rows": "rows", "result_bytes": "bytes",
                    "query_duration_ms": "milliseconds"},
    # The X-ClickHouse-Summary header of one HTTP request, complete only when
    # the request asked to wait for the end of the query.
    "HTTPSummary": {"read_rows": "rows", "read_bytes": "bytes", "written_rows": "rows",
                    "written_bytes": "bytes", "result_rows": "rows", "result_bytes": "bytes",
                    "elapsed_ns": "nanoseconds"},
    # Every QueryFinish row of system.query_log inside one observed time
    # window, summed; the window is stated with the report, never inferred.
    "QueryLogWindow": {"statements": "count", "read_rows": "rows", "read_bytes": "bytes",
                       "memory_usage": "bytes", "result_rows": "rows", "result_bytes": "bytes",
                       "query_duration_ms": "milliseconds"},
}
CH_REPORT_REQUIRED = {"QueryFinish": ("read_rows", "memory_usage", "result_rows"),
                      "HTTPSummary": ("read_rows", "read_bytes", "result_rows"),
                      "QueryLogWindow": ("statements", "read_rows", "result_rows")}
# pg_stat_database deltas over one observed window on the application database.
PG_WINDOW_FIELDS = {"statements": "count", "tup_returned": "rows", "tup_fetched": "rows",
                    "blks_read": "blocks", "blks_hit": "blocks", "xact_commit": "count"}
PG_WINDOW_REQUIRED = ("tup_returned", "blks_read", "blks_hit")
EVIDENCE_LABELS = {
    "postgresql:node": "PostgreSQL actual plan metrics",
    "postgresql:PgStatWindow": "PostgreSQL statistics window metrics",
    "clickhouse:QueryFinish": "ClickHouse QueryFinish metrics",
    "clickhouse:HTTPSummary": "ClickHouse HTTP summary metrics",
    "clickhouse:QueryLogWindow": "ClickHouse query_log window metrics",
}

def digit_field(value) -> bool:
    """A count as ClickHouse or psql prints it: a nonnegative int or digits."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    return isinstance(value, str) and 0 < len(value) <= 128 and value.isascii() and value.isdigit()

def report_kind(value, strict: bool = False) -> tuple:
    """Which literal database report a JSON object is, and its unit table.

    Returns ("engine:type", fields) or ("", {}). "excluded:query_log" marks a
    QueryFinish row that reports the measurement query itself. With strict the
    required counters must be present as numbers, as evidence demands; without
    it presence is enough, so a report with an odd field is still retained and
    the odd field is listed as invalid.
    """
    if not isinstance(value, dict):
        return "", {}
    keys = ("Actual Rows", "Actual Loops", "Actual Total Time")
    if strict:
        # Impossible counters remain inspectable in non-strict capture, but
        # cannot establish execution evidence or compatible measurement units.
        if all(finite_number(value.get(k)) and value[k] >= 0 for k in keys):
            return "postgresql:node", PG_NODE_METRICS
    elif all(k in value for k in keys):
        return "postgresql:node", PG_NODE_METRICS
    if isinstance(value.get("Plan"), dict) and any(k in value for k in PG_SUMMARY_METRICS):
        return "postgresql:summary", PG_SUMMARY_METRICS
    kind = value.get("type")
    if not isinstance(kind, str):
        # JSON Schema and many API bodies carry "type" as a list or an object.
        return "", {}
    if kind in CH_REPORT_FIELDS and value.get("query_id") and isinstance(value.get("query"), str):
        if kind == "QueryFinish" and "system.query_log" in value["query"].lower():
            return "excluded:query_log", {}
        if not strict or all(digit_field(value.get(k)) for k in CH_REPORT_REQUIRED[kind]):
            return "clickhouse:" + kind, CH_REPORT_FIELDS[kind]
        return "", {}
    if kind == "PgStatWindow" and (not strict or all(digit_field(value.get(k)) for k in PG_WINDOW_REQUIRED)):
        return "postgresql:PgStatWindow", PG_WINDOW_FIELDS
    return "", {}

def database_evidence(out: str) -> str:
    """Recognize complete execution metrics, not a mentioned EXPLAIN command."""
    if OUTPUT_INCOMPLETE_MARKER in (out or ""):
        return ""
    def inspect(value):
        if isinstance(value, dict):
            label = EVIDENCE_LABELS.get(report_kind(value, strict=True)[0])
            if label:
                return label
            items = value.values()
        elif isinstance(value, list):
            items = value
        else:
            return ""
        for child in items:
            found = inspect(child)
            if found:
                return found
        return ""
    for _, value in measurement_documents(out)[0]:
        try:
            found = inspect(value)
            if found:
                return found
        except (RecursionError, TypeError, ValueError):
            # One odd document is not evidence and must not hide the others.
            continue
    return ""


def measurement_units(out: str) -> tuple:
    """Units supplied by a recognized database report, not inferred targets.

    These identify comparable report fields only. They do not aggregate plans,
    prove which application path ran, or certify an improvement.
    """
    kind = database_evidence(out)
    if kind == "PostgreSQL actual plan metrics":
        return (("actual_rows", "rows"), ("actual_loops", "count"),
                ("actual_total_time", "milliseconds"))
    if kind == "ClickHouse QueryFinish metrics":
        units = (("read_rows", "rows"), ("memory_usage", "bytes"),
                 ("result_rows", "rows"))
        if re.search(r'["\']?query_duration_ms["\']?\s*:', out):
            units += (("query_duration_ms", "milliseconds"),)
        return units
    if kind == "ClickHouse HTTP summary metrics":
        return (("read_rows", "rows"), ("read_bytes", "bytes"),
                ("result_rows", "rows"), ("result_bytes", "bytes"))
    if kind == "ClickHouse query_log window metrics":
        return tuple(CH_REPORT_FIELDS["QueryLogWindow"].items())
    if kind == "PostgreSQL statistics window metrics":
        units = (("tup_returned", "rows"), ("tup_fetched", "rows"),
                 ("blks_read", "blocks"), ("blks_hit", "blocks"))
        if re.search(r'["\']?statements["\']?\s*:', out):
            units += (("statements", "count"),)
        return units
    return ()

# Literal field names from database diagnostics, not execution instructions.
# Plan-node counters retain their own locations. They are not summed across
# parent/child nodes or multiplied by loops, and absent fields stay absent.
PG_NODE_METRICS = {
    "Actual Rows": "rows", "Actual Loops": "count", "Actual Total Time": "milliseconds",
    "Actual Startup Time": "milliseconds", "Rows Removed by Filter": "rows",
    "Rows Removed by Join Filter": "rows", "Rows Removed by Index Recheck": "rows",
    "Shared Hit Blocks": "blocks", "Shared Read Blocks": "blocks",
    "Shared Dirtied Blocks": "blocks", "Shared Written Blocks": "blocks",
    "Local Hit Blocks": "blocks", "Local Read Blocks": "blocks",
    "Local Dirtied Blocks": "blocks", "Local Written Blocks": "blocks",
    "Temp Read Blocks": "blocks", "Temp Written Blocks": "blocks",
    "I/O Read Time": "milliseconds", "I/O Write Time": "milliseconds",
    "Temp I/O Read Time": "milliseconds", "Temp I/O Write Time": "milliseconds",
}
PG_SUMMARY_METRICS = {"Planning Time": "milliseconds", "Execution Time": "milliseconds"}

def measured_fields(out: str) -> list:
    """Bounded literal metrics, not aggregate work or proof of caller coverage."""
    if not database_evidence(out):
        return []
    reports = []
    def visit(value):
        if len(reports) >= 32:
            return
        if isinstance(value, dict):
            kind, fields = report_kind(value)
            if kind.startswith("excluded:"):
                fields = {}
            pg = kind.startswith("postgresql:")
            pg_summary = False
            metrics = {}
            for name, unit in fields.items():
                number = value.get(name)
                if isinstance(number, str) and number.isdigit():
                    number = int(number)
                if finite_number(number) and number >= 0:
                    metrics[name] = {"value": number, "unit": unit}
            if metrics:
                reports.append({"engine": "postgresql" if pg or pg_summary else "clickhouse", "fields": metrics})
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for _, value in measurement_documents(out)[0]:
        try:
            visit(value)
        except RecursionError:
            continue
    return reports

def measurement_text(value: str) -> str:
    """Redact credential-shaped values before serialization, preserving JSON."""
    # Diagnostic strings may quote escaped delimiters, or be decoded from JSON
    # containing an unpaired surrogate. Do not leak a quoted secret suffix or
    # let its encoding prevent an execution record from being collected.
    assignment = re.compile(
        r"\b([A-Za-z_][A-Za-z_0-9]*)=((?:[^\s'\"\\]|\\[^\r\n]|"
        r"'(?:\\[^\r\n]|[^'\\\r\n])*'?|\"(?:\\[^\r\n]|[^\"\\\r\n])*\"?)*)")
    value = assignment.sub(lambda m: m.group(1) + "=<withheld>"
                           if SECRET_NAME.search(m.group(1)) else m.group(0), value)
    value = redact(value)
    value = re.sub(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{20,}\b", "<withheld>", value)
    value = re.sub(r"(?i)(\bbearer\s+)\S+", r"\1<withheld>", value)
    return value.encode("utf-8", "backslashreplace").decode("utf-8")

def capture_measurement(out: str, record: dict) -> dict | None:
    """Keep one measurement's structured report, under the lock that bounds their total size."""
    with MEASUREMENT_LOCK:
        return _capture_measurement(out, record)

def _capture_measurement(out: str, record: dict) -> dict | None:
    """Retain literal reports, including identity and omissions, within this run.

    The byte allowance covers serialized retained records and the legacy metric
    view. It is not a limit on Python allocator overhead or command output.
    Counts describe recognized reports, never all queries an application ran.
    """
    global MEASUREMENT_BYTES_USED, MEASUREMENT_COUNTER
    reports = []
    seen, excluded = 0, 0
    complete = bool(record.get("output_complete", True)) and OUTPUT_INCOMPLETE_MARKER not in out
    roots, unreadable = measurement_documents(out)
    stack = [(location, value, None) for location, value in reversed(roots)]
    while stack:
        location, value, pg_identifier = stack.pop()
        if isinstance(value, dict):
            if "Query Identifier" in value:
                identifier = value["Query Identifier"]
                pg_identifier = {"location": measurement_text(location + '["Query Identifier"]'),
                                 "value": measurement_text(identifier) if isinstance(identifier, str)
                                 else identifier if isinstance(identifier, int) and not isinstance(identifier, bool)
                                 else None}
            kind, fields = report_kind(value)
            if kind == "excluded:query_log":
                excluded += 1
                kind, fields = "", {}
            pg = kind == "postgresql:node"
            pg_summary = kind == "postgresql:summary"
            ch = kind.startswith("clickhouse:")
            if fields:
                seen += 1
                if len(reports) < MEASUREMENT_REPORT_CAP:
                    metrics, invalid = {}, []
                    for name, unit in fields.items():
                        if name not in value:
                            continue
                        number = value[name]
                        if (isinstance(number, str) and len(number) <= 128
                                and number.isascii() and number.isdigit()):
                            number = int(number)
                        if finite_number(number) and number >= 0:
                            metrics[name] = {"value": number, "unit": unit}
                        else:
                            invalid.append(name)
                    item = {"engine": kind.split(":", 1)[0], "ordinal": seen,
                            "report_type": kind.split(":", 1)[1],
                            "location": measurement_text(location), "fields": metrics}
                    if invalid:
                        item["invalid_fields"] = invalid
                    if (pg or pg_summary) and pg_identifier is not None:
                        # This is the enclosing report's explicit identifier,
                        # not an inferred link to an application or workload.
                        item["query_identifier"] = dict(pg_identifier)
                    if ch or kind == "postgresql:PgStatWindow":
                        raw_id = str(value.get("query_id", ""))
                        item["query_id"] = measurement_text(raw_id)
                        if item["query_id"] != raw_id:
                            item["query_id_redacted"] = True
                            item["query_id_sha256"] = command_identity(raw_id)
                        if isinstance(value.get("query"), str):
                            item["query_sha256"] = command_identity(value["query"])
                    reports.append(item)
            stack.extend(reversed([(location + "[" + json.dumps(str(k), ensure_ascii=False) + "]", v, pg_identifier)
                                   for k, v in value.items() if isinstance(v, (dict, list))]))
        elif isinstance(value, list):
            stack.extend(reversed([(location + "[%d]" % i, v, pg_identifier) for i, v in enumerate(value)
                                   if isinstance(v, (dict, list))]))
    if not seen and record.get("kind") != "database measurement":
        return None
    MEASUREMENT_COUNTER += 1
    check_id = "check-%d" % MEASUREMENT_COUNTER
    # Context is copied from observed execution, not inferred from a query ID.
    keys = ("command", "command_key", "case_command_key", "invocation_key", "environment_key", "cwd",
            "identity", "identity_end", "at", "completed_at", "elapsed_seconds", "elapsed_kind",
            "exit_code", "timed_out", "stale", "output_complete", "case_bindings", "case_scopes")
    def clean(value):
        if isinstance(value, str):
            return measurement_text(value)
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    context = clean({k: record[k] for k in keys if k in record})
    counts = {"recognized": seen, "retained": len(reports), "omitted": seen - len(reports),
              "compact_retained": min(32, len(reports)), "compact_omitted": seen-min(32, len(reports)),
              "unknown": None if not complete or unreadable or not seen else 0,
              "unparsed_nonempty_lines": unreadable, "excluded_query_log_reports": excluded,
              "output_complete": complete, "storage_exhausted": False,
              "application_coverage": "not established; scope remains a claim"}
    data = {"check_id": check_id, "provenance": context, "capture": counts, "reports": reports}
    remaining = max(0, MEASUREMENT_BYTES_CAP - MEASUREMENT_BYTES_USED)
    def cost():
        legacy = [{"engine": r["engine"], "fields": r["fields"]} for r in reports[:32]]
        return (len(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                + len(json.dumps(legacy, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
    size = cost()
    while size > remaining and reports:
        reports.pop()
        counts.update(retained=len(reports), omitted=seen-len(reports), storage_exhausted=True)
        counts.update(compact_retained=min(32,len(reports)), compact_omitted=seen-min(32,len(reports)))
        size = cost()
    if size <= remaining:
        MEASUREMENTS[check_id] = data
        MEASUREMENT_BYTES_USED += size
    else:
        counts.update(retained=0, omitted=seen, compact_retained=0, compact_omitted=seen, storage_exhausted=True)
    record["measurement_check_id"] = check_id
    record["measurement_capture"] = counts
    # Preserve the old compact metric shape; these share field dictionaries
    # with retained reports rather than parsing a second copy of the output.
    record["metric_reports"] = [{"engine": r["engine"], "fields": r["fields"]} for r in reports[:32]]
    return data

def measurement_note(record: dict) -> str:
    """A sentence on what a measurement captured, and on what it leaves unresolved."""
    if record.get("measurement_capture_error"):
        return ("Structured measurement capture failed (%s); retained reports=0; "
                "omitted and unknown report counts are unknown. Execution provenance "
                "remains recorded; application measurement coverage is unresolved."
                % record["measurement_capture_error"])
    counts = record.get("measurement_capture")
    if not counts:
        return ""
    return ("Measurement %s: %d recognized reports; %d retained; %d omitted; "
            "unknown reports=%s%s. Reports do not establish full application coverage. "
            "Use read_measurement(check_id=%r, offset=0) for retained detail." % (
                record["measurement_check_id"], counts["recognized"], counts["retained"], counts["omitted"],
                "unknown" if counts["unknown"] is None else counts["unknown"],
                "; storage exhausted" if counts["storage_exhausted"] else "", record["measurement_check_id"]))

def execution_failure(code, out: str, timed_out: bool = False, *, checker: bool = False) -> str:
    """Classify actual diagnostic evidence without inferring service health."""
    if timed_out:
        return "timeout"
    if code == 0:
        return "none"
    if code in (126, 127) and re.search(r"command not found|not found|cannot execute|Permission denied", out, re.I):
        return "missing_or_unusable_executable"
    if re.search(r"ERR_UNKNOWN_FILE_EXTENSION|Unknown file extension|ERR_MODULE_NOT_FOUND|"
                 r"ERR_UNSUPPORTED_", out) or (checker and re.search(
            r"unknown option|unrecognized (?:option|argument)|unsupported (?:option|file extension)|"
            r"bad option|cannot find module|package \S+ does not exist|cannot find symbol|"
            r"error\[E0432\]|error\[E0433\]|unresolved import|failed to resolve", out, re.I)):
        return "unsupported_checker_configuration"
    if re.search(r"(?:psql:|DB::Exception|SQLSTATE|OperationalError|DatabaseError)", out):
        return "database_failure"
    return "application_failure" if code is not None else "unknown"


class Kit:
    """The tools the model may call, and the bookkeeping that keeps their results honest.

    Every tool goes through here: reading and editing files, running commands and database
    statements, measuring work, pinning expectations, and handing in. Each call is recorded
    against the identity of the source it ran on, so evidence from before an edit is never
    read as evidence for the code as it stands.
    """
    def __init__(self, tree: Tree, pool: ShellPool, allowance: Allowance,
                 warden: Warden | None = None, label: str = "",
                 findings: "FindingMap | None" = None) -> None:
        """Set up the tool kit over one checkout, with its allowance, its warden and its beacons.
        """
        self.tree = tree
        self.pool = pool
        self.allowance = allowance
        self.warden = warden
        self.findings = findings
        self.seen: dict[str, int] = {}
        self.workspace_stamp = self._workspace_stamp()
        self.edit_all = Beacon("editall")
        self.bg = Beacon("bgshell")
        self.conform = Beacon("conform")
        self.fence = Beacon("fence")
        self.label = label
        self.conform_state = "armed" if SUBMIT_CONFORM else "off"
        self.conform_edits = 0
        self.work_meter = WorkMeter()
        self.selfreview_state = "armed" if HIDDEN_SELFREVIEW else "off"
        self.requirement_state = "armed" if REQUIREMENT_PAUSE else "off"
        self.requirement = Beacon("requirement")
        self.case = Beacon("case")
        self.selfreview = Beacon("selfreview")
        self.selfreview_edits = 0
        self.selfreview_before: str | None = None
        self.selfreview_asked = False
        self.selfreview_reported = False
        self.selfreview_closed = False
        self.consult = Beacon("consult")
        self.consult_state = ("armed" if SELFREVIEW_CONSULT and HIDDEN_SELFREVIEW
                              else "off")
        self.consult_log: list[str] = []
        # The second reader takes the consult's place; with the consult on, it stands down.
        self.review = Beacon("review")
        self.review_state = ("armed" if REVIEW_SEAT and HIDDEN_SELFREVIEW and not SELFREVIEW_CONSULT
                             else "off")
        self.driver_seat = None
        self.pauses = 0
        self.progress_observation: str | None = None
        self.expect = Beacon("expect")
        self.expect_pauses = 0
        self.expect_required_pauses = 0
        self.green_beacon = Beacon("green")
        self.green: dict | None = None
        self.green_noted = False

    source_edits = 0

    def source_identity(self, paths=None) -> str:
        """The identity of the source this run is answering with."""
        return self.tree.source_identity(SELFREVIEW_DIFF_SEC, sorted(CREATED), paths)

    def has_edit(self) -> bool:
        """Is there an edited candidate, whichever tool produced it?

        Read from the tree rather than from a counter, so a change made with a
        shell command counts exactly as an editor call does.
        """
        try:
            return bool(self.tree.answer_paths(SELFREVIEW_DIFF_SEC))
        except BaseException:
            return bool(self.source_edits)

    def check_identity(self, command: str) -> dict | None:
        """Pin what a command is about to be run against, before it starts.

        Taken before the process exists, because a child can write to the tree
        the moment it is launched, and a background job can finish long after
        the tree it was started on has been edited away.
        """
        try:
            # One reading of the answer set, used for both the identity and
            # whether an edited candidate exists.
            paths = self.tree.answer_paths(SELFREVIEW_DIFF_SEC)
            # The SQL wrapper labels the target separately from its logical
            # statement. Capture that statement's declarations before launch;
            # the caller still records its query key explicitly, and the full
            # command/target identity remains independent below.
            case_command = (command.split(" :: ", 1)[1]
                            if command.startswith("sql ") and " :: " in command
                            else command)
            return {
                "command": redact(one_line(command))[:200],
                "command_key": command_identity(command),
                "case_command_key": command_identity(command),
                "case_bindings": case_bindings(case_command),
                "case_scopes": case_scopes(case_command),
                "check_kind": check_kind(command, "", getattr(
                    getattr(self, "warden", None), "runner", "")),
                "cwd": os.path.realpath(self.pool.cwd),
                "env": env_identity(command),
                "digest": patch_digest(self.tree.diff(SELFREVIEW_DIFF_SEC)),
                "identity": self.source_identity(paths),
                "edits": self.source_edits,
                "had_edit": bool(paths),
                "at": time.time(),
            }
        except BaseException:
            return None

    def note_verified(self, job: "Shell") -> None:
        """Record what this command established, and against which code."""
        try:
            opened = getattr(job, "opened", None)
            if opened is None:
                return
            code = job.process.poll()
            out = job._text()
            secret_target = getattr(job, "scrub_target", None) or getattr(job, "sql_target", None)
            if secret_target is not None:
                out = scrub(out, secret_target)
            if not getattr(job, "output_complete", True) and OUTPUT_INCOMPLETE_MARKER not in out:
                out = OUTPUT_INCOMPLETE_MARKER + " verification unresolved]\n" + out
            outcome, detail = check_outcome(out, job.command, code)
            runner = getattr(self.warden, "runner", "") if self.warden else ""
            record = dict(opened)
            record.update(
                invocation_key=getattr(job, "invocation_key", ""),
                environment_key=getattr(job, "environment_key", ""),
                cwd=getattr(job, "cwd", record.get("cwd")),
                completed_at=time.time(),
                elapsed_seconds=(max(0.0, time.monotonic() - job.started_monotonic)
                                 if hasattr(job, "started_monotonic") else None),
                elapsed_kind="invocation_to_collection_upper_bound",
                exit_code=code,
                output_complete=bool(getattr(job, "output_complete", True)),
                failure_type=("assertion" if outcome == "failed"
                              and assertion_failure(out) else "unknown"),
                metric_units={},
                metric_reports=[],
                execution_failure=execution_failure(code, out, bool(getattr(job, "timed_out", False))),
            )
            # Malformed diagnostic evidence cannot erase the command's
            # independently observed invocation, exit or source identity: a
            # reader that fails leaves the record without units or evidence.
            try:
                evidence = database_evidence(out)
                if getattr(job, "output_complete", True):
                    record["metric_units"] = measurement_units(out)
            except Exception as error:
                evidence = ""
                record["metric_units"] = {}
                record["measurement_capture_error"] = type(error).__name__
            # What the run recorded for this job, which for a query is the SQL
            # text and its target rather than the client's argument list.
            recorded = opened.get("command") or job.command
            kind = opened.get("check_kind") or check_kind(recorded, out, runner)
            if kind != "named regression check" and evidence:
                kind = "database measurement"
            record.update(
                kind=kind,
                status="exit_code=%s" % ("unknown" if code is None else code),
                timed_out=bool(getattr(job, "timed_out", False)),
                outcome=outcome, detail=detail,
                digest_end=patch_digest(self.tree.diff(SELFREVIEW_DIFF_SEC)),
                identity_end=(job.identity_end_reader()
                              if getattr(job, "identity_end_reader", None) else self.source_identity()),
                edits_end=self.source_edits,
            )
            # Conservative: any change to the tracked source it ran against, or
            # any deliberate edit while it ran, disqualifies the result. A shell
            # command that changes a file and restores it byte for byte while
            # the check runs is not detected; only a snapshot would catch that.
            record["stale"] = (
                record["identity_end"] != record["identity"]
                or record["edits_end"] != record["edits"])
            # Did it run against a tree that already carried an edit, and does it
            # name any file this run changed? The second is a candidate signal
            # read off the text, never a resolved claim that it exercised the
            # changed path: a name can coincide and a command can lie.
            record["after_edit"] = bool(record.get("had_edit"))
            record["names_changed_path"] = self.named_changed_paths(job.command, out)
            if (record["kind"] == "database measurement" and code == 0
                    and record["output_complete"]
                    and record["outcome"] in ("unrecognized", "ran")
                    and evidence):
                # A query plan carries no test banner. Its own metrics are the
                # result, so it is not held to the test parser's phrases.
                record["outcome"] = "measured"
                record["detail"] = evidence
            # An identity that could not be read is unknown, not a match: two
            # unreadable identities must not compare equal into evidence.
            known = bool(record["identity"]) and bool(record["identity_end"])
            record["evidence"] = bool(
                code == 0 and not record["timed_out"] and known
                and completed_record_time(record) is not None
                and record["output_complete"]
                and record["outcome"] in ("ran", "measured")
                and not record["stale"]
                and record["kind"] not in ("other command", "database query"))
            try:
                capture_measurement(out, record)
            except Exception as error:
                # Malformed diagnostic evidence cannot erase the command's
                # independently observed invocation, exit or source identity.
                record["measurement_capture_error"] = type(error).__name__
            CHECKS.append(record)
            job.measurement_record = record
            self.remember_green(record)
            self.note_case(job, out, record)
            if record["kind"] == "other command":
                return
            if record["evidence"]:
                say("[VERIFIED] %s: %s (%s) against patch %s in %s"
                    % (record["kind"], record["detail"], record["status"],
                       record["digest"], record["cwd"]))
            else:
                say("[VERIFIED] %s proved nothing usable: %s, %s%s%s%s"
                    % (record["kind"], record["status"], record["detail"],
                       ", timed out" if record["timed_out"] else "",
                       ", the files changed while it ran" if record["stale"] else "",
                       ", observation timing is unresolved"
                       if completed_record_time(record) is None else ""))
        except BaseException:
            pass

    def note_findings(self, command: str, out: str) -> None:
        """Pass one command's output to the finding record, and never let that reading end the run.
        """
        if self.findings is None:
            return
        try:
            self.findings.observe(command, out)
        except BaseException:
            self.findings.beacon.skipped("the output could not be read")

    def note_read(self, what: str) -> None:
        """Log one read, so the run shows what was looked at as well as what was changed."""
        say("[READ]%s %s" % (" " + self.label if self.label else "", what))

    def run(self, name: str, args: dict) -> str:
        # A completed command can supply its undecorated observation below.
        # Never reuse it for a subsequent tool call, including a rejected one.
        """Dispatch one tool call by name, after bringing the workspace up to date."""
        self.progress_observation = None
        if self.pool.jobs:
            self.sync_workspace()
        handler = getattr(self, "do_" + name, None)
        # A tool whose switch is off is not offered, and is not run either.
        if handler is None or name not in {t["function"]["name"] for t in TOOL_SCHEMAS}:
            raise ToolFault("no tool named %s" % name)
        return handler(args)

    def _workspace_stamp(self):
        """Cheap change detection for shell edits, including new/deleted files."""
        code, listing = git(["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            self.tree.root, 3.0)
        if code:
            return None
        digest = hashlib.sha256()
        for path in sorted(set(listing.split("\0")) - {""}):
            if path.endswith((".pyc", ".pyo")):
                continue
            try:
                stat = os.lstat(self.tree.absolute(path))
                state = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_mode)
            except OSError:
                state = None
            digest.update(repr((path, state)).encode("utf-8", "replace"))
        return digest.hexdigest()

    def sync_workspace(self, count_edits=True) -> None:
        """Notice that the files changed under this run, count it as an edit and forget stale
        reads.

        A command can change the tree as surely as the editor can, so the identity of
        the workspace is read rather than assumed from which tool was called.
        """
        stamp = self._workspace_stamp()
        if stamp is not None and stamp != self.workspace_stamp:
            if self.workspace_stamp is not None and count_edits:
                self.allowance.edits += 1
            self.seen.clear()
            self.workspace_stamp = stamp

    def guard_repeat(self, key: str) -> None:
        """Refuse a call that was already answered and whose answer cannot have changed."""
        self.seen[key] = self.seen.get(key, 0) + 1
        if self.seen[key] > REPEAT_READ_CEILING:
            raise ToolFault(
                "this exact call was already answered %d times and nothing has changed "
                "since. Scroll up and use the earlier result.%s"
                % (self.seen[key] - 1, self.open_requirement_hint())
            )

    def open_requirement_hint(self) -> str:
        """Point a stalled run at the work the statement still asks for."""
        try:
            if not REQUIREMENT_PAUSE:
                return ""
            wanted = [item for item in requirement_lines(statement_of(self.warden),
                                                         getattr(getattr(self, "tree", None), "root", ""))
                      if item["wants_edit"]]
            if not wanted:
                return ""
            return (" The statement still asks for: %s"
                    % "; ".join(item["text"][:90] for item in wanted[:2]))
        except BaseException:
            return ""

    def do_read_requirement(self, args: dict) -> str:
        """One of the statement's requirements by its number, or a range of them."""
        identifier = args.get("id")
        offset = args.get("offset", 0)
        if not isinstance(identifier, str) or not re.fullmatch(
                r"R[1-9]\d*(?:\s*-\s*R?[1-9]\d*)?|all", identifier.strip(), re.I):
            raise ToolFault("id must be a requirement ID such as R7, a range such as R3-R12, or all")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ToolFault("offset must be a nonnegative integer character offset")
        # The pattern accepts r7 as well as R7; the index spells them upper case.
        identifier = identifier.strip()
        identifier = "all" if identifier.lower() == "all" else identifier.upper()
        items = requirement_catalog(statement_of(self.warden), getattr(getattr(self, "tree", None), "root", ""))
        if identifier.lower() == "all":
            wanted = list(items)
        elif "-" in identifier:
            low, high = [int(re.sub(r"[^0-9]", "", part)) for part in identifier.split("-", 1)]
            if high < low:
                low, high = high, low
            wanted = [row for row in items if low <= int(row["id"][1:]) <= high]
            if not wanted:
                raise ToolFault("no requirements %s in this task's extracted index (%d clauses)"
                                % (identifier, len(items)))
        else:
            item = next((row for row in items if row["id"] == identifier), None)
            if item is None:
                raise ToolFault("no requirement %s in this task's extracted index (%d clauses)"
                                % (identifier, len(items)))
            wanted = [item]
        if len(wanted) > READ_REQUIREMENT_ITEMS:
            rest = wanted[READ_REQUIREMENT_ITEMS:]
            wanted = wanted[:READ_REQUIREMENT_ITEMS]
            trailer = ("\n[%d more; continue with read_requirement(id=%r)]"
                       % (len(rest), "%s-%s" % (rest[0]["id"], rest[-1]["id"])))
        else:
            trailer = ""
        if len(wanted) == 1:
            value = wanted[0]["full_text"]
            label = wanted[0]["id"]
            source = wanted[0]["source"] + (" in %s, named by the statement" % wanted[0]["document"]
                                            if wanted[0].get("document") else "")
        else:
            value = "\n\n".join(
                "%s [%s] (%s%s%s): %s" % (
                    row["id"], "change" if row["wants_edit"] else "keep", row["source"],
                    " under '%s'" % row["section"] if row.get("section") else "",
                    " in %s, named by the statement" % row["document"] if row.get("document") else "",
                    row["full_text"]) for row in wanted)
            label = "%s-%s" % (wanted[0]["id"], wanted[-1]["id"])
            source = "%d clauses" % len(wanted)
        if offset > len(value):
            raise ToolFault("%s has %d characters; offset=%d is past the end"
                            % (label, len(value), offset))
        # Reserve framing space before slicing; the assembled result fits the
        # existing read limit, including its continuation instruction.
        end = min(len(value), offset + max(1, READ_OUTPUT_CAP - 240 - len(trailer)))
        continuation = ("\nContinue with read_requirement(id=%r, offset=%d)."
                        % (identifier, end)) if end < len(value) else "\n[complete]" + trailer
        return ("%s characters %d:%d of %d; extracted %s; original statement is authoritative\n"
                % (label, offset, end, len(value), source)
                + value[offset:end] + continuation)

    def do_read_file(self, args: dict) -> str:
        """Part of a file, by line or by character offset, with its own line numbers."""
        path = str(args.get("path") or "")
        start = args.get("start")
        count = args.get("count")
        column = args.get("column", 0)
        if not isinstance(column, int) or isinstance(column, bool) or column < 0:
            raise ToolFault("column must be a nonnegative integer character offset")
        text = self.tree.read(path)
        version = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        self.guard_repeat("read:%s:%s:%s:%s:%s" % (path, start, count, column, version))
        lines = text.splitlines()
        first = max(1, int(start or 1))
        wanted = int(count) if count and int(count) > 0 else 0
        asked = min(len(lines), first + wanted - 1) if wanted else len(lines)
        if first <= len(lines):
            line = lines[first - 1]
            if column > len(line):
                raise ToolFault("%s line %d has %d characters; column=%d is past the end"
                                % (path, first, len(line), column))
            prefix = "%6d\t" % first
            full_header = "%s lines %d-%d of %d" % (path, first, first, len(lines))
            if first < asked:
                full_header += " -- pass start=%d to read on" % (first + 1)
            if column or len(full_header) + 1 + len(prefix) + len(line) > READ_OUTPUT_CAP:
                # A clipped middle cannot be recovered by advancing one line.
                # Return contiguous characters and an exact cursor instead.
                end = min(len(line), column + READ_OUTPUT_CAP)
                for _ in range(8):
                    head = "%s line %d characters %d-%d of %d" % (
                        path, first, column, end, len(line))
                    if end < len(line):
                        head += " -- pass start=%d, column=%d to read on" % (first, end)
                    elif first < asked:
                        head += " -- pass start=%d to read on" % (first + 1)
                    room = max(0, READ_OUTPUT_CAP - len(head) - len(prefix) - 1)
                    next_end = min(len(line), column + room)
                    if next_end >= end:
                        break
                    # Never grow again across a decimal-width boundary.
                    end = min(end, next_end)
                served = head + "\n" + prefix + line[column:end]
                self.note_read("read_file %s:%d characters %d-%d of %d -> %dc"
                               % (path, first, column, end, len(line), len(served)))
                return served
        rows, used, last = [], 0, first - 1
        for n in range(first, asked + 1):
            row = "%6d\t%s" % (n, lines[n - 1])
            if rows and used + len(row) + 1 > READ_OUTPUT_CAP:
                break
            if not rows:
                row = clip(row, READ_OUTPUT_CAP, "line")
            rows.append(row); used += len(row) + 1; last = n
        if not rows:
            served = ("%s is empty" % path if not lines
                      else "%s has %d line(s); start=%d is past the end"
                      % (path, len(lines), first))
            self.note_read("read_file %s:%d- of %d -> %dc"
                           % (path, first, len(lines), len(served)))
            return served
        while True:
            head = "%s lines %d-%d of %d" % (path, first, last, len(lines))
            if last < asked:
                head += " -- pass start=%d to read on" % (last + 1)
            if len(head) + 1 + used <= READ_OUTPUT_CAP:
                break
            if len(rows) > 1:
                used -= len(rows.pop()) + 1
                last -= 1
                continue
            rows[0] = clip(rows[0], max(0, READ_OUTPUT_CAP - len(head) - 1), "line")
            used = len(rows[0]) + 1
            break
        served = head + "\n" + "\n".join(rows)
        self.note_read("read_file %s:%d-%d of %d -> %dc"
                       % (path, first, last, len(lines), len(served)))
        return served

    def do_outline(self, args: dict) -> str:
        """A file's classes and functions as an outline, for finding the right place without
        reading it all.
        """
        path = str(args.get("path") or "")
        self.guard_repeat("outline:%s" % path)
        text = self.tree.read(path)
        try:
            rows = outline_source(text)
        except SyntaxError as bad:
            raise ToolFault("%s does not parse as Python around line %s, so it has "
                            "no index; read it instead" % (path, bad.lineno))
        if not rows:
            raise ToolFault("%s defines nothing, so an index of it would be empty; "
                            "read it instead" % path)
        served = clip("%s, %d lines, %d definitions\n" % (path, len(text.splitlines()),
                                                          len(rows))
                      + "\n".join(rows), SEARCH_OUTPUT_CAP, "outline")
        self.note_read("outline %s %d defs -> %dc" % (path, len(rows), len(served)))
        return served

    def do_search_text(self, args: dict) -> str:
        """Search the checkout for a pattern, in file contents or in file names."""
        pattern = str(args.get("pattern") or "")
        where = str(args.get("path") or ".")
        mode = str(args.get("mode") or "content")
        include = args.get("include")
        self.guard_repeat("grep:%s:%s:%s:%s" % (pattern, where, mode, include))
        cmd = ["grep", "-rEn", "--binary-files=without-match"]
        if mode == "files":
            cmd.append("-l")
        elif mode == "count":
            cmd.append("-c")
        for skip in (".git", "node_modules", ".venv", "__pycache__"):
            cmd.append("--exclude-dir=" + skip)
        if include:
            cmd.append("--include=" + str(include))
        if SEARCH_LIMIT and mode == "content":
            around = args.get("context")
            if around:
                cmd.append("-C%d" % max(0, min(20, int(around))))
        cmd += ["--", pattern, where]
        try:
            done = subprocess.run(
                cmd, cwd=self.tree.root, capture_output=True, text=True, timeout=60, errors="replace"
            )
            out = done.stdout or ""
            failure = ("search failed (exit_code=%s): %s" % (
                done.returncode, clip(redact(done.stderr or "no diagnostic returned").strip(),
                                      1200, "search diagnostic"))
                if done.returncode not in (0, 1) else "")
        except subprocess.TimeoutExpired as error:
            out = error.stdout or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", "replace")
            failure = "search timed out; narrow the pattern or the path"
        except OSError as error:
            raise ToolFault("search could not start: %s" %
                            clip(redact(str(error)), 1200, "search diagnostic"))
        if failure and not out.strip():
            raise ToolFault(failure)
        if not out.strip():
            self.note_read("search_text %r %s -> no matches" % (pattern, mode))
            return "no matches for %r under %s" % (pattern, where)
        if SEARCH_LIMIT:
            rows = out.splitlines()
            head = int(args.get("head_limit") or SEARCH_HEAD_LIMIT)
            if len(rows) > head:
                out = "\n".join(rows[:head]) + (
                    "\n... %d more matching lines; narrow the pattern or the path\n"
                    % (len(rows) - head))
        if failure:
            out = ("INCOMPLETE SEARCH: %s\nPartial results follow; absence of "
                   "other matches has not been established.\n%s" % (failure, out))
        served = clip(out, SEARCH_OUTPUT_CAP, "matches")
        self.note_read("search_text %r %s -> %dc" % (pattern, mode, len(served)))
        return served

    def do_find_files(self, args: dict) -> str:
        """The files whose paths match a glob, a page at a time."""
        import fnmatch
        pattern = str(args.get("pattern") or "*")
        offset = args.get("offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ToolFault("offset must be a nonnegative integer")
        self.guard_repeat("glob:%s:%d" % (pattern, offset))
        code, out = git(["ls-files", "-z"], self.tree.root, 30)
        if code != 0:
            raise ToolFault("could not list tracked files")
        paths = sorted(set(out.split("\0")) - {""})
        hits = [p for p in paths if fnmatch.fnmatch(p, pattern)]
        if not hits:
            loose = pattern if pattern.startswith("*") else "*" + pattern
            hits = [p for p in paths if fnmatch.fnmatch(p, loose)]
        if not hits:
            return "no tracked file matches %s" % pattern
        if offset >= len(hits):
            return "offset %d is at or past the end of %d matching tracked paths" % (offset, len(hits))
        rows, used = [], 0
        # Reserve the continuation notice before filling the page. Never clip
        # a path in the middle and silently advance past the missing bytes.
        reserve = 160
        for path in hits[offset:offset + 400]:
            shown = json.dumps(path, ensure_ascii=False) if any(c in path for c in "\n\r\t") else path
            if used + len(shown) + 1 > SEARCH_OUTPUT_CAP - reserve:
                break
            rows.append(shown)
            used += len(shown) + 1
        if not rows:
            raise ToolFault("one matching path exceeds the output limit; narrow the pattern or list paths through bash")
        end = offset + len(rows)
        if end < len(hits):
            rows.append("... %d more of %d matching tracked paths; pass offset=%d with the same pattern to continue."
                        % (len(hits) - end, len(hits), end))
        return "\n".join(rows)

    def do_edit(self, args: dict) -> str:
        """Replace one exact piece of text in a file, then report what the edit means.

        The old text must match exactly, so an edit can never land in a place the model
        did not read. Afterwards the file is compiled, its queries and its shape are
        read, and anything worth knowing is said back.
        """
        path = str(args.get("path") or "")
        old = str(args.get("old") or "")
        new = str(args.get("new") or "")
        every = bool(args.get("replace_all")) and REPLACE_ALL
        if not old:
            raise ToolFault("old must not be empty; use create_file to write a whole file")
        text = self.tree.read_exact(path)
        hits = text.count(old)
        if hits == 0 and "\r\n" in text:
            # The file was shown with plain line feeds. Where it really ends its
            # lines with a carriage return as well, match and write in its terms.
            paired = old.replace("\r\n", "\n").replace("\n", "\r\n")
            if text.count(paired):
                old, new = paired, new.replace("\r\n", "\n").replace("\n", "\r\n")
                hits = text.count(old)
        if hits == 0:
            raise ToolFault("that exact text is not in %s; read the file again and copy it verbatim" % path)
        if hits > 1 and not every:
            raise ToolFault(
                "that text occurs %d times in %s. Either extend it until it is unique, "
                "or pass replace_all=true if all %d should change the same way." % (hits, path, hits)
            )
        if every and hits > 1:
            self.edit_all.fired("%s x%d" % (path, hits))
            before = self.edit_all.artefact("before", text)
        else:
            before = ""
        updated = text.replace(old, new) if every else text.replace(old, new, 1)
        if updated == text:
            return "no change to %s (old and new are identical)" % path
        self.tree.write(path, updated)
        self.allowance.edits += 1
        self.source_edits += 1
        self.seen.clear()
        self.sync_workspace(count_edits=False)
        if before:
            self.edit_all.outcome(before, updated)
        note = self.compile_check(path)
        try:
            note += query_notes(path, text, updated, self.tree.root)
        except BaseException:
            pass
        note += edit_shape_notes(path, text, updated)
        note += self.clickhouse_name_note(path, text, updated)
        note += self.fence_note(path, updated)
        return "edited %s (%d occurrence%s)%s" % (path, hits if every else 1, "" if hits == 1 else "s", note)

    def do_create_file(self, args: dict) -> str:
        """Write a whole file, comparing against the bytes on disk so an unchanged write is
        reported as one.
        """
        path = str(args.get("path") or "")
        content = str(args.get("content") or "")
        # Compared against the bytes on disk, not against a tidied reading of
        # them: a file whose lines end CRLF reads back with plain line feeds,
        # so writing the same text with line feeds would look like no change
        # while the file kept its old endings and the write never happened.
        if (os.path.isfile(self.tree.absolute(path))
                and self.tree.read_exact(path) == content):
            return "no change to %s (content is identical)" % path
        self.tree.write(path, content)
        CREATED.add(path)
        self.allowance.edits += 1
        self.source_edits += 1
        self.seen.clear()
        self.sync_workspace(count_edits=False)
        return "wrote %s%s%s%s%s" % (path, self.compile_check(path), edit_shape_notes(path, "", content),
                                     self.clickhouse_name_note(path, "", content),
                                     self.fence_note(path, content))

    def compile_check(self, path: str) -> str:
        """Check that an edited file still parses, in its own language, and say so when it does
        not.
        """
        argv = SYNTAX_CHECKS.get(os.path.splitext(path)[1].lower())
        if execution_prohibited(statement_of(getattr(self, "warden", None))):
            if not path.endswith(".py"):
                return ""
            argv = [sys.executable, "-I", "-S", "-c",
                    "import sys; compile(open(sys.argv[1], 'rb').read(), sys.argv[1], 'exec')"]
        if not argv:
            return ""
        if GO_COMPILE and argv[0] == "gofmt" and not shutil.which(argv[0]) and go_tool("gofmt"):
            argv = [go_tool("gofmt"), *argv[1:]]
        if not shutil.which(argv[0]):
            return "\nSyntax verification unavailable: the configured checker executable is not installed."
        room = min(20.0, self.allowance.clock_left() - 5.0)
        if room < 3.0:
            return ""
        try:
            done = subprocess.run(
                argv + [self.tree.absolute(path)],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=room,
            )
        except subprocess.TimeoutExpired:
            return "\nSyntax verification timed out; this does not establish a syntax error."
        except OSError:
            return "\nSyntax verification could not start; the file's syntax remains unverified."
        if done.returncode == 0:
            return self.go_compile_note(path) if GO_COMPILE and path.endswith(".go") else ""
        failure = execution_failure(done.returncode, (done.stderr or "") + (done.stdout or ""), checker=True)
        if failure in ("missing_or_unusable_executable", "unsupported_checker_configuration"):
            return "\nSyntax verification unavailable (%s):\n%s" % (
                failure, clip(redact(done.stderr or done.stdout or ""), 1200, "checker diagnostic"))
        return "\n\nWARNING: the file no longer parses:\n" + clip(done.stderr or "", 1200, "error")

    def go_compile_note(self, path: str) -> str:
        """What compiling the edited file's package shows that the base commit's does not."""
        warden = getattr(self, "warden", None)
        room = min(GO_COMPILE_EDIT_SEC, self.allowance.clock_left() - FINISH_BUDGET_SEC - 5.0)
        if warden is None or room < 5.0:
            return ""
        try:
            fresh, unread = warden.go_reading([path], room)
        except (ToolFault, OSError):
            return ""
        if fresh:
            return ("\n\nWARNING: additional Go compiler diagnostics compared with the "
                    "checked baseline (compiled with tests; no test was run):\n"
                    + "\n".join(redact(line) for line in fresh[:GO_COMPILE_QUOTED])
                    + ("\nOther compile comparisons remain unverified: " + unread if unread else ""))
        if unread:
            return "\nCompile comparison unverified: %s" % unread
        return ""

    _targets: list | None = None

    def database_targets(self) -> list:
        """The databases this repository configures, discovered once."""
        if self._targets is None:
            self._targets = []
            try:
                for found in database_endpoints(self.tree.root, DB_FACTS_SEC / 2,
                                                statement_of(getattr(self, "warden", None)),
                                                dict(os.environ)):
                    if found.get("unresolved") or not found.get("host"):
                        continue
                    self._targets.append(DatabaseTarget(
                        found.get("engine") or "unknown", found["host"],
                        found.get("port") or 0, found.get("name") or "",
                        found.get("user") or "", found.get("password") or "",
                        found.get("source") or "", found.get("scheme") or ""))
            except BaseException:
                self._targets = []
        return self._targets

    def pick_target(self, wanted: str = "") -> "DatabaseTarget":
        """Choose the connection to use, or say why the choice is not this run's.

        Several plausible connections is an ambiguity to hand back, not a coin
        to flip: the caller names one rather than this guessing which database
        the task means.
        """
        targets = self.database_targets()
        if not targets:
            raise ToolFault(
                "no database connection is configured in readable form here. The "
                "repository's own settings are the authority: read them and "
                "connect through bash with the application's client.")
        if wanted:
            # An explicit identity takes precedence over a label substring.
            # Never resolve multiple matches by configuration iteration order.
            hits = [t for t in targets if wanted == t.label]
            if not hits:
                hits = [t for t in targets if wanted in (t.database, t.source)]
            if not hits:
                hits = [t for t in targets if wanted in t.label]
            if not hits:
                raise ToolFault("no configured database matches %r. Configured here: %s"
                                % (wanted, "; ".join(t.label for t in targets)))
            if len(hits) != 1:
                raise ToolFault("ambiguous database target %r matches more than one configured "
                                "connection; use a unique full label: %s"
                                % (wanted, "; ".join(sorted(t.label for t in hits))))
            return hits[0]
        if len(targets) > 1:
            raise ToolFault(
                "more than one database is configured here, so name the one you "
                "mean in `target`: %s" % "; ".join(t.label for t in targets))
        return targets[0]

    def do_case(self, args: dict) -> str:
        """Record the concrete case this run is working from.

        The quotations are checked against the statement and the file, so the
        case is anchored in real text. That is all it establishes: what the
        code does now and what the requirement needs are claims until the
        observation below is actually run.
        """
        name = args.get("case_id", "")
        if not isinstance(name, str) or ("case_id" in args and not CASE_ID.fullmatch(name)):
            raise ToolFault("case_id must start with a letter and contain at most 32 letters, digits, dots, underscores or hyphens")
        if name and name not in NAMED_CASES and len(NAMED_CASES) >= CASE_LIMIT:
            raise ToolFault("eight named cases are already recorded; replace an existing case_id explicitly")
        identifiers = args.get("requirement_ids", [])
        known = {item["id"] for item in requirement_catalog(statement_of(self.warden),
                                                               getattr(getattr(self, "tree", None), "root", ""))}
        if (not isinstance(identifiers, list) or len(identifiers) > 128
                or any(not isinstance(item, str) or item not in known for item in identifiers)
                or len(identifiers) != len(set(identifiers))):
            raise ToolFault("requirement_ids must be at most 128 distinct existing R-number IDs")
        scope = args.get("scope", {})
        if (not isinstance(scope, dict) or set(scope) - {"caller", "workload", "dataset"}
                or any(not isinstance(value, str) or len(value) > 512 for value in scope.values())):
            raise ToolFault("scope accepts caller, workload and dataset strings, at most 512 characters each")
        wanted = {
            "requirement": str(args.get("requirement") or "").strip(),
            "code_path": str(args.get("code_path") or "").strip(),
            "code_quote": str(args.get("code_quote") or "").strip(),
            "inputs": str(args.get("inputs") or "").strip(),
            "current_result": str(args.get("current_result") or "").strip(),
            "required_result": str(args.get("required_result") or "").strip(),
            "observation": str(args.get("observation") or ""),
            "unchanged": str(args.get("unchanged") or "").strip(),
        }
        if name and sum(map(len, wanted.values())) > COMMAND_CHARS_MAX + READ_OUTPUT_CAP:
            raise ToolFault("named case text exceeds the %d-character combined limit; "
                            "keep its eight text fields within that limit"
                            % (COMMAND_CHARS_MAX + READ_OUTPUT_CAP))
        role = str(args.get("asserts") or "unspecified")
        if role not in ("current_result", "required_result", "unspecified"):
            raise ToolFault("asserts must be current_result, required_result, or unspecified")
        missing = [name for name in ("requirement", "code_path", "code_quote",
                                     "current_result", "required_result",
                                     "observation") if not wanted[name].strip()]
        if missing:
            raise ToolFault("a case needs %s. Leave inputs or unchanged empty only "
                            "when the task has no row-shaped example to give."
                            % ", ".join(missing))
        statement = " ".join(statement_of(self.warden).split())
        quoted = " ".join(wanted["requirement"].split())
        if len(quoted) >= CASE_QUOTE_MIN and quoted not in statement:
            # The statement may name a repository document as its rules; a
            # quotation from that document is a quotation of the task's words.
            for _, text in referenced_documents(statement_of(self.warden),
                                                getattr(getattr(self, "tree", None), "root", "")):
                if quoted in " ".join(text.split()):
                    statement += " " + " ".join(text.split())
                    break
        if len(quoted) < CASE_QUOTE_MIN or quoted not in statement:
            raise ToolFault(
                "the requirement has to be quoted from the statement, and this is "
                "not in it: %r. Copy the words the task uses." % quoted[:120])
        try:
            source = self.tree.read(wanted["code_path"])
        except ToolFault:
            raise ToolFault("no such file in this checkout: %s" % wanted["code_path"])
        code = wanted["code_quote"]
        if len(code.strip()) < CASE_QUOTE_MIN or code.strip() not in source:
            joined = " ".join(code.split())
            if len(joined) < CASE_QUOTE_MIN or joined not in " ".join(source.split()):
                raise ToolFault(
                    "the code quotation is not in %s as written. Copy it from the "
                    "file." % wanted["code_path"])
        record = NAMED_CASES.get(name, {}) if name else CASE
        identity, at = self.source_identity(), time.time()
        replacement = dict(wanted, status="proposed", evidence="", asserts=role,
                           relationship="not established", case_id=name,
                           revision=record.get("revision", 0) + 1,
                           observation_key=command_identity(wanted["observation"]),
                           identity=identity, at=at,
                           declared_identity=identity, declared_at=at,
                           requirement_ids=list(identifiers), scope=dict(scope))
        # Validation and source reads have completed. No rejected request may
        # erase a case or advance its revision.
        record.clear()
        record.update(replacement)
        if name:
            NAMED_CASES[name] = record
        self.case.fired("recorded against %s" % wanted["code_path"])
        return ("Case recorded. The quotations match the statement and the file; "
                "that anchors it and nothing more. Run the observation you gave "
                "and inspect its output. Only a declared assertion can support "
                "or contradict its stated result; inspection alone stays "
                "unresolved:\n" + case_block(record, identity))

    def do_read_case(self, args: dict) -> str:
        """One declared case as it stands, by name, a page at a time."""
        name, offset = args.get("case_id"), args.get("offset", 0)
        if not isinstance(name, str) or (name and not CASE_ID.fullmatch(name)):
            raise ToolFault("case_id must be an existing name, or empty for the unnamed case")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ToolFault("offset must be a nonnegative integer character offset")
        record = NAMED_CASES.get(name) if name else CASE
        if not record:
            raise ToolFault("no case with that ID has been declared")
        text = case_block(record, self.source_identity())
        command = record.get("observation_key") or command_identity(record.get("observation", ""))
        revision = record.get("revision", 1)
        fields = ("command", "command_key", "case_command_key", "invocation_key", "cwd",
                  "environment_key", "identity", "identity_end", "at", "completed_at",
                  "elapsed_seconds", "elapsed_kind", "kind", "status", "outcome", "detail",
                  "stale", "timed_out", "output_complete", "execution_failure",
                  "metric_units", "metric_reports", "measurement_check_id", "measurement_capture", "measurement_capture_error", "case_bindings", "case_scopes")
        observations = [{key: row[key] for key in fields if key in row}
                        for row in CHECKS
                        if row.get("case_bindings", {}).get(name) == revision
                        and (row.get("case_command_key") or row.get("command_key")) == command]
        text += ("\nRetained observations bound to this revision: %d. These include superseded "
                 "readings; current status above uses chronological selection. Command output "
                 "capture remains bounded. Scope and requirement links are model claims.\n" % len(observations)
                 + redact(json.dumps(observations, ensure_ascii=False, sort_keys=True, indent=2)))
        if offset > len(text):
            raise ToolFault("offset is past the end of the current case record")
        end = min(len(text), offset + READ_OUTPUT_CAP - 320)
        continuation = ("\nContinue with read_case(case_id=%r, offset=%d)." % (name, end)
                        if end < len(text) else "\n[complete]")
        return ("case %s revision %s characters %d:%d of %d; claims and observations are distinct\n" %
                (name or "unnamed", record.get("revision", 1), offset, end, len(text))
                + text[offset:end] + continuation)

    def do_read_measurement(self, args: dict) -> str:
        """One measurement's structured report by its identifier, a page at a time."""
        check_id, offset = args.get("check_id"), args.get("offset", 0)
        if not isinstance(check_id, str) or not re.fullmatch(r"check-[1-9][0-9]*", check_id):
            raise ToolFault("check_id must be an existing measurement check-N identifier")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ToolFault("offset must be a nonnegative integer character offset")
        record = next((r for r in CHECKS if r.get("measurement_check_id") == check_id), None)
        if record is None:
            raise ToolFault("no measurement with that check_id was recorded in this run")
        data = MEASUREMENTS.get(check_id)
        if data is None:
            data = {"check_id": check_id, "capture": record["measurement_capture"], "reports": [],
                    "storage": "retention allowance exhausted; no structured detail stored"}
        text = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
        if offset > len(text):
            raise ToolFault("offset is past the end of this measurement record")
        end = min(len(text), offset + READ_OUTPUT_CAP - 512)
        current = self.source_identity()
        state = ("current-source observation" if current and record.get("identity") == current
                 and record.get("identity_end") == current and not record.get("stale")
                 else "different or unknown source; not current verification")
        continuation = ("\nContinue with read_measurement(check_id=%r, offset=%d)." % (check_id, end)
                        if end < len(text) else "\n[complete]")
        return ("Measurement %s; %s; characters %d:%d of %d. Query IDs and plan locations "
                "identify reported items, not caller coverage.\n" % (check_id, state, offset, end, len(text))
                + text[offset:end] + continuation)

    def requirement_coverage(self) -> dict:
        """For each stated requirement: what was declared about it, what was observed, what is
        unresolved.
        """
        rows = requirement_catalog(statement_of(getattr(self, "warden", None)),
                                   getattr(getattr(self, "tree", None), "root", ""))
        current = self.source_identity()
        result = {row["id"]: {"declared": [], "observed_assertions": [], "unresolved": []} for row in rows}
        for name, case in declared_cases():
            for identifier in case.get("requirement_ids", []):
                if identifier not in result:
                    continue
                item = {"case_id": name, "revision": case.get("revision", 1)}
                result[identifier]["declared"].append(item)
                usable = case_status(case, current) == "observed" and case.get("asserts") == "required_result"
                result[identifier]["observed_assertions" if usable else "unresolved"].append(item)
        return result

    def note_case(self, job: "Shell", out: str, record: dict) -> None:
        """Collect only prelaunch-bound revisions, using the check ledger's order."""
        try:
            ran = (record.get("case_command_key") or record.get("command_key")
                   or command_identity(job.command))
            current = None
            for name, case in declared_cases():
                revision = case.get("revision", 1)
                wanted = case.get("observation_key") or command_identity(case.get("observation", ""))
                def bound(row):
                    if "case_bindings" in row:
                        bindings = row["case_bindings"]
                        return isinstance(bindings, dict) and bindings.get(name) == revision
                    # Direct callers predating revision snapshots can address
                    # only the unnamed case. Every live invocation now includes
                    # the field, even when empty, so it never uses late binding.
                    # Unknown-era direct observations can mark an unnamed case
                    # attempted. The ordering check below prevents them from
                    # supporting a declaration they do not demonstrably follow.
                    return not name
                if not bound(record) or wanted != ran:
                    continue
                if current is None:
                    current = self.source_identity()
                rows = [row for row in CHECKS
                        if bound(row)
                        and (row.get("case_command_key") or row.get("command_key")) == wanted]
                if record not in rows:
                    rows.append(record)
                # Select every independent context, rather than allowing the
                # last callback's environment or SQL target to hide another.
                readings = current_records(rows, current)
                observed = current
                if not readings:
                    observed = case.get("observed_identity") or record.get("identity", "")
                    readings = current_records(rows, observed) or [record]
                    if readings == [record]:
                        observed = record.get("identity", "")
                role = case.get("asserts", "unspecified")
                verdicts = set()
                for row in readings:
                    ordered = (completed_record_time(row) is not None
                               and row.get("at", 0) >= case.get("declared_at", case.get("at", 0)))
                    if not ordered or not record_context(row) or not stable_record(row):
                        verdicts.add("unresolved")
                    elif usable_record(row) and row.get("outcome") == "ran":
                        verdicts.add("supports")
                    elif meaningful_failure(row):
                        verdicts.add("contradicts")
                    else:
                        verdicts.add("unresolved")
                if record in readings:
                    case["observed_output"] = redact(clip(out, 600, "case output"))
                case["evidence"] = redact("; ".join(
                    "%s: %s; source %s" % (row.get("status", "exit unknown"),
                        clip(row.get("detail", "unrecognized output"), 160, "detail"),
                        row.get("identity") or "unknown") for row in readings[-4:]))
                case["identity"] = observed  # Legacy observation alias.
                case["observed_identity"] = observed
                case["status"] = "attempted"
                if role == "unspecified":
                    case["relationship"] = "unresolved: command outcome does not establish either claimed result"
                elif len(verdicts) > 1:
                    case["relationship"] = (
                        "unresolved: observations disagree across overlapping or unordered "
                        "execution intervals or independent contexts; collection order "
                        "cannot establish success")
                elif verdicts == {"supports"}:
                    case["status"] = "observed"
                    case["relationship"] = "supports the declared %s assertion; assertion coverage remains the caller's claim" % role
                elif verdicts == {"contradicts"}:
                    case["status"] = "contradicted"
                    case["relationship"] = "the declared %s assertion failed; this does not prove the other result" % role
                    if role == "required_result":
                        case["relationship"] += "; on original code this may support the suspected defect"
                else:
                    case["relationship"] = "unresolved: stale, timed out, source/order unknown, or no usable assertion result"
                self.case.fired("%s revision %s: %s after %s" % (
                    name or "unnamed", revision, case["status"], record.get("command", "")[:60]))
        except BaseException:
            pass

    def do_sql(self, args: dict) -> str:
        """Run one statement against the database the application configures.

        The wrapper requests a transaction and rollback. Report their completion
        only from a successful client and affirmative server acknowledgements;
        arbitrary external or nontransactional effects are not covered by that.
        """
        forbidden = execution_prohibited(statement_of(getattr(self, "warden", None)))
        if forbidden:
            raise ToolFault("database execution is prohibited by the statement: %s. "
                            "Use source inspection and independent static checks."
                            % forbidden)
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolFault("query must not be empty")
        # psql -c accepts either SQL or one internal backslash command. Keep
        # this tool on its SQL path before credentials reach that client.
        if query.startswith("\\"):
            raise ToolFault("psql client commands are not SQL statements; send SQL "
                            "rather than a leading backslash command")
        if len(query) > SQL_CHARS_MAX:
            raise ToolFault("that statement is %d characters long; send the part "
                            "you need to measure" % len(query))
        target = self.pick_target(str(args.get("target") or ""))
        if target.engine == "clickhouse" and CH_HTTP:
            if target.scheme in ("http", "https") or (not target.scheme and target.port in (8123, 8443)):
                return self.sql_clickhouse(target, query, args)
            raise ToolFault(
                "%s is configured without an HTTP endpoint (ClickHouse serves HTTP on "
                "port 8123 by default), and this run's ClickHouse client speaks HTTP only. "
                "Name an HTTP endpoint in `target` if one is configured, or use the "
                "application's own client through bash, with the settings at %s."
                % (target.label, target.source))
        if target.engine != "postgresql":
            raise ToolFault(
                "%s is configured here and this run carries clients for postgresql "
                "(psql) and clickhouse (HTTP) only. Use the application's own client "
                "through bash, with the settings at %s." % (target.engine, target.source))
        if not shutil.which("psql"):
            raise ToolFault(
                "no psql client is installed in this environment. Use the "
                "application's own database client or shell through bash.")
        control = transaction_control(query)
        if control:
            raise ToolFault(
                "%s is transaction control, and this tool already runs the "
                "statement inside a transaction that it rolls back. A %s reached "
                "the server would keep the statement's work instead, so send the "
                "statement without it." % (control, control))
        analyze = bool(args.get("analyze"))
        explain = bool(args.get("explain")) or analyze
        machine = str(args.get("format") or "").strip().lower() == "json"
        inner = query.rstrip().rstrip(";")
        if explain and not inner.upper().startswith("EXPLAIN"):
            # ANALYZE runs the statement; a plain plan does not. They are not
            # offered as one switch for that reason. The text form stays the
            # default because it is what a reader expects; JSON is available
            # for a machine-readable tree, and the text form's node counters
            # are retained as literal reports as well.
            inner = (("EXPLAIN (ANALYZE, BUFFERS%s) %s" % (", FORMAT JSON" if machine else "", inner))
                     if analyze else ("EXPLAIN%s %s" % (" (FORMAT JSON)" if machine else "", inner)))
        room = self.allowance.clock_left() - FINISH_BUDGET_SEC
        if room < 5.0:
            raise ToolFault("not enough of the run left to wait on a query")
        try:
            asked = float(args.get("timeout") or SQL_BUDGET_SEC)
        except (TypeError, ValueError):
            raise ToolFault("timeout must be a number of seconds")
        if asked != asked:
            raise ToolFault("timeout must be a number of seconds")
        budget = max(5.0, min(asked, room))
        # The server stops the statement on its own a little after this tool
        # gives up waiting. Without that, a statement nobody is waiting for
        # keeps running, and with it the lane every test run needs.
        timeout_ms = int(max(5.0, budget) * 1000 + 2000)
        statements = [
            "SET statement_timeout = %d" % timeout_ms,
            "BEGIN",
            inner,
            "SAVEPOINT %s" % OPEN_PROBE,
            "ROLLBACK",
        ]
        # Several statements are sent one by one, each as its own -c, so every
        # result prints in order on any psql version. Each is SQL for the
        # server: one that starts with a backslash would be read by the client
        # as its own command instead, so it is refused before that.
        parts = sql_statements(inner) if SQL_SCRIPT and not explain else [inner]
        script = len(parts) >= 2
        if script:
            for part in parts:
                if part.startswith("\\"):
                    raise ToolFault("psql client commands are not SQL statements; every "
                                    "statement in the call must be SQL")
            statements = statements[:2] + parts + statements[3:]
        command = " ".join(shlex.quote(word) for word in target.statement_argv(statements))
        grace = min(LANE_GRACE_SEC, budget, max(0.0, room))
        opened = self.check_identity("sql %s :: %s" % (target.label, query))
        if isinstance(opened, dict):
            opened["command"] = scrub(opened.get("command", ""), target)
            opened["case_command_key"] = command_identity(query)
            opened["case_bindings"] = case_bindings(query)
            opened["case_scopes"] = case_scopes(query)
        self.progress_observation = ""  # A launch/queue handle is not a result.
        try:
            job = self.pool.start(command, pack_venv=False,
                                  hard_timeout=min(room, budget + LANE_GRACE_SEC),
                                  lane=SUITE_LANE, lane_wait=grace,
                                  env_extra=target.environment_extra())
        except LaneBusy:
            return ("not started. A test run is using the same database; collect "
                    "it first, then run this query.")
        job.opened = opened
        job.sql_target = target
        job.sql_analyze = analyze
        job.sql_script = script
        job.case_command = query
        target.begin_attempt()
        done, out = job.wait(budget)
        if not done:
            self.bg.fired("query still running as %s" % job.name)
            return scrub(job.result(out), target) + (
                "\n\n[still running as %s; collect it with bash_poll. The server "
                "stops it shortly after this budget, which frees the lane a test "
                "run needs]" % job.name)
        result = self.finish_shell(job, out)
        return result if getattr(job, "sql_reported", False) else self.sql_result(job, out, result)

    def sql_clickhouse(self, target: "DatabaseTarget", query: str, args: dict) -> str:
        """One statement over ClickHouse HTTP, measured, in this process."""
        write = bool(args.get("write"))
        explain = bool(args.get("explain"))
        analyze = bool(args.get("analyze"))
        second = single_statement(query)
        if second:
            raise ToolFault(second)
        room = self.allowance.clock_left() - FINISH_BUDGET_SEC
        if room < 5.0:
            raise ToolFault("not enough of the run left to wait on a query")
        try:
            asked = float(args.get("timeout") or SQL_BUDGET_SEC)
        except (TypeError, ValueError):
            raise ToolFault("timeout must be a number of seconds")
        if asked != asked:
            raise ToolFault("timeout must be a number of seconds")
        budget = max(5.0, min(asked, room))
        grace = min(LANE_GRACE_SEC, budget, max(0.0, room))
        opened = self.check_identity("sql %s :: %s" % (target.label, query))
        if isinstance(opened, dict):
            opened["command"] = scrub(opened.get("command", ""), target)
            opened["case_command_key"] = command_identity(query)
            opened["case_bindings"] = case_bindings(query)
            opened["case_scopes"] = case_scopes(query)
        self.progress_observation = ""  # A launch/queue handle is not a result.
        probe = HttpProbe("sql %s :: %s" % (target.label, query),
                          getattr(getattr(self, "pool", None), "cwd", None) or os.getcwd(), target,
                          {"readonly": "0" if write else "2", "explain": explain})
        try:
            self.pool.reserve(SUITE_LANE, probe, grace)
        except LaneBusy:
            return ("not started. A test run is using the same database; collect "
                    "it first, then run this query.")
        probe.opened = opened
        probe.sql_target = target
        probe.sql_analyze = analyze
        probe.sql_write = write
        probe.case_command = query
        target.begin_attempt()
        try:
            text, first = ClickHouseHttp(target).run_measured(
                query, budget=budget, write=write, explain=explain,
                room=max(0.0, room - budget))
        except BaseException as error:
            text = "-- the HTTP client failed before a response: %s" % type(error).__name__
            first = {"status": None, "error_code": None, "timed_out": False,
                     "transport_error": type(error).__name__}
        probe.complete(text, 0 if first.get("status") == 200 else 1, bool(first.get("timed_out")))
        target.observe_http(first.get("status"), first.get("error_code"),
                            bool(first.get("timed_out")), first.get("transport_error") or "")
        return self.finish_shell(probe, text)

    def sql_result(self, job: "Shell", out: str, result: str) -> str:
        """Complete SQL observations equally for immediate and polled jobs."""
        target = job.sql_target
        job.sql_reported = True
        if PLAN_NOTES:
            try:
                plan = None
                for _, document in measurement_documents(out)[0]:
                    if isinstance(document, (dict, list)) and json.dumps(document)[:2000].find("Node Type") >= 0:
                        plan = document
                        break
                notes = plan_shape_notes(sql_text=getattr(job, "case_command", ""), plan=plan,
                                         plan_text=out if getattr(job, "sql_analyze", False)
                                         or "Clauses:" in out or "Join" in out else "")
                result += plan_notes_block(notes)
            except BaseException:
                pass
        if target.engine == "clickhouse":
            # The HTTP attempt was observed when its response arrived; there
            # is no client transcript to read tags from and nothing to roll back.
            if getattr(job, "sql_write", False):
                transaction = ("ClickHouse statements are not transactional and this one was "
                               "sent with write=true: a change that succeeded persisted and "
                               "is not rolled back")
            else:
                transaction = ("ClickHouse statements are not transactional; this one was sent "
                               "read-only (readonly=2), so the server refused any write")
            return scrub("%s\n%s\n%s" % (target.describe(), transaction,
                         clip(scrub(result, target), SQL_OUTPUT_CAP, "query output")), target)
        target.observe_client(job.process.poll(), out,
                              bool(getattr(job, "timed_out", False)))
        if target.observations["rollback"] is True:
            transaction = ("the wrapper transaction was rolled back"
                           + (" after ANALYZE executed it" if job.sql_analyze else "")
                           + "; external and nontransactional effects are not established")
        elif PROBE_REFUSED.search(out):
            transaction = ("the server rejected the open-transaction probe. The "
                           "wrapper rollback is unconfirmed; whether any work was "
                           "committed or rolled back is unknown")
        else:
            transaction = "transaction completion and rollback are unconfirmed"
        if getattr(job, "sql_script", False) and (job.process.poll() or 0) != 0:
            transaction += ("; the client stopped at the first failing statement and "
                            "disconnected before ROLLBACK was sent. The server does not "
                            "commit a transaction that is open at disconnect")
        return scrub("%s\n%s\n%s" % (target.describe(), transaction,
                     clip(scrub(result, target), SQL_OUTPUT_CAP, "query output")), target)

    def measure_dir(self) -> str:
        """A private directory for measurement work, made once and removed when the pool closes.
        """
        where = getattr(self, "_measure_dir", None)
        if not where or not os.path.isdir(where):
            where = tempfile.mkdtemp(prefix="measure")
            self._measure_dir = where
            self.pool.own_cleanup(lambda path=where: shutil.rmtree(path, ignore_errors=True))
        return where

    @staticmethod
    def window_query_id() -> str:
        """A query id for one measurement window: unique here, and the same on every run of this
        problem.
        """
        global MEASURE_WINDOW_COUNTER
        MEASURE_WINDOW_COUNTER += 1
        # One process, one database: a counter is unique here, and the same on
        # every run of the same problem.
        return "%sw%d" % (CH_QUERY_ID_PREFIX, MEASURE_WINDOW_COUNTER)

    def pg_stats(self, target: "DatabaseTarget", budget: float = 10.0) -> dict | None:
        """pg_stat_database counters of the application database, or None."""
        if not shutil.which("psql"):
            return None
        env = dict(os.environ)
        env.update(target.environment_extra())
        try:
            done = subprocess.run(target.statement_argv([PG_STATS_SQL]) + ["-At"], capture_output=True,
                                  text=True, errors="replace", timeout=budget, env=env)
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            return None
        try:
            stats = json.loads((done.stdout or "").strip().splitlines()[0])
        except (ValueError, IndexError):
            return None
        try:
            more = subprocess.run(target.statement_argv([PG_STATEMENTS_SQL]) + ["-At"], capture_output=True,
                                  text=True, errors="replace", timeout=budget, env=env)
            if more.returncode == 0 and (more.stdout or "").strip().isdigit():
                stats["statements"] = int(more.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        return stats if isinstance(stats, dict) else None

    def window_begin(self, target) -> dict:
        """Open a measurement window on the configured database, and say whether it opened."""
        if target is None:
            return {"kind": "none", "ok": False, "note": "no configured database to read work from"}
        if target.engine == "clickhouse" and target.scheme != "native" and CH_HTTP:
            reply = ClickHouseHttp(target).request("SELECT now64(6) AS t0 FORMAT JSONEachRow", budget=5.0,
                                                   query_id=self.window_query_id(), readonly=2, database=False)
            try:
                t0 = json.loads((reply.get("body") or "").strip().splitlines()[0])["t0"]
            except (ValueError, IndexError, KeyError, TypeError):
                return {"kind": "clickhouse", "ok": False,
                        "note": "system time not readable: %s" % ClickHouseHttp.describe(reply)}
            return {"kind": "clickhouse", "ok": True, "t0": t0}
        if target.engine == "postgresql":
            stats = self.pg_stats(target)
            return {"kind": "postgresql", "ok": stats is not None, "before": stats,
                    "note": "" if stats is not None else "pg_stat_database not readable through psql"}
        return {"kind": "none", "ok": False, "note": "no work reader for %s" % target.engine}

    def window_end(self, target, begun: dict, command: str, room: float) -> tuple:
        """(report text to attach, aggregate dict or None) for one finished run."""
        if not begun.get("ok"):
            return "-- database work window unavailable: %s\n" % begun.get("note", ""), None
        identifier = "window-%d" % MEASURE_WINDOW_COUNTER
        if begun["kind"] == "clickhouse":
            client = ClickHouseHttp(target)
            budget = max(2.0, min(CH_DETAIL_SEC, room))
            flush = client.request("SYSTEM FLUSH LOGS", budget=budget, query_id=self.window_query_id(),
                                   readonly=None, database=False)
            if flush.get("status") != 200:
                return ("-- query_log window unavailable: SYSTEM FLUSH LOGS %s\n"
                        % client.describe(flush)), None
            total = client.request(CH_WINDOW_SQL, budget=budget, query_id=self.window_query_id(),
                                   readonly=2, params={"t0": begun["t0"]}, database=False)
            try:
                aggregate = json.loads((total.get("body") or "").strip().splitlines()[0])
            except (ValueError, IndexError, TypeError):
                return "-- query_log window unavailable: %s\n" % client.describe(total), None
            report = {"type": "QueryLogWindow", "query_id": identifier, "query": command,
                      "window_start": begun["t0"]}
            for key in CH_REPORT_FIELDS["QueryLogWindow"]:
                if key in aggregate:
                    report[key] = aggregate[key]
            lines = [json.dumps(report, ensure_ascii=False)]
            rows = client.request(CH_WINDOW_ROWS_SQL, budget=budget, query_id=self.window_query_id(),
                                  readonly=2, params={"t0": begun["t0"]}, database=False)
            if rows.get("status") == 200:
                lines.extend((rows.get("body") or "").strip().splitlines()[:MEASURE_LOG_ROWS])
            aggregate = {k: aggregate.get(k) for k in CH_REPORT_FIELDS["QueryLogWindow"]}
            for key, value in list(aggregate.items()):
                if isinstance(value, str) and value.isdigit():
                    aggregate[key] = int(value)
            return "\n".join(lines) + "\n", aggregate
        if begun["kind"] == "postgresql":
            # Backends publish their counters after a short interval; wait for it.
            time.sleep(min(1.2, max(0.0, room)))
            after = self.pg_stats(target)
            if after is None:
                return "-- pg_stat_database window unavailable after the command\n", None
            before = begun.get("before") or {}
            report = {"type": "PgStatWindow", "query_id": identifier, "query": command}
            aggregate = {}
            for key in PG_WINDOW_FIELDS:
                if key in after and key in before:
                    try:
                        aggregate[key] = int(after[key]) - int(before[key])
                        report[key] = str(max(0, aggregate[key]))
                    except (TypeError, ValueError):
                        continue
            return json.dumps(report, ensure_ascii=False) + "\n", aggregate
        return "", None

    def measure_run(self, command: str, where: str, role: str, target, scope: dict,
                    pair_id: str, scale: str, budget: float, room: float) -> dict:
        """One recorded run of the command in one checkout, with its work window."""
        label = "measure %s :: %s" % (role, command)
        opened = self.check_identity(label)
        if isinstance(opened, dict):
            opened["check_kind"] = "database measurement"
            opened["case_command_key"] = command_identity(command)
            opened["case_bindings"] = case_bindings(command)
            opened["case_scopes"] = case_scopes(command) + [
                {"case_id": "measure", "revision": 0, "requirement_ids": [], "scope": dict(scope)}]
            opened["tree_role"] = role
            opened["measure_pair"] = pair_id
            opened["measure_scale"] = scale
            if role == "baseline":
                opened["identity"] = pristine_identity(self.tree.base)
                opened["had_edit"] = False
                opened["cwd"] = os.path.realpath(where)
                opened["digest"] = patch_digest("")
        spool = os.path.join(self.measure_dir(), "%s-%s-%s.out" % (pair_id, scale, role))
        wrapped = "( %s ) > %s 2>&1; s=$?; tail -c %d %s; exit $s" % (
            command, shlex.quote(spool), MEASURE_SPOOL_TAIL, shlex.quote(spool))
        window = self.window_begin(target)
        try:
            job = self.pool.start(wrapped, hard_timeout=budget + LANE_GRACE_SEC, lane=SUITE_LANE,
                                  lane_wait=min(LANE_GRACE_SEC, budget), cwd=where)
        except LaneBusy:
            return {"record": None, "reason": "not started: a test run holds the database lane",
                    "exit": None, "wall": 0.0, "window": None, "tail": "", "timed_out": False}
        job.opened = opened
        job.scrub_target = target
        job.case_command = command
        if role == "baseline":
            base = self.tree.base
            job.identity_end_reader = lambda: pristine_identity(base) if worktree_clean(where) else ""
        done, out = job.wait(budget)
        timed_out = False
        if not done:
            job.timed_out = True
            timed_out = True
            job.stop()
        wall = max(0.0, time.monotonic() - job.started_monotonic)
        doc, aggregate = self.window_end(target, window, command, room)
        window_note = ""
        if aggregate is None and target is not None:
            window_note = doc.strip().lstrip("- ").strip()
            window_note = window_note.split("unavailable: ", 1)[-1] if "unavailable: " in window_note else window_note
        job.attach(doc, front=True)
        text = self.finish_shell(job, job._text())
        record = getattr(job, "measurement_record", None)
        tail = scrub(out[-MEASURE_TAIL_SHOWN:], target) if out else ""
        # The whole output, with host timing made steady and the checkout's
        # path written as the working tree's, so the two runs can be compared
        # line by line.
        comparable = steady(scrub(out, target) if out else "").replace(where, self.tree.root)
        return {"record": record, "reason": "", "exit": job.process.poll(), "wall": wall,
                "window": aggregate, "window_note": window_note, "tail": tail, "timed_out": timed_out,
                "text": text, "output": comparable}

    def do_measure(self, args: dict) -> str:
        """The same command in a base checkout and in the working tree, with work counts."""
        forbidden = execution_prohibited(statement_of(getattr(self, "warden", None)))
        if forbidden:
            raise ToolFault("database execution is prohibited by the statement: %s. "
                            "Use source inspection and independent static checks." % forbidden)
        command = str(args.get("command") or "").strip()
        if not command:
            raise ToolFault("command must not be empty")
        if len(command) > COMMAND_CHARS_MAX:
            raise ToolFault("that command is %d characters long. Write it to a script with "
                            "create_file and measure a short command that runs it." % len(command))
        blocked = HISTORY_GIT.search(command)
        if blocked:
            raise ToolFault("git %s is not available here, and measure already takes the baseline "
                            "reading in a separate checkout of the base commit." % blocked.group(1))
        outward = self.network_refusal(command)
        if outward:
            raise ToolFault("%s is not available here: this run has no network beyond the "
                            "repository's own database endpoints." % outward)
        changing = self.database_write_refusal(command)
        if changing:
            raise ToolFault(changing)
        scope = {}
        for key in ("caller", "workload", "dataset"):
            value = " ".join(str(args.get(key) or "").split())
            if not value or len(value) > 512:
                raise ToolFault("%s is required (at most 512 characters): say which code path, "
                                "which request or workload, and which fixture this measures. They "
                                "are recorded with the readings as your claims." % key)
            scope[key] = value
        baseline = args.get("baseline", True) is not False
        scales = args.get("scales") or []
        if not isinstance(scales, list) or len(scales) > MEASURE_SCALES_MAX:
            raise ToolFault("scales must be a list of at most %d entries" % MEASURE_SCALES_MAX)
        plan = []
        for entry in scales:
            if not isinstance(entry, dict) or not str(entry.get("label") or "").strip():
                raise ToolFault("each scale needs a label, a positive size and an optional setup command")
            size = entry.get("size")
            if not finite_number(size) or isinstance(size, bool) or size <= 0:
                raise ToolFault("scale %r needs a positive numeric size" % entry.get("label"))
            setup = str(entry.get("setup") or "").strip()
            if setup and (HISTORY_GIT.search(setup) or self.network_refusal(setup, exempt=False)):
                raise ToolFault("scale %r: its setup command is not available here" % entry.get("label"))
            plan.append({"label": re.sub(r"[^\w.-]+", "_", str(entry["label"]))[:32],
                         "size": float(size), "setup": setup})
        if not plan:
            plan = [{"label": "default", "size": None, "setup": ""}]
        try:
            asked = float(args.get("timeout") or MEASURE_RUN_SEC)
        except (TypeError, ValueError):
            raise ToolFault("timeout must be a number of seconds")
        if asked != asked:
            raise ToolFault("timeout must be a number of seconds")
        target = None
        target_note = ""
        try:
            target = self.pick_target(str(args.get("target") or ""))
        except ToolFault as error:
            target_note = str(error)
        room = self.allowance.clock_left() - FINISH_BUDGET_SEC - 30.0
        if room < 10.0:
            raise ToolFault("not enough of the run left to measure; make the change you already "
                            "have evidence for, or submit")
        budget = max(5.0, min(asked, SHELL_BUDGET_CEILING_SEC, room))
        global MEASURE_COUNTER
        MEASURE_COUNTER += 1
        pair_id = "m%d" % MEASURE_COUNTER
        where, where_note = (None, "no baseline requested")
        shared: list = []
        if baseline:
            if self.warden is None:
                where_note = "no baseline: this run has no checkout manager"
            else:
                where, where_note = self.warden.ensure_pristine()
                if where:
                    shared = link_dependency_dirs(self.tree.root, where)
        results = []
        self.progress_observation = ""
        for stage in plan:
            entry = {"label": stage["label"], "size": stage["size"], "before": None, "after": None,
                     "skipped": ""}
            results.append(entry)
            left = self.allowance.clock_left() - FINISH_BUDGET_SEC - 30.0
            if left < 10.0:
                entry["skipped"] = "not run: budget"
                continue
            if stage["setup"]:
                try:
                    job = self.pool.start(stage["setup"], hard_timeout=min(budget, left), lane=SUITE_LANE,
                                          lane_wait=min(LANE_GRACE_SEC, budget))
                except LaneBusy:
                    entry["skipped"] = "not run: a test run holds the database lane during setup"
                    continue
                job.opened = self.check_identity(stage["setup"])
                job.scrub_target = target
                done, out = job.wait(min(budget, left))
                if not done:
                    job.timed_out = True
                    job.stop()
                self.finish_shell(job, job._text())
                if job.process.poll() != 0:
                    entry["skipped"] = "not run: setup exited %s" % job.process.poll()
                    continue
            if where:
                left = self.allowance.clock_left() - FINISH_BUDGET_SEC - 30.0
                if left < 10.0:
                    entry["skipped"] = "baseline not run: budget"
                else:
                    entry["before"] = self.measure_run(command, where, "baseline", target, scope,
                                                       pair_id, stage["label"], min(budget, left),
                                                       max(0.0, left - min(budget, left)))
            left = self.allowance.clock_left() - FINISH_BUDGET_SEC - 30.0
            if left < 10.0:
                entry["skipped"] = (entry["skipped"] + "; " if entry["skipped"] else "") + "candidate not run: budget"
                continue
            entry["after"] = self.measure_run(command, self.tree.root, "candidate", target, scope,
                                              pair_id, stage["label"], min(budget, left),
                                              max(0.0, left - min(budget, left)))
        return self.measure_report(command, scope, pair_id, where, where_note, shared, results,
                                   target, target_note)

    def measure_report(self, command: str, scope: dict, pair_id: str, where, where_note: str,
                       shared: list, results: list, target, target_note: str) -> str:
        """The readings side by side, what paired, and what none of it establishes."""
        # Judged per run by its pair id: another run's pair can share this
        # command's text without saying anything about these two readings.
        paired = {id(record) for record in self.paired_measures()}
        lines = ["measure %s: `%s` (caller=%s, workload=%s, dataset=%s)" % (
            pair_id, " ".join(command.split())[:200], scope["caller"], scope["workload"], scope["dataset"])]
        base_sha = (self.tree.base or "")[:7] or "?"
        lines.append("baseline: %s%s%s; candidate: the working tree as it stands" % (
            "original code at %s in a " % base_sha if where else "", where_note,
            " (shared: %s)" % ", ".join(shared) if shared else ""))
        read_any = any((run or {}).get("window") is not None
                       for entry in results for run in (entry["before"], entry["after"]))
        ran_any = any(run is not None and not run.get("reason")
                      for entry in results for run in (entry["before"], entry["after"]))
        if target is None:
            lines.append("database work: not read (%s); wall time and exit status only" % target_note)
        elif ran_any and not read_any:
            lines.append("database work: not read from %s (its window could not be opened); wall time and "
                         "exit status only" % target.label)
        else:
            lines.append("database work read from %s (%s)" % (
                target.label, "system.query_log over the run's window" if target.engine == "clickhouse"
                else "pg_stat_database deltas over the run's window" if target.engine == "postgresql"
                else "no reader for this engine"))
        metrics = ("statements", "read_rows", "read_bytes", "memory_usage", "result_rows",
                   "result_bytes", "query_duration_ms", "tup_returned", "tup_fetched", "blks_read", "blks_hit")
        per_side: dict = {"before": [], "after": []}
        for entry in results:
            lines.append("scale %s%s:" % (entry["label"], " (size %s)" % thousands(entry["size"]) if entry["size"] else ""))
            if entry["skipped"]:
                lines.append("  %s" % entry["skipped"])
            before, after = entry["before"], entry["after"]
            if before is None and after is None:
                continue
            values = {}
            for side, run in (("before", before), ("after", after)):
                window = (run or {}).get("window") or {}
                values[side] = dict(window)
                if run is not None:
                    values[side]["wall_seconds"] = sig1_float(run["wall"])
                    values[side]["exit_code"] = run["exit"]
                    per_side[side].append((entry["size"], values[side]))
            rows = [m for m in metrics if any(m in values.get(s, {}) for s in ("before", "after"))]
            rows += ["wall_seconds", "exit_code"]
            lines.append("  %-18s %16s %16s   %s" % ("metric", "before", "after", "after/before"))
            for metric in rows:
                b = values.get("before", {}).get(metric) if before is not None else None
                a = values.get("after", {}).get(metric) if after is not None else None
                lines.append("  %-18s %16s %16s   %s" % (
                    metric, thousands(b) if before is not None else "not run",
                    thousands(a) if after is not None else "not run",
                    ratio_text(b, a) if metric != "exit_code" else ""))
            for side, run in (("before", before), ("after", after)):
                if run is None:
                    continue
                if run.get("reason"):
                    lines.append("  %s: %s" % (side, run["reason"]))
                elif run.get("timed_out"):
                    lines.append("  %s: timed out; its reading is not usable" % side)
                elif run.get("exit") not in (0, None):
                    lines.append("  %s: exited %s; its reading is recorded but not usable" % (side, run["exit"]))
                if target is not None and not run.get("reason") and run.get("window") is None:
                    lines.append("  %s: database work not read (%s)" % (side, run.get("window_note") or "no reading"))
            ids = [(side, (run or {}).get("record") or {}) for side, run in (("baseline", before), ("candidate", after))]
            lines.append("  records: %s" % (", ".join(
                "%s (%s)" % (record.get("measurement_check_id") or "no measurement record", side)
                for side, record in ids if record) or "none"))
            after_record = (after or {}).get("record") or {}
            if id(after_record) in paired:
                lines.append("  pair recorded: both readings carry the same units and workload; the hand-in reading can compare them")
            else:
                why = ("no baseline reading" if before is None else
                       before.get("reason") or ("baseline timed out" if before.get("timed_out") else
                       "baseline exited %s" % before["exit"] if before.get("exit") not in (0, None) else
                       "units differ between the two readings" if (before.get("record") or {}).get("metric_units")
                       != after_record.get("metric_units") else
                       "the working tree carries no edit yet, so the candidate reading is the original code"
                       if not after_record.get("after_edit") else
                       "the candidate reading is not usable (%s)" % after_record.get("detail", "")))
                lines.append("  pair not recorded: %s" % why)
            for side, run in (("before", before), ("after", after)):
                if run and run.get("tail", "").strip():
                    lines.append("  %s output tail: %s" % (side, " ".join(steady(run["tail"]).split())[-MEASURE_TAIL_SHOWN:]))
            if before is not None and after is not None and "output" in before and "output" in after:
                lines.append("  " + output_comparison(before["output"], after["output"]))
        for side in ("before", "after"):
            readings = [(size, values) for size, values in per_side[side] if size]
            for (s1, v1), (s2, v2) in zip(readings, readings[1:]):
                if not s1 or s2 == s1:
                    continue
                size_ratio = s2 / s1
                parts = []
                for metric in ("read_rows", "statements", "tup_returned", "wall_seconds"):
                    b, a = v1.get(metric), v2.get(metric)
                    if isinstance(b, (int, float)) and isinstance(a, (int, float)) and b > 0:
                        growth = a / b
                        parts.append("%s x%.3g%s" % (metric, growth,
                                     " (more than twice the size ratio)" if growth > 2 * size_ratio else ""))
                if parts:
                    lines.append("growth (%s): size x%.3g -> %s" % (side, size_ratio, ", ".join(parts)))
        lines.append("not established: which application path each run exercised; that every statement "
                     "in the window belongs to this command (a time window on the shared database, this "
                     "tool's own queries excluded); cache and data state between runs; result correctness "
                     "(compare outputs yourself); baseline dependency state (shared, not rebuilt)")
        return "\n".join(lines)

    def do_bash(self, args: dict) -> str:
        """Run one shell command in the checkout, under this run's rules.

        A command that would reach the network, rewrite history or change the live
        database is refused with the reason. Everything else runs in the background with
        its own timeout and lane, so a long command can be collected later rather than
        blocking the run.
        """
        self.sync_workspace()
        command = str(args.get("command") or "")
        if not command.strip():
            raise ToolFault("command must not be empty")
        if len(command) > COMMAND_CHARS_MAX:
            raise ToolFault(
                "that command is %d characters long. Write the content to a file with "
                "create_file and run a short command that uses it." % len(command))
        forbidden = execution_prohibited(statement_of(self.warden))
        if forbidden:
            checked = static_python_command(command, self.tree)
            if checked is None:
                raise ToolFault(
                    "Database execution is prohibited by the task: %s. "
                    "Arbitrary shell commands and application tests cannot be "
                    "established as database-free. Use read/edit/search tools "
                    "or a syntax-only Python check, such as python -m py_compile "
                    "path.py. No repository code is executed by that check."
                    % forbidden)
            command = checked
        outward = self.network_refusal(command)
        if outward:
            self.fence.fired("refused %r" % outward)
            raise ToolFault(
                "%s is not available here: this run has no network beyond the "
                "repository's own database endpoints%s. Work from the repository in "
                "front of you -- its code, its history and its tests -- and derive the "
                "change from that." % (outward, (
                    " (curl or wget to a host and port read from its configuration is "
                    "allowed%s)" % ("; the sql tool reaches them too" if DB_TOOL else ""))
                    if NETWORK_DB_EXEMPT else ""))
        blocked = HISTORY_GIT.search(command)
        if blocked:
            raise ToolFault(
                "git %s is not available here. Your changes are collected from the working "
                "tree as it stands, so moving or discarding them loses the work. Read-only "
                "git (status, diff, log, grep, show, ls-files) is fine.%s" % (blocked.group(1), (
                    " To compare against the original code, use measure: it runs a command in "
                    "a separate checkout of the base commit." if DB_TOOL and MEASURE_TOOL else ""))
            )
        changing = self.database_write_refusal(command)
        if changing:
            raise ToolFault(changing)
        want_bg = bool(args.get("background")) and ASYNC_SHELL
        asked = float(args.get("timeout") or 120)
        room = self.allowance.clock_left() - FINISH_BUDGET_SEC
        if room < 5.0:
            raise ToolFault(
                "not enough of the run left to wait on a command; make the "
                "change you already have evidence for, or submit")
        budget = max(5.0, min(asked, SHELL_BUDGET_CEILING_SEC, room))
        lane = SUITE_LANE if suite_shaped(command) else None
        grace = min(LANE_GRACE_SEC, budget, max(0.0, room)) if lane else 0.0
        holder = self.pool.lane_busy(lane) if lane else None
        opened = self.check_identity(command)
        self.progress_observation = ""  # Completion supplies the observation.
        try:
            job = self.pool.start(command, hard_timeout=room, lane=lane,
                                  lane_wait=grace)
        except LaneBusy:
            name = holder.name if holder else "another run"
            self.bg.fired("held back a suite run behind %s" % name)
            return (
                "not started. A test run is already going as %s, and two runs "
                "of the same suite share one database and one set of fixtures, "
                "so running both measures neither. Collect %s with bash_poll "
                "first, then run this if you still need it." % (name, name))
        job.opened = opened
        if want_bg:
            self.bg.fired("started %s: %s" % (job.name, redact(command)[:120]))
            return "started in the background as %s; collect it with bash_poll" % job.name
        done, out = job.wait(budget)
        self.sync_workspace()
        if done:
            return self.finish_shell(job, out)
        self.bg.fired("kept %s alive past %.0fs: %s"
                      % (job.name, budget, redact(command)[:120]))
        return (
            job.result(out)
            + "\n\n[still running after the %.0fs wait, moved to the background as %s; "
            "keep working and collect it later with bash_poll]" % (asked, job.name)
        )

    def database_write_refusal(self, command: str) -> str:
        """Why this line may not run: it would change a database outside the sql tool, or ""."""
        if not DB_WRITE_FENCE or not re.search(r"psql|clickhouse|curl|wget", command or ""):
            return ""
        stated = {" ".join(argv) for argv in command_lines(statement_of(getattr(self, "warden", None)))}
        for engine, argv, parts, whole in client_invocations(command):
            if " ".join(argv) in stated:
                continue
            verb = statements_write([*parts, "COMMIT"] if whole else parts)
            if verb:
                return self.database_write_text(verb, engine)
        for invocation in network_invocations(command) or []:
            if invocation["tool"] in ("curl", "wget"):
                verb = statements_write(network_payloads(invocation))
                if verb:
                    return self.database_write_text(verb, "clickhouse")
        return ""

    @staticmethod
    def database_write_text(verb: str, engine: str) -> str:
        """The refusal for a command that would change the live database, and what to do instead.
        """
        return ("Not run: this sends %s to the live database, where the change stays after the "
                "command ends. The answer is the patch, and a change made in the database is not "
                "part of it; the baseline checkout, measure and the project's tests also read this "
                "same database. To see what the statement does, use the sql tool%s." % (verb, (
                    " (a PostgreSQL statement runs there inside a transaction that is rolled back), "
                    "or send BEGIN, the statements and ROLLBACK in one psql call"
                    if engine == "postgresql" else
                    "; write=true is how it makes a lasting ClickHouse change deliberately")))

    def network_refusal(self, command: str, exempt: bool = True) -> str:
        """The network program this line may not run, or "" when it may run.

        Every command in the line is read, wrappers and shell -c scripts
        included. With the database exemption, curl and wget may run when every
        destination they would contact is a configured database endpoint.
        """
        if not NETWORK_FENCE or not NETWORK_WORD.search(command or ""):
            return ""
        invocations = network_invocations(command)
        if invocations is None:
            # A line that cannot be read as shell words is refused whenever it names a
            # network program anywhere, not only where the old pattern looked.
            found = NETWORK_COMMAND.search(command) or NETWORK_TOOL_WORD.search(command)
            return found.group(1) if found else ""
        if SUBSTITUTION.search(command) and (invocations or NETWORK_COMMAND.search(command)):
            # A substitution can run a network command, or feed one a
            # destination, that the words on the line do not show.
            return invocations[0]["tool"] if invocations else NETWORK_COMMAND.search(command).group(1)
        if not invocations:
            return ""
        if exempt and NETWORK_DB_EXEMPT:
            allowed = self.network_exemption(invocations)
            if allowed:
                self.fence.fired("allowed %s" % allowed)
                return ""
        return next((i["tool"] for i in invocations if i["tool"] not in ("curl", "wget")),
                    invocations[0]["tool"])

    def network_exemption(self, invocations: list) -> str:
        """Why these network commands may run: every destination is a configured database.

        Only curl and wget, with no environment assignment in front of them and
        no option that adds a destination the line does not show; each URL, or
        URL variable, must resolve to the host and port of an endpoint this
        repository configures. Anything else stays refused.
        """
        if not invocations or any(i["tool"] not in ("curl", "wget") for i in invocations):
            return ""
        try:
            targets = self.database_targets()
        except BaseException:
            return ""
        if not targets:
            return ""
        hosts = {(str(t.host).lower(), t.port or DEFAULT_PORTS.get(t.engine, 0)) for t in targets}
        reached = []
        for invocation in invocations:
            if invocation["assignments"]:
                return ""
            destinations = network_destinations(invocation["tool"], invocation["args"])
            if not destinations:
                return ""
            for word in destinations:
                verified = verified_destination(word, hosts)
                if not verified:
                    return ""
                reached.append(verified)
        tools = ", ".join(dict.fromkeys(i["tool"] for i in invocations))
        return "%s to %s (configured database endpoint)" % (tools, ", ".join(dict.fromkeys(reached)))

    def do_bash_poll(self, args: dict) -> str:
        """Collect a background command, waiting for it when the run still has time."""
        job = self.pool.get(str(args.get("job") or ""))
        waiter = getattr(job, "wait", None)
        if callable(waiter) and not job.finished():
            room = self.allowance.clock_left() - FINISH_BUDGET_SEC
            if room > 1.0:
                try:
                    waiter(min(BASH_POLL_WAIT_SEC, room))
                except BaseException as error:
                    self.bg.fired("the poll's wait ended early: %s" % type(error).__name__)
        self.sync_workspace()
        # Read after polling so a completed job's final bytes are available.
        done = job.finished()
        out = job._text()
        if done:
            return self.finish_shell(job, out)
        self.progress_observation = ""  # Live jobs suppress recovery in drive.
        if getattr(job, "sql_target", None) is not None:
            out = scrub(out, job.sql_target)
        return "[running; collect again with bash_poll]\n" + (clip(out, SHELL_OUTPUT_CAP, "partial output") or "(no output)")

    def finish_shell(self, job: Shell, out: str) -> str:
        """Close out a finished command: record it, read its output and say what it means."""
        self.pool.jobs.pop(job.name, None)
        try:
            self.sync_workspace()
            target = getattr(job, "scrub_target", None) or getattr(job, "sql_target", None)
            if target is not None:
                out = scrub(out, target)
            if not getattr(job, "output_complete", True) and OUTPUT_INCOMPLETE_MARKER not in out:
                out = OUTPUT_INCOMPLETE_MARKER + " verification unresolved]\n" + out
            command = scrub(job.command, target) if target is not None else job.command
            report_shell(job, out)
            self.consult_record(command, out, job.process.poll())
            self.note_findings(command, out)
            self.note_verified(job)
            record = getattr(job, "measurement_record", {})
            opened = getattr(job, "opened", None) or {}
            # Shell timing, job names and measurement handles are presentation
            # metadata. Their changing values do not make identical evidence
            # new. Preserve actual output bytes and execution/source context;
            # even a changed duration printed by the application stays new.
            self.progress_observation = command_identity(json.dumps([
                "command observation", command, getattr(job, "cwd", getattr(self.pool, "cwd", "")),
                getattr(job, "environment_key", ""), opened.get("identity"),
                record.get("identity_end"), self.source_edits,
                job.process.poll(), bool(getattr(job, "timed_out", False)),
                bool(getattr(job, "output_complete", True)), command_identity(out),
            ], ensure_ascii=True))
            result = job.result(out)
            measurement = measurement_note(getattr(job, "measurement_record", {}))
            if measurement:
                result += "\n" + measurement
            # How long the collection took is recorded with the check, not shown:
            # the model's reading of a result should not change with the host.
            failure = execution_failure(job.process.poll(), out, bool(getattr(job, "timed_out", False)))
            if failure in ("missing_or_unusable_executable", "unsupported_checker_configuration"):
                result += ("\nTooling failure: this does not establish database availability or application correctness. "
                           "Use an available repository client or check command; no replacement command was run.")
                sources = sorted({redact(str(t.source)) for t in (getattr(self, "_targets", None) or [])})
                if sources:
                    result += "\nPreviously discovered connection sources: " + ", ".join(sources[:3])[:600]
            opened = getattr(job, "opened", None)
            if opened and opened.get("identity") != self.source_identity():
                result += "\nSource changed during this invocation. Finish setup or restoration, then run a separate stable check; this result cannot verify the final source."
            return self.sql_result(job, out, result) if getattr(job, "sql_target", None) else result
        finally:
            job.stop()

    def do_submit(self, args: dict) -> str:
        """Hand in the answer, unless something holds it back.

        An answer that would be an empty patch is refused outright. Otherwise the holds
        are asked in turn, and the run goes on; when nothing holds it, this is the end
        of the run.
        """
        refused = empty_answer_note(self)
        if refused:
            say("[HANDIN] submit refused: the answer would be an empty patch")
            return refused
        held = self.handin_hold()
        if held:
            return held
        faults = self.warden.verdict() if self.warden else []
        if not faults:
            note = self.handin_pause()
            if note:
                return note
            raise Finished(str(args.get("summary") or ""))
        return ("Not handed in. Review these recorded concerns against the "
                "current task and command results; a diagnostic alone does "
                "not establish an unmet requirement:\n\n"
                + "\n\n".join(faults[:3])
                + "\n\nThere is budget left. Fix this and call submit again.")

    def handin_hold(self) -> str:
        """A hold that happens once before the answer goes, or "".

        A command the model started whose result it has not read is named once
        while the run can still act on it. The next submit hands in either way.
        """
        try:
            if (self.allowance.clock_left() < HANDIN_HOLD_MIN_SEC
                    or self.allowance.money_left() <= 0):
                return ""
            if JOB_HANDIN and not getattr(self, "job_held", False):
                running = self.unread_jobs()
                if running:
                    self.job_held = True
                    job = running[0]
                    return ("Not handed in yet: %s (`%s`) is still running, and "
                            "its result has not been read. Its output so far:\n%s\n"
                            "Collect it with bash_poll, or call submit again to hand in "
                            "without it." % (job.name, redact(one_line(job.command))[:160],
                                             clip(redact(steady(job._tail())), 1500, "output")))
        except (ToolFault, OSError, AttributeError):
            return ""
        return ""

    def unread_jobs(self) -> list:
        """Commands the model started that are still running, oldest first."""
        own = getattr(self.warden, "job", None) if self.warden else None
        return sorted((job for job in list(self.pool.jobs.values())
                       if isinstance(job, Shell) and job is not own
                       and hasattr(job, "opened") and not job.finished()),
                      key=lambda job: job.started)

    def review_reading(self) -> str:
        """The second reader's anchored doubts as a note for the driver, or "".

        Runs once per run, at the first hand-in that reaches it, while the run
        still has time and money to act on what it says.
        """
        if self.review_state != "armed":
            return ""
        try:
            if (self.allowance.clock_left() < REVIEW_MIN_WALL_SEC
                    or self.allowance.money_left() < REVIEW_MIN_USD
                    or not self.allowance.edits):
                self.review.skipped("too little left to act on a reading, or nothing changed")
                return ""
            diff = self.tree.diff(SELFREVIEW_DIFF_SEC)
            if not diff.strip():
                self.review.skipped("the answer holds no change yet")
                return ""
        except Exception as error:
            self.review.skipped("could not start: %s" % type(error).__name__)
            return ""
        self.review_state = "done"
        seed = getattr(self.driver_seat, "seed", None)
        snapshots = {path: text for path, text in ((p, self.text_of(p)) for p in diff_paths(diff))
                     if text is not None}
        try:
            stop, findings, closed = run_review(statement_of(self.warden), self.tree, self.pool,
                                                self.allowance, self.warden, self.review, seed, diff,
                                                "\n".join(self.consult_log))
        except Exception as error:
            self.review.skipped("gave up: %s" % type(error).__name__)
            return ""
        note = self.closing_verdict(closed, snapshots) if closed else ""
        if not findings:
            return note or (REVIEW_EMPTY if stop == "done" else "")
        return note + REVIEW_SPLICE % clip(redact(review_findings_text(findings)), 6000, "review")

    def closing_verdict(self, closed: list, snapshots: dict) -> str:
        """Keep the reader's correction only when the task's stated check passes on it."""
        where = ", ".join(sorted(set(closed)))
        statement = statement_of(self.warden)
        commands = [stated_runner(statement, self.tree.root)]
        commands += [" ".join(argv) for argv in check_commands(statement)]
        commands = [c for c in commands if c][:2]
        if not commands:
            self.review.fired("correction kept unchecked: the statement names no check")
            return CLOSER_UNCHECKED % where
        outcomes = [(command, self.closing_run(command)) for command in commands]
        failed = [(c, out) for c, out in outcomes if "exit_code=0" not in out]
        if not failed:
            self.review.fired("correction kept: %s" % "; ".join(redact(c) for c, _ in outcomes))
            return CLOSER_KEPT % (where, "; ".join(redact(c) for c, _ in outcomes))
        restored = [self.restore_text(path, text) for path, text in snapshots.items()]
        self.seen.clear()
        command, out = failed[0]
        last = (out.strip().splitlines() or [""])[-1][:200]
        self.review.fired("correction reverted (%d file(s) restored): %s" % (sum(restored), redact(command)))
        return CLOSER_REVERTED % (where, redact(command), redact(last))

    def text_of(self, path: str) -> str | None:
        """The file's exact bytes as text, or None when it cannot be read."""
        try:
            return self.tree.read_exact(path)
        except Exception:
            return None

    def restore_text(self, path: str, text: str) -> bool:
        """Put a file back as it was before the reader's correction."""
        try:
            self.tree.write(path, text)
        except Exception:
            return False
        return True

    def closing_run(self, command: str) -> str:
        """One stated check, run to completion within the closer's wall, as text."""
        deadline = time.monotonic() + CLOSER_CHECK_SEC
        try:
            out = self.run("bash", {"command": command, "timeout": CLOSER_CHECK_SEC})
            job = re.search(r"started in the background as (\S+)", out)
            while job and "exit_code=" not in out and time.monotonic() < deadline:
                out = self.run("bash_poll", {"job": job.group(1)})
            return out
        except (ToolFault, Finished) as fault:
            return "error: %s" % fault
        except Exception as error:
            return "error: %s: %s" % (type(error).__name__, error)

    def expectation_required_note(self) -> str | None:
        """Hold the hand-in once on a database task that lists its rows and has checked no expectation."""
        if not (EXPECT_TOOL and EXPECT_REQUIRED) or self.expect_required_pauses >= EXPECT_REQUIRED_MAX:
            return None
        if self.allowance.clock_left() < EXPECT_REQUIRED_MIN_SEC or self.allowance.money_left() <= 0:
            return None
        statement = statement_of(self.warden)
        if not stated_requirements(statement) or not self.probe_needs_real_data():
            return None
        current = self.source_identity()
        if any(r.get("met") and r.get("identity") == current for r in EXPECTATIONS):
            return None
        self.expect_required_pauses += 1
        self.expect.fired("held the hand-in: no expectation checked on the current code")
        return EXPECT_REQUIRED_TEXT

    def handin_pause(self) -> str | None:
        """The one thing standing between this answer and the hand-in, or None when nothing does.

        A tree the run already knows is wrong is not sent to the second reader, because
        the hold goes back on its own and the reader's budget is better spent on an
        answer that might be right.
        """
        self.work_meter.close(self)
        held = self.expectation_note() or self.green_note() or self.expectation_required_note()
        if held:
            # A tree the run already knows is wrong is not read by the second
            # reader; the hold goes back on its own.
            return self.work_meter.extend(self, held)
        note = self._handin_pause()
        reading = self.review_reading()
        if reading and reading != REVIEW_EMPTY and not note:
            # Anchored doubts are a pause of their own when no other fired.
            note = "Not handed in yet: a second reading of the change is below."
            self.pauses += 1
        if reading and note:
            note += reading
        if note:
            # The readings of the change as it stands go with whichever pause is
            # asked: only one is, and a statement that lists requirements asks
            # the requirement reading rather than the self-review.
            note += self.unbound_reading() + self.clickhouse_names_reading()
            advisory = getattr(self.warden, "contract_advisories", []) if self.warden else []
            if advisory:
                note += "\n\nAdvisory contract observations (not additional requirements):\n" + clip(
                    redact("\n".join(advisory)), 2000, "advisory detail")
            whitespace = self.warden.whitespace_advisory() if self.warden else ""
            if whitespace:
                note += "\n\n" + whitespace
        return self.work_meter.extend(self, note)

    def do_expect(self, args: dict) -> str:
        """Run one probe and compare its rows with what the model says the task's sentence requires."""
        statement = statement_of(self.warden)
        quote = str(args.get("requirement_quote") or "").strip()
        if not quoted_in(statement, quote):
            raise ToolFault("requirement_quote must be the task's own words, copied exactly: at least "
                            "%d characters that appear in the statement" % EXPECT_QUOTE_MIN)
        expected = args.get("expected")
        if not isinstance(expected, list) or len(expected) > EXPECT_ROWS_MAX:
            raise ToolFault("expected must be a list of rows, each a list of values (an empty list means "
                            "the probe must return no rows), at most %d rows" % EXPECT_ROWS_MAX)
        rows = [row if isinstance(row, list) else [row] for row in expected]
        sql = str(args.get("sql") or "").strip()
        command = str(args.get("command") or "").strip()
        if bool(sql) == bool(command):
            raise ToolFault("give exactly one probe: sql or command")
        if sql and not DB_TOOL:
            raise ToolFault("the sql tool is not offered in this run; give a command probe instead")
        if command and ROWS_SUPPLIED.search(command) and self.probe_needs_real_data():
            raise ToolFault("a database is configured here, so the probe must read the application's "
                            "data: give sql, or a command that runs the application's own code against "
                            "it. A command that supplies its own rows or replaces the data access can "
                            "only agree with the reading that wrote it; the rows you pin must come from "
                            "the sentence applied to rows you have read")
        ordered = bool(args.get("ordered"))
        label = " ".join(str(args.get("label") or quote[:40]).split())[:60]
        if sql:
            out = self.do_sql({"query": sql, "target": str(args.get("target") or ""), "timeout": EXPECT_PROBE_SEC})
            probe = "sql :: " + " ".join(sql.split())
        else:
            out = self.do_bash({"command": command, "timeout": EXPECT_PROBE_SEC})
            probe = "bash :: " + " ".join(command.split())
        if PROBE_NOT_RUN.search(out):
            return ("the probe did not run to completion, so the expectation is not settled; collect "
                    "or rerun it, then pin the expectation again.\n%s" % clip(out, 2000, "probe output"))
        if PROBE_CUT.search(out):
            return ("the probe's output was cut for length, so the expectation is not settled; use a "
                    "probe that returns fewer rows, or aggregate.\n%s" % clip(out, 2000, "probe output"))
        returned = table_rows(out, sql=bool(sql))
        difference = rows_differ(returned, rows, ordered)
        EXPECTATIONS.append({"key": quote + " :: " + probe, "label": label, "quote": quote, "probe": probe,
                             "args": {"requirement_quote": quote, "sql": sql, "command": command,
                                      "target": str(args.get("target") or ""), "expected": rows,
                                      "ordered": ordered, "label": label},
                             "ordered": ordered, "expected": len(rows), "returned": len(returned),
                             "met": not difference, "difference": difference, "edits": self.allowance.edits,
                             "identity": self.source_identity(), "time": time.time()})
        self.expect.fired("%s %s :: %s" % ("met" if not difference else "unmet", label, redact(probe)[:120]))
        head = ("expectation met: the probe returned the %d row(s) you pinned for that sentence" % len(returned)
                if not difference else "expectation NOT met: %s" % difference)
        return "%s\n(quoting: \"%s\")\nprobe output, as read:\n%s" % (
            head, quote[:200], clip(out, 4000, "probe output"))

    def probe_needs_real_data(self) -> bool:
        """Is there application data a probe could read? Discovery failures count as no."""
        try:
            return bool(self.database_targets())
        except BaseException:
            return False

    def expectation_note(self) -> str | None:
        """Hold the hand-in while a pinned expectation fails on the code as it stands."""
        if not EXPECT_TOOL or not EXPECTATIONS or self.expect_pauses >= EXPECT_PAUSES_MAX:
            return None
        if self.allowance.clock_left() < HANDIN_HOLD_MIN_SEC or self.allowance.money_left() <= 0:
            return None
        latest: dict = {}
        for record in EXPECTATIONS:
            latest[record["key"]] = record
        current = self.source_identity()
        unmet = [r for r in latest.values() if not r["met"] and r["identity"] == current]
        if unmet:
            self.expect_pauses += 1
            self.expect.fired("held the hand-in: %d unmet on the current code" % len(unmet))
            return EXPECT_UNMET % "\n".join("- %s: %s (probe: %s)" % (r["label"], r["difference"], redact(r["probe"])[:120])
                                            for r in unmet[:4])
        ran_now = any(r["identity"] == current for r in latest.values())
        stale = [r for r in latest.values() if not r["met"] and r["identity"] != current]
        if stale and not ran_now:
            self.expect_pauses += 1
            self.expect.fired("held the hand-in: %d unmet before the last edit, nothing run since" % len(stale))
            return EXPECT_STALE % (len(stale), ", ".join(r["label"] for r in stale[:4]))
        return None

    def remember_green(self, record: dict) -> None:
        """Keep the tree that just passed the task's named check, as a patch."""
        if (not GREEN_MEMORY or record.get("kind") != "named regression check"
                or not record.get("evidence") or not record.get("after_edit") or record.get("stale")):
            return
        incomplete: list = []
        try:
            patch = self.tree.diff(SELFREVIEW_DIFF_SEC, failed=incomplete)
        except BaseException:
            return
        if incomplete or not patch.strip() or len(patch) > GREEN_PATCH_CHARS:
            return
        self.green = {"patch": patch, "command": str(record.get("command") or ""),
                      "identity": record.get("identity_end") or record.get("identity") or "",
                      "edits": self.allowance.edits}
        self.green_beacon.fired("kept the tree after edit %d: %s"
                                % (self.allowance.edits, redact(self.green["command"])[:100]))

    def green_note(self) -> str | None:
        """Hold the hand-in once when the kept passing tree is gone and its check now fails."""
        if not GREEN_MEMORY or not self.green or self.green_noted:
            return None
        if self.allowance.clock_left() < HANDIN_HOLD_MIN_SEC or self.allowance.money_left() <= 0:
            return None
        current = self.source_identity()
        if current == self.green["identity"]:
            return None
        kept = " ".join(self.green["command"].split())
        failing = [r for r in current_records(CHECKS, current)
                   if r.get("kind") == "named regression check" and r.get("outcome") == "failed"
                   and completed_record_time(r) is not None
                   and " ".join(str(r.get("command") or "").split()) == kept]
        if not failing:
            return None
        self.green_noted = True
        self.green_beacon.fired("held the hand-in: the kept tree's check fails now")
        return GREEN_LOST % (self.green["edits"], redact(self.green["command"])[:120])

    def do_restore_green(self, args: dict) -> str:
        """Put the files back to the last state that passed the task's named check."""
        why = " ".join(str(args.get("why") or "").split())[:200]
        if not self.green:
            raise ToolFault("no passing state has been kept in this run")
        if self.allowance.clock_left() < GREEN_RESTORE_MIN_SEC:
            raise ToolFault("too little of the run is left to restore and re-check a tree")
        incomplete: list = []
        current = self.tree.diff(SELFREVIEW_DIFF_SEC, failed=incomplete)
        if incomplete:
            raise ToolFault("the tree as it stands could not be read completely; restore was not attempted")
        self.tree.restore(30.0)
        code, out = self.apply_patch(self.green["patch"])
        if code != 0:
            recovered, back = self.apply_patch(current) if current.strip() else (0, "")
            self.count_restore()
            if recovered == 0:
                return ("the kept patch did not apply cleanly (%s); the files are back as they were before "
                        "this call, nothing was lost" % clip(out, 400, "git"))
            return ("the kept patch did not apply cleanly (%s), and the previous work could not be put back "
                    "either (%s); the tree is at the original now, re-apply your edits"
                    % (clip(out, 300, "git"), clip(back, 300, "git")))
        self.count_restore()
        self.green_beacon.fired("restored the tree kept after edit %d%s" % (self.green["edits"], (": " + why) if why else ""))
        return ("restored the tree to the state after edit %d that passed %s; run the check again before handing in"
                % (self.green["edits"], redact(self.green["command"])[:120]))

    def apply_patch(self, patch: str) -> tuple:
        """Apply a patch to the tree with git; (exit code, git's words)."""
        handle, path = tempfile.mkstemp(prefix="ridges-green-", suffix=".diff")
        try:
            with os.fdopen(handle, "w", encoding="utf-8", errors="surrogateescape") as fh:
                fh.write(patch)
            return git(["apply", "--whitespace=nowarn", path], self.tree.root, 30.0)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def count_restore(self) -> None:
        """Count a restore as an edit, so what follows is judged against the tree as it now stands.
        """
        self.allowance.edits += 1
        self.source_edits += 1
        self.seen.clear()
        self.sync_workspace(count_edits=False)

    def _handin_pause(self) -> str | None:
        """The next hold in the hand-in ladder, with each state asked only once."""
        if not HIDDEN_SELFREVIEW:
            return self.selfreview_note() or self.conform_note()
        for state, name in HANDIN_PAUSES:
            if getattr(self, state, "off") == "asked":
                return getattr(self, name)()
        if self.pauses >= HANDIN_PAUSES_MAX:
            return None
        for _, name in HANDIN_PAUSES:
            note = getattr(self, name)()
            if note:
                self.pauses += 1
                return note
        return None

    def selfreview_close(self, why: str) -> list[str]:
        """Close the second reading when the run ends, whatever ended it."""
        try:
            self.work_meter.close(self)
            return self._selfreview_close(why)
        except BaseException as error:
            try:
                self.selfreview.skipped("the closing reading ended on %s"
                                        % type(error).__name__)
            except BaseException:
                pass
            return []

    def _selfreview_close(self, why: str) -> list[str]:
        """Record what the second reading bought, and whether it stayed inside the change's own
        region.
        """
        if self.selfreview_closed or not self.selfreview_asked:
            if self.selfreview_state == "armed":
                self.selfreview_state = "done"
                self.selfreview.skipped("the answer went out before the "
                                        "review could be offered: %s" % why)
            return []
        self.selfreview_closed = True
        self.selfreview_state = "done"
        self.selfreview.fired("closing on %s" % why)
        self.selfreview_outcome()
        if self.warden is None:
            self.selfreview.skipped("nothing here reads where the change landed")
            return []
        done, faults = self.warden.final_faults(SELFREVIEW_CLOSE_SEC)
        if not done:
            self.selfreview.skipped("where the change landed was not read")
            return []
        if faults:
            self.selfreview.fired("what the review bought lands outside the "
                                  "region: %s" % faults[0][:160])
        else:
            self.selfreview.fired("%d of 2 reader(s): what the review bought "
                                  "stays inside the region" % done)
        return faults

    def consult_record(self, command: str, out: str, returncode: int | None = None) -> None:
        """Keep one command's result for the readers that will be asked about it later."""
        self.work_meter.record(command, out, returncode)
        if self.consult_state != "armed" and self.review_state != "armed":
            return
        tail = ""
        for line in reversed((out or "").splitlines()):
            if line.strip() and line.strip() != STILL_RUNNING:
                tail = line.strip()
                break
        self.consult_log.append("$ %s\n    %s"
                                % (redact(" ".join((command or "").split()))[:SHELL_REPORT_CAP],
                                   tail[:SHELL_REPORT_CAP]))
        del self.consult_log[:-CONSULT_HISTORY_LINES]

    def consult_ask(self, statement: str, diff: str) -> str:
        """Ask a fresh seat, with no transcript, what it makes of the statement and the change.
        """
        seat = Seat(self.allowance, models=[CONSULT_MODEL], patient=False,
                    impatient_sec=CONSULT_CALL_SEC,
                    effort=CONSULT_EFFORT, reply_ceiling=CONSULT_MAX_TOKENS)
        reply = seat.ask(
            [{"role": "system", "content": CONSULT_BRIEF},
             {"role": "user", "content": CONSULT_REQUEST % (
                 clip(statement or "", CONSULT_SECTION_CHARS, "the task"),
                 clip(diff or "", CONSULT_SECTION_CHARS, "the patch") or "(nothing yet)",
                 clip("\n".join(self.consult_log), CONSULT_SECTION_CHARS,
                      "what has been run") or "(nothing yet)")}],
            None)
        return str(reply.get("content") or "").strip()

    def consult_splice(self, note: str, diff: str) -> str:
        """Add a fresh reading of the change to a hand-in note, once, when the run can afford it.
        """
        if self.consult_state != "armed":
            return note
        self.consult_state = "done"
        self.consult.reached(self.allowance.calls, self.allowance.spent,
                             self.allowance.clock_left())
        if self.allowance.clock_left() < CONSULT_MIN_WALL_SEC:
            self.consult.skipped("too little of the run left to ask and still "
                                 "act on the answer")
            return note
        if not diff:
            self.consult.skipped("there is no reading of the change to show")
            return note
        calls, spent = self.allowance.calls, self.allowance.spent
        answer, why = "", ""
        try:
            answer = self.consult_ask(statement_of(self.warden), diff)
        except BaseException as error:
            why = "the reader did not answer: %s" % type(error).__name__
        if not (answer or why):
            why = "the reader answered with nothing"
        self.consult.calls = self.allowance.calls - calls
        self.consult.usd = self.allowance.spent - spent
        self.consult.bill()
        if why:
            self.consult.skipped(why)
            return note
        self.consult.fired("consulted: %s answer" % length_class(answer))
        return note + CONSULT_SPLICE % clip(answer, CONSULT_REPLY_CHARS,
                                            "the reader's answer")

    def named_changed_paths(self, command: str, out: str) -> list:
        """Files this run changed whose path or name appears in a check.

        A candidate signal only. It cannot establish that the check exercised
        the changed code, and it is never reported as if it had.
        """
        try:
            text = "%s\n%s" % (command or "", out or "")
            hits = []
            for path in self.tree.changed_paths(SELFREVIEW_DIFF_SEC):
                stem = os.path.splitext(os.path.basename(path))[0]
                if path in text or (len(stem) > 3 and stem in text):
                    hits.append(path)
            return hits[:4]
        except BaseException:
            return []

    def evidence_summary(self) -> dict:
        """What the checks in this run observed, separated from what they imply.

        A passing check is evidence about the tree it ran on and the command it
        ran. It becomes evidence about the requested change only if it ran after
        the change existed; whether it exercised that change is a candidate
        reading, kept separate here rather than assumed.
        """
        current = self.source_identity()
        # Evidence is about the tree as it stands. An earlier pass stays in the
        # record for reading, but it is not evidence for a source that has
        # changed since, and an unknown identity is never a match.
        usable = current_usable_records(CHECKS, current)
        after = [record for record in usable if record.get("after_edit")]
        naming = [record for record in after if record.get("names_changed_path")]
        return {
            "usable": usable,
            "kinds": {record.get("kind") for record in usable},
            "after_edit": after,
            "naming_edit": naming,
            "distinguished": self.observed_change(current),
            "measurement_pairs": self.measurement_pairs(current),
            "failed_now": [record for record in current_records(CHECKS, current)
                           if record.get("outcome") == "failed"
                           and stable_record(record)
                           and record.get("identity") == current],
            "measurements": [record for record in usable
                             if record.get("kind") == "database measurement"],
            "ran_anything": bool(CHECKS),
        }

    def checks_by_command(self, kind: str = "") -> dict:
        """Records grouped by the complete invocation and its context."""
        grouped: dict = {}
        for record in CHECKS:
            if record.get("kind") == "other command":
                continue
            if kind and record.get("kind") != kind:
                continue
            context = record_context(record)
            if context:
                grouped.setdefault(context, []).append(record)
        return grouped

    def observed_change(self, current: str = "") -> list:
        """Commands that failed on other source and pass on this candidate.

        A pair counts only when the passing half is about the tree as it stands:
        an older pair says the run once had something working, not that what is
        about to be handed in works. The direction matters too, and an arbitrary
        command is not a check, so neither clears the question.
        """
        current = current or self.source_identity()
        return observed_change_records(CHECKS, current)

    def measurement_pairs(self, current: str = "") -> list:
        """Measurements of one command on either side of the change.

        Two readings of the same state are one observation repeated, not a
        comparison, so the pair must straddle the edit.
        """
        current = current or self.source_identity()
        if not current:
            return []
        pairs = []
        current_successes = current_usable_records(CHECKS, current)
        for records in self.checks_by_command("database measurement").values():
            after = [r for r in records if r.get("after_edit")
                     and r.get("identity") == current
                     and r.get("outcome") == "measured" and r in current_successes]
            before = [r for r in records if r.get("outcome") == "measured"
                      and r.get("identity") != current and usable_record(r)]
            for current_reading in after:
                units = current_reading.get("metric_units")
                scope = workload_context(current_reading)
                if units and scope is not None and any(previous.get("metric_units") == units
                                 and workload_context(previous) == scope
                                 and earlier_record(previous, current_reading)
                                 for previous in before):
                    pairs.append(current_reading["command"])
                    break
        for after in self.paired_measures(current, current_successes):
            if after["command"] not in pairs:
                pairs.append(after["command"])
        return pairs

    def paired_measures(self, current: str = "", current_successes=None) -> list:
        """Candidate readings of the measure tool that pair with their own baseline.

        Readings the measure tool took in a separate base checkout carry a
        different working directory, so they never share a command group with
        the live tree's reading. They pair by the explicit pair id, under every
        other condition of measurement_pairs: units, workload, order, usability
        and a baseline identity that is not the candidate's.
        """
        current = current or self.source_identity()
        if not current:
            return []
        if current_successes is None:
            current_successes = current_usable_records(CHECKS, current)
        found = []
        for after in CHECKS:
            if not (after.get("measure_pair") and after.get("tree_role") == "candidate"
                    and after in current_successes and after.get("outcome") == "measured"
                    and after.get("after_edit")):
                continue
            for previous in CHECKS:
                if (previous.get("measure_pair") == after["measure_pair"]
                        and previous.get("tree_role") == "baseline"
                        and previous.get("outcome") == "measured"
                        and usable_record(previous) and previous.get("identity") != current
                        and previous.get("metric_units") == after.get("metric_units")
                        and workload_context(previous) is not None
                        and workload_context(previous) == workload_context(after)
                        and earlier_record(previous, after)):
                    found.append(after)
                    break
        return found

    CARRY_CHARS = 2000

    def carried_note(self) -> str:
        """The run's own findings, in a form a transcript shrink cannot blur.

        Case claims and observations remain separate. Bound each field instead
        of clipping the assembled block: clipping its middle can erase the
        current failure while retaining older passes at the end.
        """
        try:
            def bounded(value, cap):
                value = " ".join(str(value).split())
                if len(value) <= cap:
                    return value
                head = (cap - 5) // 2
                return value[:head] + " ... " + value[-(cap - 5 - head):]

            lines = []
            current = self.source_identity()
            coverage = self.requirement_coverage()
            if coverage:
                uncovered = [key for key, value in coverage.items() if not value["declared"]]
                unresolved = [key for key, value in coverage.items() if value["unresolved"]]
                lines.append("requirement coverage is claimed: %d without a linked case; %d with unresolved assertions. IDs: %s%s" % (
                    len(uncovered), len(unresolved), ",".join((uncovered + unresolved)[:6]),
                    "; more omitted" if len(uncovered) + len(unresolved) > 6 else ""))
            cases = declared_cases()
            cases.sort(key=lambda entry: case_status(entry[1], current) == "observed")
            if len(cases) > 1 or NAMED_CASES:
                lines.append("cases: assertion coverage remains the caller's claim")
                for name, case in cases:
                    state = case.get("status", "proposed")
                    if state != "proposed" and case.get("observed_identity", case.get("identity")) != current:
                        state += "; observation superseded"
                    quoted = case.get("declared_identity", case.get("identity"))
                    refs = case.get("requirement_ids", [])
                    lines.append("case %s r%s: %s; quote %s; asserts %s; refs %s%s" % (
                        name or "unnamed", case.get("revision", 1), state,
                        "current" if current and quoted == current else "other/unknown",
                        case.get("asserts", "unspecified"), ",".join(refs[:2]) or "none",
                        "; +%d omitted" % (len(refs) - 2) if len(refs) > 2 else ""))
                lines.append("%d case detail block(s) omitted; read_case retrieves them. Unresolved cases listed first." % len(cases))
            elif CASE:
                state = case_status(CASE, current)
                if not current or CASE.get("declared_identity", CASE.get("identity")) != current:
                    state += "; quotations/claims came from different or unknown source"
                fields = [
                    ("status", state, 110),
                    ("quotation source", CASE.get("declared_identity", CASE.get("identity")) or "unknown", 32),
                    ("observation source", CASE.get("observed_identity", CASE.get("identity")) or "unknown", 32),
                    ("assertion role", CASE.get("asserts", "unspecified"), 18),
                    ("relationship", CASE.get("relationship", "not established"), 145),
                    ("code", CASE.get("code_path", ""), 85),
                    ("requirement", CASE.get("requirement", ""), 85),
                    ("quoted code", CASE.get("code_quote", ""), 55),
                    ("inputs", CASE.get("inputs", ""), 60),
                    ("claimed now", CASE.get("current_result", ""), 75),
                    ("claimed required", CASE.get("required_result", ""), 75),
                    ("observation", CASE.get("observation", ""), 115),
                    ("unchanged", CASE.get("unchanged", ""), 60),
                ]
                lines.append("case: claimed results are not established by matching quotations")
                for label, value, cap in fields:
                    lines.append("  %s: %s" % (label, bounded(value, cap)))
                refs = CASE.get("requirement_ids", [])
                lines.append("  refs: %s%s; declaration/observations: read_case(case_id='')." % (
                    ",".join(refs[:2]) or "none", "; +%d omitted" % (len(refs) - 2) if len(refs) > 2 else ""))
            changed = self.tree.changed_paths(SELFREVIEW_DIFF_SEC)
            if changed:
                lines.append("changed so far: " + bounded(", ".join(changed[:6]), 110))
            observations = [record for record in CHECKS
                            if record.get("kind") != "other command"]
            current_successes = current_usable_records(observations, current)
            latest_readings = current_records(observations, current)
            uncertain = [r for r in latest_readings if r not in current_successes]
            if observations:
                lines.append("current-source checks: %d usable; %d unresolved or failed. Each verifies only its invocation."
                             % (len(current_successes), len(uncertain)))
            # A recent unrelated pass cannot displace an unresolved reading.
            # The counts above disclose observations omitted from the details.
            prioritized = (list(reversed(uncertain)) + list(reversed(current_successes))
                           + [r for r in reversed(observations) if r not in latest_readings])
            prefix = ("\nFindings this run recorded, carried forward because the "
                      "history above was abridged:\n")
            room = self.CARRY_CHARS - len(prefix) - len("\n".join(lines)) - 90
            count = min(4, len(prioritized), max(0, room // 260))
            per_record = room // count if count else 0
            for record in prioritized[:count]:
                if record not in latest_readings:
                    state = "SUPERSEDED: source changed or later comparable check"
                elif record in current_successes:
                    state = "current usable result; still the tree as it stands"
                else:
                    state = "current unresolved/failed; not usable as success"
                if record.get("measurement_check_id"):
                    state += "; reports " + record["measurement_check_id"]
                    capture = record.get("measurement_capture", {})
                    state += "; retained=%s; omitted=%s; unknown=%s" % (
                        capture.get("retained", "unknown"), capture.get("omitted", "unknown"),
                        "unknown" if capture.get("unknown") is None else capture["unknown"])
                elif record.get("measurement_capture_error"):
                    state += "; measurement capture failed; omitted/unknown counts unresolved"
                fixed = "observed: %s -- %s -- source %s (%s)" % (
                    "%s", bounded(record.get("status", "unknown"), 40),
                    bounded(record.get("identity") or "unknown", 16), state)
                available = max(50, per_record - len(fixed) - 8)
                command = bounded(record.get("command") or "unknown", max(35, available * 2 // 3))
                detail = bounded(record.get("detail") or "unknown", max(15, available - len(command)))
                lines.append((fixed % command) + "; " + detail)
            if len(prioritized) > count:
                lines.append("%d additional recorded observation(s) omitted from details." % (len(prioritized) - count))
            if not lines:
                return ""
            return prefix + "\n".join(lines)
        except BaseException:
            return ""

    def requirement_note(self) -> str | None:
        """One pass over the statement's requirements before the answer goes.

        It uses what this run already recorded: which checks produced a usable
        result and of what kind. Passing the check the task says already passes
        is not evidence of the change it asks for, and an optimization without a
        before and after measurement has not been shown to optimize anything.
        This asks once and does not hold the patch: it is a reading, not a gate.
        """
        if self.requirement_state != "armed":
            return None
        self.requirement_state = "done"
        try:
            statement = statement_of(self.warden)
            items = requirement_lines(statement, getattr(getattr(self, "tree", None), "root", ""))
            kinds = task_kinds(statement)
            if not items and not kinds:
                self.requirement.skipped("the statement states no requirement to read back")
                return None
            if self.allowance.clock_left() < SELFREVIEW_MIN_WALL_SEC:
                self.requirement.skipped("too little of the run left to act on the answer")
                return None
            if not self.has_edit():
                self.requirement.skipped("no edit to hold up to the requirements")
                return None
            evidence = self.evidence_summary()
            obligation = measurement_obligation(
                statement, getattr(getattr(self, "tree", None), "root", ""),
                observed_database=bool(evidence.get("measurements")))
            gaps = []
            if not evidence["usable"]:
                historical_passes = [record for record in CHECKS if usable_record(record)]
                if (historical_passes
                        and all(not record.get("after_edit")
                                and record.get("identity") != self.source_identity()
                                for record in historical_passes)):
                    gaps.append("every check that passed ran before this edit "
                                "existed, so none of them looked at it")
                else:
                    gaps.append("no current-source check has an unsuperseded usable "
                                "result; historical passes do not verify the "
                                "candidate as it stands")
            else:
                if not evidence["after_edit"]:
                    gaps.append("every check that passed ran before this edit "
                                "existed, so none of them looked at it")
                elif not evidence["distinguished"]:
                    if not evidence["naming_edit"]:
                        gaps.append("no check that passed even names the file this "
                                    "run changed, so nothing observed the changed "
                                    "path")
                    else:
                        gaps.append("the checks that passed share a name with the "
                                    "file this run changed, which is not the same "
                                    "as exercising it: no command reported anything "
                                    "different before and after the change")
                if (evidence["kinds"] <= {"named regression check", "static check"}
                        and not evidence["distinguished"]):
                    # Matching the runner the statement names says which command
                    # it is, not what it did before the change. What the run
                    # knows about its status beforehand comes from a record of
                    # it on the original code -- and there is a difference
                    # between not having one and having one that passed, which
                    # this used to report the same way.
                    before = sorted({record.get("command") for record in CHECKS
                                     if not record.get("after_edit")
                                     and usable_record(record)
                                     and record.get("identity") != self.source_identity()
                                     and any(record_context(record) == record_context(passed)
                                             and record_context(record)
                                             and earlier_record(record, passed)
                                             for passed in evidence["usable"])})
                    if before:
                        gaps.append("%s passed before this edit as well as after "
                                    "it, so it does not tell the two apart. A "
                                    "check that fails on the original code and "
                                    "passes on this one would"
                                    % ", ".join(before[:2]))
                    else:
                        gaps.append("the checks that passed are the command the "
                                    "task names and static checks, and this run "
                                    "has no result for either on the original "
                                    "code, so neither shows the change took "
                                    "effect. Running the same command against "
                                    "the code as it was is what would")
            cases = declared_cases()
            case_identity = self.source_identity() if cases else ""
            for name, case in cases:
                status = case_status(case, case_identity)
                if status != "observed":
                    gaps.append("case %s revision %s is still %s: %s; observation %s"
                                % (name or "unnamed", case.get("revision", 1), status,
                                   case.get("relationship", "not established")[:180],
                                   case.get("observation", "")[:70]))
            if evidence["failed_now"]:
                gaps.append("a check on the tree as it stands is failing (%s)"
                            % evidence["failed_now"][-1].get("detail"))
            forbidden = execution_prohibited(statement)
            if forbidden:
                # Nothing was measured because the task said not to measure.
                # Naming that as a gap asks for the prohibited work again.
                self.requirement.skipped(
                    "the statement forbids executing queries: %s" % forbidden)
            elif obligation and ABSOLUTE_OBLIGATION.search(obligation):
                # An absolute bound needs no original-code baseline, but a
                # reading still has to measure the stated field and code path.
                if not evidence["measurements"]:
                    gaps.append("the statement sets a bound on database work "
                                "(%s) and no measurement of the tree as it "
                                "stands was recorded" % obligation)
                else:
                    gaps.append("a current database reading is recorded; compare "
                                "its relevant field and units with the stated "
                                "bound (%s), and establish that it measures the "
                                "application path. A reading alone does not "
                                "establish that bound" % obligation)
            elif obligation:
                if not evidence["measurement_pairs"]:
                    gaps.append("the statement asks for less database work or names a "
                                "measured quantity (%s) and no measurement of the same "
                                "command on both sides of the change was recorded, so "
                                "there is nothing to compare%s"
                                % (obligation, " (measure runs one command in a separate base "
                                   "checkout and in the working tree and records both)"
                                   if DB_TOOL and MEASURE_TOOL else ""))
                else:
                    gaps.append("comparable database readings were recorded, "
                                "but a pair alone does not prove improvement; "
                                "compare the relevant values in matching units "
                                "against the statement (%s) and confirm the "
                                "same fixtures and application path were used"
                                % obligation)
            if not gaps:
                self.requirement.fired("evidence covers %s" % ", ".join(sorted(evidence["kinds"])))
                return None
            # Enrich the existing one-shot review; coverage alone introduces
            # no additional mandatory pause or inference request.
            coverage = self.requirement_coverage()
            uncovered = [key for key, value in coverage.items() if not value["declared"]]
            unresolved = [key for key, value in coverage.items() if value["unresolved"]]
            if uncovered:
                gaps.append("no case declares coverage for %s%s; this is unverified coverage, not proof of a defect" % (
                    ", ".join(uncovered[:12]), " (%d more omitted)" % (len(uncovered) - 12) if len(uncovered) > 12 else ""))
            if unresolved:
                gaps.append("linked assertions remain unresolved for %s; passing another caller or workload does not clear them" % ", ".join(unresolved[:12]))
            gaps.append("case-to-requirement and workload links are model claims; observed assertions verify only their recorded invocations")
            self.requirement.fired("%d gap(s) over %d stated requirement(s)"
                                   % (len(gaps), len(items)))
            listed = "\n".join(
                "  - %s [%s] %s" % (item["id"], "change" if item["wants_edit"] else "keep", item["text"])
                for item in items[:8]) or "  (the statement states none in these words)"
            count = len(requirement_catalog(statement, getattr(getattr(self, "tree", None), "root", "")))
            if count > min(8, len(items)):
                listed += "\n  %d clause(s) omitted; read_requirement retrieves complete R1..R%d text." % (
                    count - min(8, len(items)), count)
            return ("Not handed in yet - one reading against the statement, and it "
                    "happens only once. %s.\n\nThese are its requirements as written, "
                    "changes and constraints together; the statement itself remains "
                    "the authority:\n%s\n\nFor each, say either which check of yours "
                    "distinguishes the new behaviour from the old, or what in the "
                    "statement or the code makes it already true. If a service you "
                    "need is unavailable, say which requirement is unverified and "
                    "why, and do not invent a substitute for it. Then call submit "
                    "again." % ("; ".join(gaps), listed))
        except BaseException as error:
            self.requirement.skipped("reading ended on %s" % type(error).__name__)
            return None

    def selfreview_note(self) -> str | None:
        """Ask the run to read its own change once before handing it in, and record what came of
        it.
        """
        if self.selfreview_state == "asked":
            self.selfreview_state = "done"
            self.selfreview_closed = True
            self.selfreview.fired("resubmitted after %d further edit(s)"
                                  % (self.allowance.edits - self.selfreview_edits))
            self.selfreview_outcome()
            return None
        if self.selfreview_state != "armed":
            return None
        if self.allowance.edits == 0:
            self.selfreview_state = "done"
            self.selfreview.skipped("nothing has been changed to look at")
            return None
        if self.allowance.clock_left() < SELFREVIEW_MIN_WALL_SEC or self.allowance.money_left() <= 0:
            self.selfreview_state = "done"
            self.selfreview.skipped("too little of the run left to act on the answer")
            return None
        used = self.selfreview_share()
        used = used if used is not None else 0.0
        self.selfreview_state = "asked"
        self.selfreview_asked = True
        self.selfreview_edits = self.allowance.edits
        self.selfreview.fired("hand-in paused at %.2f of the allowance" % used)
        diff = ""
        try:
            diff = self.tree.diff(SELFREVIEW_DIFF_SEC)
            self.selfreview_before = self.selfreview.artefact("before", diff)
        except BaseException as error:
            self.selfreview_before = None
            self.selfreview.skipped("the answer before the review could not be "
                                    "read: %s" % type(error).__name__)
        return self.consult_splice((
            "Not handed in yet - review the actual diff against the current "
            "task and the surrounding code you read. Identify any concrete "
            "mismatch, citing the task clause or changed line that supports "
            "it. Do not invent additional requirements.\n\n"
            "Use the recorded command results to distinguish what was "
            "observed from what remains unverified. A permitted check may "
            "resolve a concrete uncertainty; respect all execution "
            "restrictions in the task. Correct any supported defect, then "
            "call submit again. If no supported concern remains, submit "
            "without adding unrelated checks."
        ), diff)

    def unbound_reading(self) -> str:
        """A note on anything still unaccounted for at hand-in, or empty when the reading failed.
        """
        try:
            return unbound_at_hand_in(self.tree)
        except BaseException:
            return ""

    def clickhouse_names_target(self) -> "DatabaseTarget | None":
        """The one configured database when it is a ClickHouse server this run reaches over HTTP."""
        if not CH_NAMES or not CH_HTTP or execution_prohibited(statement_of(getattr(self, "warden", None))):
            return None
        targets = self.database_targets()
        if len(targets) != 1 or targets[0].engine != "clickhouse":
            return None
        target = targets[0]
        if target.scheme in ("http", "https") or (not target.scheme and target.port in (8123, 8443)):
            return target
        return None

    def fence_note(self, path: str, after: str) -> str:
        """What the hand-in readers would say about this edit, said now, once each.

        The scope and contract readers run on the same file at hand-in. Hearing
        them at edit time saves the turns between a change that lands outside
        its bounds and the refusal that would follow.
        """
        if not EDIT_NOTES or self.warden is None:
            return ""
        try:
            before = self.warden.original(path)
            if before is None:
                return ""
            statement = statement_of(self.warden)
            marks = list(self.warden.line_faults(
                path, before, after, hunk_spans(self.warden.fenced_diff(path)), stated_methods(statement)))
            hard, _ = contract_violations(path, before, after, statement)
            marks += [("contract " + re.sub(r"\d+", "", text)[:60], text) for text in hard]
        except BaseException:
            return ""
        seen = self.__dict__.setdefault("edit_notes_seen", set())
        said = []
        for tag, text in marks:
            key = re.sub(r":\d+$", "", tag)
            if key in seen:
                continue
            seen.add(key)
            said.append(text)
        if not said:
            return ""
        return "\nAbout this edit (the same reading applies at hand-in):\n" + clip(
            redact("\n".join(said[:3])), 1500, "edit notes")

    def clickhouse_name_note(self, path: str, before: str, after: str) -> str:
        """Names the SQL this edit added calls that the ClickHouse server says it does not have, once per run."""
        try:
            target = self.clickhouse_names_target()
            if target is None:
                return ""
            reported = CH_NAME_STATE.setdefault("reported", set())
            names = [name for name in edit_sql_calls(path, before, after) if name not in reported]
            found = clickhouse_absent_names(target, names) if names else []
        except BaseException:
            return ""
        if not found:
            return ""
        reported.update(name for name, _ in found)
        return ("\n\nThe SQL this edit added calls %s that this ClickHouse server does not "
                "have. Asked directly (a bare call, read-only): %s"
                % ("a name" if len(found) == 1 else "names", clickhouse_names_text(found)))

    def clickhouse_names_reading(self) -> str:
        """A paragraph naming the calls the change's SQL still makes that the server does not have."""
        try:
            target = self.clickhouse_names_target()
            if target is None:
                return ""
            names: list = []
            fresh = (self.tree._untracked(10.0) or set()) - (getattr(self.tree, "untracked_at_start", None) or set())
            for path in (self.tree.changed_paths(10.0) + sorted(fresh))[:12]:
                try:
                    before = self.tree.at_base(path, 10.0)
                except ToolFault:
                    before = ""
                for name in edit_sql_calls(path, before, self.tree.read(path)):
                    if name not in names:
                        names.append(name)
            found = clickhouse_absent_names(target, names)
        except BaseException:
            return ""
        return ("\n\nOne reading of the change as it stands, to settle first: its SQL calls "
                "what this ClickHouse server does not have: %s" % clickhouse_names_text(found)) if found else ""

    def selfreview_outcome(self) -> None:
        """Record what the second reading changed, or why the pair cannot be compared."""
        if self.selfreview_reported:
            return
        self.selfreview_reported = True
        if self.selfreview_before is None:
            self.selfreview.skipped("no reading of the answer before the "
                                    "review, so the pair is not comparable")
            return
        try:
            after = self.tree.diff(SELFREVIEW_DIFF_SEC)
        except BaseException as error:
            self.selfreview.skipped("the answer could not be read back: %s"
                                    % type(error).__name__)
            return
        self.selfreview.outcome(self.selfreview_before, after)

    def selfreview_share(self) -> float | None:
        """How far through its allowance the run is, as a share, or None when that cannot be told.
        """
        try:
            budget = self.allowance.deadline - self.allowance.started
            if not finite_number(budget) or budget <= 0:
                return None
            return max(0.0, self.allowance.elapsed()) / budget
        except BaseException:
            return None

    def conform_note(self) -> str | None:
        """Ask once, before the answer goes, that every stated detail be pointed at the code that
        satisfies it.
        """
        if self.conform_state == "asked":
            self.conform_state = "done"
            self.conform.fired("resubmitted after %d further edit(s)"
                               % (self.allowance.edits - self.conform_edits))
            return None
        if self.conform_state != "armed":
            return None
        if self.allowance.clock_left() < CONFORM_MIN_WALL_SEC:
            self.conform_state = "done"
            self.conform.skipped("too little of the run left to act on the answer")
            return None
        self.conform_state = "asked"
        self.conform_edits = self.allowance.edits
        self.conform.fired("hand-in paused to re-read the statement's requirements")
        return ("Not handed in yet - one check before it goes, and it happens only "
                "once. Re-read the problem statement and collect every specific "
                "detail it requires: orderings, boundaries, defaults, exact values, "
                "which failures are tolerated. For each one, point at the code that "
                "satisfies it as the tree now stands - a line of your diff, or code "
                "that was already right and needed no change. Fix any requirement "
                "nothing satisfies; a detail the statement spells out holds to the "
                "letter. Then confirm your diff changed nothing else's meaning: a "
                "block you moved still does exactly what it did. When both hold, "
                "call submit again.")

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
QUOTED_RE = re.compile(r"[`'\"]([A-Za-z_][A-Za-z0-9_.]{2,})(\(\))?[`'\"]")
COMMON_WORDS = frozenset(
    """the this that with from when what which should would could have been
    test tests file files line lines code error errors return returns value
    values method function class module import python true false none self
    argument arguments result results object objects string strings expected""".split()
)

def quoted_parts(name: str, called: bool = False) -> list[str]:
    """The searchable parts of a backticked name: the whole of it, and its segments worth matching.
    """
    parts = name.split(".")
    if len(parts) == 1:
        return parts
    segments = [p for p in parts if p and p not in COMMON_WORDS]
    return segments if called else [name] + segments

def sweep_order(terms: set, quoted: set) -> list[str]:
    """Search terms in the order worth trying: the statement's own backticked names first, longest
    first.
    """
    return sorted(terms, key=lambda t: (t not in quoted, -len(t), t))

def candidate_files(tree: Tree, statement: str, beacon: Beacon,
                    limit: int = 12) -> list[str]:
    """The files most likely to matter, found by searching the checkout for the statement's own
    words.

    This costs no model call, so the opening message can name the places worth reading
    before the first reply.
    """
    quoted = {p for name, call in QUOTED_RE.findall(statement)
              for p in quoted_parts(name.lower(), bool(call))}
    terms = {w.lower() for w in WORD_RE.findall(statement)} - COMMON_WORDS
    terms |= quoted
    terms = {t for t in terms if len(t) > 3}
    if not terms:
        beacon.skipped("the problem statement carried no usable identifier")
        return []
    scores: dict[str, float] = {}
    looked = 0
    deadline = time.monotonic() + PRELOCATE_BUDGET_SEC
    for term in sweep_order(terms, quoted)[:40]:
        left = deadline - time.monotonic()
        if looked >= 30 or left <= 1.0:
            break
        try:
            done = subprocess.run(
                ["git", "grep", "-lFi", "--", term],
                cwd=tree.root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=min(PRELOCATE_TERM_SEC, left),
            )
        except (subprocess.TimeoutExpired, OSError):
            looked += 1
            continue
        looked += 1
        hits = [p for p in (done.stdout or "").splitlines() if p]
        if not hits or len(hits) > 60:
            continue
        weight = (1.0 / len(hits)) * (3.0 if term in quoted else 1.0)
        for path in hits:
            penalty = 0.25 if ("test" in path.lower() or path.startswith("docs/")) else 1.0
            scores[path] = scores.get(path, 0.0) + weight * penalty
    ranked = [p for p, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:limit]
    beacon.fired("%d terms over %d files -> %d candidates" % (len(terms), len(scores), len(ranked)))
    return ranked

def repo_sketch(tree: Tree) -> str:
    """A few lines describing the checkout: its size, its top directories and the languages in it.
    """
    parts = []
    code, listing = git(["ls-files"], tree.root, 30)
    files = listing.splitlines() if code == 0 else []
    tops: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for path in files:
        tops[path.split("/", 1)[0]] = tops.get(path.split("/", 1)[0], 0) + 1
        ext = os.path.splitext(path)[1] or "(none)"
        kinds[ext] = kinds.get(ext, 0) + 1
    parts.append("%d tracked files." % len(files))
    parts.append(
        "Top level: " + ", ".join("%s (%d)" % (k, v) for k, v in sorted(tops.items(), key=lambda kv: -kv[1])[:12])
    )
    parts.append(
        "Extensions: " + ", ".join("%s (%d)" % (k, v) for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:8])
    )
    markers = ("pyproject.toml", "setup.cfg", "tox.ini", "Makefile")
    if RIDGES_SCOPE_FOLLOWS_STATEMENT:
        markers += ("go.mod", "package.json", "Cargo.toml", "pom.xml")
    for marker in markers:
        if os.path.isfile(os.path.join(tree.root, marker)):
            parts.append("Present: " + marker)
    return "\n".join(parts)

BRIEF = """You are fixing a defect in a checked-out repository. You have shell access
and file tools. When you are done, the working tree is the answer: your changes are read
straight off it, so leave the fix in place and call submit.

Work within the current task's permissions and the repository's conventions.
Keep changes relevant to the requested behavior. Preserve existing interfaces
and dependencies unless the task calls for a change. A quieter check alone is
not evidence that a defect is fixed: verify the requested behavior, including
when the task explicitly calls for a compatibility annotation or test change.
Do not fabricate successful results or special-case a checker to conceal a defect.

CRITICAL - spend turns carefully. Every reply costs one exchange with the model, and
exchanges are the scarcest thing you have. Put every tool call that does not depend on another one into the
SAME reply. Reading four files is four calls in one reply, not four replies. Searching for
three patterns is three calls in one reply. Only wait for a result when the next thing you
do genuinely depends on it.

Do not sit idle while a slow command runs. Start a test suite with background=true, keep
reading code, and collect it with bash_poll when you need the answer.

HOW TO READ THE PROBLEM

Read the current statement and locate the relevant code, its callers and the
repository's documentation and tests. Establish the required behavior and explain
the mismatch using those sources or an observed result. Choose the implementation
after that investigation. Preserve the behavior and interfaces the task requires.

WHEN THE SAME DEFECT REPEATS

A defect is often not one broken place but one broken rule, violated in many independent
places. Repairing some of them leaves the same bug live at the rest, and a caller that hits
an unrepaired site sees no improvement at all. So do not discover the sites one at a time.

1. Get the COMPLETE list first, with one command whose output you can re-run later --
   a grep, a linter, a compile step. Count the hits before you start.
2. Clear them in batches. When the same mechanical change applies at many independent
   places, put as many as one reply will hold into a single reply, and use replace_all
   when the span to change is identical within a file.
3. Re-run THE SAME command. You are not finished when the edits look right; you are
   finished when that command reports nothing left. Check the count against step 1.

BEFORE YOU SUBMIT

Run the tests that cover what you touched, and re-run the command from step 1 above if the
task had repeated sites. A check that ran before your last edit says nothing about the
code as it stands now; run it again after the edit. Where a sentence of the task decides
which rows come back, pin it with expect: quote the sentence, give a probe that reads the
application's data and the rows the sentence requires of the rows you have read (units and
types as the data has them, not as you assume), and run it on the code as it stands; the
hand-in waits while an expectation you pinned fails. Then read your own diff and ask
whether it changes anything the problem did not ask for. Go through the statement's
requirements one by one and point each at a hunk that meets it, or at a line you kept
because it already did. On a task that lists what the rows must be, the hand-in waits
until at least one expectation has been checked on the application's data with the
code as it stands. Leave the file's imports as they were unless the problem is about
them, and run the repository's own lint command when it names one. When you move code,
move it as it is; rewording along the way is a second change nobody asked for. Call
submit with a one-line summary of what you changed."""

DB_WORKFLOW_EDGES = """Before editing a query, settle which engine and which query layer
(ORM, query builder, raw SQL, migration) the code in front of you uses, and write the
change in that layer's own terms. Then work out what the result must be for the cases a
plain reading skips: empty input, NULL values, a relationship that repeats a row, rows
that do not belong, the filters and the ordering asked for, how many rows come back and
which columns. A printed SQL string shows what the code builds, not what it returns:
run it. With the sql tool you can insert scenario rows, query them and read every result
inside one transaction that is rolled back, so those cases can be tried against the
real database without leaving anything behind. expect turns one such case into a check
that quotes the task's sentence and compares the rows the probe returns with the rows
that sentence requires.
"""

def without_sweep(brief: str) -> str:
    """The brief with the repeated-defect section removed, for a run with that behaviour switched
    off.
    """
    start = brief.find("WHEN THE SAME DEFECT")
    end = brief.find("BEFORE YOU SUBMIT")
    return brief[:start] + brief[end:] if 0 <= start < end else brief

DB_WORKFLOW_BASE = """

DATABASE TASK WORKFLOW
Read the current task, relevant application code, schema, documentation and tests
to establish the required database behavior. Choose checks from that evidence,
using the repository's own client and configuration. Base any claimed improvement
on comparable measurements of the application path. Respect execution restrictions
and report unavailable services or unverified behavior accurately. A generated SQL
string or a permissive stand-in alone does not establish application correctness.
"""
# The tools are named only when their switches offer them.
DB_WORKFLOW_TOOLS = ""
if DB_TOOL:
    DB_WORKFLOW_TOOLS += (
        "The sql tool speaks PostgreSQL (through psql, rolled back) and ClickHouse (over\n"
        "HTTP, read-only unless write=true); each statement comes back with its own work\n"
        "counts. " if CH_HTTP else
        "The sql tool speaks PostgreSQL (through psql, rolled back); each statement comes\n"
        "back with its own work counts. ")
if DB_TOOL and MEASURE_TOOL:
    DB_WORKFLOW_TOOLS += (
        "To compare with the original code, use measure: it runs one command in a\n"
        "separate checkout of the base commit and in the working tree and records both\n"
        "readings. Do not copy files or check out revisions in the working tree to obtain a\n"
        "baseline.\n")
elif DB_WORKFLOW_TOOLS:
    DB_WORKFLOW_TOOLS = DB_WORKFLOW_TOOLS.rstrip() + "\n"
DB_WORKFLOW_COMMON = DB_WORKFLOW_BASE + DB_WORKFLOW_TOOLS
DB_WORKFLOW_NO_EXECUTION = """When the statement forbids running queries, that
settles how the work is shown, not what it has to achieve: the change it asks
for is still the change. Do not execute statements against the database and do
not fabricate numbers for what one would have returned. Work from what is in
front of you -- the query the code builds, the schema and migrations in the
repository, the indexes declared there, and what the surrounding tests already
assert. Say which parts of the runtime behaviour are therefore unverified, and
name the command that would settle each of them if it were permitted.
"""
def database_workflow(forbidden: str = "") -> str:
    """Investigation guidance and a current execution restriction, if present."""
    if forbidden:
        # Naming the query tools here would ask for the execution the statement
        # forbids; the restriction replaces them.
        return (DB_WORKFLOW_BASE + DB_WORKFLOW_NO_EXECUTION
                + "\nThe statement's own words on this: \"%s\"\n" % forbidden)
    # The cases to settle come after the tools they are settled with, and stay
    # out of DB_WORKFLOW_COMMON, whose length is held to what it was.
    return DB_WORKFLOW_COMMON + ("\n" + DB_WORKFLOW_EDGES if DB_TOOL else "")

def compose_brief(statement: str, root: str) -> str:
    """The brief this run gives the model: the standing one, plus the database guidance when it
    applies.
    """
    text = BRIEF if SWEEP_WORKFLOW else without_sweep(BRIEF)
    if database_context(statement, root):
        text += database_workflow(execution_prohibited(statement))
    return text

# ---------------------------------------------------------------------------
# What the task asks for, and what this run knows before it spends a turn.
#
# The brief already tells the model to trace the query, keep a record and check
# the behaviour. None of that survives a transcript shrink and none of it is
# free: every fact the model has to discover costs a reasoning turn. These
# helpers derive the cheap part deterministically, once, from the statement and
# the checkout, and are recomputed rather than remembered so compaction cannot
# drop them.
# ---------------------------------------------------------------------------

TASK_KIND_PATTERNS = (
    ("optimization", re.compile(
        r"\boptimiz|\boptimis|\bn\+1\b|\bperformance\b|\bfaster\b|\bslow(?:er|ly)?\b|"
        r"\bscal(?:e|es|ing|ability)\b|\btoo many quer|\bnumber of quer|"
        r"\bmust not (?:scale|grow)\b|\bbounded\b|\bone database query\b|"
        r"\bexpensive\b|\bdatabase work\b", re.I)),
    ("repair", re.compile(
        r"\brepair\b|\bincorrect(?:ly)?\b|\bwrong(?:ly)?\b(?!(?:\s+|-)(?:types?|shapes?|domains?|inputs?)\b)|"
        r"\bsilently\b|\bomits?\b|"
        r"\bdouble[- ]count|\boff by\b|\bshould (?:return|be|report)\b|\bfails to\b|"
        r"\binstead of\b|\bmisreports?\b", re.I)),
    ("generation", re.compile(
        r"\bauthor\b|\bintroduce\b|\bimplement\b|\badd (?:an?|the) \w+|"
        r"\bnew (?:method|annotation|migration|index|column)\b|\bannotate\b", re.I)),
)

KIND_DECLARATION = re.compile(r"^\s*(?:work|task|kind|type|category)\s*:\s*([A-Za-z]+)", re.I | re.M)
KIND_WORDS = {"optimization": "optimization", "optimisation": "optimization", "optimize": "optimization",
              "optimise": "optimization", "performance": "optimization", "repair": "repair",
              "fix": "repair", "bugfix": "repair", "correction": "repair", "generation": "generation",
              "authoring": "generation", "implement": "generation", "implementation": "generation",
              "feature": "generation"}
KIND_IMPERATIVE = (
    ("optimization", re.compile(r"^(?:optimi[sz]e|speed\s+up|reduce|cut|lower|bound)\b", re.I)),
    ("repair", re.compile(r"^(?:fix|repair|correct|resolve|stop)\b", re.I)),
    ("generation", re.compile(r"^(?:implement|add|author|introduce|write|create|build|provide)\b", re.I)),
)
# A statement that bounds the database work in its own words asks for that
# work to shrink or stay bounded, whatever its first verb is.
KIND_WORK_BOUND = re.compile(
    r"\b(?:read|scan)s?\s+at\s+most\b|\bat most (?:one|\d+) (?:database |server-executed |sql )?"
    r"(?:quer(?:y|ies)|statements?|rows)\b|\bwork budget\b|\brows read\b|\bone database query\b", re.I)
# What a statement says about the code's state names the kind of work as well.
KIND_STATE = (
    ("generation", re.compile(r"\b(?:is|are) missing\b|\bdoes not exist yet\b|\bnot (?:yet )?implemented\b", re.I)),
    ("repair", re.compile(r"\bdoes not meet its (?:production )?contract\b|\bnot meeting its contract\b|"
                          r"\b(?:support|operators?|users?|customers?|analysts?|reviewers?)\s+report(?:s|ed)?\s+"
                          r"(?:show|that|indicate)\b", re.I)),
)
# A clause about the hidden examples or private tests describes the checker,
# not the work asked for; its verbs do not vote.
CHECKER_CLAUSE = re.compile(r"^\s*(?:the\s+)?(?:hidden|private)\s+(?:examples?|tests?|fixtures?|checks?|validation)\b", re.I)
# "X must implement the application contract" states whose words are the
# contract, not what kind of work is asked; it does not count as a kind.
KIND_CONTRACT_PHRASE = re.compile(
    r"\bmust (?:implement|satisfy|meet) the (?:[\w-]+\s+){0,6}?(?:contract|behavior|behaviour)\b", re.I)
METADATA_LINE = re.compile(r"^\s*[A-Z][A-Za-z]*(?:\s+[A-Za-z()/-]+){0,2}:\s")
HEADING_MARK = re.compile(r"^\s{0,3}#{1,6}\s+")

def task_kinds(statement: str) -> list[str]:
    """Advisory work categories derived from the current instruction's words.

    Current declarations and the first explicit imperative rank ahead of word
    counts. Mixed work retains every mentioned category; these labels neither
    prescribe a solution nor establish requirements or verification success.
    """
    # Use semantic clauses so soft wrapping and introductory formatting do
    # not change which current instruction supplies the advisory category.
    clauses = list(instruction_clauses(statement))
    first: list = []
    about_database = database_context(statement)
    declared = next((match for clause in clauses
                     if (match := KIND_DECLARATION.search(clause))), None) if about_database else None
    if declared and KIND_WORDS.get(declared.group(1).lower()):
        first.append(KIND_WORDS[declared.group(1).lower()])
    if not first and about_database and clauses and HEADING_MARK.match(clauses[0]):
        # The title's own verb names the work: "Repair descendant counts",
        # "Optimize the device filter", "Author prefix annotations".
        title = HEADING_MARK.sub("", clauses[0]).split()
        if title and KIND_WORDS.get(title[0].lower()):
            first.append(KIND_WORDS[title[0].lower()])
    if not first and about_database and any(
            (found := KIND_WORK_BOUND.search(c)) and not negated_kind(c, found) for c in clauses):
        first.append("optimization")
    if not first and about_database:
        # Read the first explicit imperative wherever it occurs in the
        # current instruction; no fixed opening window determines authority.
        for clause in clauses:
            for kind, pattern in KIND_IMPERATIVE:
                if pattern.match(HEADING_MARK.sub("", clause).strip()):
                    first.append(kind)
                    break
            if first:
                break
    if not first and about_database:
        for clause in clauses:
            stated = [(found.start(), kind) for kind, pattern in KIND_STATE
                      if (found := pattern.search(clause))]
            if stated:
                # "Operators report that ... is missing": the phrase opening the
                # clause names the work; the later one is its detail.
                first.append(min(stated)[1])
                break
    text = " ".join(KIND_CONTRACT_PHRASE.sub(" ", HEADING_MARK.sub("", c)) for c in clauses
                    if not (METADATA_LINE.match(c) and len(c.split()) <= 12) and not CHECKER_CLAUSE.match(c))
    scored = []
    for kind, pattern in TASK_KIND_PATTERNS:
        hits = sum(1 for found in pattern.finditer(text) if not negated_kind(text, found))
        if hits:
            scored.append((hits, kind))
    ordered = [kind for _, kind in sorted(scored, reverse=True)]
    return first + [kind for kind in ordered if kind not in first]

# A kind word the statement negates is not a vote for that kind: "performance is
# not a concern", "optimization is out of scope", "this is not a performance
# task", "do not optimize". Only words that name a kind are read this way; "not
# bounded" is a complaint about scaling, not a negation of it.
NEGATABLE_KIND = re.compile(r"optimi|performance|faster|slow|scal|repair|incorrect|wrong|implement|author|introduce|annotate", re.I)
KIND_NEGATION_BEFORE = re.compile(r"\b(?:not|no|never|isn't|aren't|don't|doesn't|without|nor)\b(?:\s+\w+){0,2}\s*$", re.I)
KIND_NEGATION_AFTER = re.compile(
    r"^\w*\s+(?:is|are|was|were)\s+(?:not\b|out\s+of\s+scope\b|irrelevant\b|unimportant\b)|"
    r"^[^.;!?]{0,20}\bout\s+of\s+scope\b|"
    r"^\w*\s+(?:is|are)\s+(?:not\s+)?(?:a\s+)?non[- ]?goal\b", re.I)

def negated_kind(text: str, found) -> bool:
    """Is this match of a work kind negated by the words around it, such as a sentence refusing it?
    """
    if not NEGATABLE_KIND.search(found.group(0)):
        return False
    before = re.split(r"[.;!?]", text[max(0, found.start() - 40):found.start()])[-1]
    after = re.split(r"[.;!?]", text[found.end():found.end() + 60])[0]
    return bool(KIND_NEGATION_BEFORE.search(before) or KIND_NEGATION_AFTER.search(after))

REQUIREMENT_MODAL = re.compile(
    r"\b(?:must(?: not)?|may not|never|only|exactly|no more than|at most|"
    r"at least|has to|have to|should(?: not)?|keep|keeps|retain|retains|"
    r"preserve|preserves|do not|don't|cannot|uses one)\b", re.I)
# A clause asking for something to stay as it was. Deliberately narrower than
# the ledger's guard words: "keep two decimal places" asks for a change, while
# "keep the rest of the file unchanged" does not, and both start with "keep".
REQUIREMENT_GUARD = re.compile(
    r"\bunchanged\b|\buntouched\b|\bintact\b|\bas (?:it|they) (?:is|are|was|were)\b|"
    r"\bstays? the same\b|\bmust not (?:change|be changed|be modified|differ)\b|"
    r"\bdo(?:es)? not (?:change|modify|edit|touch|alter)\b|"
    r"\bdon't (?:change|modify|edit|touch)\b|"
    r"\bleave\b[^.;]{0,40}\balone\b|\brest of (?:the|its) file\b|"
    r"\bno (?:other|further) (?:change|edit)s?\b", re.I)
REQUIREMENT_BULLET_MARK = re.compile(r"^\s*(?:[-+*\u2022]|[0-9]{1,9}[.)])\s+")
# A statement states its central requirement as an instruction as often as with
# a modal: "Return a lazy, composable queryset whose count is ...". A short,
# closed list of instruction verbs, not a growing taxonomy.
REQUIREMENT_IMPERATIVE = re.compile(
    r"^(?:Return|Report|Count|Annotate|Add|Replace|Rewrite|Write|Implement|"
    r"Produce|Compute|Emit|Ensure|Make|Move|Limit|Restrict|Confine|Keep|Use|"
    r"Preserve|Retain|Avoid|Do not|Don't|Never|Only)\b", re.I)
# Where a long clause may be cut, so a trailing negation is not lost silently.
REQUIREMENT_BREAK = re.compile(r"[;,]\s|\sand\s|\sor\s")
REQUIREMENT_CHARS = 170

def asks_to_keep(item: str) -> bool:
    """Does this requirement ask that something stay as it is, rather than asking for a change?
    """
    return bool(REQUIREMENT_GUARD.search(item or ""))
REQUIREMENT_MAX = 12
READ_REQUIREMENT_ITEMS = 20

def requirement_excerpt(item: str) -> str:
    """A requirement clipped to its cap at a sentence or clause break, with a marker where it was
    cut.
    """
    if len(item) <= REQUIREMENT_CHARS:
        return item
    marker = " [...]"
    head = item[:REQUIREMENT_CHARS - len(marker)]
    breaks = [found.end() for found in REQUIREMENT_BREAK.finditer(head)]
    cut = breaks[-1] if breaks else head.rfind(" ") + 1
    return (head[:cut].rstrip(" ,;") if cut > 40 else head.rstrip()) + marker

REQUIREMENT_INDEX_FULL = flag("RIDGES_REQUIREMENT_INDEX_FULL")
REQUIREMENT_INDEX_CHARS = 6000
REQUIRED_SECTION = re.compile(r"required behavio|acceptance|requirements?|constraints?|contract|rules|must", re.I)
# A clause the statement labels itself, "3." or "D03:", is one item of it.
NUMBERED_ITEM = re.compile(r"^\s{0,3}(?:\d{1,9}[.)]|[A-Z]{1,2}\d{2,3}:)\s+")
HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
# A repository document the statement itself names as carrying its rules.
REFERENCED_DOCUMENT = re.compile(r"(?<![\w/.-])([A-Za-z0-9_][\w./-]*\.(?:md|txt|rst|adoc))\b")
REFERENCE_CONTEXT = re.compile(
    r"\bcontract|\brequirement|\bspecification|\bauthoritative|\bpreserv|\bclarification|"
    r"\bfrozen|\brules?\b|\bmust follow|\baccording to|\bdefined in|\bdescribed in|\bsee\b", re.I)
REFERENCED_DOCUMENT_CAP = 3
REFERENCED_DOCUMENT_BYTES = 64_000

def referenced_documents(statement: str, root: str) -> list[tuple]:
    """(path, text) for each repository document the statement names as rules.

    Only a file the current instructions name, only inside the checkout, only
    a readable text document. The statement's words are what authorises the
    read; a file is never adopted because of its name alone.
    """
    if not root or not REQUIREMENT_INDEX_FULL:
        return []
    # Read once per run: R-numbers are handed to the model at the start, so a
    # document edited later must not renumber them. The first read happens
    # while the opening message is built, before any edit.
    key = (os.path.realpath(root), statement or "")
    if key in REFERENCED_READ:
        return list(REFERENCED_READ[key])
    out: list = []
    seen: set = set()
    for clause in instruction_clauses(statement or ""):
        if not REFERENCE_CONTEXT.search(clause):
            continue
        for name in REFERENCED_DOCUMENT.findall(clause):
            path = name.strip("`'\"")
            if path.lower() in seen or len(out) >= REFERENCED_DOCUMENT_CAP:
                continue
            seen.add(path.lower())
            try:
                base = os.path.realpath(root)
                full = os.path.realpath(os.path.join(base, path))
                if os.path.commonpath([base, full]) != base or not os.path.isfile(full):
                    continue
                if os.path.getsize(full) > REFERENCED_DOCUMENT_BYTES:
                    continue
                with open(full, "r", encoding="utf-8", errors="replace") as handle:
                    out.append((path, handle.read()))
            except (OSError, ValueError):
                continue
    REFERENCED_READ[key] = list(out)
    return out

REFERENCED_READ: dict = {}

def requirement_catalog(statement: str, root: str = "") -> list[dict]:
    """The requirements the statement states outright.

    Its bullets and its modal sentences, nothing inferred: a guess belongs in a
    hypothesis, not in a list the run treats as given. Each is tagged as one
    that asks for a change or one that asks for something to stay as it is,
    and with the heading it sits under. When the statement names a repository
    document as its rules, that document's own bullets and modal sentences
    follow, marked with the document they came from.
    """
    found: list[dict] = []
    seen: set = set()

    def add(text: str, source: str, section: str = "", document: str = "", line: int = 0) -> None:
        item = REQUIREMENT_BULLET_MARK.sub("", " ".join((text or "").split()))
        key = item.rstrip(".;: ").lower()
        if len(item) < 8 or key in seen:
            return
        seen.add(key)
        found.append({"id": "R%d" % (len(found) + 1), "text": requirement_excerpt(item),
                      "full_text": item, "source": source, "section": section,
                      "document": document, "line": line,
                      # Advisory only. A clause can ask for a change and bound it
                      # at the same time, so this tag never filters the list.
                      "wants_edit": not asks_to_keep(item)})

    def read(text: str, document: str = "") -> None:
        # Use the same current-instruction provenance as task permissions. Keep
        # source line positions so a wrapped qualifier remains attached to its
        # bullet; code/examples must not absorb the instruction after a fence.
        original_lines = (text or "").splitlines()
        current_lines = [""] * len(original_lines)
        for number, clause in instruction_clauses(text, with_lines=True):
            if not current_lines[number]:
                raw = original_lines[number]
                current_lines[number] = raw[:len(raw) - len(raw.lstrip())] + clause
            else:
                current_lines[number] += " " + clause
        current_text = "\n".join(current_lines)
        headings: list = []
        for number, raw in enumerate(original_lines):
            heading = HEADING_LINE.match(raw)
            if heading:
                headings.append((number, heading.group(1)))

        starts = [start for start, _ in headings]

        def section_at(number: int) -> str:
            index = bisect.bisect_right(starts, number) - 1
            return headings[index][1] if index >= 0 else ""

        # Bullets keep their line so the section they sit under is known.
        first_lines = {}
        for number, line in enumerate(current_lines):
            key = REQUIREMENT_BULLET_MARK.sub("", " ".join(line.split())).rstrip(".;: ").lower()
            if key and key not in first_lines:
                first_lines[key] = number
        # The first line (in order) whose key starts with a given head of up to
        # 60 characters, built once: scanning every key per item was quadratic.
        prefixes: dict = {}
        for key, number in first_lines.items():
            for size in range(1, min(len(key), 60) + 1):
                prefixes.setdefault(key[:size], number)
        for item in stated_requirements(current_text):
            key = REQUIREMENT_BULLET_MARK.sub("", " ".join(item.split())).rstrip(".;: ").lower()
            number = first_lines.get(key)
            if number is None and key:
                # A wrapped bullet starts with its first line's text: take the longest line key
                # the item begins with (they meet at a word boundary) before the looser match.
                cuts = [i for i, char in enumerate(key) if char == " "]
                number = next((first_lines[key[:i]] for i in reversed(cuts) if key[:i] in first_lines), None)
            if number is None:
                number = prefixes.get(key[:60], 0) if key else next(iter(first_lines.values()), 0)
            add(item, "bullet", section_at(number), document, number + 1)
        # Numbered items are a list of requirements as much as dashed ones; a
        # wrapped item keeps its continuation lines until a blank line or the
        # next item.
        item_lines: list = []
        for number, line in enumerate(current_lines + [""]):
            if NUMBERED_ITEM.match(line):
                if item_lines:
                    add(" ".join(item_lines[1:]), "bullet", section_at(item_lines[0]), document, item_lines[0] + 1)
                item_lines = [number, NUMBERED_ITEM.sub("", line).strip()]
            elif item_lines and line.strip() and not HEADING_LINE.match(line) and not LEDGER_BULLET.match(line):
                item_lines.append(line.strip())
            elif item_lines:
                add(" ".join(item_lines[1:]), "bullet", section_at(item_lines[0]), document, item_lines[0] + 1)
                item_lines = []
        _, paragraphs = statement_markdown_blocks(current_text)
        for paragraph in paragraphs:
            head = " ".join(paragraph.split())[:60].lower()
            number = prefixes.get(head[:40], 0) if head[:40] else 0
            for sentence in FENCE_SENTENCE.split(paragraph):
                line = NUMBERED_ITEM.sub("", " ".join(sentence.split()))
                if line.startswith("$ "):
                    continue
                if REQUIREMENT_MODAL.search(line):
                    add(line, "sentence", section_at(number), document, number + 1)
                elif REQUIREMENT_IMPERATIVE.match(line):
                    add(line, "instruction", section_at(number), document, number + 1)

    read(statement)
    for path, text in referenced_documents(statement, root):
        read(text, path)
    return found

def requirement_order(items: list) -> list:
    """Requirements first, then changes asked for, then what must stay; stable."""
    if not REQUIREMENT_INDEX_FULL:
        return list(items)

    def rank(item: dict) -> int:
        if (REQUIRED_SECTION.search(item.get("section") or "") and not item.get("document")):
            return 0
        return 1 if item.get("wants_edit") else 2
    return sorted(items, key=rank)

def compact_ids(ids: list) -> str:
    """Requirement numbers written compactly, with runs of consecutive ones given as ranges."""
    numbers = sorted({int(i[1:]) for i in ids if re.fullmatch(r"R\d+", i)})
    out: list = []
    start = previous = None
    for number in numbers:
        if start is None:
            start = previous = number
        elif number == previous + 1:
            previous = number
        else:
            out.append("R%d" % start if start == previous else "R%d-R%d" % (start, previous))
            start = previous = number
    if start is not None:
        out.append("R%d" % start if start == previous else "R%d-R%d" % (start, previous))
    return ", ".join(out)

def range_example(omitted: list, count: int) -> str:
    """A read_requirement range starting at the first omitted clause, in range syntax."""
    first = int(omitted[0][1:]) if omitted else 1
    low = max(1, min(first, count - 1))
    return "R%d-R%d" % (low, max(low, min(count, low + READ_REQUIREMENT_ITEMS - 1)))

def requirement_lines(statement: str, root: str = "") -> list[dict]:
    """The bounded display of the complete, task-local requirement catalog."""
    return requirement_order(requirement_catalog(statement, root))[:REQUIREMENT_MAX]

DSN_RE = re.compile(
    r"\b(postgres(?:ql)?|clickhouse|mysql|mariadb)://(?:[^@\s/]*@)?"
    r"([A-Za-z0-9_.-]+)(?::(\d{2,5}))?", re.I)
# The same engine is spelled many ways by drivers, DSN schemes and settings
# modules. One canonical name per engine, so a client is chosen by what the
# server is rather than by which library the application happens to use.
ENGINE_ALIASES = {
    "postgres": "postgresql", "postgresql": "postgresql", "postgresql_psycopg2": "postgresql",
    "psycopg2": "postgresql", "psycopg": "postgresql", "postgis": "postgresql", "pg": "postgresql",
    "pgsql": "postgresql", "asyncpg": "postgresql", "pgx": "postgresql", "npgsql": "postgresql",
    "clickhouse": "clickhouse", "ch": "clickhouse", "clickhousedb": "clickhouse",
    "clickhouse_connect": "clickhouse", "clickhouse_driver": "clickhouse", "clickhouses": "clickhouse",
    "mysql": "mysql", "mariadb": "mysql", "mysql2": "mysql", "trilogy": "mysql", "sqlite": "sqlite", "sqlite3": "sqlite",
}

SQL_ENGINES = ("postgresql", "clickhouse", "mysql", "sqlserver", "mssql", "cockroachdb")

def normalize_engine(name: str) -> str:
    """The canonical engine for a driver, scheme or settings spelling."""
    text = str(name or "").strip().lower()
    text = text.rsplit(".", 1)[-1].split("+", 1)[0]
    return ENGINE_ALIASES.get(text, text or "unknown")
# The value is the whole token, and a token followed by "(" is a call that
# computes the setting (config('DB_HOST'), get_secret(...)), not the setting.
SETTING_RE = re.compile(
    r"['\"](ENGINE|HOST|PORT|NAME|USER)['\"]\s*:\s*['\"]?([A-Za-z0-9_.\-/]+)(?![A-Za-z0-9_.\-/]|\s*\()", re.I)
# Read separately and with a wider value class, because a password may hold
# punctuation the fields above never do. Its value is never reported.
SETTING_SECRET = re.compile(r"['\"]PASSWORD['\"]\s*:\s*['\"]([^'\"]*)['\"]", re.I)
DB_SETTING_FILES = ("settings.py", "configuration.py", "config.py", "database.py",
                    "db.py", "settings_test.py", "local_settings.py")
DEFAULT_PORTS = {"postgres": 5432, "postgresql": 5432, "clickhouse": 8123,
                 "mysql": 3306, "mariadb": 3306}
CLICKHOUSE_NATIVE_PORTS = (9000, 9440)
DATABASE_WORD = re.compile(r"DATABASES?\b|DATABASE_URL|CLICKHOUSE|\bDSN\b", re.I)
# A host read out of source must look like a host, not like the expression that
# computes one: os.environ['DB_HOST'] is an unresolved value, not a hostname.
LITERAL_HOST = re.compile(r"^(?:[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?|/[^\s]*)$")
# An expression, not a hostname: call or index syntax, an environment lookup, a settings or
# config attribute path, or an ALL-CAPS identifier (a constant or variable name). Substrings
# alone are not enough: config-db and db.environment.local are ordinary hostnames.
CODE_FRAGMENT = re.compile(
    r"[()\[\]]|\$\{|(?i:\bos\.|getenv|\benviron\b|\benv\b|^(?:settings|config|conf)\.)|"
    r"(?:^|\.)[A-Z][A-Z0-9]*_[A-Z0-9_]*(?:$|\.)|\.[A-Z]{4,}[A-Z0-9]*(?:\.|$)")

def literal_value(value: str) -> str:
    """The value if the source states it outright, else "" for unresolved."""
    text = (value or "").strip()
    if not text or CODE_FRAGMENT.search(text) or not LITERAL_HOST.match(text):
        return ""
    return text

DYNAMIC_SETTING = re.compile(
    r"os\.environ|os\.getenv|getenv\(|environ\[|config\(|Env\(|env\(|"
    r"dj_database_url|from_url\(|settings\.", re.I)
SETTING_FILE_CAP = 12

def contained_text(root: str, path: str, limit: int) -> str | None:
    """A file's text when it resolves inside the checkout, else None.

    Configuration and source are read to learn where the database is. A link
    that leads out of the checkout is not the repository's configuration.
    """
    try:
        base = os.path.realpath(root)
        full = os.path.realpath(os.path.join(base, path))
        if os.path.commonpath([base, full]) != base or not os.path.isfile(full):
            return None
        with open(full, "r", errors="replace") as handle:
            return handle.read(limit)
    except (OSError, ValueError):
        return None

def url_host(host: str) -> str:
    """A host as a URL spells it: an IPv6 literal goes in brackets."""
    return "[%s]" % host if ":" in (host or "") and not host.startswith("[") else host

def settings_file_endpoints(root: str, budget: float = 6.0) -> list[dict]:
    """Where a Python settings module says the database is, read as text only.

    Nothing here imports or evaluates a settings module: a literal is reported
    as a literal, and a value the application computes at run time is reported
    as unresolved rather than guessed at.
    """
    deadline = time.time() + max(1.0, budget)
    out, seen = [], set()
    code, listing = git(["ls-files", "--cached", "--others", "--exclude-standard",
                         "-z", "--"], root, max(1.0, budget))
    names = [p for p in listing.split("\0") if p] if code == 0 else []
    examined = 0
    for path in names:
        if os.path.basename(path) not in DB_SETTING_FILES:
            continue
        if examined >= SETTING_FILE_CAP or time.time() >= deadline:
            break
        examined += 1
        text = contained_text(root, path, 200_000)
        if text is None:
            continue
        engine = host = name = user = ""
        port = 0
        # A settings file names hosts and passwords for caches and queues as
        # well. When it has a DATABASES block, read that block and nothing
        # before it, so the first HOST in the file is not taken for the
        # database's.
        block = text
        marker = re.search(r"\bDATABASES?\s*=", text)
        if marker:
            block = text[marker.start():marker.start() + 6000]
        found_secret = SETTING_SECRET.search(block)
        # Held for the client's environment only; never logged or reported.
        password = found_secret.group(1) if found_secret else ""
        for key, value in SETTING_RE.findall(block):
            key = key.upper()
            if key == "ENGINE" and not engine:
                engine = normalize_engine(value)
            elif key == "USER" and not user:
                user = literal_value(value)
            elif key == "HOST" and not host:
                host = literal_value(value)
            elif key == "PORT" and not port and value.isdigit():
                port = int(value)
            elif key == "NAME" and not name:
                name = value
        for scheme, dsn_host, dsn_port in DSN_RE.findall(block if marker else text):
            engine = engine or normalize_engine(scheme)
            host = host or dsn_host
            port = port or int(dsn_port or DEFAULT_PORTS.get(scheme.lower(), 0) or 0)
        if not host:
            if DATABASE_WORD.search(text) and DYNAMIC_SETTING.search(text):
                key = ("unresolved", path)
                if key not in seen:
                    seen.add(key)
                    out.append({"engine": engine or "unknown", "host": "",
                                "port": 0, "name": name, "source": path,
                                "unresolved": True})
            continue
        port = port or DEFAULT_PORTS.get((engine or "").lower(), 0)
        # The database name is part of the identity: an application database and
        # its test database sit on the same host and port and are not the same
        # connection.
        key = (host, port, name)
        if key in seen:
            continue
        seen.add(key)
        out.append({"engine": engine or "unknown", "host": host, "port": port,
                    "name": name, "source": path, "user": user,
                    "password": password})
    return out[:3]

ENV_ENDPOINTS = flag("RIDGES_ENV_ENDPOINTS")
# Variable names applications and their deployments use for a database
# connection. The family (prefix) says which engine when the value does not.
ENV_DB_VARIABLE = re.compile(
    r"^(?:(CLICKHOUSE|PG|POSTGRES|POSTGRESQL|DATABASE|DB)_?"
    r"(URL|URI|DSN|HOST|HOSTADDR|PORT|DATABASE|DB|DBNAME|NAME|USER|USERNAME|PASSWORD|PASS)"
    r"|(DATABASE_URL|DB_URL|DSN))$")
ENV_FAMILY_ENGINE = {"CLICKHOUSE": "clickhouse", "PG": "postgresql", "POSTGRES": "postgresql",
                     "POSTGRESQL": "postgresql", "DATABASE": "", "DB": ""}
ENDPOINT_URL = re.compile(
    r"^\s*(?:jdbc:)?([A-Za-z][A-Za-z0-9+.-]*)://(?:([^:@/\s]*)(?::([^@/\s]*))?@)?"
    r"(\[[0-9a-fA-F:.]+\]|[A-Za-z0-9_.-]+)(?::(\d{1,5}))?(?:/([^?\s#;]*))?(?:\?([^\s#]*))?")
ENV_FILE_NAMES = (".env", ".env.example", ".env.sample", ".env.local", ".env.development",
                  ".env.test", ".env.dist", ".env.defaults")
ENV_FILE_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$")
CONFIG_FILE_NAMES = ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
                     "docker-compose.override.yml", "appsettings.json", "appsettings.Development.json",
                     "application.properties", "application.yml", "application.yaml",
                     "application-dev.properties", "application-dev.yml", "database.yml",
                     "alembic.ini", "config.toml", "settings.toml", "app.toml")
SOURCE_DEFAULT_EXTENSIONS = (".ts", ".mts", ".cts", ".js", ".mjs", ".cjs", ".py", ".go",
                             ".cs", ".rs", ".java", ".kt", ".rb")
SOURCE_DEFAULT_WORDS = re.compile(r"process\.env|os\.(?:environ|getenv)|Getenv|GetEnvironmentVariable|"
                                  r"env::var|System\.getenv|ENV\[|ENV\.fetch", re.I)
SOURCE_DEFAULT_FILE_CAP = 40
ENDPOINT_CAP = 4
STATEMENT_DEFAULT = re.compile(
    r"\b([A-Z][A-Z0-9_]{2,})\s+defaults?\s+to\s+`?([^\s`,;]+)`?"
    r"((?:\s*(?:,|and)\s*[A-Z][A-Z0-9_]{2,}\s+to\s+`?[^\s`,;]+`?)*)", re.I)
STATEMENT_DEFAULT_MORE = re.compile(r"([A-Z][A-Z0-9_]{2,})\s+to\s+`?([^\s`,;]+)`?", re.I)
# Fallback literal after a variable lookup, per language. Group 1 is the
# variable, group 2 the literal. Comments are blanked before matching.
SOURCE_DEFAULT_PATTERNS = {
    "node": (re.compile(r"process\.env(?:\.([A-Z][A-Z0-9_]*)|\[['\"]([A-Z][A-Z0-9_]*)['\"]\])"
                        r"\s*(?:\?\?|\|\|)\s*(['\"`])([^'\"`\n]+)\3"),),
    "python": (re.compile(r"os\.(?:environ\.get|getenv)\(\s*['\"]([A-Z][A-Z0-9_]*)['\"]\s*,\s*"
                          r"(['\"])([^'\"\n]+)\2\s*\)"),),
    "go": (re.compile(r"[gG]et[eE]nv\w*\(\s*\"([A-Z][A-Z0-9_]*)\"\s*,\s*\"([^\"\n]+)\"\s*\)"),
           re.compile(r"cmp\.Or\(\s*os\.Getenv\(\"([A-Z][A-Z0-9_]*)\"\)\s*,\s*\"([^\"\n]+)\"\s*\)")),
    "csharp": (re.compile(r"GetEnvironmentVariable\(\s*\"([A-Z][A-Z0-9_]*)\"\s*\)\s*\?\?\s*\"([^\"\n]+)\""),),
    "rust": (re.compile(r"env::var\(\s*\"([A-Z][A-Z0-9_]*)\"\s*\)[^;\n]{0,80}?unwrap_or(?:_else)?\(\s*"
                        r"(?:\|_?\|\s*)?(?:String::from\(|\s*)?\"([^\"\n]+)\""),),
    "java": (re.compile(r"getOrDefault\(\s*\"([A-Z][A-Z0-9_]*)\"\s*,\s*\"([^\"\n]+)\"\s*\)"),
             re.compile(r"System\.getenv\(\s*\"([A-Z][A-Z0-9_]*)\"\s*\)\)?[^;\n]{0,120}?"
                        r"(?:orElse\(|\?:\s*|:\s*)\"([^\"\n]+)\""),),
    "ruby": (re.compile(r"ENV\.fetch\(\s*['\"]([A-Z][A-Z0-9_]*)['\"]\s*,\s*['\"]([^'\"\n]+)['\"]\s*\)"),
             re.compile(r"ENV\[['\"]([A-Z][A-Z0-9_]*)['\"]\]\s*\|\|\s*['\"]([^'\"\n]+)['\"]")),
}
SOURCE_LOOKUP_PATTERNS = {
    "node": re.compile(r"process\.env(?:\.([A-Z][A-Z0-9_]*)|\[['\"]([A-Z][A-Z0-9_]*)['\"]\])"),
    "python": re.compile(r"os\.(?:environ(?:\.get)?|getenv)[\[(]\s*['\"]([A-Z][A-Z0-9_]*)['\"]"),
    "go": re.compile(r"os\.Getenv\(\s*\"([A-Z][A-Z0-9_]*)\""),
    "csharp": re.compile(r"GetEnvironmentVariable\(\s*\"([A-Z][A-Z0-9_]*)\""),
    "rust": re.compile(r"env::var\(\s*\"([A-Z][A-Z0-9_]*)\""),
    "java": re.compile(r"System\.getenv\(\s*\"([A-Z][A-Z0-9_]*)\""),
    "ruby": re.compile(r"ENV(?:\[|\.fetch\()\s*['\"]([A-Z][A-Z0-9_]*)['\"]"),
}
GO_ENV_FALLBACK = re.compile(
    r"(\w+)\s*:?=\s*os\.Getenv\(\s*\"([A-Z][A-Z0-9_]*)\"\s*\)[\s\S]{0,400}?"
    r"if\s+\1\s*==\s*\"\"\s*\{\s*\1\s*=\s*\"([^\"\n]+)\"")
CLIENT_OPTION = re.compile(r"\b(username|user|password|database|host|port)\s*[:=]\s*(['\"`])([^'\"`\n]*)\2")
SOURCE_LANGUAGE = {".ts": "node", ".mts": "node", ".cts": "node", ".js": "node", ".mjs": "node",
                   ".cjs": "node", ".py": "python", ".go": "go", ".cs": "csharp", ".rs": "rust",
                   ".java": "java", ".kt": "java", ".rb": "ruby"}

def blank_comments(text: str, language: str) -> str:
    """Comments blanked, strings kept: a default lives in a string literal."""
    if language in ("python", "ruby"):
        return re.sub(r"^\s*#[^\n]*", "", text, flags=re.M)
    return re.sub(r"/\*.*?\*/|(?<![:'\"`])//[^\n]*", lambda m: " " * len(m.group(0)), text, flags=re.S)

def endpoint_from_url(value: str, variable: str = "") -> dict | None:
    """One connection read out of a URL literal, or None when it is not one."""
    found = ENDPOINT_URL.match(str(value or ""))
    if not found:
        return None
    scheme, user, password, host, port, database, query = found.groups()
    scheme = scheme.lower()
    base = scheme.split("+", 1)[0]
    transport = scheme.split("+", 1)[1] if "+" in scheme else ""
    family = (ENV_DB_VARIABLE.match(variable or "") or [None, None])[1] if variable else None
    if base in ("http", "https"):
        engine = ENV_FAMILY_ENGINE.get(family or "", "") if variable else ""
        if not engine:
            engine = "clickhouse" if re.search(r"clickhouse", variable or "", re.I) else "unknown"
        transport, base_scheme = base, base
    else:
        engine = normalize_engine(base)
        base_scheme = ""
        if engine not in SQL_ENGINES:
            # redis://, amqp://, mongodb://, kafka://: other services the
            # application talks to, not a database this run can query.
            return None
    if engine == "unknown" and not variable:
        return None
    options = dict(urllib.parse.parse_qsl(query or "", keep_blank_values=True)) if query else {}
    name = urllib.parse.unquote(database or "") or options.get("database") or options.get("dbname") or ""
    number = int(port) if port else 0
    if number > 65535:
        return None
    endpoint_scheme = ""
    if engine == "clickhouse":
        if base_scheme:
            endpoint_scheme = base_scheme
        elif transport in ("http", "https"):
            endpoint_scheme = transport
        elif number in CLICKHOUSE_NATIVE_PORTS or transport == "native":
            endpoint_scheme = "native"
        else:
            endpoint_scheme = "http"
        if not number:
            number = 8443 if endpoint_scheme == "https" else 9000 if endpoint_scheme == "native" else 8123
    elif not number:
        number = DEFAULT_PORTS.get(engine, 0)
    return {"engine": engine, "host": host.strip("[]"), "port": number, "name": name,
            "user": urllib.parse.unquote(user or "") or options.get("user", ""),
            "password": urllib.parse.unquote(password or "") or options.get("password", ""),
            "scheme": endpoint_scheme}

def env_variable_family(name: str) -> tuple:
    """(family, kind) for a database variable name, or ("", "")."""
    found = ENV_DB_VARIABLE.match(name or "")
    if not found:
        return "", ""
    if found.group(3):
        return {"DATABASE_URL": "DATABASE", "DB_URL": "DB", "DSN": "DB"}[found.group(3)], "URL"
    kind = found.group(2)
    kind = {"URI": "URL", "DSN": "URL", "HOSTADDR": "HOST", "DB": "NAME", "DBNAME": "NAME",
            "DATABASE": "NAME", "USERNAME": "USER", "PASS": "PASSWORD"}.get(kind, kind)
    return found.group(1), kind

def bind(bindings: dict, name: str, value: str, source: str, rank: int) -> None:
    """Keep the lowest-ranked value per variable; the first at equal rank."""
    if not env_variable_family(name)[0] or value is None:
        return
    value = str(value).strip().strip("'\"")
    if not value:
        return
    current = bindings.get(name)
    if current is None or rank < current["rank"]:
        bindings[name] = {"value": value, "source": source, "rank": rank}

def env_file_bindings(root: str, bindings: dict, deadline: float) -> None:
    """Database settings read from the project's own environment files, within the time allowed.
    """
    for name in ENV_FILE_NAMES:
        if time.time() >= deadline:
            return
        text = contained_text(root, name, 100_000)
        if text is None:
            continue
        for line in text.splitlines():
            found = ENV_FILE_LINE.match(line)
            if found and not line.lstrip().startswith("#"):
                value = found.group(2).split(" #", 1)[0].strip()
                bind(bindings, found.group(1), value, "%s %s" % (name, found.group(1)), 1)

def config_file_endpoints(root: str, bindings: dict, deadline: float) -> list[dict]:
    """Connections stated in deployment and framework configuration files."""
    out = []
    code, listing = git(["ls-files", "--cached", "--others", "--exclude-standard", "-z", "--"],
                        root, max(1.0, deadline - time.time()))
    names = [p for p in listing.split("\0") if p] if code == 0 else []
    examined = 0
    for path in names:
        base = os.path.basename(path)
        if base not in CONFIG_FILE_NAMES and not (base.startswith("appsettings") and base.endswith(".json")) \
                and not (base.startswith("application") and base.endswith((".properties", ".yml", ".yaml"))):
            continue
        if examined >= SETTING_FILE_CAP or time.time() >= deadline or path.count("/") > 3:
            break
        examined += 1
        text = contained_text(root, path, 200_000)
        if text is None:
            continue
        label = path
        if base.startswith("appsettings") or base.endswith(".json"):
            for key, value in re.findall(r"\"([A-Za-z_][\w.-]*)\"\s*:\s*\"([^\"\n]*)\"", text):
                if re.search(r"\bHost\s*=", value, re.I):
                    parts = {k.strip().lower(): v.strip() for k, v in
                             (item.split("=", 1) for item in value.split(";") if "=" in item)}
                    host = parts.get("host") or parts.get("server")
                    if host:
                        out.append({"engine": "clickhouse" if "clickhouse" in key.lower() + value.lower()
                                    else "postgresql", "host": host,
                                    "port": int(parts["port"]) if parts.get("port", "").isdigit() else 0,
                                    "name": parts.get("database") or parts.get("db") or "",
                                    "user": parts.get("username") or parts.get("user id") or parts.get("user") or "",
                                    "password": parts.get("password", ""), "scheme": "",
                                    "source": "%s ConnectionStrings.%s" % (label, key)})
                found = endpoint_from_url(value, key.upper() if env_variable_family(key.upper())[0] else "")
                if found:
                    found["source"] = "%s %s" % (label, key)
                    out.append(found)
            continue
        if base.endswith(".properties"):
            props = dict(re.findall(r"^\s*([\w.-]+)\s*[=:]\s*(.*?)\s*$", text, flags=re.M))
            for key, value in props.items():
                if key.endswith((".url", ".jdbc-url", ".jdbcUrl")):
                    found = endpoint_from_url(value.replace("jdbc:ch:", "jdbc:clickhouse:"))
                    if found:
                        prefix = key.rsplit(".", 1)[0]
                        found["user"] = found["user"] or props.get(prefix + ".username", "") or props.get(prefix + ".user", "")
                        found["password"] = found["password"] or props.get(prefix + ".password", "")
                        found["source"] = "%s %s" % (label, key)
                        out.append(found)
            continue
        if base == "alembic.ini" or base.endswith(".toml"):
            for key, value in re.findall(r"^\s*([\w.-]*url)\s*=\s*['\"]?([^'\"\n]+)", text, flags=re.M | re.I):
                found = endpoint_from_url(value)
                if found:
                    found["source"] = "%s %s" % (label, key)
                    out.append(found)
            if base.endswith(".toml"):
                section = re.search(r"^\[(?:database|db|postgres|postgresql|clickhouse)[^\]]*\]\s*\n((?:(?!\[)[^\n]*\n?){1,20})",
                                    text, flags=re.M | re.I)
                if section:
                    fields = dict(re.findall(r"^\s*(\w+)\s*=\s*['\"]?([^'\"\n]+?)['\"]?\s*$", section.group(1), flags=re.M))
                    host = fields.get("host") or fields.get("hostname")
                    if host and literal_value(host):
                        engine = "clickhouse" if "clickhouse" in section.group(0).lower() else normalize_engine(
                            fields.get("engine") or fields.get("driver") or "postgresql")
                        out.append({"engine": engine, "host": host, "port": int(fields["port"]) if fields.get("port", "").isdigit() else 0,
                                    "name": fields.get("name") or fields.get("database") or fields.get("dbname") or "",
                                    "user": fields.get("user") or fields.get("username") or "",
                                    "password": fields.get("password", ""), "scheme": "", "source": "%s [section]" % label})
            continue
        if base.endswith((".yml", ".yaml")):
            # Line-based reading of compose services and framework YAML: a
            # KEY=VALUE or key: value line binds a variable or a field.
            compose = base.startswith(("docker-compose", "compose"))
            for line in text.splitlines():
                stripped = line.strip().lstrip("- ")
                found = re.match(r"([A-Z][A-Z0-9_]*)\s*[=:]\s*(.+?)\s*$", stripped)
                if found and env_variable_family(found.group(1))[0]:
                    value = found.group(2).strip().strip("'\"")
                    if not re.match(r"^\$\{?[A-Z_]+", value):
                        bind(bindings, found.group(1), value, "%s %s" % (label, found.group(1)), 2)
                    continue
                if not compose:
                    found = re.match(r"(url|jdbc-url|jdbcUrl)\s*:\s*(.+?)\s*$", stripped)
                    if found:
                        endpoint = endpoint_from_url(found.group(2).strip("'\"").replace("jdbc:ch:", "jdbc:clickhouse:"))
                        if endpoint:
                            endpoint["source"] = "%s %s" % (label, found.group(1))
                            out.append(endpoint)
            if base == "database.yml":
                fields = {}
                for key in ("adapter", "host", "port", "database", "username", "password"):
                    found = re.search(r"^\s+%s:\s*['\"]?([^'\"\n#]+?)['\"]?\s*$" % key, text, flags=re.M)
                    if found and not found.group(1).strip().startswith("<%"):
                        fields[key] = found.group(1).strip()
                if fields.get("host") and literal_value(fields["host"]):
                    out.append({"engine": normalize_engine(fields.get("adapter") or "postgresql"),
                                "host": fields["host"], "port": int(fields["port"]) if fields.get("port", "").isdigit() else 0,
                                "name": fields.get("database", ""), "user": fields.get("username", ""),
                                "password": fields.get("password", ""), "scheme": "", "source": label})
    return out

def source_default_bindings(root: str, bindings: dict, unresolved: dict, deadline: float) -> None:
    """Fallback literals the application's own code supplies for its variables.

    A lookup with no fallback is remembered as unresolved for that variable, so
    a run can say which variable the application reads without guessing it.
    """
    code, listing = git(["ls-files", "--cached", "--others", "--exclude-standard", "-z", "--"],
                        root, max(1.0, deadline - time.time()))
    names = [p for p in listing.split("\0") if p] if code == 0 else []
    examined = 0
    for path in names:
        extension = os.path.splitext(path)[1].lower()
        language = SOURCE_LANGUAGE.get(extension)
        if not language or "/test" in "/" + path.lower() or "node_modules/" in path or "vendor/" in path:
            continue
        if examined >= SOURCE_DEFAULT_FILE_CAP or time.time() >= deadline:
            break
        text = contained_text(root, path, 200_001)
        if text is None:
            continue
        if len(text) > 200_000 or not SOURCE_DEFAULT_WORDS.search(text):
            continue
        examined += 1
        clean = blank_comments(text, language)
        for pattern in SOURCE_DEFAULT_PATTERNS.get(language, ()):
            for found in pattern.finditer(clean):
                groups = [g for g in found.groups() if g is not None]
                if len(groups) < 2:
                    continue
                name, literal = groups[0], groups[-1]
                if name.isupper():
                    bind(bindings, name, literal, "default in %s for %s" % (path, name), 3)
        if language == "go":
            for _, name, literal in GO_ENV_FALLBACK.findall(clean):
                bind(bindings, name, literal, "default in %s for %s" % (path, name), 3)
        lookup = SOURCE_LOOKUP_PATTERNS.get(language)
        if lookup:
            for found in lookup.finditer(clean):
                name = next((g for g in found.groups() if g), "")
                if env_variable_family(name)[0] and name not in bindings:
                    unresolved.setdefault(name, path)
        # Client options given as literals next to the lookup, for the engine
        # the file names. These complete a URL-less connection; they never
        # override an explicit variable.
        engine_word = "clickhouse" if re.search(r"clickhouse", text, re.I) else (
            "pg" if re.search(r"\bpg\b|postgres|psycopg|asyncpg|npgsql|pgx|sqlx", text, re.I) else "")
        if engine_word:
            family = "CLICKHOUSE" if engine_word == "clickhouse" else "PG"
            for key, _, literal in CLIENT_OPTION.findall(clean):
                kind = {"username": "USER", "user": "USER", "password": "PASSWORD",
                        "database": "DATABASE", "host": "HOST", "port": "PORT"}[key.lower()]
                name = family + ("" if family == "PG" else "_") + kind
                if literal or kind == "PASSWORD":
                    bind(bindings, name, literal if literal else "", "%s option in %s" % (key, path), 3) if literal else None

def statement_default_bindings(statement: str, bindings: dict) -> None:
    """Defaults the current task states outright: "NAME defaults to VALUE"."""
    for clause in instruction_clauses(statement or ""):
        for found in STATEMENT_DEFAULT.finditer(clause):
            pairs = [(found.group(1), found.group(2))] + STATEMENT_DEFAULT_MORE.findall(found.group(3) or "")
            for name, value in pairs:
                # A sentence-final period belongs to the prose, not the value.
                bind(bindings, name.upper(), value.rstrip(".,;:)"),
                     "statement: %s" % " ".join(clause.split())[:120], 4)

def endpoints_from_bindings(bindings: dict) -> list[dict]:
    """One endpoint per variable family, URL first, then host/port/name parts."""
    families: dict = {}
    for name, item in bindings.items():
        family, kind = env_variable_family(name)
        families.setdefault(family, {})[kind] = (name, item)
    out = []
    for family, kinds in families.items():
        engine = ENV_FAMILY_ENGINE.get(family, "")
        sources = []
        endpoint = None
        if "URL" in kinds:
            name, item = kinds["URL"]
            endpoint = endpoint_from_url(item["value"], name)
            if endpoint:
                sources.append("%s" % item["source"])
        if endpoint is None and "HOST" in kinds:
            name, item = kinds["HOST"]
            host = literal_value(item["value"])
            if host:
                port = 0
                if "PORT" in kinds and kinds["PORT"][1]["value"].isdigit():
                    port = int(kinds["PORT"][1]["value"])
                scheme = ""
                if engine == "clickhouse":
                    scheme = "native" if port in CLICKHOUSE_NATIVE_PORTS else "http"
                    port = port or 8123
                endpoint = {"engine": engine or "unknown", "host": host,
                            "port": port or DEFAULT_PORTS.get(engine, 0), "name": "",
                            "user": "", "password": "", "scheme": scheme}
                sources.append(item["source"])
        if endpoint is None:
            continue
        for kind, field in (("NAME", "name"), ("USER", "user"), ("PASSWORD", "password")):
            if kind in kinds and not endpoint.get(field):
                endpoint[field] = kinds[kind][1]["value"]
                if kind != "PASSWORD":
                    sources.append(kinds[kind][1]["source"])
        if engine and endpoint.get("engine") in ("", "unknown"):
            endpoint["engine"] = engine
        endpoint["source"] = "; ".join(dict.fromkeys(sources))
        out.append(endpoint)
    return out

def merge_endpoints(*lists) -> list[dict]:
    """Distinct connections by (host, port, name); the first source wins."""
    out: list = []
    for found in lists:
        for endpoint in found:
            if endpoint.get("unresolved"):
                out.append(endpoint)
                continue
            key = (endpoint.get("host"), endpoint.get("port"), endpoint.get("name"))
            twin = next((e for e in out if not e.get("unresolved")
                         and (e.get("host"), e.get("port"), e.get("name")) == key), None)
            if twin is None:
                twin = next((e for e in out if not e.get("unresolved") and e.get("host") == key[0]
                             and e.get("port") == key[1] and ("" in (e.get("name"), key[2]))), None)
                if twin is not None and not twin.get("name"):
                    twin["name"] = key[2]
            if twin is None:
                out.append(dict(endpoint))
                continue
            for field in ("user", "password", "scheme", "engine"):
                if not twin.get(field) or twin.get(field) == "unknown":
                    twin[field] = endpoint.get(field) or twin.get(field)
            if endpoint.get("source") and endpoint["source"] not in twin.get("source", ""):
                twin["source"] = "%s; %s" % (twin.get("source", ""), endpoint["source"])
    return out

def database_endpoints(root: str, budget: float = 6.0, statement: str = "",
                       environ: dict | None = None) -> list[dict]:
    """Where this repository and its run say the database is, read as text.

    Sources in order of authority: the process environment this run was given,
    env files, deployment and framework configuration, the fallback literal the
    application's own code supplies for a variable, and a default the current
    task states. Python settings modules are read as before. A value computed
    at run time stays unresolved rather than guessed.
    """
    deadline = time.time() + max(1.0, budget)
    settings = settings_file_endpoints(root, max(0.5, budget / 2))
    if not ENV_ENDPOINTS:
        return settings[:3]
    bindings: dict = {}
    unresolved: dict = {}
    try:
        if environ:
            for name, value in environ.items():
                bind(bindings, name, value, "environment %s" % name, 0)
        env_file_bindings(root, bindings, deadline)
        configured = config_file_endpoints(root, bindings, deadline)
        source_default_bindings(root, bindings, unresolved, deadline)
        statement_default_bindings(statement, bindings)
    except BaseException:
        configured = []
    derived = endpoints_from_bindings(bindings)
    merged = merge_endpoints(settings, configured, derived)
    for endpoint in merged:
        if not endpoint.get("unresolved") and not endpoint.get("port"):
            endpoint["port"] = DEFAULT_PORTS.get(endpoint.get("engine") or "", 0)
    families_resolved = {env_variable_family(n)[0] for n in bindings}
    for name, path in unresolved.items():
        family = env_variable_family(name)[0]
        if family in families_resolved:
            continue
        engine = ENV_FAMILY_ENGINE.get(family, "") or "unknown"
        merged.append({"engine": engine, "host": "", "port": 0, "name": "",
                       "source": "%s reads %s" % (path, name), "variable": name, "unresolved": True})
        families_resolved.add(family)
    resolved = [e for e in merged if not e.get("unresolved")][:ENDPOINT_CAP]
    return resolved + [e for e in merged if e.get("unresolved")][:2]

def clickhouse_ping(host: str, port: int, timeout: float = 2.0) -> str:
    """Whether a ClickHouse HTTP endpoint answers /ping. That is all this is."""
    if not host or not port:
        return "no host and port to probe in the settings"
    try:
        with urllib.request.urlopen(urllib.request.Request(
                "http://%s:%d/ping" % (url_host(host), port)), timeout=timeout) as response:
            body = response.read(64).decode("utf-8", "replace").strip()
        return "HTTP /ping answered %r; not an authenticated database check" % body[:16]
    except (urllib.error.URLError, OSError, ValueError) as error:
        return ("HTTP /ping failed (%s); the application's own client may still "
                "reach it, so treat database checks as unverified until one runs"
                % type(error).__name__)

SQL_BUDGET_SEC = 60.0
SQL_OUTPUT_CAP = 6000
CH_QUERY_ID_PREFIX = "ridges-"
CH_QUERY_COUNTER = 0
CH_BODY_CAP = 1_000_000
CH_DETAIL_SEC = 10.0
CH_ERROR_CODE = re.compile(r"Code:\s*(\d+)\.")
MEASURE_RUN_SEC = 120.0
MEASURE_SCALES_MAX = 6
MEASURE_SPOOL_TAIL = 60_000
MEASURE_TAIL_SHOWN = 400
MEASURE_DIFF_SHOWN = 160


def output_comparison(before: str, after: str) -> str:
    """One line on whether the two runs printed the same thing.

    Same output is a fact about behaviour the command exercised; a first
    differing line shows where the change made itself felt. Neither is a
    verdict on correctness, which is the caller's to establish.
    """
    old = (before or "").splitlines()
    new = (after or "").splitlines()
    if old == new:
        return "outputs identical: the base and working-tree runs printed the same %d line(s)" % len(old)
    for number, (a, b) in enumerate(zip(old, new, strict=False), 1):
        if a != b:
            return ("outputs differ; first differing line %d: before=%r after=%r"
                    % (number, a[:MEASURE_DIFF_SHOWN], b[:MEASURE_DIFF_SHOWN]))
    longer, extra = ("after", new) if len(new) > len(old) else ("before", old)
    return ("outputs differ; the %s run printed %d more line(s), first extra: %r"
            % (longer, abs(len(new) - len(old)), extra[min(len(old), len(new))][:MEASURE_DIFF_SHOWN]))
# Dependency directories a separate checkout may share with the live tree.
# Build outputs (target, build, bin, obj) are not shared: they hold the
# candidate's compiled code, and the baseline must not run that.
MEASURE_LINKED_DIRS = (".venv", "venv", "node_modules", "vendor", ".tox", ".gradle", ".bundle")
MEASURE_COPIED_FILE_BYTES = 1_000_000
MEASURE_COUNTER = 0
MEASURE_WINDOW_COUNTER = 0
MEASURE_LOG_ROWS = 200
CH_WINDOW_SQL = (
    "SELECT count() AS statements, sum(read_rows) AS read_rows, sum(read_bytes) AS read_bytes, "
    "max(memory_usage) AS memory_usage, sum(result_rows) AS result_rows, sum(result_bytes) AS result_bytes, "
    "sum(query_duration_ms) AS query_duration_ms FROM system.query_log WHERE type = 'QueryFinish' "
    "AND event_time_microseconds >= {t0:DateTime64(6)} AND query_id NOT LIKE 'ridges-%' "
    "AND is_initial_query = 1 FORMAT JSONEachRow")
CH_WINDOW_ROWS_SQL = (
    "SELECT type, query_id, query, read_rows, read_bytes, memory_usage, result_rows, result_bytes, "
    "query_duration_ms FROM system.query_log WHERE type = 'QueryFinish' "
    "AND event_time_microseconds >= {t0:DateTime64(6)} AND query_id NOT LIKE 'ridges-%' "
    "AND is_initial_query = 1 ORDER BY event_time_microseconds LIMIT " + str(MEASURE_LOG_ROWS)
    + " FORMAT JSONEachRow")
PG_STATS_SQL = ("SELECT row_to_json(s) FROM (SELECT xact_commit, tup_returned, tup_fetched, blks_read, "
                "blks_hit FROM pg_stat_database WHERE datname = current_database()) s")
PG_STATEMENTS_SQL = ("SELECT coalesce(sum(calls), 0) FROM pg_stat_statements WHERE dbid = "
                     "(SELECT oid FROM pg_database WHERE datname = current_database())")
CH_QUERY_LOG_SQL = (
    "SELECT type, query_id, query, read_rows, read_bytes, memory_usage, result_rows, "
    "result_bytes, query_duration_ms FROM system.query_log WHERE query_id = {id:String} "
    "AND type = 'QueryFinish' ORDER BY event_time_microseconds DESC LIMIT 1 FORMAT JSONEachRow")

def sql_statements(query: str) -> list[str]:
    """The statements in the text, split where its code has a semicolon.

    Text, bodies and comments are not read for the split, so a semicolon inside
    a string stays inside its statement. Sent one by one, every statement's
    result prints in order on any psql, and a scenario can insert rows, query
    them and read each result before the tool's ROLLBACK leaves nothing.
    """
    text = query or ""
    code = sql_code_only(text, keep_positions=True)
    parts, begin = [], 0
    for index, char in enumerate(code):
        if char == ";":
            parts.append(text[begin:index])
            begin = index + 1
    parts.append(text[begin:])
    return [part.strip() for part in parts if part.strip()]


def single_statement(query: str) -> str:
    """"" when the text is one statement, else why it is not."""
    code = sql_code_only((query or "").rstrip().rstrip(";"))
    index = code.find(";")
    if index < 0:
        return ""
    return ("one statement per call; a second statement begins after the semicolon "
            "at character %d. Send the part you need to measure." % (index + 1))

def sql_has_format(query: str) -> bool:
    """Does this statement already name its own output format, so one must not be added?"""
    return bool(re.search(r"\bFORMAT\s+[A-Za-z]+\s*;?\s*$", sql_code_only(query or ""), re.I))

class ClickHouseHttp:
    """ClickHouse over its HTTP interface, one request per statement.

    Every statement goes read-only unless the caller allows a write, waits for
    the end of the query so the server's summary header is complete, carries
    a query id of this run's own, and is bounded server-side by
    max_execution_time. The password travels in a header only.
    """

    def __init__(self, target: "DatabaseTarget") -> None:
        """Bind an HTTP client to one configured database."""
        self.target = target

    def request(self, sql: str, *, budget: float, query_id: str = "", readonly=2,
                params: dict | None = None, settings: dict | None = None,
                database: bool = True) -> dict:
        """Send one statement over HTTP and return what came back, with the status and timings.

        Every request carries a query id, a read-only setting and its own budget, so it
        can be found afterwards in the server's own records and cannot outlive the time
        it was given.
        """
        import socket
        query = {}
        if database and self.target.database:
            query["database"] = self.target.database
        if query_id:
            query["query_id"] = query_id
        if readonly is not None:
            query["readonly"] = str(readonly)
        query["max_execution_time"] = str(int(max(1.0, budget)))
        query["wait_end_of_query"] = "1"
        query["default_format"] = "JSONEachRow"
        for key, value in (settings or {}).items():
            query[key] = str(value)
        for key, value in (params or {}).items():
            query["param_" + key] = str(value)
        url = self.target.http_base + "/?" + urllib.parse.urlencode(query)
        headers = {"Content-Type": "text/plain; charset=utf-8",
                   "X-ClickHouse-User": self.target.user or "default"}
        secret = self.target.secret()
        if secret:
            headers["X-ClickHouse-Key"] = secret
        began = time.monotonic()
        result = {"status": None, "body": "", "summary": None, "format": "", "error_code": None,
                  "elapsed": 0.0, "timed_out": False, "transport_error": "", "truncated": False}

        def read(response) -> str:
            data = response.read(CH_BODY_CAP + 1)
            if len(data) > CH_BODY_CAP:
                result["truncated"] = True
                data = data[:CH_BODY_CAP]
            return data.decode("utf-8", "replace")

        def summary_of(headers_in) -> dict | None:
            raw = headers_in.get("X-ClickHouse-Summary") if headers_in else None
            if not raw:
                return None
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                return None
            return value if isinstance(value, dict) else None

        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, data=sql.encode("utf-8"), headers=headers, method="POST"),
                    timeout=budget + 5.0) as response:
                result.update(status=response.status, body=read(response),
                              summary=summary_of(response.headers),
                              format=response.headers.get("X-ClickHouse-Format", ""))
        except urllib.error.HTTPError as error:
            body = ""
            try:
                body = read(error)
            except (OSError, ValueError):
                pass
            code = error.headers.get("X-ClickHouse-Exception-Code") if error.headers else None
            found = CH_ERROR_CODE.search(body or "")
            code = int(code) if code and str(code).isdigit() else int(found.group(1)) if found else None
            result.update(status=error.code, body=body, summary=summary_of(error.headers),
                          error_code=code, timed_out=(code == 159 or error.code == 408))
        except (urllib.error.URLError, OSError, ValueError, socket.timeout) as error:
            reason = getattr(error, "reason", error)
            result.update(transport_error=type(reason).__name__ if reason is not None else type(error).__name__,
                          timed_out=isinstance(reason, (socket.timeout, TimeoutError))
                          or "timed out" in str(error).lower())
        result["elapsed"] = time.monotonic() - began
        return result

    def run_measured(self, sql: str, *, budget: float, write: bool = False,
                     explain: bool = False, room: float = 0.0) -> tuple:
        """The statement's summary line, its query_log row when readable, then rows."""
        global CH_QUERY_COUNTER
        CH_QUERY_COUNTER += 1
        query_id = "%s%d-%x" % (CH_QUERY_ID_PREFIX, CH_QUERY_COUNTER, time.monotonic_ns() & 0xFFFFFF)
        statement = sql.rstrip().rstrip(";")
        if explain and not statement.lstrip().upper().startswith("EXPLAIN"):
            statement = "EXPLAIN PLAN actions=1 " + statement
        first = self.request(statement, budget=budget, query_id=query_id,
                             readonly=None if write else 2)
        lines = []
        summary = first.get("summary")
        if isinstance(summary, dict):
            report = {"type": "HTTPSummary", "query_id": query_id, "query": statement}
            report.update({k: summary[k] for k in CH_REPORT_FIELDS["HTTPSummary"] if k in summary})
            lines.append(json.dumps(report, ensure_ascii=False))
        else:
            lines.append("-- no X-ClickHouse-Summary header in the response; the statement's own "
                         "work is reported only by system.query_log below, if readable")
        detail_room = min(CH_DETAIL_SEC, max(0.0, room))
        if first.get("status") == 200 and not explain and detail_room >= 2.0:
            flush = self.request("SYSTEM FLUSH LOGS", budget=detail_room, readonly=None, database=False)
            if flush.get("status") == 200:
                detail = self.request(CH_QUERY_LOG_SQL, budget=detail_room, readonly=2,
                                      params={"id": query_id}, database=False)
                row = (detail.get("body") or "").strip().splitlines()
                if detail.get("status") == 200 and row:
                    lines.append(row[0])
                else:
                    lines.append("-- query_log detail unavailable: %s" % self.describe(detail))
            else:
                lines.append("-- query_log detail unavailable: SYSTEM FLUSH LOGS %s"
                             % self.describe(flush))
        elif first.get("status") == 200 and not explain:
            lines.append("-- query_log detail not requested: no time left for it")
        body = first.get("body") or ""
        if first.get("status") == 200:
            fmt = first.get("format") or ("JSONEachRow" if not sql_has_format(statement) else "as written")
            if explain:
                rendered = []
                for line in body.splitlines():
                    try:
                        rendered.append(str(json.loads(line).get("explain", line)))
                    except (ValueError, AttributeError):
                        rendered.append(line)
                body = "\n".join(rendered)
            size = len(body.encode("utf-8", "replace"))
            if size > SQL_OUTPUT_CAP:
                body = body[:SQL_OUTPUT_CAP] + "\n[rows truncated: %d of %d bytes shown%s]" % (
                    SQL_OUTPUT_CAP, size, "; response itself was cut at %d bytes" % CH_BODY_CAP
                    if first.get("truncated") else "")
            lines.append("-- %s (FORMAT %s, %d bytes, %.3fs)" % (
                "plan" if explain else "rows", fmt, size, first.get("elapsed", 0.0)))
            lines.append(body if body.strip() else "(no rows)")
        else:
            lines.append("-- error: %s" % self.describe(first, with_text=False))
            if body.strip():
                lines.append(clip(body, 2000, "server message"))
            if first.get("timed_out"):
                kill = self.request("KILL QUERY WHERE query_id = {id:String} SYNC", budget=3.0,
                                    readonly=None, params={"id": query_id}, database=False)
                lines.append("-- timed out after the %ds budget (max_execution_time); kill requested: %s"
                             % (int(budget), "done" if kill.get("status") == 200 else self.describe(kill)))
        return "\n".join(lines), first

    @staticmethod
    def describe(result: dict, with_text: bool = True) -> str:
        """What happened to one request, in a sentence: the server's code and its first line, or
        the transport fault.
        """
        if result.get("status") is None:
            return "no HTTP response (%s)%s" % (result.get("transport_error") or "transport error",
                                                 "; timed out" if result.get("timed_out") else "")
        code = result.get("error_code")
        head = (result.get("body") or "").strip().splitlines()
        text = head[0][:160] if head and with_text else ""
        return "HTTP %s%s%s" % (result["status"], " exception code %s" % code if code else "",
                                 ": " + text if text and result["status"] != 200 else "")

def link_dependency_dirs(root: str, where: str, budget: float = 10.0) -> list:
    """Share installed dependencies with a separate checkout; never build outputs.

    Top-level dependency directories present in the live tree and absent in
    the checkout are linked, and small ignored files at the top level (an
    .env, a local config) are copied, so the original code can start there.
    What was shared is reported with the reading.
    """
    shared: list = []
    if not root or not where or os.path.realpath(root) == os.path.realpath(where):
        return shared
    for name in MEASURE_LINKED_DIRS:
        source = os.path.join(root, name)
        target = os.path.join(where, name)
        if os.path.isdir(source) and not os.path.lexists(target):
            try:
                os.symlink(source, target)
                shared.append(name + "/ (linked)")
            except OSError:
                continue
    code, listing = git(["ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--"],
                        root, max(1.0, budget))
    if code == 0:
        copied = 0
        for path in [p for p in listing.split("\0") if p]:
            if "/" in path or copied >= 50:
                continue
            source = os.path.join(root, path)
            target = os.path.join(where, path)
            try:
                if (os.path.isfile(source) and not os.path.islink(source)
                        and os.path.getsize(source) <= MEASURE_COPIED_FILE_BYTES
                        and not os.path.lexists(target)):
                    shutil.copy2(source, target)
                    shared.append(path + " (copied)")
                    copied += 1
            except OSError:
                continue
    return shared

def thousands(value) -> str:
    """A number for reading: grouped digits, three decimals for a fraction, and n/a for nothing.
    """
    if isinstance(value, bool) or value is None:
        return "n/a"
    if isinstance(value, int):
        return "{:,}".format(value)
    if isinstance(value, float):
        return "{:,.3f}".format(value)
    return str(value)

def ratio_text(before, after) -> str:
    """The ratio of after to before, or the reason that ratio says nothing."""
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)) or isinstance(before, bool):
        return ""
    if before == 0:
        return "n/a (before is 0)" if after else "0 -> 0"
    return "x%.3g" % (after / before)

class HttpProbe:
    """One in-process database request, shaped like a finished Shell job.

    The record path reads a job's command, timing, exit status, text and
    identity keys; this supplies them for a request that never spawned a
    process, so a ClickHouse statement is recorded exactly as a psql one.
    """
    counter = 0

    def __init__(self, command: str, cwd: str, target: "DatabaseTarget", settings: dict) -> None:
        """Open a probe for one statement against one configured database."""
        HttpProbe.counter += 1
        self.name = "http%d" % HttpProbe.counter
        self.command = command
        self.cwd = os.path.realpath(cwd)
        self.started = time.time()
        self.started_monotonic = time.monotonic()
        self.invocation_key = command_identity(json.dumps([command, sorted(settings.items())],
                                                          ensure_ascii=True, default=str))
        self.environment_key = environment_identity({"target": target.label, "user": target.user,
                                                     **{k: str(v) for k, v in settings.items()}})
        self.timed_out = False
        self.output_complete = True
        self.closed = False
        self._out = ""
        self._code = None
        self.process = self

    def poll(self):
        """The probe's exit code, or None while it has not finished."""
        return self._code

    def complete(self, text: str, code: int, timed_out: bool = False) -> None:
        """Record the probe's output, its code and whether it ran out of time."""
        self._out, self._code, self.timed_out = text, code, timed_out

    def _text(self) -> str:
        """Everything the probe returned."""
        return self._out

    def _tail(self, cap: int = SHELL_OUTPUT_CAP) -> str:
        """The last of what the probe returned."""
        return self._out[-cap:]

    def finished(self) -> bool:
        """Has the probe finished?"""
        return self._code is not None

    def wait(self, timeout: float) -> tuple:
        """Whether the probe has finished, with what it returned; it does not block."""
        return self.finished(), self._out

    def result(self, out: str) -> str:
        """The probe's status line and its output, clipped, as the model will read it."""
        status = "running" if self._code is None else "exit_code=%d" % self._code
        if self.timed_out:
            status += " timed_out=true"
        return "[%s]\n%s" % (status, clip(steady(out), SHELL_OUTPUT_CAP, "query output") or "(no output)")

    def stop(self) -> None:
        """Close the probe."""
        self.closed = True
SQL_CHARS_MAX = 200_000
PG_CONNECT_TIMEOUT = "5"
OPEN_PROBE = "sql_tool_open"
DOLLAR_TAG = re.compile(r"\$[A-Za-z_][A-Za-z_0-9]*\$|\$\$")
# END is left out: it closes a CASE expression as well, and a body that uses it
# for a block is inside dollar quotes, which are blanked before this is applied.
CONTROL_WORD = re.compile(
    r"\b(COMMIT|ROLLBACK|ABORT|SAVEPOINT|RELEASE|START\s+TRANSACTION|BEGIN|"
    r"PREPARE\s+TRANSACTION)\b", re.I)

def sql_code_only(query: str, keep_positions: bool = False) -> str:
    """The statement with its text, its bodies and its comments blanked.

    A word inside a string, a dollar-quoted body or a comment is not a keyword,
    so a scan that reads them finds control where there is none and refuses a
    statement that was fine. With keep_positions every blanked span keeps its
    length, so an index into the result is an index into the statement.
    """
    out = []
    index, size = 0, len(query)
    while index < size:
        char = query[index]
        start = index
        if char == "'":
            # A backslash escapes the next character only in an E'' literal.
            # In an ordinary one the server takes it literally, so 'C:\\' ends
            # there; reading it as an escape would run past the closing quote
            # and swallow whatever follows the statement.
            before = "".join(out[-2:])
            escapes = bool(before) and before[-1] in "eE" and not (
                len(before) > 1 and (before[-2].isalnum() or before[-2] == "_"))
            index += 1
            while index < size:
                if escapes and query[index] == "\\" and index + 1 < size:
                    index += 2
                    continue
                if query[index] == "'":
                    if index + 1 < size and query[index + 1] == "'":
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            out.append(" " * (index - start) if keep_positions else " ")
            continue
        if char == '"':
            index += 1
            while index < size and query[index] != '"':
                index += 1
            index += 1
            out.append(" " * (index - start) if keep_positions else " ")
            continue
        if char == "-" and query[index:index + 2] == "--":
            while index < size and query[index] != "\n":
                index += 1
            out.append(" " * (index - start) if keep_positions else " ")
            continue
        if char == "/" and query[index:index + 2] == "/*":
            depth, index = 1, index + 2
            while index < size and depth:
                if query[index:index + 2] == "/*":
                    depth, index = depth + 1, index + 2
                elif query[index:index + 2] == "*/":
                    depth, index = depth - 1, index + 2
                else:
                    index += 1
            out.append(" " * (index - start) if keep_positions else " ")
            continue
        if char == "$":
            opened = DOLLAR_TAG.match(query, index)
            if opened:
                tag = opened.group(0)
                closed = query.find(tag, opened.end())
                index = size if closed < 0 else closed + len(tag)
                out.append(" " * (index - start) if keep_positions else " ")
                continue
        out.append(char)
        index += 1
    return "".join(out)

def transaction_control(query: str) -> str:
    """The transaction-control word this statement would run, if any."""
    found = CONTROL_WORD.search(sql_code_only(query or ""))
    return " ".join(found.group(1).split()).upper() if found else ""

PROBE_REFUSED = re.compile(r"SAVEPOINT can only be used in transaction blocks", re.I)

PLAN_NOTES = flag("RIDGES_PLAN_NOTES")
PLAN_NOTES_SEEN: set = set()
PLAN_NOTE_CAP = 6
SQL_WORD = re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE|WITH|MERGE)\b", re.I)
SQL_SHAPE_WORD = re.compile(r"\bFROM\b|\bINTO\b|\bJOIN\b", re.I)
# String literals of the languages the corpus uses, so SQL embedded in
# application code can be read: template/back-tick, triple-quoted, verbatim,
# raw and ordinary strings. The interpolation holes become one placeholder.
SQL_LITERAL = re.compile(
    r"`((?:[^`\\]|\\.)*)`|\"\"\"([\s\S]*?)\"\"\"|\'\'\'([\s\S]*?)\'\'\'|"
    r"@\"((?:[^\"]|\"\")*)\"|r#\"([\s\S]*?)\"#|\"((?:[^\"\\\n]|\\.)*)\"|\'((?:[^\'\\\n]|\\.)*)\'")
SQL_CLAUSE_END = re.compile(r"\b(?:WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT|UNION|SETTINGS|FORMAT|HAVING|WINDOW|QUALIFY)\b|[;)]", re.I)
COLUMN_EQUALITY = re.compile(r"[\w.\"`]+\s*=\s*[\w.\"`]+")
CONSTANT_PREDICATE = re.compile(r"^\s*(?:1\s*=\s*1|TRUE|1)\s*$", re.I)

def sql_fragments_from_source(text: str) -> list[str]:
    """SQL statements embedded as string literals in application code."""
    found = []
    for match in SQL_LITERAL.finditer(text or ""):
        literal = next((g for g in match.groups() if g is not None), "")
        if len(literal) < 12 or not SQL_WORD.search(literal) or not SQL_SHAPE_WORD.search(literal):
            continue
        literal = re.sub(r"\$\{[^}]*\}|\{[A-Za-z_][\w:().]*\}|%\(\w+\)s|#\{[^}]*\}", " __param__ ", literal)
        found.append(literal)
    return found

def select_list(code: str) -> str:
    """The text between the first SELECT and its own FROM, at depth zero."""
    start = re.search(r"\bSELECT\b", code, re.I)
    if not start:
        return ""
    depth, index = 0, start.end()
    while index < len(code):
        char = code[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif (depth == 0 and code[index:index + 4].upper() == "FROM"
              and not (code[index - 1].isalnum() or code[index - 1] == "_")
              and not (index + 4 < len(code) and (code[index + 4].isalnum() or code[index + 4] == "_"))):
            return code[start.end():index]
        index += 1
    return code[start.end():]

JOIN_BOUNDARY = re.compile(
    r"(?:(?:NATURAL\s+)?(?:LEFT|RIGHT|FULL|INNER|CROSS|ANY|ALL|ASOF|SEMI|ANTI|GLOBAL|ARRAY)(?:\s+OUTER)?\s+)*JOIN\b|"
    r"(?:WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT|UNION|SETTINGS|FORMAT|HAVING|WINDOW|QUALIFY|PREWHERE)\b", re.I)
QUALIFIED_COLUMN = re.compile(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b")

def join_segment(code: str, start: int) -> str:
    """The text from start to the next join or clause keyword outside parentheses."""
    depth, index = 0, start
    while index < len(code):
        char = code[index]
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif char == ";" and depth == 0:
            break
        elif (depth == 0 and char.isalpha() and (index == 0 or not (code[index - 1].isalnum() or code[index - 1] == "_"))
              and JOIN_BOUNDARY.match(code, index)):
            break
        index += 1
    return code[start:index]

def outside_parentheses(text: str) -> str:
    """The text with everything inside parentheses blanked, positions kept."""
    out, depth = [], 0
    for char in text:
        if char == "(":
            depth += 1
            out.append(" ")
        elif char == ")":
            depth = max(0, depth - 1)
            out.append(" ")
        else:
            out.append(" " if depth else char)
    return "".join(out)

def correlated_item(code: str, start: int) -> bool:
    """Whether the FROM item at start is LATERAL, or a function of an earlier item's column.

    PostgreSQL treats a function call in FROM as implicitly lateral, so
    unnest(u.tags) is evaluated per row of u, not multiplied with it.
    """
    rest = code[start:start + 400].lstrip()
    if re.match(r"LATERAL\b", rest, re.I):
        return True
    call = re.match(r"([A-Za-z_][\w.]*)\s*\(", rest)
    if not call or call.group(1).upper() in ("SELECT", "VALUES"):
        return False
    depth, index = 0, call.end() - 1
    while index < len(rest):
        if rest[index] == "(":
            depth += 1
        elif rest[index] == ")":
            depth -= 1
            if depth == 0:
                break
        index += 1
    return bool(QUALIFIED_COLUMN.search(rest[call.end():index]))

def lateral_on(code: str, position: int) -> bool:
    """Whether the ON clause at position closes a JOIN LATERAL item."""
    depth, index = 0, position - 1
    while index >= 3:
        char = code[index]
        if char == ")":
            depth += 1
        elif char == "(":
            if depth == 0:
                return False
            depth -= 1
        elif (depth == 0 and code[index - 3:index + 1].upper() == "JOIN"
              and (index - 4 < 0 or not (code[index - 4].isalnum() or code[index - 4] == "_"))):
            return bool(re.match(r"JOIN\s+LATERAL\b", code[index - 3:], re.I))
        index -= 1
    return False

def plan_shape_notes(sql_text: str = "", plan=None, plan_text: str = "") -> list[str]:
    """Describe observed counters in a decoded PostgreSQL execution plan.

    SQL source and textual step names alone do not establish execution work,
    cardinality, or a required repair. The optional text arguments remain
    accepted for existing callers, but only the supplied plan supplies notes.
    """
    del sql_text, plan_text  # Compatibility arguments supply no observations.
    notes: list = []
    if plan is not None:
        stack = [(plan, "$")]
        while stack:
            node, where = stack.pop()
            if isinstance(node, dict):
                if node.get("Node Type") == "Nested Loop":
                    inner = (node.get("Plans") or [None, None])[1:2]
                    inner = inner[0] if inner else None
                    loops = inner.get("Actual Loops") if isinstance(inner, dict) else None
                    if isinstance(loops, (int, float)) and loops > 1000:
                        notes.append("Nested Loop at %s ran its inner side %d times (Actual Loops "
                                     "of %s)" % (where, int(loops), inner.get("Node Type", "?")))
                removed, rows = node.get("Rows Removed by Filter"), node.get("Actual Rows")
                loops = node.get("Actual Loops")
                loops = loops if isinstance(loops, (int, float)) and not isinstance(loops, bool) and loops > 1 else 1
                if (isinstance(removed, (int, float)) and isinstance(rows, (int, float))
                        and not isinstance(removed, bool) and not isinstance(rows, bool)):
                    # PostgreSQL reports both as per-loop averages.
                    removed, rows = removed * loops, rows * loops
                    if removed >= max(10000, 100 * max(rows, 0)):
                        notes.append("%s at %s discarded %s%d rows by filter%s to return %s%d" % (
                            node.get("Node Type", "node"), where, "about " if loops > 1 else "",
                            int(removed), " across %d loops" % int(loops) if loops > 1 else "",
                            "about " if loops > 1 else "", int(rows)))
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        stack.append((value, "%s.%s" % (where, key)))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    stack.append((value, "%s[%d]" % (where, index)))
    unique: list = []
    for note in notes:
        if note not in unique:
            unique.append(note)
    return unique[:PLAN_NOTE_CAP]

def plan_notes_block(notes: list) -> str:
    """The execution-plan counters that were observed, said as observations rather than as a
    verdict.
    """
    if not notes:
        return ""
    return ("\nObserved execution-plan counters (not a correctness or improvement verdict):\n"
            + "\n".join("  - " + note for note in notes[:PLAN_NOTE_CAP]))

def edit_shape_notes(path: str, before: str, after: str) -> str:
    """Editing source supplies no execution-plan observation."""
    del path, before, after  # Retain the existing callback's call signature.
    return ""

# The names an edit's SQL calls are read against the ClickHouse server the
# repository configures. The server's own catalog says which names it lists; a
# name it does not list is put to the server as a bare call, and only the
# server's answer that it has no such function is reported, in its own words.
CH_NAMES = flag("RIDGES_CH_NAMES")
CH_NAME_FLOOR = 200
CH_NAME_SEC = 5.0
CH_NAME_LIMIT = 12
CH_UNKNOWN_FUNCTION = 46
CH_NAME_STATE: dict = {}
CH_CATALOG_QUERIES = (
    ("functions", "SELECT name, case_insensitive, is_aggregate FROM system.functions"),
    ("combinators", "SELECT name FROM system.aggregate_function_combinators"),
    ("table_functions", "SELECT name FROM system.table_functions"),
    ("types", "SELECT name FROM system.data_type_families"),
    ("engines", "SELECT name FROM system.table_engines"),
    ("keywords", "SELECT keyword AS name FROM system.keywords"))
# Words that stand before "(" in SQL without calling anything, for a server
# whose keyword table cannot be read.
SQL_PAREN_WORDS = frozenset((
    "SELECT", "FROM", "WHERE", "IN", "EXISTS", "ANY", "ALL", "SOME", "OVER", "FILTER", "WITHIN",
    "VALUES", "AS", "ON", "USING", "AND", "OR", "NOT", "BY", "INTERVAL", "JOIN", "INTO", "WITH",
    "CASE", "WHEN", "THEN", "ELSE", "LIMIT", "OFFSET", "HAVING", "UNION", "EXCEPT", "INTERSECT",
    "LIKE", "ILIKE", "BETWEEN", "IS", "NULL", "DISTINCT", "SETTINGS", "PREWHERE", "QUALIFY",
    "WINDOW", "TABLE", "COLUMNS", "ARRAY", "GLOBAL", "LATERAL", "RETURNING", "SET", "KEY", "TTL"))
SQL_DEFINITION = re.compile(
    r"^\s*(?:CREATE|ALTER|ATTACH|DETACH|DROP|RENAME|TRUNCATE|OPTIMIZE|EXCHANGE|GRANT|REVOKE)\b", re.I)
SQL_CALLED = re.compile(r"(?<![\w.`\"$@:])([A-Za-z_][A-Za-z0-9_]*)\s*\(")

def sql_called_names(sql: str) -> list:
    """Names followed by "(" in a statement's code, in order, once each.

    Strings, comments and quoted names are not read, a qualified name is not a
    plain call, and a table named with its column list after INTO or TABLE is
    not a call at all.
    """
    code = sql_code_only(sql or "")
    names: list = []
    for match in SQL_CALLED.finditer(code):
        head = re.search(r"([A-Za-z_]\w*)\s*$", code[:match.start()])
        if head and head.group(1).upper() in ("INTO", "TABLE"):
            continue
        if match.group(1) not in names:
            names.append(match.group(1))
    return names

def edit_sql_calls(path: str, before: str, after: str) -> list:
    """Names the SQL an edit added calls: in the statements its added lines are part of."""
    added = [text for _, text in added_lines(before, after) if text.strip()]
    if not added:
        return []
    if path.lower().endswith(".sql"):
        statements = [part for part in re.split(r";\s*(?:\n|$)", after) if part.strip()]
    else:
        # A literal on one line is inside an added line; a literal over several
        # lines holds the added line. Both kinds are read.
        statements = sql_fragments_from_source("\n".join(added))
        statements += [fragment for fragment in sql_fragments_from_source(after)
                       if fragment not in statements and any(line.strip() in fragment for line in added)]
    names: list = []
    for statement in statements:
        if SQL_DEFINITION.match(statement):
            continue
        for name in sql_called_names(statement):
            called = re.compile(r"(?<![\w.])%s\s*\(" % re.escape(name))
            if name not in names and any(called.search(line) for line in added):
                names.append(name)
    return names

class ClickHouseCatalog:
    """What one ClickHouse server lists as functions, and the words that precede "(" without calling."""

    def __init__(self, tables: dict) -> None:
        """Read one server's catalog into the sets a name can be checked against."""
        rows = tables.get("functions") or []
        self.functions = {row["name"] for row in rows}
        self.any_case = {row["name"].lower() for row in rows if row.get("case_insensitive")}
        self.aggregates = {row["name"] for row in rows if row.get("is_aggregate")}
        self.suffixes = sorted({row["name"] for row in tables.get("combinators") or []}, key=len, reverse=True)
        words = {row["name"].lower() for key in ("table_functions", "types", "engines")
                 for row in tables.get(key) or []}
        for row in tables.get("keywords") or [{"name": word} for word in SQL_PAREN_WORDS]:
            words.update(part.lower() for part in row["name"].split())
        self.not_calls = words

    def trusted(self) -> bool:
        """Did enough of the catalog come back to be worth consulting?"""
        return len(self.functions) >= CH_NAME_FLOOR

    def lists(self, name: str) -> bool:
        """Whether the server lists the name as a function, or it is no call there."""
        folded = name.lower()
        return (name in self.functions or folded in self.any_case or folded in self.not_calls
                or self.combined(name, 3))

    def combined(self, name: str, depth: int) -> bool:
        """Whether the name is an aggregate function carrying combinator suffixes."""
        for suffix in self.suffixes:
            stem = name[:-len(suffix)] if name.endswith(suffix) else ""
            if stem and (stem in self.aggregates or (depth > 1 and self.combined(stem, depth - 1))):
                return True
        return False

def clickhouse_catalog(target: "DatabaseTarget") -> "ClickHouseCatalog | None":
    """The server's catalog, read once per run; None when it cannot be read or is too small to trust."""
    if "catalog" in CH_NAME_STATE:
        return CH_NAME_STATE["catalog"] or None
    client = ClickHouseHttp(target)
    tables: dict = {}
    for key, sql in CH_CATALOG_QUERIES:
        result = client.request(sql, budget=CH_NAME_SEC, readonly=2, database=False)
        if result.get("status") != 200:
            if key == "functions":
                break
            continue  # an older server without this table
        found = []
        for line in (result.get("body") or "").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                found.append(row)
        tables[key] = found
    catalog = ClickHouseCatalog(tables)
    CH_NAME_STATE["catalog"] = catalog if catalog.trusted() else False
    return CH_NAME_STATE["catalog"] or None

def clickhouse_absent(target: "DatabaseTarget", name: str) -> str:
    """The server's own answer when it has no function by this name, else ""."""
    answers = CH_NAME_STATE.setdefault("answers", {})
    if name in answers:
        return answers[name]
    if len(answers) >= CH_NAME_LIMIT:
        return ""
    result = ClickHouseHttp(target).request("SELECT %s()" % name, budget=CH_NAME_SEC,
                                            readonly=2, database=False)
    if result.get("error_code") != CH_UNKNOWN_FUNCTION:
        answers[name] = ""
        return ""
    said = (result.get("body") or "").strip()
    try:
        wrapped = json.loads(said)  # the HTTP interface wraps it for a JSON output format
        said = wrapped["exception"] if isinstance(wrapped, dict) and isinstance(wrapped.get("exception"), str) else said
    except ValueError:
        pass
    said = re.sub(r"\s*\(version [^()]*(?:\([^()]*\))?\)\s*$", "", " ".join(said.split()))
    answers[name] = said[:300] or "no function by that name (code %d)" % CH_UNKNOWN_FUNCTION
    return answers[name]

def clickhouse_absent_names(target: "DatabaseTarget", names: list) -> list:
    """(name, the server's answer) for each name the server says it does not have."""
    catalog = clickhouse_catalog(target)
    if catalog is None:
        return []
    found = []
    for name in names:
        if not catalog.lists(name):
            said = clickhouse_absent(target, name)
            if said:
                found.append((name, said))
    return found

def clickhouse_names_text(found: list) -> str:
    """Every name the server said it does not have, each with its answer (at most CH_NAME_LIMIT are asked)."""
    return "; ".join("%s (the server answers: %s)" % (name, said) for name, said in found)

class DatabaseTarget:
    """One database this repository configures, and what is known about it.

    Configuration, reachability, authentication, execution and rollback are
    separate observations. None means unknown, not successful. A failed or
    unfinished attempt must not inherit successful observations from an old one.
    The password is held here and put only into a client's environment; it is
    never formatted into a log line, a model message or a process argument.
    """

    def __init__(self, engine: str, host: str, port: int, database: str,
                 user: str = "", password: str = "", source: str = "",
                 scheme: str = "") -> None:
        """Record one configured database, with its engine normalised and its password kept out of
        sight.
        """
        self.engine = normalize_engine(engine)
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self._password = password
        self.source = source
        # "http"/"https" for a ClickHouse HTTP endpoint, "native" for its
        # binary port; empty when the configuration did not say.
        self.scheme = scheme
        self.state = "configured"
        self.detail = "read from %s" % source
        self.observations = {"configured": True, "reachable": None,
                             "authenticated": None, "executed": None,
                             "rollback": None}

    @property
    def http_base(self) -> str:
        """The HTTP origin of a ClickHouse endpoint, without credentials."""
        scheme = self.scheme if self.scheme in ("http", "https") else "http"
        return "%s://%s:%d" % (scheme, url_host(self.host), self.port or 8123)

    def observe_http(self, status, error_code, timed_out: bool = False,
                     transport_error: str = "") -> None:
        """Record one HTTP attempt's actual outcome, nothing inherited.

        A 200 establishes that the server ran the statement; an exception code
        says which step refused it. Nothing is rolled back over HTTP, so the
        rollback observation stays unknown rather than claiming a transaction.
        """
        self.begin_attempt()
        if timed_out:
            self.detail = "client timed out; execution is unconfirmed"
            self.observations["reachable"] = True if status is not None else None
            return
        if status is None:
            self.state = "failed"
            self.detail = "no HTTP response (%s); reachability is unconfirmed" % (
                transport_error or "transport error")
            self.observations.update(reachable=False, executed=False)
            return
        if status == 200:
            self.state, self.detail = "executed", "the server ran the statement over HTTP"
            self.observations.update(reachable=True, authenticated=True, executed=True)
            return
        self.state = "failed"
        self.detail = "HTTP %s%s; successful execution is unconfirmed" % (
            status, (" exception code %s" % error_code) if error_code else "")
        self.observations.update(reachable=True, executed=False)
        if error_code in (194, 516):
            self.observations["authenticated"] = False
        elif error_code is not None:
            self.observations["authenticated"] = True

    def begin_attempt(self) -> None:
        """Start a fresh attempt: nothing is known about this target until it answers."""
        self.state, self.detail = "unknown", "client running; execution is unconfirmed"
        for name in ("reachable", "authenticated", "executed", "rollback"):
            self.observations[name] = None

    def observe_client(self, code: int | None, out: str, timed_out: bool = False) -> None:
        """Record this attempt's actual exit status and positive acknowledgements.

        Client completion is distinct from query completion. Successful psql
        with ON_ERROR_STOP establishes the requested commands completed. A
        failure can leave effects unknown, even when connection setup succeeded.
        """
        self.begin_attempt()
        if OUTPUT_INCOMPLETE_MARKER in (out or ""):
            self.detail = "client output incomplete; execution and rollback are unconfirmed"
            return
        tags = [line.strip() for line in (out or "").splitlines()]
        if timed_out or code is None:
            self.detail = "client timed out" if timed_out else "client exit status is unknown"
            return
        if code == 0:
            self.state, self.detail = "executed", "psql completed the requested SQL successfully"
            self.observations.update(reachable=True, authenticated=True, executed=True)
            # Require an affirmative final rollback tag as well as the open
            # transaction probe, not merely absence of one particular error.
            if ("BEGIN" in tags and "SAVEPOINT" in tags and tags
                    and tags[-1] == "ROLLBACK" and not PROBE_REFUSED.search(out)):
                self.observations["rollback"] = True
            return
        self.state, self.detail = "failed", "psql exited %s; successful execution is unconfirmed" % code
        if "BEGIN" in tags:
            self.observations.update(reachable=True, authenticated=True)
        elif re.search(r"^psql:.*(?:password authentication failed|no pg_hba.conf entry)",
                       out or "", re.I | re.M):
            self.observations.update(reachable=True, authenticated=False, executed=False)
        elif re.search(r"^psql:.*(?:connection[^\n]*refused|could not translate host name)",
                       out or "", re.I | re.M):
            self.observations.update(reachable=False, executed=False)

    @property
    def label(self) -> str:
        """A name for this connection that carries no secret."""
        return "%s://%s:%s/%s" % (self.engine, self.host or "?", self.port or "?",
                                  self.database or "?")

    def describe(self) -> str:
        """The target as the model reads it: where it is, who connects, and what has been observed.
        """
        who = (" as %s" % self.user) if self.user else ""
        facts = ", ".join("%s=%s" % (name, "unknown" if value is None else str(value).lower())
                          for name, value in self.observations.items())
        return "%s%s [%s: %s; %s]" % (self.label, who, self.state, self.detail, facts)

    def secret(self) -> str:
        """The password for this target, for the one place that has to pass it to a client."""
        return self._password

    def argv(self, sql_path: str = "") -> list:
        """A client command line with no password in it."""
        if self.engine != "postgresql":
            return []
        argv = ["psql", "-X", "-v", "ON_ERROR_STOP=1", "-P", "pager=off",
                "-h", str(self.host), "-p", str(self.port or 5432)]
        if self.user:
            argv += ["-U", self.user]
        if self.database:
            argv += ["-d", self.database]
        return argv + (["-f", sql_path] if sql_path else [])

    def statement_argv(self, statements: list) -> list:
        """A client command line that sends these statements and nothing else.

        Each is passed with -c. The SQL tool rejects a leading backslash because
        psql also accepts one internal client command in that option. Backslash
        text embedded in SQL remains part of the server's SQL input.
        """
        argv = self.argv()
        for statement in statements:
            argv += ["-c", statement]
        return argv

    def environment_extra(self) -> dict:
        """Only what the client needs beyond the inherited environment."""
        if self.engine != "postgresql":
            return {}
        extra = {"PGCONNECT_TIMEOUT": PG_CONNECT_TIMEOUT}
        if self._password:
            extra["PGPASSWORD"] = self._password
        return extra

SECRET_DISTINCTIVE = 12

def scrub(text: str, target: "DatabaseTarget | None") -> str:
    """Remove a known secret from anything about to be shown or recorded.

    A short password is often an ordinary word -- a role, a database, a table
    prefix -- and replacing every occurrence of it would leave the reply saying
    <withheld>_ipam_vlan, which hides the answer rather than the secret. So a
    short one is withheld where it stands as a word of its own, and a long one,
    which no identifier is going to contain by accident, wherever it appears.
    """
    out = redact(text or "")
    secret = target.secret() if target else ""
    if not secret or secret not in out:
        return out
    if len(secret) >= SECRET_DISTINCTIVE:
        return out.replace(secret, "<withheld>")
    return re.sub(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(secret),
                  "<withheld>", out)

def endpoint_answers(host: str, port: int, timeout: float = 2.0) -> str:
    """Whether a TCP port accepts a connection. That is all this establishes.

    It is not an authenticated connection, not the application's own client, and
    not a statement about the schema or fixtures being present. A refusal is not
    proof the application cannot connect either: the client may use a socket, a
    different host inside the run, or credentials this never touches.
    """
    if not host or not port:
        return "no host and port to probe in the settings"
    import socket
    began = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return ("a TCP port accepted a connection in %dms; not an "
                    "authenticated database check" % int(1000 * (time.time() - began)))
    except OSError as error:
        return ("TCP connect failed (%s); the application's own client may still "
                "reach it, so treat database checks as unverified until one runs"
                % type(error).__name__)

def symbol_sites(root: str, names: list, budget: float = 6.0) -> list[str]:
    """Where the repository defines the symbols the target code mentions.

    One allowance for the whole sweep, not one per search: six searches each
    handed the remaining budget would spend six times the budget. No floor
    either, so an expired allowance starts nothing.
    """
    out = []
    deadline = time.time() + max(0.0, budget)
    for name in names[:6]:
        left = deadline - time.time()
        if left <= 0.2:
            break
        code, hits = git(["grep", "-n", "-E", "--untracked", "--",
                          r"^\s*(class|def|func|function)\s+%s\b" % re.escape(name)],
                         root, left)
        if code == 0:
            for line in hits.splitlines()[:2]:
                path, _, rest = line.partition(":")
                number, _, _ = rest.partition(":")
                out.append("%s: candidate definition at %s:%s (textual match)"
                       % (name, path, number))
    return out[:8]

CALLER_CAP = 5

def caller_sites(root: str, method: str, budget: float = 6.0) -> list[str]:
    """Who calls the method the statement names."""
    if not method:
        return []
    if budget <= 0.2:
        return []
    code, hits = git(["grep", "-n", "-F", "--untracked", "--", "." + method + "("], root,
                     budget)
    if code != 0:
        return []
    return [line.split(":", 2)[0] + ":" + line.split(":", 2)[1]
            for line in hits.splitlines()][:CALLER_CAP]

TARGET_SYMBOL = re.compile(r"\b([A-Z][A-Za-z0-9_]{2,})\b")

def db_facts(statement: str, tree: Tree, budget: float = 20.0) -> list[str]:
    """The cheap facts a database task needs, gathered without a model call."""
    if not database_context(statement, tree.root):
        return []
    deadline = time.time() + max(2.0, budget)
    facts: list[str] = []

    def room(cap: float) -> float:
        return max(0.5, min(cap, deadline - time.time()))

    kinds = task_kinds(statement)
    facts.append("task words describe: %s" % (", ".join(kinds) if kinds
                                              else "no database task kind named"))
    path = stated_file(statement, tree.root)
    methods = stated_methods(statement)
    if path:
        facts.append("file the statement names: %s" % path)
        try:
            source = tree.read(path)
        except BaseException:
            source = ""
        region = fenced_region(source, methods) if source and methods else None
        if region:
            name, low, high = region
            facts.append("method the statement names: %s at lines %d-%d" % (name, low, high))
            body = "\n".join(source.splitlines()[low - 1:high])
            wanted = [w for w in dict.fromkeys(TARGET_SYMBOL.findall(body))
                      if w not in ("True", "False", "None")]
            if time.time() < deadline:
                facts.extend(symbol_sites(tree.root, wanted, room(6.0)))
        if methods and time.time() < deadline:
            callers = caller_sites(tree.root, methods[-1][1], room(6.0))
            facts.append("textual matches for `.%s(`: %s (candidates, not a "
                         "resolved call graph)" % (
                             methods[-1][1],
                             ", ".join(callers) if callers else "none in this checkout"))
    for endpoint in database_endpoints(tree.root, room(6.0), statement, dict(os.environ)):
        if endpoint.get("unresolved"):
            facts.append("database in %s: named but computed at run time, so this "
                         "inspection did not resolve it%s" % (
                             endpoint["source"],
                             "; left unverified under the task's execution restriction"
                             if execution_prohibited(statement) else
                             " -- read it in the running application if a check needs it"))
            continue
        if time.time() >= deadline:
            facts.append("database settings found in %s; no time left in this "
                         "inspection to probe it" % endpoint["source"])
            continue
        facts.append("database in %s: %s host=%s port=%s name=%s -- configured; %s%s" % (
            endpoint["source"], endpoint["engine"], endpoint["host"],
            endpoint["port"] or "?", endpoint["name"] or "?",
            ("not probed because the task prohibits database execution"
             if execution_prohibited(statement) else
             clickhouse_ping(endpoint["host"], endpoint["port"], room(2.0))
             if endpoint.get("engine") == "clickhouse" and endpoint.get("scheme") != "native"
             else endpoint_answers(endpoint["host"], endpoint["port"], room(2.0))),
            "; authentication and a query are separate steps, so use the sql tool "
            "or the application's client before treating it as usable"
            if DB_TOOL and not execution_prohibited(statement) else ""))
    runner = stated_runner(statement, tree.root)
    if runner:
        facts.append("check command the statement gives: %s" % runner)
    return facts

DB_PROBE = flag("RIDGES_DB_PROBE")
DB_PROBE_SEC = 30.0
DB_PROBE_RETRY_SEC = 2.0
DB_PROBE_ONE_SEC = 8.0
PG_TABLE_COUNT = ("select count(*) from information_schema.tables where table_schema "
                  "not in ('pg_catalog', 'information_schema')")
CH_TABLE_COUNT = ("SELECT count() FROM system.tables WHERE database NOT IN "
                  "('system', 'INFORMATION_SCHEMA', 'information_schema')")
PROBE_STARTING = re.compile(r"connection refused|could not connect|starting up|"
                            r"is the server running|Connection reset|timeout expired|"
                            r"Name or service not known|Temporary failure", re.I)


def probe_once(target, budget: float) -> tuple[str, str]:
    """(fact, "") when the database answered, ("", why) when it did not yet.

    The fact is the server's version and how many application tables it holds.
    A refusal to connect is a reason to ask again; a failed login is not, and
    comes back as the fact itself.
    """
    if target.engine == "postgresql":
        if not shutil.which("psql"):
            return ("database probe: no psql client here; the application's own client "
                    "is the way to reach %s" % target.label, "")
        env = dict(os.environ)
        env.update(target.environment_extra())
        argv = [*target.argv(), "-At", "-c", "select version()", "-c", PG_TABLE_COUNT]
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=max(2.0, budget), env=env,
                                  check=False)
        except (subprocess.TimeoutExpired, OSError) as error:
            return "", "the client did not answer: %s" % type(error).__name__
        out = scrub((done.stdout or "") + (done.stderr or ""), target)
        if done.returncode != 0:
            if PROBE_STARTING.search(out):
                return "", one_line(out)[:160]
            return ("database probe: %s did not accept the configured credentials or database (%s)"
                    % (target.label, one_line(out)[:160])), ""
        lines = [line.strip() for line in (done.stdout or "").splitlines() if line.strip()]
        if len(lines) < 2:
            return "", "the answer was incomplete"
        version, count = scrub(lines[0], target), lines[-1]
    elif target.engine == "clickhouse" and target.scheme in ("http", "https", "") :
        client = ClickHouseHttp(target)
        version = client.request("SELECT version()", budget=max(2.0, budget / 2), readonly=2)
        if version.get("status") != 200:
            why = scrub(one_line(str(version.get("body") or version.get("error") or "no answer"))[:160], target)
            if version.get("status") is None or PROBE_STARTING.search(why):
                return "", why
            return "database probe: %s did not accept the request (%s)" % (target.label, why), ""
        tables = client.request(CH_TABLE_COUNT, budget=max(2.0, budget / 2), readonly=2)
        if tables.get("status") != 200:
            return "", "the table count did not answer"
        version, count = "ClickHouse " + one_line(version.get("body") or "").strip(), one_line(tables.get("body") or "").strip()
    else:
        return "", "no client here for %s" % target.engine
    try:
        tables_count = int(count)
    except ValueError:
        return "", "the table count was not a number"
    where = target.database or "the configured database"
    if tables_count == 0:
        return ("database probe: %s answers (%s); 0 application tables in %s, so no schema is "
                "loaded yet: the repository's setup or test command is what creates it"
                % (target.label, version[:80], where)), ""
    return ("database probe: %s answers (%s); %d application table(s) in %s"
            % (target.label, version[:80], tables_count, where)), ""


def database_probe(kit, statement: str, budget: float = DB_PROBE_SEC) -> str:
    """One authenticated reading of the configured database, or "".

    A database still starting is asked again until it answers or the budget
    ends, so what the run reads at the start does not depend on how fast the
    host brought it up.
    """
    if not DB_PROBE or execution_prohibited(statement):
        return ""
    try:
        target = kit.pick_target()
    except Exception:
        return ""
    deadline = time.monotonic() + max(2.0, budget)
    why = "no answer"
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            break
        fact, why = probe_once(target, min(DB_PROBE_ONE_SEC, left))
        if fact:
            return fact
        time.sleep(min(DB_PROBE_RETRY_SEC, max(0.0, deadline - time.monotonic())))
    return ("database probe: %s gave no answer within %.0f s (%s); it is configured, not "
            "shown reachable" % (target.label, budget, why or "no answer"))


# "uses one database query" is a bound on the finished code: one reading of the
# finished code answers it. "must not scale", "reduce", "faster" compare two
# states, and only a before and an after answer those.
ABSOLUTE_OBLIGATION = re.compile(
    r"\bone database query\b|\bsingle query\b|\bat most \d+ quer|"
    r"\bno more than \d+ quer|\bexactly \d+ quer|"
    r"\bat most (?:one|\d+) (?:database |server-executed |sql )?(?:quer(?:y|ies)|statements?)\b|"
    r"\bone database statement\b|\bsingle (?:database |sql )?statement\b", re.I)

QUERY_OBLIGATION = re.compile(
    r"\bone database query\b|\bsingle query\b|\bnumber of quer(?:y|ies)\b|"
    r"\bquery count\b|\bmust not (?:scale|grow)\b|\bbounded by\b|"
    r"\bwithout (?:an? )?(?:extra|additional) quer|\bn\+1\b", re.I)
# "bounded by max_length", "leg n+1", "before optimization" bound or name the
# domain; they bound the database work only in a clause that names that work.
# "N+1" written as the idiom is, in capitals, names the query problem itself.
BOUND_NEEDS_NOUN = re.compile(r"bounded by|n\s?\+\s?1|optimi|speed\s+up", re.I)
QUERY_IDIOM = re.compile(r"N\s?\+\s?1")


def domain_bound(words: str, clause: str) -> bool:
    """Do these matched words bound the domain rather than the database work?"""
    return bool(BOUND_NEEDS_NOUN.search(words) and not QUERY_IDIOM.search(words)
                and not DATABASE_NOUN.search(clause))
DATABASE_NOUN = re.compile(
    r"\b(?:quer(?:y|ies)|statements?|round\s?trips?|scans?|rows?|database|sql|index(?:es)?|"
    r"latenc(?:y|ies)|performance|slow(?:er)?|fast(?:er)?|clickhouse|postgres(?:ql)?|transfer)\b", re.I)

def query_obligation(statement: str, root: str = "", *, observed_database: bool = False) -> str:
    """The statement's own words binding how much database work is allowed.

    Read from the text, not from the task label: a correctness repair can carry
    a one-query obligation, and calling it a repair must not drop that.
    """
    if not database_context(statement, root, observed_database=observed_database):
        return ""
    for line in instruction_clauses(statement):
        found = (QUERY_OBLIGATION.search(line or "")
                 or ABSOLUTE_OBLIGATION.search(line or ""))
        if found and domain_bound(found.group(0), line):
            continue
        if found:
            return " ".join(line.split())[:REQUIREMENT_CHARS]
    return ""

def requirement_block(statement: str, root: str = "") -> str:
    """The statement's own requirements, as a list this run will come back to."""
    catalog = requirement_catalog(statement, root)
    ordered = requirement_order(catalog)
    items = ordered if REQUIREMENT_INDEX_FULL else ordered[:REQUIREMENT_MAX]
    if not items:
        return ""
    lines = []
    used = 0
    shown: set = set()
    for item in items:
        line = "  %s [%s] %s%s" % (item["id"], "change" if item["wants_edit"] else "keep",
                                   item["text"],
                                   " (from %s, named by the statement)" % item["document"]
                                   if item.get("document") else "")
        if REQUIREMENT_INDEX_FULL and lines and used + len(line) > REQUIREMENT_INDEX_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
        shown.add(item["id"])
    count = len(catalog)
    clipped = sum(row["text"].endswith(" [...]") for row in items if row["id"] in shown)
    omitted = [row["id"] for row in catalog if row["id"] not in shown]
    if omitted or clipped:
        lines.append("  %d clause(s) omitted%s; %d displayed clause(s) shortened. "
                     "Use read_requirement(id, offset) for complete text: one id such as R7, "
                     "a range such as %s, or 'all' (at most 20 items per call); R1..R%d exist."
                     % (len(omitted), " (%s)" % compact_ids(omitted) if omitted else "", clipped,
                        range_example(omitted, count), count))
    return ("\nWhat the statement requires, quoted from it. The tag is a reading, "
            "not a division: a clause can ask for a change and bound it at once, "
            "and a clause missing from this list is still a requirement. The "
            "statement above is the authority; this is an index into it. Anything "
            "you add to the list is your inference, not its requirement:\n"
            + "\n".join(lines))

OPENING_FILE = flag("RIDGES_OPENING_FILE")
OPENING_FILE_CHARS = 16_000
OPENING_FILE_MARGIN = 20


def opening_file(statement: str, tree: Tree) -> str:
    """The file the statement names, as it stands, numbered like read_file shows it.

    The model reads this file first in every run; handing it over with the
    problem saves that turn. A large file is cut to the named method with a
    margin, or to its head, and says so.
    """
    if not OPENING_FILE:
        return ""
    path = stated_file(statement, getattr(tree, "root", ""))
    if not path:
        return ""
    try:
        source = tree.read(path)
    except BaseException:
        return ""
    if not source.strip():
        return ""
    lines = source.splitlines()
    first, last, note = 1, len(lines), ""
    if len(source) > OPENING_FILE_CHARS:
        region = fenced_region(source, stated_methods(statement))
        if region:
            _, low, high = region
            first, last = max(1, low - OPENING_FILE_MARGIN), min(len(lines), high + OPENING_FILE_MARGIN)
            note = " (lines %d-%d of %d, around the method the statement names; read_file shows the rest)" % (
                first, last, len(lines))
        else:
            kept = source[:OPENING_FILE_CHARS].count("\n")
            first, last = 1, max(1, kept)
            note = " (the first %d of %d lines; read_file shows the rest)" % (last, len(lines))
    body = "\n".join("%6d\t%s" % (number, lines[number - 1]) for number in range(first, last + 1))
    return "\nThe file the statement names, %s, as it stands now%s:\n%s" % (path, note, body)


def opening_message(statement: str, tree: Tree, hints: list[str],
                    facts: list | None = None, file_block: str = "") -> str:
    """The first message the model sees: the task, the repository, the requirements and the facts
    read for free.
    """
    blocks = ["Problem to fix:\n\n" + statement.strip(), "\nRepository at a glance:\n" + repo_sketch(tree)]
    requirements = requirement_block(statement, getattr(tree, "root", ""))
    if requirements:
        blocks.append(requirements)
    if facts:
        blocks.append("\nRead out of this checkout before the run started, so you "
                      "do not have to spend turns on it. Facts, not instructions:\n"
                      + "\n".join("  " + line for line in facts))
    if file_block:
        blocks.append(file_block)
    if hints:
        blocks.append(
            "\nFiles whose contents overlap the rare terms in the problem, most overlap first. "
            "This is a starting point produced by text matching, not an answer:\n"
            + "\n".join("  " + p for p in hints)
        )
    return "\n".join(blocks)

CARRY_HEAD = "\nFindings this run recorded, carried forward"
NOTES_HEAD = "Earlier completed steps (abridged"
ASSISTANT_NOTE_HEAD = "Earlier assistant note (unverified; may be superseded by later source or observations):\n"

def shrink_transcript(messages: list[dict], cap: int, beacon: Beacon,
                      carry=None) -> bool:
    """Shrink the transcript to fit the cap.

    `carry` may be a callable: it is resolved only once shrinking is actually
    needed, so its git inspection does not run ahead of every request.

    Two things are never treated as history. The turn being answered: the model
    has not read those results yet, so clipping them hands it a stub for the
    file it just asked for. And what this function itself inserts: room for it
    is set aside first, because fitting the cap and then inserting is how the
    transcript ends up over the cap again and in the last-resort pass.
    """
    def size():
        # Tool arguments and message metadata also consume the model's context.
        return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))

    def place(wanted) -> int:
        for index, message in enumerate(messages):
            if message is wanted:
                return index
        return len(messages)

    total = size()
    if total <= cap:
        return False
    beacon.fired("transcript %d characters over cap %d" % (total, cap))
    if callable(carry):
        carry = carry()
    original = total
    # What an earlier shrink left is replaced, not stacked: the findings are
    # rebuilt from the live records, and the abridged history is folded into
    # the note written below.
    notes = []
    assistant_note = ""
    for message in list(messages[2:]):
        body = message.get("content")
        if message.get("role") != "user" or not isinstance(body, str):
            continue
        if body.startswith(CARRY_HEAD):
            messages.pop(place(message))
        elif body.startswith(NOTES_HEAD):
            notes.extend(line for line in body.split("\n")[1:] if line.strip())
            messages.pop(place(message))
        elif body.startswith(ASSISTANT_NOTE_HEAD):
            assistant_note = body[len(ASSISTANT_NOTE_HEAD):]
            messages.pop(place(message))
    notes_room = min(2000, max(128, cap // 10))
    assistant_room = min(600, max(100, cap // 15))
    reserve = (notes_room + 200 + assistant_room + len(ASSISTANT_NOTE_HEAD) + 40
               + (len(json.dumps(carry, ensure_ascii=False)) + 40 if carry else 0))
    goal = max(cap // 2, cap - reserve)
    answering = None
    for message in reversed(messages[2:]):
        if message.get("role") == "assistant":
            answering = message
            break
    # Retain short evidence excerpts instead of replacing results with blank notices.
    for message in messages[2:max(2, min(place(answering), len(messages) - 4))]:
        if size() <= goal:
            break
        if message.get("role") != "tool":
            continue
        body = str(message.get("content") or "")
        if len(body) > 1200:
            message["content"] = clip(body, 1200, "earlier tool output; reread if needed")
    # Remove old assistant/tool exchanges together: never leave orphan tool IDs.
    # Keep the system instructions, task, and the turn being answered intact here.
    while size() > goal and len(messages) > 3 and place(answering) > 2:
        start = 2
        message = messages[start]
        end = start + 1
        calls = message.get("tool_calls") or []
        if calls:
            wanted = {c.get("id") for c in calls}
            found = set()
            while end < len(messages) and messages[end].get("role") == "tool":
                found.add(messages[end].get("tool_call_id"))
                end += 1
            if not wanted.issubset(found):
                break
        if end >= len(messages) or end > place(answering):
            break
        for old in messages[start:end]:
            if old.get("role") == "assistant" and str(old.get("content") or "").strip():
                # Retain the model's latest removed working note verbatim and
                # explicitly as a claim, not a finding reconstructed from tools.
                assistant_note = str(old["content"]).strip()
            for call in old.get("tool_calls") or []:
                function = call.get("function") or {}
                notes.append("%s %s" % (function.get("name", "tool"),
                                       clip(str(function.get("arguments") or ""), 240)))
            if old.get("role") == "tool":
                body = str(old.get("content") or "")
                notes.append(clip(body, 360, "prior result"))
        del messages[start:end]
    carried_message = None
    if carry:
        # Its own message, and held by reference: the pass below clips content
        # from the middle, which would leave these findings as headless lines.
        carried_message = {"role": "user", "content": carry}
        messages.insert(2, carried_message)
    if notes:
        # A plain user note cannot break assistant/tool pairing.
        messages.insert(2, {"role": "user", "content":
            NOTES_HEAD + "; files remain available to reread):\n"
            + clip("\n".join(notes), notes_room, "earlier history")})
    assistant_message = None
    if assistant_note:
        assistant_message = {"role": "user", "content": ASSISTANT_NOTE_HEAD
                             + clip(assistant_note, assistant_room, "assistant note")}
        messages.insert(2, assistant_message)
    # A single large completed tool call can itself exceed the cap. Compress its
    # historical payload explicitly; retain valid JSON and the original call IDs.
    if size() > cap:
        limit = max(256, min(1200, cap // max(8, len(messages) * 2)))
        completed = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
        for message in messages[2:place(answering)]:
            if message is carried_message or message is assistant_message:
                continue
            if message.get("content"):
                message["content"] = clip(str(message["content"]), limit, "earlier content; reread if needed")
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments") or ""
                if call.get("id") in completed and len(arguments) > limit:
                    function["arguments"] = json.dumps({
                        "_history_compacted": True,
                        "argument_excerpt": clip(arguments, limit, "historical arguments")})
    if size() > cap and answering is not None:
        # Nothing older is left to give. The turn being answered is itself
        # larger than the cap, so its results share what room there is, and
        # say so, rather than the request being refused for its size.
        fresh = [m for m in messages[place(answering):] if m.get("role") == "tool"
                 and len(str(m.get("content") or "")) > 1200]
        if fresh:
            spare = cap - (size() - sum(len(str(m["content"])) for m in fresh))
            share = max(1200, spare // len(fresh) - 200)
            for message in fresh:
                message["content"] = clip(
                    str(message["content"]), share,
                    "this result, which is larger than the transcript allows; "
                    "read it again in smaller parts")
    total = size()
    if total > cap:
        beacon.skipped("protected task or pending exchange exceeds context cap")
    say("[" + Beacon.tag("trim") + "] freed %d characters, transcript now %d" % (original - total, total))
    return total < original

PLAN_BRIEF = """You are planning, not editing. You can read this repository but you cannot change it.

Another agent has been reading this task for several turns and has not managed to change
anything yet. Your job is to work out where the change belongs and say so.

You have a reading budget of %d characters of tool output; what is left of it is reported
back to you after every turn. A ranged read costs what it returns, so read narrowly: find
the file first, then the lines.

Call set_plan whenever your understanding improves. It overwrites the previous note, and the
note MAY BE READ AT ANY MOMENT -- so keep it worth reading from the first call onward rather
than saving it for the end. Call done_planning as soon as you have enough; you are not
required to spend the budget.

A good note names files and line ranges and says what has to become true. It does not
restate the task."""
PLAN_TOOL_NAMES = ("read_file", "search_text", "find_files", "outline")
PLAN_EXTRA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_plan",
            "description": "Record the current best plan, replacing any earlier one. Call this "
                           "as soon as you have something worth handing over, and again "
                           "whenever it improves.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Files, line ranges, and what has to become true."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done_planning",
            "description": "Stop planning and hand the note over. Call this as soon as the plan "
                           "is good enough; nothing is gained by spending the whole budget.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

def plan_tools() -> list[dict]:
    """The read-only tools the planning seat is offered."""
    return [s for s in TOOL_SCHEMAS
            if s["function"]["name"] in PLAN_TOOL_NAMES] + PLAN_EXTRA_TOOLS

def run_plan(statement: str, tree: Tree, pool: ShellPool, allowance: Allowance,
             beacon: Beacon, turn: int) -> str:
    """Ask a fresh seat for a short plan of attack before the work starts, and return it."""
    beacon.reached(turn, allowance.spent, allowance.clock_left())
    if not PLAN_SEAT:
        beacon.skipped("not switched on for this run")
        return ""
    spent_at_entry, calls_at_entry = allowance.spent, allowance.calls
    ceiling = spent_at_entry + allowance.soft_usd * PLAN_SPEND_SHARE
    kit = Kit(tree, pool, allowance, label="PLAN")
    seat = Seat(allowance, models=[PLAN_MODEL], patient=False)
    messages = [{"role": "system", "content": PLAN_BRIEF % PLAN_READ_BUDGET},
                {"role": "user", "content": statement}]
    note, read, stop, step = "", 0, "turns", 0
    used_call_ids: set = set()
    try:
        for step in range(1, PLAN_TURN_CAP + 1):
            if read >= PLAN_READ_BUDGET:
                stop = "budget"
                break
            if allowance.spent >= ceiling or allowance.money_left() <= 0:
                stop = "spend"
                break
            reply = seat.ask(messages, plan_tools())
            calls = usable_calls(reply.get("tool_calls"), used_call_ids)
            entry = {"role": "assistant", "content": str(reply.get("content") or "")}
            if calls:
                entry["tool_calls"] = recorded_calls(calls)
            messages.append(entry)
            if not calls:
                stop = "silent"
                break
            finished = False
            for index, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    function = {}
                name = str(function.get("name") or "")
                try:
                    args = json.loads(function.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments were not an object")
                except Exception as error:
                    result = "could not read the arguments: %s" % error
                else:
                    if name == "done_planning":
                        finished = True
                        result = "planning ended"
                    elif name == "set_plan":
                        note = str(args.get("text") or "")[:PLAN_NOTE_CHARS]
                        result = "plan recorded, %d characters" % len(note)
                    else:
                        try:
                            result = kit.run(name, args)
                        except (ToolFault, Finished) as fault:
                            result = "error: %s" % fault
                        except Exception as error:
                            result = "error: %s: %s" % (type(error).__name__, error)
                served = clip(str(result), READ_OUTPUT_CAP)
                read += len(served)
                messages.append({"role": "tool", "tool_call_id": call_ident(call, index),
                                 "content": served})
            if finished:
                stop = "done"
                break
            messages.append({"role": "user", "content": "Reading budget: %d used, %d left."
                             % (read, max(0, PLAN_READ_BUDGET - read))})
    except Exception as error:
        stop = "error"
        say("[" + Beacon.tag("plan") + "] gave up: %s: %s" % (type(error).__name__, str(error)[:200]))
    beacon.calls = allowance.calls - calls_at_entry
    beacon.usd = allowance.spent - spent_at_entry
    beacon.fired("stopped=%s steps=%d read=%dc note=%dB" % (stop, step, read, len(note)))
    empty = beacon.artefact("before", "")
    beacon.outcome(empty, note.strip())
    if note.strip():
        say("[" + Beacon.tag("plan") + "] note %s" % note.strip().replace("\n", " | ")[:PLAN_NOTE_CHARS])
    beacon.bill()
    return note.strip()

# A submit decides whether the run ends, so it runs after the reply's other
# calls, and only the first one in a reply is evaluated: a second was written
# before the model could read what the first returns.
SUBMIT_LAST = flag("RIDGES_SUBMIT_LAST")
SUBMIT_HELD = ("Not run: this reply already calls submit once. Read what that call "
               "returned, then call submit again if the answer is ready.")

def call_order(calls: list) -> tuple[list, set]:
    """The order a reply's calls run in, and the indexes of submits that do not run."""
    if not SUBMIT_LAST:
        return list(range(len(calls))), set()

    def name(call) -> str:
        function = call.get("function") if isinstance(call, dict) else None
        return str(function.get("name") or "") if isinstance(function, dict) else ""
    submits = [index for index, call in enumerate(calls) if name(call) == "submit"]
    others = [index for index, call in enumerate(calls) if name(call) != "submit"]
    return others + submits, set(submits[1:])

def drive(statement: str, tree: Tree, pool: ShellPool, allowance: Allowance,
          findings: "FindingMap | None" = None, seed_salt: int = 0) -> None:
    """Work the problem: the loop that reads, edits, checks and hands in.

    One seat drives with the tools; the warden holds the answer to the statement; the
    beacons record every decision. The loop ends when the answer is handed in, when the
    turns run out, or when the allowance does, and the second reading is closed out whatever
    ended it. A seed salt lets a later derivation of the same problem start from a different
    sampling state.
    """
    seat = Seat(allowance)
    seat.seed = (request_seed(statement) + seed_salt) % (2 ** 31)
    warden = Warden(tree, pool, allowance, statement)
    warden.arm()
    kit = Kit(tree, pool, allowance, warden, findings=findings)
    kit.driver_seat = seat
    locate = Beacon("prelocate")
    trim = Beacon("trim")
    sweep = Beacon("sweep")
    plan = Beacon("plan")
    locate.reached(0, allowance.spent, allowance.clock_left())
    if PRELOCATE:
        hints = candidate_files(tree, statement, locate)
    else:
        locate.skipped("not switched on for this run")
        hints = []
    sweep.reached(0, allowance.spent, allowance.clock_left())
    if SWEEP_WORKFLOW:
        sweep.fired("clause present")
    else:
        sweep.skipped("not switched on for this run")
    kit.conform.reached(0, allowance.spent, allowance.clock_left())
    if not SUBMIT_CONFORM:
        kit.conform.skipped("not switched on for this run")
    kit.selfreview.reached(0, allowance.spent, allowance.clock_left())
    if not HIDDEN_SELFREVIEW:
        kit.selfreview.skipped("not switched on for this run")
    if findings is not None:
        findings.beacon.reached(0, allowance.spent, allowance.clock_left())
    else:
        Beacon("findings").skipped("not switched on for this run")
    facts_beacon = Beacon("dbfacts")
    facts_beacon.reached(0, allowance.spent, allowance.clock_left())
    facts: list = []
    if DB_FACTS:
        try:
            facts = db_facts(statement, tree, DB_FACTS_SEC)
            facts_beacon.fired("%d fact(s) read without a model call" % len(facts))
        except BaseException as error:
            facts_beacon.skipped("could not be read: %s" % type(error).__name__)
    else:
        facts_beacon.skipped("not switched on for this run")
    if DB_FACTS:
        try:
            probe = database_probe(kit, statement)
            if probe:
                facts.append(probe)
        except BaseException as error:
            facts_beacon.skipped("the database probe failed: %s" % type(error).__name__)
    try:
        file_block = opening_file(statement, tree)
    except BaseException:
        file_block = ""
    # The opening message carries the task, the repository and whatever facts
    # were read without a model call. Nothing scans the declared file for
    # shapes a defect is said to take: that list was built before this run had
    # read the task, and pasting it here started the model on places this run
    # had picked rather than on the task in front of it.
    opening = opening_message(statement, tree, hints, facts, file_block)
    messages: list[dict] = [
        {"role": "system", "content": compose_brief(statement, tree.root)},
        {"role": "user", "content": opening},
    ]
    cap = transcript_cap_chars(seat.current(), allowance.ceiling_usd)
    capped_for = seat.current()
    say("[LOOP] transcript cap set for %s" % seat.current())
    blanks = 0
    used_call_ids: set = set()
    ending = "the loop reached its last turn"
    try:
        pressed = 0
        observations: set[str] = set()
        evidence_revision = 0
        pressed_revision = -1
        previous_progress = True
        stalled_turns = 0
        wrapped_up = False
        for turn in range(1, TURN_CEILING + 1):
            if pool.jobs:
                kit.sync_workspace()
            warden.collect()
            halt = allowance.halt_reason()
            if halt:
                say("[LOOP] stopping on %s at turn %d" % (halt, turn))
                ending = "the run stopped on %s" % halt
                return
            stalled_turns = 0 if previous_progress else stalled_turns + 1
            before_edit = allowance.edits == 0
            if (turn > FIRST_EDIT_DEADLINE_TURN
                    and pressed < EDIT_PRESSES_MAX and not previous_progress
                    and pressed_revision != evidence_revision
                    and (before_edit or stalled_turns >= 3)
                    and not any(job.process.poll() is None for job in pool.jobs.values())):
                pressed += 1
                pressed_revision = evidence_revision
                if before_edit:
                    say("[LOOP] %d turns without an edit; pressing for one (#%d)"
                        % (turn - 1, pressed))
                    press = (
                        "No edit yet. State the best-supported cause and the smallest "
                        "change that would address it. If one fact is still missing, "
                        "run a targeted check to settle it, then implement the fix. "
                        "Keep enough budget for tests and final diff review."
                    )
                else:
                    say("[LOOP] %d turns without a new observation; recovery #%d"
                        % (stalled_turns, pressed))
                    press = (
                        "Investigation stalled. Recent calls added no new observation. "
                        "Use the evidence already collected to identify the remaining "
                        "uncertainty and choose a check that can distinguish its causes, "
                        "or make a change supported by that evidence. An edit is not "
                        "required by this reminder. If the requested work is complete, "
                        "review the diff and submit. Keep the remaining budget for "
                        "relevant verification; do not repeat an unchanged check without "
                        "a reason from the current task."
                    )
                if before_edit and pressed == 1:
                    note = run_plan(statement, tree, pool, allowance, plan, turn)
                    if note:
                        press += (
                            "\n\nA second agent read the repository and left this note. "
                            "It did not run anything and may be wrong -- check it against "
                            "the file before you act on it.\n\n" + note
                        )
                messages.append({"role": "user", "content": press})
            if seat.current() != capped_for:
                # The seat changed since the cap was set: fit the transcript to
                # the seat that is about to read it, not the one that left.
                cap = transcript_cap_chars(seat.current(), allowance.ceiling_usd)
                capped_for = seat.current()
                say("[LOOP] seat changed under us; transcript cap set for %s"
                    % seat.current())
            if TRANSCRIPT_CAP:
                if shrink_transcript(messages, cap, trim, kit.carried_note):
                    kit.seen.clear()
            reply = seat.ask(messages, TOOL_SCHEMAS)
            calls = usable_calls(reply.get("tool_calls"), used_call_ids)
            text = str(reply.get("content") or "")
            if calls and not PARALLEL_TOOLS:
                dropped = len(calls) - 1
                calls = calls[:1]
                if dropped:
                    say("[" + Beacon.tag("batch") + "] not switched on for this run, dropped %d call(s)" % dropped)
            entry = {"role": "assistant", "content": text if calls else (text or "")}
            if calls:
                entry["tool_calls"] = recorded_calls(calls)
            messages.append(entry)
            if not calls:
                previous_progress = False
                blanks += 1
                # Three replies in a row with no tool call, with or without
                # text, mean this seat is not working the problem: another
                # seat gets the transcript, and the cap is refitted before it
                # is asked.
                if blanks >= BLANK_REPLY_CEILING and seat.retire(seat.current()):
                    say("[LOOP] %d replies without a tool call; changed seats" % blanks)
                    blanks = 0
                    continue
                if not text.strip() and blanks >= BLANK_REPLY_CEILING:
                    if empty_answer(tree, statement, allowance):
                        # Ending here would hand in an empty patch.
                        say("[LOOP] %d blank replies and no seat left; the answer "
                            "would be an empty patch, so the run goes on" % blanks)
                        blanks = 0
                    else:
                        say("[LOOP] %d blank replies and no seat left" % blanks)
                        ending = "no seat left to answer"
                        return
                messages.append(
                    {
                        "role": "user",
                        "content": "That reply carried no tool call. Take the next concrete step, "
                        "or call submit if the change is complete.",
                    }
                )
                continue
            blanks = 0
            say("[LOOP] turn %d: %d tool call(s)" % (turn, len(calls)))
            previous_progress = False
            order, held = call_order(calls)
            answers: dict = {}
            for index in order:
                call = calls[index]
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    function = {}
                name = str(function.get("name") or "")
                observed = False
                if index in held:
                    answers[index] = SUBMIT_HELD
                    continue
                try:
                    args = json.loads(function.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments were not an object")
                except Exception as error:
                    result = "could not read the arguments: %s" % error
                else:
                    try:
                        result = kit.run(name, args)
                        observed = True
                    except Finished as done:
                        say("[LOOP] submit at turn %d: %s" % (turn, str(done)[:200]))
                        ending = "the answer was handed in"
                        return
                    except ToolFault as fault:
                        result = "error: %s" % fault
                    except Exception as error:
                        result = "error: %s: %s" % (type(error).__name__, error)
                if observed and str(result).strip():
                    # New observations, including returned test failures, can
                    # justify further investigation. Repeating the same call
                    # and result does not justify repeated edit instructions.
                    identity = kit.progress_observation
                    if identity is None:
                        identity = hashlib.sha256(json.dumps(
                            [name, args, str(result)], sort_keys=True,
                            ensure_ascii=True).encode()).hexdigest()
                    if identity and identity not in observations:
                        observations.add(identity)
                        evidence_revision += 1
                        previous_progress = True
                answers[index] = clip(str(result), READ_OUTPUT_CAP)
            # Every call is answered, in the order the reply made them.
            for index, call in enumerate(calls):
                if index in answers:
                    messages.append({"role": "tool", "tool_call_id": call_ident(call, index),
                                     "content": answers[index]})
            if (not wrapped_up
                    and (turn >= TURN_CEILING - 10
                         or allowance.clock_left() < max(SELFREVIEW_MIN_WALL_SEC * 2,
                            (allowance.deadline - allowance.started) * 0.20)
                         or allowance.money_left() < allowance.soft_usd * 0.15)):
                wrapped_up = True
                messages.append(
                    {
                        "role": "user",
                        "content": "Reserve the remaining budget for verification. Finish the "
                        "current change, run the task's relevant tests, inspect database "
                        "behavior where required, review the diff, and call submit.",
                    }
                )
        say("[LOOP] hit the turn ceiling")
    except BaseException as error:
        ending = "the run ended on %s" % type(error).__name__
        raise
    finally:
        kit.selfreview_close(ending)

def apply_patch_text(root: str, patch: str, budget: float = 30.0) -> tuple:
    """Apply a patch to the tree at root with git; (exit code, git's words)."""
    handle, path = tempfile.mkstemp(prefix="ridges-pick-", suffix=".diff")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", errors="surrogateescape") as fh:
            fh.write(patch)
        return git(["apply", "--whitespace=nowarn", path], root, budget)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

def clean_tree(tree: Tree, budget: float = 60.0) -> bool:
    """Put the tree back to its base and remove the files this run created."""
    try:
        for path in sorted(CREATED):
            target = tree.absolute(path)
            if os.path.isfile(target):
                os.unlink(target)
    except (OSError, ToolFault) as error:
        say("[SECOND] a created file could not be removed: %s" % type(error).__name__)
    CREATED.clear()
    return tree.restore(budget)

def capture_derivation(tree: Tree, allowance: Allowance, label: str) -> dict | None:
    """The answer as it stands, with the checks and expectations that vouched for it."""
    incomplete: list = []
    try:
        patch = tree.diff(SELFREVIEW_DIFF_SEC, failed=incomplete)
    except BaseException:
        return None
    if incomplete or not patch.strip():
        return None
    checks: list = []
    for record in CHECKS:
        command = " ".join(str(record.get("command") or "").split())
        if record.get("kind") == "named regression check" and command and command not in checks:
            checks.append(command)
    expectations = [dict(r["args"]) for r in EXPECTATIONS if r.get("args")]
    return {"label": label, "patch": patch, "checks": checks, "expectations": expectations,
            "spent": allowance.spent, "created": sorted(CREATED)}

def derivation_standing(statement: str, tree: Tree, pool: ShellPool, allowance: Allowance,
                     candidate: dict, checks: list, expectations: list, beacon: Beacon,
                     finish_by: float) -> tuple:
    """(checks failed, expectations unmet): lower is better; ties keep the first."""
    worst = (99, 99)
    if not clean_tree(tree, 30.0):
        return worst
    code, out = apply_patch_text(tree.root, candidate["patch"], 30.0)
    if code != 0:
        beacon.fired("%s did not apply: %s" % (candidate["label"], out.strip()[:80]))
        return worst
    warden = Warden(tree, pool, allowance, statement)
    kit = Kit(tree, pool, allowance, warden)
    failed = unmet = 0
    for command in checks[:PICK_CHECKS_MAX]:
        room = finish_by - time.monotonic()
        if room < 10.0:
            break
        try:
            said = kit.do_bash({"command": command, "timeout": min(PICK_CHECK_SEC, room)})
        except (ToolFault, Spent):
            failed += 1
            continue
        status = re.search(r"\[exit_code=(\d+)\]", said or "")
        if not status or status.group(1) != "0" or PROBE_NOT_RUN.search(said or ""):
            failed += 1
    for args in expectations[:PICK_EXPECTATIONS_MAX]:
        if finish_by - time.monotonic() < 10.0:
            break
        try:
            said = kit.do_expect(dict(args))
        except (ToolFault, Spent):
            unmet += 1
            continue
        if not str(said).startswith("expectation met"):
            unmet += 1
    return (failed, unmet)

def second_derivation(statement: str, tree: Tree, pool: ShellPool, allowance: Allowance) -> None:
    """Work the problem again from a clean tree, then leave the better answer in the tree."""
    beacon = Beacon("second")
    beacon.reached(allowance.calls, allowance.spent, allowance.clock_left())
    first = capture_derivation(tree, allowance, "first")
    if first is None:
        if allowance.clock_left() < SECOND_MIN_SEC or allowance.money_left() < SECOND_MIN_USD:
            beacon.skipped("no first answer, and too little of the run left to try again")
            return
        beacon.fired("no first answer after $%.3f and %d calls; working the problem again from a clean tree"
                     % (allowance.spent, allowance.calls))
        try:
            if clean_tree(tree, 60.0):
                reset_checks()
                drive(statement, tree, pool, allowance, None, seed_salt=1)
        except Spent as stop:
            say("[SECOND] out of allowance: %s" % stop)
        except BaseException as error:
            say("[SECOND] ended on %s: %s" % (type(error).__name__, str(error)[:120]))
        return
    if not SECOND_DERIVATION:
        beacon.skipped("not switched on for this run")
        return
    if allowance.clock_left() < SECOND_MIN_SEC:
        beacon.skipped("too little of the run left: %.0fs" % allowance.clock_left())
        return
    if allowance.money_left() < SECOND_MIN_USD or allowance.spent > SECOND_MAX_SPENT_USD:
        beacon.skipped("too little money left, or the first answer already cost $%.3f" % allowance.spent)
        return
    beacon.fired("working the problem again from a clean tree after $%.3f and %d calls"
                 % (allowance.spent, allowance.calls))
    try:
        winner = compare_derivations(statement, tree, pool, allowance, first, beacon)
    except BaseException as error:
        # Whatever went wrong in the comparison, the first answer was already
        # in: it goes back into the tree and out.
        say("[SECOND] the comparison ended on %s: %s; the first answer stands"
            % (type(error).__name__, str(error)[:120]))
        winner = first
    reset_checks()
    clean_tree(tree, 60.0)
    code, out = apply_patch_text(tree.root, winner["patch"], 30.0)
    if code != 0:
        say("[SECOND] the %s answer did not apply back (%s); trying the first" % (winner["label"], out.strip()[:80]))
        if winner is not first:
            clean_tree(tree, 30.0)
            apply_patch_text(tree.root, first["patch"], 30.0)

def compare_derivations(statement: str, tree: Tree, pool: ShellPool, allowance: Allowance,
                        first: dict, beacon: Beacon) -> dict:
    """Work the problem again and return the answer that goes out; the first on any doubt."""
    if not clean_tree(tree, 60.0):
        beacon.skipped("the tree could not be put back")
        return first
    reset_checks()
    try:
        drive(statement, tree, pool, allowance, None, seed_salt=1)
    except Spent as stop:
        say("[SECOND] out of allowance: %s" % stop)
    except BaseException as error:
        say("[SECOND] ended on %s: %s" % (type(error).__name__, str(error)[:120]))
    second = capture_derivation(tree, allowance, "second")
    room = allowance.clock_left() - FINISH_BUDGET_SEC
    finish_by = time.monotonic() + max(30.0, min(PICK_SEC, room))
    if second is None or second["patch"] == first["patch"]:
        beacon.fired("the second answer is %s; the first stands" % ("missing" if second is None else "identical"))
        winner = first
    elif room < PICK_MIN_SEC:
        beacon.fired("no room left to compare the two answers (%.0fs); the first stands" % room)
        winner = first
    else:
        checks: list = []
        for raw in first["checks"] + second["checks"] + [stated_runner(statement, tree.root)]:
            command = " ".join((raw or "").split())
            if command and command not in checks:
                checks.append(command)
        seen: set = set()
        expectations: list = []
        for args in first["expectations"] + second["expectations"]:
            key = json.dumps(args, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                expectations.append(args)
        standings = []
        for candidate in (first, second):
            standing = derivation_standing(statement, tree, pool, allowance, candidate, checks, expectations,
                                     beacon, finish_by)
            standings.append(standing)
            say("[SECOND] %s: checks failed %d, expectations unmet %d"
                % (candidate["label"], *standing))
        winner = second if standings[1] < standings[0] else first
        if (THIRD_DERIVATION and standings[0] == standings[1]
                and allowance.clock_left() >= SECOND_MIN_SEC and allowance.money_left() >= SECOND_MIN_USD
                and allowance.spent <= THIRD_MAX_SPENT_USD):
            beacon.fired("the two answers differ but tie on every check; working the problem a third time")
            third = None
            if clean_tree(tree, 60.0):
                reset_checks()
                try:
                    drive(statement, tree, pool, allowance, None, seed_salt=2)
                except Spent as stop:
                    say("[SECOND] third derivation out of allowance: %s" % stop)
                except BaseException as error:
                    say("[SECOND] third derivation ended on %s: %s" % (type(error).__name__, str(error)[:120]))
                third = capture_derivation(tree, allowance, "third")
            if third is None:
                beacon.fired("no third answer; the pick between two stands")
            elif third["patch"] == first["patch"]:
                winner = first
                beacon.fired("the third answer agrees with the first")
            elif third["patch"] == second["patch"]:
                winner = second
                beacon.fired("the third answer agrees with the second")
            elif allowance.clock_left() - FINISH_BUDGET_SEC < PICK_MIN_SEC:
                beacon.fired("no room left to compare the third answer; the pick between two stands")
            else:
                finish_by = time.monotonic() + max(30.0, min(PICK_SEC, allowance.clock_left() - FINISH_BUDGET_SEC))
                standing = derivation_standing(statement, tree, pool, allowance, third, checks, expectations,
                                               beacon, finish_by)
                say("[SECOND] third: checks failed %d, expectations unmet %d" % standing)
                ranked = sorted(zip([*standings, standing], (first, second, third), (0, 1, 2), strict=False),
                                key=lambda item: (item[0], item[2]))
                winner = ranked[0][1]
        beacon.fired("picked the %s answer" % winner["label"])
    return winner

STATEMENT_ECHO_LINES = 200
STATEMENT_ECHO_WIDTH = 400

def echo_statement(statement: str) -> None:
    """Write the task statement into the run log, so a run can be read without the task to hand.
    """
    lines = statement.splitlines()
    say("[TASK] %d line(s), %d char(s)" % (len(lines), len(statement)))
    for line in lines[:STATEMENT_ECHO_LINES]:
        say("[TASK] | %s" % line[:STATEMENT_ECHO_WIDTH])
    if len(lines) > STATEMENT_ECHO_LINES:
        say("[TASK] | ... %d more line(s)" % (len(lines) - STATEMENT_ECHO_LINES))

def agent_main(input: dict) -> str:
    """Solve the problem in the checkout and return the answer as a patch.

    The entry point: it sets up the allowance, the checkout and the shell pool, drives the
    work, then builds the answer the way it will be read. The tree is put back to the base
    commit before the patch is finished, by-products and anything outside the task's
    boundary are dropped, and a patch that does not apply is salvaged section by section.
    Whatever goes wrong, a string comes back rather than an exception.
    """
    pool = None
    tree = None
    try:
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        reset_checks()
        allowance = Allowance()
        root = os.getcwd()
        tree = Tree(root)
        pool = ShellPool(root)
        statement = str((input or {}).get("problem_statement") or "").strip()
        findings = FindingMap(root) if FINDING_MAP else None
        say("[RUN] budget=$%.3f clock=%.0fs build=v58 driver=%s relief=%s last=%s review=%s plan=%s "
            "seed=%s effort=%s flags=review:%d closer:%d expect:%d green:%d parallel:%d plan:%d "
            "second:%d third:%d required:%d"
            % (allowance.ceiling_usd, allowance.clock_left(), DRIVER_MODEL, RELIEF_MODEL, LAST_RESORT_MODEL,
               REVIEW_MODEL, PLAN_MODEL, request_seed(statement), REASONING_EFFORT, int(REVIEW_SEAT),
               int(CLOSER_SEAT), int(EXPECT_TOOL), int(GREEN_MEMORY), int(PARALLEL_TOOLS), int(PLAN_SEAT),
               int(SECOND_DERIVATION), int(THIRD_DERIVATION), int(EXPECT_REQUIRED)))
    except BaseException as error:
        if pool is not None:
            pool.close()
        if tree is not None:
            try:
                tree.close()
            except BaseException:
                pass
        try:
            say("[RUN] startup failed: %s" % type(error).__name__)
        except BaseException:
            pass
        return ""
    try:
        echo_statement(statement)
    except BaseException:
        pass
    try:
        drive(statement, tree, pool, allowance, findings)
    except Spent as stop:
        say("[RUN] out of allowance: %s" % stop)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        say("[RUN] crashed: %s: %s" % (type(error).__name__, error))
    try:
        second_derivation(statement, tree, pool, allowance)
    except BaseException as error:
        say("[SECOND] skipped: %s: %s" % (type(error).__name__, str(error)[:120]))
    finish_by = time.monotonic() + max(
        0.0, allowance.clock_left() + FINISH_BUDGET_SEC)

    def finish_room(cap: float) -> float:
        room = min(cap, finish_by - time.monotonic())
        if room < 5.0:
            raise TimeoutError("finish budget exhausted")
        return room
    pool.close()
    patch = ""
    try:
        patch = tree.diff(max(5.0, min(60.0, finish_by - time.monotonic())))
    except BaseException:
        import traceback
        traceback.print_exc()
    handed_digest = patch_digest(patch)
    # The identity of the source as handed over, taken before the tree is put
    # back, so it can be compared with what the checks actually ran against.
    try:
        handed_identity = tree.source_identity(SELFREVIEW_DIFF_SEC, sorted(CREATED))
    except BaseException:
        handed_identity = ""
    vouched = False
    try:
        vouched = tree.restore(finish_room(60.0))
    except BaseException:
        pass
    if patch.strip():
        try:
            patch = byproduct_trim(patch, Beacon("byproduct"), statement, root)
        except BaseException as error:
            say("[%s] skipped: %s" % (Beacon.tag("byproduct"), type(error).__name__))
    if PATCH_ENVELOPE and patch.strip():
        try:
            patch = envelope_or_whole(patch, Beacon("envelope"), statement, root)
        except BaseException as error:
            say("[%s] skipped: the envelope could not be read: %s"
                % (Beacon.tag("envelope"), type(error).__name__))
    usable = None
    if vouched:
        try:
            usable = tree.applies(patch, finish_room(30.0))
        except BaseException:
            pass
    else:
        say("[PATCH] the tree was not confirmed put back; the answer goes "
            "out whole and unchecked")
    if usable is False and patch.strip():
        try:
            rescued = tree.salvage(patch, finish_room(45.0))
        except BaseException as error:
            say("[PATCH] salvage failed: %s" % type(error).__name__)
            rescued = ""
        if rescued:
            patch, usable = rescued, True
    if findings is not None:
        try:
            findings.report(patch)
        except BaseException:
            findings.beacon.skipped("the record could not be worked out")
    try:
        say("[SHAPE] " + patch_shape(patch))
    except BaseException:
        pass
    try:
        usable_checks = current_usable_records(CHECKS, handed_identity)
        failed_checks = [record for record in current_records(CHECKS, handed_identity)
                         if record.get("outcome") == "failed"
                         and completed_record_time(record) is not None]
        for record in CHECKS:
            say("[VERIFIED] ran %s :: %s :: %s :: %s :: env %s :: cwd %s"
                % (record.get("kind"), record.get("command"),
                   record.get("status"), record.get("detail"),
                   record.get("env"), record.get("cwd")))
        if not usable_checks:
            say("[VERIFIED] no unsuperseded clean check verifies the current "
                "answer (%d command(s) recorded, %d current failure(s))"
                % (len(CHECKS), len(failed_checks)))
            historical_passes = [record for record in CHECKS if usable_record(record)]
            # Explain actual source drift only. An old pass superseded by a
            # same-source failure is a failed recheck, not a different source.
            if (handed_identity and not failed_checks and historical_passes
                    and all(record.get("identity") != handed_identity
                            for record in historical_passes)):
                previous = [record for record in historical_passes
                            if record in current_usable_records(CHECKS, record["identity"])]
                if previous:
                    best = previous[-1]
                    say("[VERIFIED] the answer handed in (%s) is NOT the one that "
                        "passed (%s via %s); its source changed after that check"
                        % (handed_digest, best["digest"], best["command"]))
        else:
            best = usable_checks[-1]
            if failed_checks:
                say("[VERIFIED] %d current-source clean check(s) and %d current "
                    "failure(s); these command-specific observations do not "
                    "establish an overall pass"
                    % (len(usable_checks), len(failed_checks)))
            else:
                say("[VERIFIED] current-source check passed (%s, %s: %s); "
                    "this verifies only that invocation"
                    % (best["digest"], best["kind"], best["detail"]))
            if patch_digest(patch) != handed_digest:
                say("[VERIFIED] the text handed in was trimmed or salvaged after "
                    "the check, so it is not byte-for-byte what ran")
            # A fail-then-pass pair is about one command, in one place, in
            # that order, on the source being handed in. Matching command names
            # across the whole run says none of those things, and paired with
            # the line above it produced a log that called the same answer both
            # superseded and proven.
            moved = observed_change_records(CHECKS, handed_identity)
            if moved and not failed_checks and patch_digest(patch) == handed_digest:
                say("[VERIFIED] %s failed earlier in this run and passes on the "
                    "source handed in (%s), same invocation, environment and directory; "
                    "that is an observed change for those commands in this "
                    "environment" % (", ".join(sorted(set(moved))[:3]), handed_digest))
            elif all(record["kind"] in ("named regression check", "static check")
                     for record in usable_checks):
                say("[VERIFIED] the current successful observations are named "
                    "regression or static checks; no observed fail-to-pass "
                    "change was established for these invocations")
    except BaseException:
        pass
    try:
        say("[FOLD] " + fold_report(patch))
    except BaseException:
        pass
    try:
        say("[" + Beacon.tag("retry") + "] ladders=%d retried=%d recovered=%d old_would_bench=%d "
            "old_would_exhaust=%d walled=%d quiet=%d unreachable=%d unusable=%d absent=%d "
            "limited=%d"
            % (RETRY_TALLY["ladders"], RETRY_TALLY["retried"],
               RETRY_TALLY["recovered"], RETRY_TALLY["old_would_bench"],
               RETRY_TALLY["old_would_exhaust"], RETRY_TALLY["walled"],
               RETRY_TALLY["quiet"], RETRY_TALLY["unreachable"],
               RETRY_TALLY["unusable"], RETRY_TALLY["absent"],
               RETRY_TALLY["limited"]))
    except BaseException:
        pass
    say(
        "[RUN] done in %.0fs, $%.4f over %d calls, %d edits, patch %dB, usable=%s"
        % (allowance.elapsed(), allowance.spent, allowance.calls, allowance.edits,
           len(patch), {True: "yes", False: "no"}.get(usable, "unknown"))
    )
    try:
        tree.close()
    except BaseException as error:
        say("[TREE] private preservation snapshot cleanup unconfirmed: %s" % type(error).__name__)
    return patch
