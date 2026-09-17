from __future__ import annotations
import ast
import collections
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
FALLBACK_BASE_URL = "https://openrouter.ai/api/v1"
SYNTAX_CHECKS = {
    ".py": [sys.executable, "-c",
            "import sys; compile(open(sys.argv[1], 'rb').read(), sys.argv[1], 'exec')"],
    ".js": ["node", "--check"],
    ".mjs": ["node", "--check"],
    ".cjs": ["node", "--check"],
    ".ts": ["node", "--check"],
    ".rb": ["ruby", "-c"],
    ".php": ["php", "-l"],
    ".pl": ["perl", "-c"],
    ".lua": ["luac", "-p"],
    ".sh": ["bash", "-n"],
    ".bash": ["bash", "-n"],
    ".go": ["gofmt", "-e"],
}
DRIVER_MODEL = os.getenv("RIDGES_AGENT_MODEL", "openai/gpt-5.6-luna")
RELIEF_MODEL = os.getenv("RIDGES_RELIEF_MODEL", "")
PLAN_MODEL = os.getenv("RIDGES_PLAN_MODEL", "openai/gpt-5.6-luna")
SEAT_CACHE_TERMS = {
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
    "@preset/luna-high": (0.200e-6, 1.200e-6),
    "openai/gpt-5.6-terra": (2.000e-6, 12.000e-6),
    "google/gemini-3.7-flash": (0.375e-6, 1.875e-6),
    "deepseek/deepseek-v4-flash-0731": (0.440e-6, 1.320e-6),
    "tencent/hy3": (0.132e-6, 0.528e-6),
}
UNKNOWN_TOKEN_PRICE = (1.0e-6, 4.0e-6)
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
READ_OUTPUT_CAP = 24_000
SEARCH_HEAD_LIMIT = 250
SHELL_OUTPUT_CAP = 8_000
SEARCH_OUTPUT_CAP = 8_000
TEMPERATURE = 0.0

def one_line(value: object) -> str:
    return " ".join(str(value or "").split())

# This function creates a short, stable 8-character fingerprint for a message
# Convert the important part of a message into text -> SHA-256 hash it -> keep the first 8 characters
def reply_fingerprint(message: dict) -> str:
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

# This function checks whether value is a real, finite Python number represented as an int or float
def finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf"))

# This function converts a valid finite number into an integer. If the value is not a valid finite number, it returns -1
def whole_number(value: object) -> int:
    return int(value) if finite_number(value) else -1

COUNT_CEILING = 100_000_000

# This function returns big value between COUNT_CEILING and whole_number(value)
def counted(value: object) -> int:
    return min(COUNT_CEILING, max(0, whole_number(value)))

# This function extracts the number of reasoning tokens from a usage dictionary and returns it as an integer
def reasoning_tokens(usage: dict) -> int:
    details = (usage or {}).get("completion_tokens_details")
    return whole_number(details.get("reasoning_tokens")
                        if isinstance(details, dict) else None)

# This function splits prompt tokens into: non-cached prompt tokens and cached prompt tokens. It returns them as a tuple
def prompt_split(usage: dict) -> tuple:
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

# This function find how many prompt-cache tokens were written/created in an API usage object
# The important part is CACHE_WRITE_FIELDS. 
def cache_written(usage: dict) -> int:
    usage = usage or {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    for field in CACHE_WRITE_FIELDS:
        for holder in (usage, details):
            if holder.get(field) is not None:
                return whole_number(holder.get(field))
    return -1

# This function reads an environment variable and inteprets it as a boolean flag
def flag(name: str, default: str = "1") -> bool:
    return (os.getenv(name) or default).strip().lower() not in ("0", "no", "off", "false", "")

PARALLEL_TOOLS = flag("RIDGES_PARALLEL_TOOLS")
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
HIDDEN_SELFREVIEW = flag("RIDGES_HIDDEN_SELFREVIEW", "1")
WORK_METER = flag("RIDGES_WORK_METER", "1")
SELFREVIEW_CONSULT = flag("RIDGES_SELFREVIEW_CONSULT", "0")
STATED_BASELINE = flag("RIDGES_STATED_BASELINE", "0")
LEDGER_READBACK = flag("RIDGES_LEDGER_READBACK", "0")
TELEMETRY_SLOTS = flag("RIDGES_TELEMETRY_SLOTS")
RIDGES_SCOPE_FOLLOWS_STATEMENT = flag("RIDGES_SCOPE_FOLLOWS_STATEMENT", "1")
SCAN_ORDER = flag("RIDGES_SCAN_ORDER")
SCAN_ABSENT = flag("RIDGES_SCAN_ABSENT", "0")
DEAD_CONJUNCT = flag("RIDGES_DEAD_CONJUNCT")
SCAN_LEDGER = flag("RIDGES_SCAN_LEDGER", "0")
UNTOUCHED_ROWS = flag("RIDGES_UNTOUCHED_ROWS", "0")
PACK_VENV = flag("RIDGES_PACK_VENV", "0")
TAINTED_BASELINE = flag("RIDGES_TAINTED_BASELINE")
FOLD_ANCHORS = flag("RIDGES_FOLD_ANCHORS", "0")
MOVE_VERBATIM = flag("RIDGES_MOVE_VERBATIM")
SUBMIT_CONFORM = flag("RIDGES_SUBMIT_CONFORM", "0")
PLAN_SEAT = flag("RIDGES_PLAN_SEAT", "0")
SUITE_SCOPE = flag("RIDGES_SUITE_SCOPE")
SUITE_IMPORTLIB = flag("RIDGES_SUITE_IMPORTLIB", "0")
SUITE_SHIM = flag("RIDGES_SUITE_SHIM")
SUITE_READABLE = flag("RIDGES_SUITE_READABLE", "0")
NETWORK_FENCE = flag("RIDGES_NETWORK_FENCE")
SEARCH_LIMIT = flag("RIDGES_SEARCH_LIMIT", "0")
OUTLINE = flag("RIDGES_OUTLINE", "0")
FINDING_MAP = flag("RIDGES_FINDING_MAP")

# This function reads an environment variable and tries to convert it into a finite float. If anything is wrong, it returns the provided default
def num_env(name: str, default: float) -> float:
    try:
        value = float((os.getenv(name) or "").strip())
    except (TypeError, ValueError):
        return default
    return value if value == value and value not in (float("inf"), float("-inf")) else default

# This function safely prints a message to the console
def say(message: str) -> None:
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        pass

# The Beacon class is basically a small telemetry/logger object for tracking one named stage of a larger system
# It records things as like which stage is running, whether it was reached/skipped/fired, hashes of text before/after changes, whether content changed, API call count and cost in USD
# The class itself does not perform the actual work, It mainly reports what happened
class Beacon:
    SLOTS = ("bgshell", "conform", "editall", "fence", "findings", "ledger",
             "plan", "prelocate", "preview", "sweep", "trim", "verbatim",
             "warden", "batch", "idnorm", "retry", "scan", "scanledger",
             "untouched", "scope", "checks",
             "lint", "bound", "envelope", "contract",
             "editfence",
             "selfreview",
             "consult",
             "steady", "meter", "scope_statement")

    def __init__(self, slug: str) -> None:
        self.name = slug.lower()
        self.slug = self.tag(self.name)
        self.calls = 0
        self.usd = 0.0

    @classmethod
    def tag(cls, name: str) -> str:
        if not TELEMETRY_SLOTS:
            return "SCOPE" if name == "scope_statement" else name.upper()
        try:
            return "M%02d" % (cls.SLOTS.index(name) + 1)
        except ValueError:
            return name.upper()

    def reached(self, step: int, spent: float, clock: float) -> None:
        say("[%s] reached step=%d spent=$%.4f clock=%.0fs" % (self.slug, step, spent, clock))

    def skipped(self, reason: str) -> None:
        say("[%s] skipped: %s" % (self.slug, reason))

    def fired(self, detail: str) -> None:
        say("[%s] fired: %s" % (self.slug, detail[:400]))

    def artefact(self, when: str, blob: str) -> str:
        import hashlib
        digest = hashlib.sha256((blob or "").encode("utf-8", "replace")).hexdigest()[:8]
        say("[%s] %s %s %dB" % (self.slug, when, digest, len(blob or "")))
        return digest

    def outcome(self, before_digest: str, after: str) -> None:
        import hashlib
        digest = hashlib.sha256((after or "").encode("utf-8", "replace")).hexdigest()[:8]
        changed = "yes" if digest != before_digest else "no"
        say("[%s] after %s %dB changed=%s" % (self.slug, digest, len(after or ""), changed))

    def bill(self) -> None:
        say("[%s] cost calls=%d usd=%.4f" % (self.slug, self.calls, self.usd))

# This defines a custom exception names Spent
class Spent(Exception):
    pass

# This defines a custom exception names ReadExpired
class ReadExpired(Exception):
    pass

# This `Allowance` class is a resource budget tracker for the agent
# It tracks three main things:
# - Time: how long the agent is allowed to run
# - Money: how much API/model cost is allowed to spend
# - Activity: number of model calls, edits and tests
class Allowance:
    quoted_calls = 0

    def __init__(self) -> None:
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
        return self.deadline - time.time()

    def money_left(self) -> float:
        return self.soft_usd - self.spent

    def elapsed(self) -> float:
        return time.time() - self.started

    def halt_reason(self) -> str:
        if self.clock_left() <= 0:
            return "wall clock"
        if self.money_left() <= 0:
            return "budget"
        return ""

    def charge(self, model: str, usage: dict) -> float:
        self.calls += 1
        quoted = usage.get("cost")
        if finite_number(quoted) and 0 <= quoted <= self.ceiling_usd:
            self.spent += float(quoted)
            self.quoted_calls += 1
            return float(quoted)
        prompt = counted(usage.get("prompt_tokens"))
        completion = counted(usage.get("completion_tokens"))
        details = usage.get("prompt_tokens_details") or {}
        cached = counted(details.get("cached_tokens")) if isinstance(details, dict) else 0
        fresh = max(0, prompt - cached)
        in_price, out_price = MODEL_PRICING.get(model, UNKNOWN_TOKEN_PRICE)
        cache_price = SEAT_CACHE_TERMS.get(model, UNKNOWN_CACHE_TERMS)[0]
        cost = fresh * in_price + cached * cache_price + completion * out_price
        self.spent += cost
        return cost

# This function returns a usable ID for a tool/function call
def call_ident(call: dict, index: int) -> str:
    ident = call.get("id")
    return ident if isinstance(ident, str) and ident else "call_%d" % index

# This function cleans and normalizes a list of tool/function calls before they are recorded or stored
def recorded_calls(calls: list) -> list:
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

# This function calculates the maximum transcript size, in characters, that the agent should keep for a model
def transcript_cap_chars(model: str, ceiling_usd: float) -> int:
    cache_price, window = SEAT_CACHE_TERMS.get(model, UNKNOWN_CACHE_TERMS)
    affordable = (ceiling_usd * TRANSCRIPT_SPEND_SHARE) / (TURNS_PLANNED * cache_price)
    tokens = min(affordable, window * 0.6)
    return int(max(TRANSCRIPT_FLOOR_CHARS, tokens * CHARS_PER_TOKEN))

FINISH_REASONS = ("stop", "length", "tool_calls", "content_filter", "error", "")

# This function reports the UTF-8 byte size of some text
def foreign(text: str) -> str:
    body = "" if text is None else str(text)
    return "%dB" % len(body.encode("utf-8", "replace")) if body else "empty"

SECRET_SHAPED = re.compile(r"\b(?:sk|pk|Bearer)[-_ ][A-Za-z0-9_\-]{6,}|[A-Za-z0-9_\-]{28,}")
REFUSAL_REASON_CHARS = 200
REASON_CODES = (402, 429)

# This function takes an error/refusal detail string and turns it into a clean, safe, short reason message
def refusal_reason(detail: str) -> str:
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

# This function reads the HTTP Retry-After header and converts it into a non-negative number of seconds
def retry_after_seconds(headers) -> float:
    try:
        said = headers.get("Retry-After")
    except Exception:
        return 0.0
    try:
        return max(0.0, float(str(said).strip()))
    except (TypeError, ValueError):
        return 0.0

class SeatRefused(Exception):
    pass

class SeatTimedOut(SeatRefused):
    pass

class SeatAbsent(SeatRefused):
    pass

class SeatRateLimited(SeatRefused):
    pass

# This function builds a list of API base URLs to use, based on environment variables
def base_urls() -> list[str]:
    out = []
    injected = (os.getenv("OPENROUTER_BASE_URL") or "").strip().rstrip("/")
    if injected:
        out.append(injected)
    proxy = (os.getenv("SANDBOX_PROXY_URL") or "").strip().rstrip("/")
    if proxy and proxy + "/api/v1" not in out:
        out.append(proxy + "/api/v1")
    if not out:
        out.append(FALLBACK_BASE_URL)
    return out

# This `Seat` class is basically the LLM/API request manager for the agent
# It decides:
# - which model to use,
# - which API base URL to call,
# - how long each request may run,
# - how to retry,
# - when to switch to another model,
# - how to react to 401/403/404/429/timeouts,
# - how much each successful call cost,
# - and when to stop trying completely
# The class starts by storing the available models, timeout behavior, reasoning settings, API endpoints, and API key
class Seat:
    roster: list = []
    patient = True
    impatient_sec = 60.0
    effort = REASONING_EFFORT
    reply_ceiling = REPLY_TOKEN_CEILING

    def __init__(self, allowance: Allowance, models: list | None = None,
                 patient: bool = True, impatient_sec: float = 60.0,
                 effort: str | None = None,
                 reply_ceiling: int | None = None) -> None:
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
        self.roster = list(self.models)
        self.timeouts: dict = {}
        self.bases = base_urls()
        self.key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("RIDGES_OPENROUTER_API_KEY")
            or os.getenv("AI_PROXY_KEY")
            or ""
        )

    def current(self) -> str:
        return self.models[0]

    def retire(self, model: str) -> bool:
        if model in self.models and len(self.models) > 1:
            self.models.remove(model)
            if model in self.roster:
                self.roster.remove(model)
            say("[SEAT] retired %s, now on %s" % (model, self.models[0]))
            return True
        return False

    def ask(self, messages: list[dict], tools: list[dict] | None) -> dict:
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
        left = self.allowance.clock_left()
        if not self.allowance.edits:
            share = left
        else:
            share = min(left, max(left * CALL_CLOCK_SHARE, SEAT_CALL_TIMEOUT_SEC))
        return share, time.monotonic(), left

    def _attempt(self, model: str, messages: list[dict], tools: list[dict] | None,
                 budget=None) -> dict:
        body = {
            "model": model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": self.reply_ceiling,
        }
        if self.effort == "off":
            body["reasoning"] = {"enabled": False}
        elif self.effort:
            body["reasoning"] = {"effort": self.effort, "exclude": True}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        payload = json.dumps(body).encode("utf-8")
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
        for attempt in range(attempts):
            if self.allowance.clock_left() <= 5:
                raise Spent("clock ran out mid-request")
            asked = 0
            walled = 0
            quiet = 0
            unusable = 0
            limited = 0
            absent = 0
            advised = 0.0
            sweep = []
            granted = ceiling
            for base in self.bases:
                room = room_left()
                if room < SEAT_RETRY_FLOOR_SEC:
                    break
                asked += 1
                parsed = None
                try:
                    request_grant = min(granted, room if timed_out else room_now())
                    request = urllib.request.Request(
                        base + "/chat/completions", data=payload, headers=headers
                    )
                    with urllib.request.urlopen(
                            request, timeout=request_grant) as response:
                        parsed = json.loads(response.read().decode("utf-8", "replace"))
                except urllib.error.HTTPError as error:
                    try:
                        detail = error.read()[:400].decode("utf-8", "replace")
                    except Exception:
                        detail = ""
                    finally:
                        try:
                            error.close()
                        except Exception:
                            pass
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
                    if error.code in (400, 402, 413):
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
                    if attempt:
                        self._tally(attempt, old_bench_at, recovered=True)
                    return booked
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
            time.sleep(nap)
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
        choices = parsed.get("choices") or []
        if not choices:
            raise Spent("reply carried no choices")
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise Spent("reply carried no message object")
        usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
        quoted_before = self.allowance.quoted_calls
        cost = self.allowance.charge(model, usage)
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
        if finish not in ("stop", "tool_calls", ""):
            named = finish if finish in FINISH_REASONS else "other"
            say("[SEAT] %s reply ended on %s after %d token(s)"
                % (model, named, counted(usage.get("completion_tokens"))))
        return message

GIT_TIMED_OUT = 124

# This function is a safe wrapper around running Git commands from Python
# It takes:
# - `args`: Git arguments, like ["status"] or ["diff", "--stat"]
# - `cwd``: directory where Git should run
# - `timeout`: maximum allowed runtime, default 60 seconds
def git(args: list[str], cwd: str, timeout: float = 60.0) -> tuple[int, str]:
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

# This `Tree` class is a Git working-tree manage
# Its job is to safely read/write files inside a repository, detect changes, generate diffs, verify whether patches apply, salvage valid parts of broken patches, and restore the repository back to its original sate
#                  Tree
#                   │
#       ┌───────────┼────────────┐
#       ↓           ↓            ↓
#     Files        Git         Patches
#       │           │            │
#     read        diff         check
#     write       restore      salvage
#     path-safe   changed      apply-test
# The important idea is that when `Tree` is crated, it remembers the repository's original Git commit and the files that were already untracked.
# Later it can compare against that original state or restore it
class Tree:
    def __init__(self, root: str) -> None:
        self.root = root
        code, out = git(["rev-parse", "HEAD"], root, 30)
        self.base = out.strip() if code == 0 else ""
        self.untracked_at_start = self._untracked() or set()

    def _untracked(self, budget: float = 30.0) -> set | None:
        code, out = git(["ls-files", "--others", "--exclude-standard", "-z"],
                        self.root, max(1.0, budget))
        return {p for p in out.split("\0") if p} if code == 0 else None

    def absolute(self, path: str) -> str:
        root = os.path.normpath(self.root)
        joined = os.path.normpath(os.path.join(root, path))
        if joined != root and not joined.startswith(root + os.sep):
            raise ToolFault("path escapes the repository: %s" % path)
        return joined

    def read(self, path: str) -> str:
        full = self.absolute(path)
        if not os.path.isfile(full):
            raise ToolFault("no such file: %s" % path)
        with open(full, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()

    def write(self, path: str, text: str) -> None:
        full = self.absolute(path)
        os.makedirs(os.path.dirname(full) or self.root, exist_ok=True)
        with open(full, "w", encoding="utf-8") as handle:
            handle.write(text)

    def changed_paths(self, budget: float = 15.0) -> list[str]:
        if not self.base:
            return []
        code, out = git(["diff", "--name-only", self.base], self.root, budget)
        if code != 0:
            raise ToolFault("could not list the changed files: %s" % out.strip()[:200])
        return [line.strip() for line in out.splitlines() if line.strip()]

    def at_base(self, path: str, budget: float = 15.0) -> str:
        code, out = git(["show", "%s:%s" % (self.base, path)], self.root, budget)
        if code != 0:
            raise ToolFault("could not read %s as it was: %s" % (path, out.strip()[:200]))
        return out

    def generated_exclusions(self, budget: float = 15.0) -> list[str]:
        code, listing = git(["ls-tree", "-r", "--name-only", "-z", self.base], self.root, budget)
        if code:
            return []
        baseline = set(listing.split("\0")) | self.untracked_at_start
        code, listing = git(["ls-files", "--others", "--exclude-standard", "-z"], self.root, budget)
        if code:
            return []
        return [":(top,literal,exclude)" + path for path in listing.split("\0")
                if path and path not in baseline and
                (set(path.split("/")[:-1]) & {"vendor", "dist", "target", "node_modules"}
                 or path.split("/")[-1] == "go.sum")]

    def diff(self, budget: float = 60.0) -> str:
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
        args = ["diff", "--binary", "--no-color"] + ([self.base] if self.base else [])
        code, out = git(args, self.root, left())
        if code != 0:
            if deadline - time.monotonic() < 10.0:
                say("[TREE] diff failed (%s) and there is no time to ask again: %s"
                    % (code, out.strip()[:200]))
                return ""
            say("[TREE] diff failed (%s), retrying once: %s" % (code, out.strip()[:200]))
            code, out = git(args, self.root, left())
            if code != 0:
                say("[TREE] diff failed again: %s" % out.strip()[:200])
                return ""
        return out

    def applies(self, patch: str, budget: float = 30.0) -> bool | None:
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
        if "index.lock" not in (said or ""):
            return False
        try:
            os.remove(os.path.join(self.root, ".git", "index.lock"))
        except OSError:
            return False
        say("[TREE] removed a lock a stopped process left behind")
        return True

    def restore(self, budget: float = 60.0) -> bool:
        deadline = time.monotonic() + budget
        args = (["reset", "--hard", self.base] if self.base
                else ["checkout", "--", "."])
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
                if os.path.isdir(full):
                    shutil.rmtree(full)
                else:
                    os.remove(full)
                gone += 1
            except OSError as error:
                say("[TREE] could not remove %s: %s" % (path, error))
        vouched = code == 0 and listed is not None and gone == len(added)
        say("[TREE] restored to %s, removed %d of %d path(s) the run created%s"
            % (self.base[:8] or "?", gone, len(added),
               "" if vouched else " -- not confirmed"))
        return vouched

class ToolFault(Exception):
    pass

CLIP_NOTE = "\n... [%d characters of %s elided] ...\n"

# This function shortens a long string to fit within a maximum character limit, but instead of keeping only the beginning, it preserves both the start and the end and inserts a note in the middle saying that some content was removed
def clip(text: str, cap: int, label: str = "output") -> str:
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

# This function creates a short one-line summary of a shell command after it runs
def report_shell(job: "Shell", out: str) -> None:
    tail = ""
    for line in reversed((out or "").splitlines()):
        if line.strip() and line.strip() != STILL_RUNNING:
            tail = line.strip()
            break
    say("[SHELL] %.1fs %dc :: %s :: %s"
        % (time.time() - job.started, len(out or ""),
           " ".join(job.command.split())[:SHELL_REPORT_CAP],
           tail[:SHELL_REPORT_CAP]))

FINDING_CHECK = re.compile(r"^\s*ruff\s+check\b")
FINDING_ARROW = re.compile(r"^\s*-->\s+(\S+?):(\d+):\d+\s*$", re.M)
FINDING_CONCISE = re.compile(r"^(\S+?):(\d+):\d+:\s", re.M)
HUNK_HEAD = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,(\d+))? @@")

# This function cleans up a file path so it becomes a short relative-looking path
def path_tail(name: str, root: str = "") -> str:
    text = str(name or "").replace("\\", "/")
    base = str(root or "").replace("\\", "/").rstrip("/")
    if base and text.startswith(base + "/"):
        text = text[len(base) + 1:]
    while text.startswith("./"):
        text = text[2:]
    return text

# This function extracts file + line-number findings from some tool output
def findings_from_text(out: str, root: str = "") -> dict:
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

# This function parses JSON-formatted findings and converts them into the same structure as your previous findings_from_text() function
def findings_from_json(text: str, root: str = "") -> dict:
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

# This function splits one large Git patch into separate patch sections, one per file
def split_by_file(patch: str) -> list[str]:
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

# This function extracts the file path from the first line of one Git diff section
def section_path(section: str) -> str:
    head = section.split("\n", 1)[0]
    if not head.startswith("diff --git "):
        return ""
    body = head[len("diff --git "):].strip()
    halves = body.split(" b/", 1)
    if len(halves) == 2:
        return halves[1].strip().strip('"')
    return body.strip().strip('"')

# This function checks whether a file path looks like junk / generated / unwanted content according to three configured rules
def junk_path(path: str) -> bool:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return False
    if parts[-1] in ENVELOPE_JUNK_NAMES:
        return True
    if any(part in ENVELOPE_JUNK_DIRS for part in parts):
        return True
    return parts[-1].endswith(ENVELOPE_JUNK_ENDS)

# This function checks one Git diff section and returns a short explanation if that section looks like an "envelope" change - meaning a structural or suspicious repository-level change rather than a normal code edit
def envelope_reason(section: str) -> str:
    path = section_path(section)
    for line in section.splitlines():
        if line.startswith("@@"):
            break
        if line.startswith("new file mode"):
            return "adds %s" % path
        if line.startswith("deleted file mode"):
            return "deletes %s" % path
        if line.startswith("rename from ") or line.startswith("rename to "):
            return "renames %s" % path
        if line.startswith("GIT binary patch"):
            return "carries a binary blob for %s" % path
    if junk_path(path):
        return "touches %s, which is build or scratch output" % path
    return ""

# This function removes file permission/model changes from one Git diff section, while keeping the rest of the patch intact
def demoded(section: str) -> tuple[str, str]:
    kept, taken = [], ""
    for line in section.splitlines(keepends=True):
        head = line.rstrip("\r\n")
        if head.startswith("old mode ") or head.startswith("new mode "):
            taken = head.strip()
            continue
        kept.append(line)
    if not taken:
        return section, ""
    return "".join(kept), ("drops the mode change on %s (%s)"
                           % (section_path(section), taken))

# This function is a patch filter. It takes a full Git patch and removes sections that look out-of-scope or undersirable, while preserving ordinary code edits
def envelope_trim(patch: str, beacon: "Beacon", statement: str = "") -> str:
    if RIDGES_SCOPE_FOLLOWS_STATEMENT and not statement_bounded(statement):
        beacon.skipped("statement leaves the path set open")
        return patch
    sections = split_by_file(patch)
    kept, dropped = [], []
    for section in sections:
        why = envelope_reason(section)
        if why:
            dropped.append(why)
            continue
        trimmed, note = demoded(section)
        if note:
            dropped.append(note)
        if "\n@@" in trimmed:
            kept.append(trimmed)
        elif note:
            dropped[-1] = note.replace("drops the mode change on",
                                       "drops the mode-only section for")
    if not dropped:
        beacon.skipped("nothing to drop from %d section(s)" % len(sections))
        return patch
    joined = "".join(kept)
    if not joined.strip():
        beacon.fired("every one of %d section(s) would go; kept the answer "
                     "whole: %s" % (len(sections), "; ".join(dropped[:3])))
        return patch
    before = beacon.artefact("before", patch)
    beacon.fired("dropped %d of %d section(s): %s"
                 % (len(dropped), len(sections), "; ".join(dropped[:3])))
    beacon.outcome(before, joined)
    return joined

CONTRACT_BANNED_NAMES = frozenset((
    "__import__", "breakpoint", "compile", "eval", "exec", "getattr",
    "globals", "locals", "open", "setattr", "vars"))
CONTRACT_BANNED_NODES = ("AsyncFunctionDef", "Await", "ClassDef", "Delete",
                         "DictComp", "For", "GeneratorExp", "Global",
                         "Lambda", "ListComp", "Match", "Nonlocal", "SetComp",
                         "Try", "While", "Yield", "YieldFrom")
CONTRACT_REFUSED_NODES = ("Import", "ImportFrom", "Raise", "With")

# This function checks whether a file path points to a Python file
def contract_readable(path: str) -> bool:
    return path.endswith(".py")

# This function analyzes Python source code with Python's ast module and builds three maps describing every function/method in the file
def python_functions(text: str) -> tuple[dict, dict, dict]:
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

# This function finds functions that are nested inside another function and records which outer function contains them
def nested_definitions(tree) -> dict:
    import ast
    found: dict = {}

    def visit(node, holder: str, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            named = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef))
            name = (prefix + "." + child.name).lstrip(".") if named else prefix
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if holder:
                    found[child] = holder
                visit(child, name, name)
            else:
                visit(child, holder, name)
    visit(tree, "", "")
    return found

# This function compares two versions of text and returns the line numbers that changed in the old version and the new version
def changed_line_numbers(before: str, after: str) -> tuple[set, set]:
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

# This function checks whether edits to a Python file violate a set of contract/scrope rules
def contract_violations(path: str, before: str, after: str,
                        statement: str = "") -> tuple[list, list]:
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
    if RIDGES_SCOPE_FOLLOWS_STATEMENT and not statement_bounded(statement):
        return hard, soft
    try:
        owner, heads, first = python_functions(after)
        _, was_heads, was_first = python_functions(before)
    except (SyntaxError, ValueError, RecursionError):
        return hard, soft
    _, now = changed_line_numbers(before, after)
    for name in sorted(set(heads) & set(was_heads)):
        if heads[name] == was_heads[name]:
            continue
        soft.append(
            "%s: the signature or decorators of %s are not the ones that were "
            "there. Where a statement restricts the change to one method, its "
            "header is usually compared field by field with the original and any "
            "difference is refused. Unless the statement says otherwise, keep "
            "the def line as it was and put the change in the body."
            % (path, name))
    touched = sorted({owner[line] for line in now if line in owner})
    for name in touched:
        if name in was_first and first.get(name) != was_first.get(name):
            soft.append(
                "%s: the first statement of %s is not the one that was there. "
                "Where a statement restricts the change to one method, that "
                "first line -- often a local import -- is compared verbatim and "
                "any other one is refused." % (path, name))
    tree = ast.parse(after)
    nested = nested_definitions(tree)
    refused_nodes = (statement_refused_nodes(statement)
                     if RIDGES_SCOPE_FOLLOWS_STATEMENT else CONTRACT_REFUSED_NODES)
    seen: set = set()
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if line not in now:
            continue
        kind = type(node).__name__
        if owner.get(line):
            refused = ""
            if node in nested:
                refused = "a %s nested inside %s" % (kind, nested[node])
            elif kind in refused_nodes:
                refused = "the construct %s" % kind
            if refused:
                if refused not in seen:
                    seen.add(refused)
                    hard.append(
                        "%s line %d introduces %s. This construct is refused "
                        "inside the bounded method by the active contract "
                        "rules. Keep the existing "
                        "def and express the change with assignments, if "
                        "branches, calls and return statements." %
                        (path, line, refused))
                continue
        bad = ""
        if (kind in CONTRACT_BANNED_NODES
                or (owner.get(line) and kind in CONTRACT_REFUSED_NODES)
                or (RIDGES_SCOPE_FOLLOWS_STATEMENT and owner.get(line)
                    and kind in ("AsyncFor", "AsyncWith", "TryStar"))):
            bad = "the construct %s" % kind
        elif isinstance(node, ast.Name) and node.id in CONTRACT_BANNED_NAMES:
            bad = "the name %s" % node.id
        elif isinstance(node, ast.Name) and "__" in node.id:
            bad = "the name %s" % node.id
        elif isinstance(node, ast.Attribute) and "__" in node.attr:
            bad = "the attribute %s" % node.attr
        if bad and bad not in seen:
            seen.add(bad)
            soft.append(
                "%s line %d introduces %s. A statement that restricts the "
                "change to one method commonly forbids that inside it; one that "
                "sets no such restriction does not." % (path, line, bad))
    return hard, soft

# This function parses a Git unified diff and builds a dictionary showing which original-file line numbers were touched by the patch
def patch_touched_lines(patch: str) -> dict:
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

# This helper cleans one filename/path taken from a Git diff header
def _diff_side(name: str, prefix: str) -> str:
    text = name.strip()
    if text == "/dev/null" or not text:
        return ""
    if text.startswith(prefix):
        text = text[len(prefix):]
    return path_tail(text)

# This function measures how close the patch's touched lines are to known findings
def finding_distances(findings: dict, touched: dict) -> list:
    out = []
    for where in sorted(touched):
        rows = findings.get(where)
        if not rows:
            continue
        for line in sorted(touched[where]):
            out.append(min(abs(line - row) for row in rows))
    return out

# This function creates a human-readable summary comparing reported findings with the lines changed by a patch
def finding_record(findings: dict, touched: dict) -> str:
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

# This FindingMap class is a collector and reporter for checker/linter findings
# It watches shell commands, recognizes commands that look like checks, extract file/line findings from their output, stores those findings, and later compares them against the lines touched by the final patch
class FindingMap:
    def __init__(self, root: str) -> None:
        self.root = root
        self.rows: dict = {}
        self.reads = 0
        self.beacon = Beacon("findings")

    def observe(self, command: str, out: str) -> None:
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
        if not self.rows:
            self.beacon.skipped("no reading of the check was recorded")
            return
        self.beacon.fired(finding_record(self.rows, patch_touched_lines(patch)))

_VENV_BIN = ""

# This code finds a usable Python virtual-environment `bin` dirctory from a small list of likely locations, remembers the result, and return it
def venv_bin() -> str:
    global _VENV_BIN
    if _VENV_BIN == "":
        _VENV_BIN = "-"
        for where in ("/opt/venv/bin", "/usr/local/venv/bin", ".venv/bin", "../venv/bin"):
            python = os.path.join(where, "python")
            if os.path.isfile(python) and os.access(python, os.X_OK):
                _VENV_BIN = os.path.abspath(where)
                break
    return "" if _VENV_BIN == "-" else _VENV_BIN

# This function returns the path of the Python intepreter that is currently runnning the program
def repo_python() -> str:
    return sys.executable or "python3"

PACKAGES_AT_ONCE = 1

# This function builds an environment-variable dictionary for subprocess commands, with two main adjustments
def command_env(pack_venv: bool = True) -> dict:
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

# This Shell class is a background shell-command runner with logging, timeout control, and output capture
class Shell:
    counter = 0

    def __init__(self, command: str, cwd: str, pack_venv: bool = True,
                 hard_timeout: float | None = None) -> None:
        Shell.counter += 1
        self.name = "job%d" % Shell.counter
        self.command = command
        self.started = time.time()
        env = command_env(pack_venv)
        where = venv_bin() if pack_venv and PACK_VENV else ""
        if where:
            command = "export PATH=%s:$PATH\n%s" % (shlex.quote(where), command)
        self.sink = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", errors="replace", suffix=".out", delete=False
        )
        self.process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=cwd,
            env=env,
            stdout=self.sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.watchdog = None
        if hard_timeout is not None:
            self.watchdog = threading.Timer(max(1.0, hard_timeout), self._kill)
            self.watchdog.daemon = True
            self.watchdog.start()

    def _kill(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                try:
                    self.process.kill()
                except Exception:
                    pass

    def _text(self) -> str:
        try:
            self.sink.flush()
            with open(self.sink.name, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        except OSError:
            return ""

    def _tail(self, cap: int = SHELL_OUTPUT_CAP) -> str:
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
        try:
            self.process.wait(timeout=max(1.0, timeout))
            return True, self._text()
        except subprocess.TimeoutExpired:
            return False, self._text()

    def finished(self) -> bool:
        return self.process.poll() is not None

    def drain(self) -> str:
        if self.process.poll() is None:
            return self._text() + "\n" + STILL_RUNNING
        return self._text()

    def stop(self) -> None:
        if self.watchdog is not None:
            self.watchdog.cancel()
        self._kill()
        try:
            self.sink.close()
            os.unlink(self.sink.name)
        except OSError:
            pass

# This `ShellPool` class is a manager for multiple background Shell jobs
# It keeps track of started shell commands by name, lets other code retreives a specific job, and can shut down and clean up all jobs at once
class ShellPool:
    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self.jobs = {}

    def start(self, command: str, pack_venv: bool = True,
              hard_timeout: float | None = None) -> Shell:
        job = Shell(command, self.cwd, pack_venv, hard_timeout)
        self.jobs[job.name] = job
        return job

    def get(self, name: str) -> Shell:
        job = self.jobs.get(name)
        if job is None:
            raise ToolFault("no background job named %s" % name)
        return job

    def close(self) -> None:
        for job in list(self.jobs.values()):
            try:
                job._kill()
                report_shell(job, job._tail())
                job.stop()
            except Exception:
                pass
        self.jobs.clear()

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file, optionally a line range. Prefer a range once you know where you are looking. A read that does not fit stops early and the header says where to start again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repository root."},
                    "start": {"type": "integer", "description": "First line, 1-based."},
                    "count": {"type": "integer", "description": "How many lines to return."},
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
                "properties": {"pattern": {"type": "string"}},
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

# This function looks up the schema/definition for a tool by its name
def _schema(name: str) -> dict:
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
    r"\bgit\s+(?:-[^\s]+\s+)*(commit|stash|checkout|switch|restore|reset|clean|revert|rebase|merge|cherry-pick|push)\b"
)
NETWORK_COMMAND = re.compile(
    r"(?:^|[|&;]|\$\(|`)\s*(?:sudo\s+)?"
    r"(curl|wget|nc|ncat|telnet|ssh|scp|rsync|ftp|"
    r"git\s+(?:fetch|pull|clone|remote|ls-remote|submodule))(?![\w-])"
)
TEST_PATH = re.compile(
    r"(^|/)conftest\.py$|(^|/)tests?(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")
CHECK_TEST_PATH = TEST_PATH
if RIDGES_SCOPE_FOLLOWS_STATEMENT:
    TEST_PATH = re.compile(TEST_PATH.pattern + r"|_test\.go$|\.(?:test|spec)\.(?:js|ts)$")
NOQA_DIRECTIVE = re.compile(r"#\s*(?:(?:ruff|flake8)\s*:\s*)?noqa\b", re.I)
FAILED_TEST = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.M)
PASSED_COUNT = re.compile(r"(\d+) passed")
FENCED_BLOCK = re.compile(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)```", re.S)
CODE_SPAN = re.compile(r"`([^`\n]+)`")
COMMAND_SEPARATOR = frozenset(("&&", "||", "|", "|&", "&", ";", ";;"))
REDIRECT_IN_TOKEN = re.compile(r"\d*(?:>>|>|<<|<)&?")

# This function checks whether a single shell token contains a redirection operator and returns
def split_redirect(token: str) -> tuple:
    found = REDIRECT_IN_TOKEN.search(token)
    if not found:
        return token, False
    return token[:found.start()], not token[found.end():]

# This function scans a text `statement` for code blocks/inline code, tokenizes the shell commands it finds, and returns only commands that look like:
# ruff check...
def command_lines(statement: str) -> list:
    found = []
    text = statement or ""
    for block in FENCED_BLOCK.findall(text) + CODE_SPAN.findall(text):
        try:
            tokens = shlex.split(block, comments=True)
        except ValueError:
            continue
        current = []
        skip = False
        for token in tokens + [";"]:
            if skip:
                skip = False
            elif token in COMMAND_SEPARATOR:
                if len(current) > 2 and current[:2] == ["ruff", "check"]:
                    found.append(current)
                current = []
            else:
                head, skip = split_redirect(token)
                if head:
                    current.append(head)
    return found

# This block is building a fallback system for running Python test suits even when optional dependecies are missing
# The first part defines regexes and thresholds used to understand pytest output. 
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
SUITE_SHIM_MIN_GREEN = 0.5
SHIM_SOURCE = '''"""Stands in for a distribution this image does not carry.

It answers the import, and it survives being *used* by the package that is
importing it, which is not the same as being a mock of it.

Raising on every use was the first design, and the reasoning was sound as far as
it went: a stand-in that answered attribute access lets a test that really
exercises the absent package pass for a reason unconnected to the project, and a
refusal aimed at an answer that was right is the expensive direction.  What that
reasoning missed is where the import usually is.  A test package's own
`__init__` routinely *calls* what it imports -- to register a plugin, install a
hook, enable a checker -- and a stand-in that raises there fails before a single
test is collected.  That is not one test lost to a missing package.  It is every
test under that package, which is the whole reading.

So attribute access gives back something callable that returns its argument when
it is used as a decorator and another one of itself otherwise.  Nothing it does
is an answer about the project.  A test that really exercises the absent package
gets the same nothing in both readings, so it lands on both sides of the
difference and cancels exactly as a raising stand-in would; the difference is
only that everything beside it still runs.

The guard that makes this safe is not in here.  It is the admission check on the
recovered reading: a reading that comes back mostly red over stand-ins is a
reading about the stand-ins, and is refused whatever this module does.
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

# This function checks whether Python can located a module/package by name
def resolvable(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except BaseException:
        return True

# This function extracts top-level missing Python module names from command/text output, removes duplicates, and keeps only modules that really appear to be unavailable
def missing_modules(out: str) -> list:
    seen = []
    for name in MISSING_MODULE.findall(out or ""):
        if "." in name:
            continue
        if name not in seen and name.isidentifier() and not resolvable(name):
            seen.append(name)
    return seen

# This function extracts missing Python distribution/package names from output and returns a unique cleaned list
def missing_dists(out: str) -> list:
    seen = []
    for name in MISSING_DIST.findall(out or ""):
        if name not in seen and re.fullmatch(r"[A-Za-z0-9._-]+", name):
            seen.append(name)
    return seen

# This function checks whether path is inside root, after resolving both to their real absolute filesystem locations
def inside(path: str, root: str) -> bool:
    try:
        path, root = os.path.realpath(path), os.path.realpath(root)
        if (os.path.splitdrive(path)[0].lower()
                != os.path.splitdrive(root)[0].lower()):
            return False
        return os.path.commonpath([path, root]) == root
    except (ValueError, OSError, TypeError):
        return True

# This function creates fake Python distribution metadata folders for package names that are missing, and returns the names it successfully created.
def write_dist_records(names: list, where: str) -> list:
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

# This function creates fake Python packages for missing module names by writing the SHIM_SOURCE code into each package’s __init__.py.
def write_shims(names: list, where: str) -> list:
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
    " Before changing anything, name it: which kind of input that this file "
    "already handles does the patched code treat differently from the "
    "original, and on which line? If nothing does, the cause is not the "
    "change in behaviour and the named tests are where to look."
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

# This block configures timing, safety/review, and consultation limits for the agent, while length_class() simply labels consultation text as short, middling, or long based on character count.
def length_class(text: str) -> str:
    size = len(text or "")
    if size < CONSULT_ANSWER_SHORT:
        return "a short"
    return "a middling" if size < CONSULT_ANSWER_LONG else "a long"

# This function safely extracts a .statement string from a warden object
def statement_of(warden) -> str:
    return (warden.statement if warden is not None else "") or ""

CONSULT_BRIEF = """You are asked one question about work another agent has just done, and you will not be asked again. You cannot change anything and you have no tools. What you write is handed to the agent holding the pen; it decides what to do with it.

What that work has to satisfy is wider than the behaviour its own brief describes. A change can hand back the right answer for every input and still be wrong on a property nobody wrote down: how much work is done to produce the answer and whether that grows with the size of what is asked for; how many separate round trips it takes; whether a result is worked out when it is asked for or when it is defined; whether the structures that already exist to make a lookup cheap are used, or gone past; the order of what comes back; which end points of a range are included.

Name the ones that could apply to THIS change, and no others. For each, give the exact command that settles it against the real thing this project runs against -- the service the code talks to, started the way this project already starts it. A substitute written in place of that thing settles nothing: it records what it was asked for, and what has to hold is what the real one did. Where the real thing keeps its own account of the work it carried out, the command that reads that account back is the one to give.

Be short and be concrete: the property, then the command, then what its output would have to say. Say nothing about what you cannot see."""
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
LEDGER_BULLET = re.compile(r"^[-*][ \t]+(.*)$")
LEDGER_FENCE = re.compile(r"^[ \t]*(?:```|~~~)")
LEDGER_MIN = 3
LEDGER_MAX_ASKS = 2
FENCE_MAX_REFUSALS = 2
HANDIN_PAUSES = (("selfreview_state", "selfreview_note"),
                 ("untouched_state", "untouched_note"),
                 ("ledger_state", "ledger_note"),
                 ("conform_state", "conform_note"))
HANDIN_PAUSES_MAX = 1

# Review another agent's patch, but do not edit anything. Identity hidden behavioral properties that may matter and tell the main agent exactly how to verify them against the real system
def bullet_runs(text: str) -> list[list[str]]:
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

# This function takes all the bullet points found by bullet_runs() and flattens them into one simple list of requirement strings
def stated_requirements(text: str) -> list[str]:
    return [item for run in bullet_runs(text) for item in run]

LEDGER_GUARD = re.compile(
    r"\b(preserve|preserves|keep|keeps|retain|retains|remain|remains|"
    r"continue|continues|unchanged|intact|still|do not|don't|must not|"
    r"never|leave|leaves|untouched)\b", re.I)

# This code decides whether a requirements bullet is asking the agent to change something or merely to preserve existing behavior
def wants_an_edit(item: str) -> bool:
    return not LEDGER_GUARD.search(item or "")

# This function builds a one-time reminder message telling the agent to compare its diff against every stated requirement before submitting
def read_back(items: list[str]) -> str:
    lines = "\n".join("  %d. %s" % (n + 1, item) for n, item in enumerate(items))
    return ("Before this goes in, put the diff beside what the task asked for. "
            "These are its own words:\n%s\n"
            "Anything on that list with nothing in the diff answering for it "
            "is not done yet; do those now. Where the list asks that existing "
            "behaviour be kept, leaving that code alone is how it is met. If "
            "every item already has something answering for it, hand in again "
            "-- this is asked once." % lines)

# This function checks whether a file path looks likfe a Python source file that Python can normally import
def importable(path: str) -> bool:
    return any(path.endswith(suffix) for suffix in importlib.machinery.SOURCE_SUFFIXES)

NESTED_SOURCE = "src"

# This function returns the directories that should be treated as Python package/source roots for a repository
def package_roots(root: str) -> list[str]:
    nested = os.path.join(root, NESTED_SOURCE)
    return [root, nested] if os.path.isdir(nested) else [root]

EXCLUDING_OPTION = frozenset(("--exclude", "--extend-exclude", "--force-exclude"))

# This function extracts the actual target arguments from a command-line argument list, while skipping options and, for certain options, also skipping the value that follows them
def check_targets(argv: list) -> list:
    targets, skip = [], False
    for token in argv[2:]:
        if skip:
            skip = False
        elif token.startswith("-"):
            skip = token in EXCLUDING_OPTION
        else:
            targets.append(token)
    return targets

# This function tries to identify one specific file that the task statement explicitly tells Ruff to check
def declared_file(statement: str, root: str) -> str | None:
    found = set()
    for argv in command_lines(statement):
        for token in check_targets(argv):
            if not RIDGES_SCOPE_FOLLOWS_STATEMENT and not token.endswith(".py"):
                continue
            if CHECK_TEST_PATH.search(token):
                continue
            candidate = os.path.join(root, token)
            if inside(candidate, root) and os.path.isfile(candidate):
                found.add(token)
    return found.pop() if len(found) == 1 else None

# This function extracts certain interesting backtick-quoted literals from the task statement, while filtering out things that look like commands, paths, identifiers, function calls, or other structural code
def named_literals(statement: str) -> list[str]:
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

# This function reads a unified diff and returns the line ranges touched by each diff hunk
def touched_lines(diff: str) -> list[tuple[int, int]]:
    spans = []
    for start, length in HUNK_RE.findall(diff or ""):
        first = int(start)
        spans.append((first, first + (int(length) if length else 1) - 1))
    return spans

ROW_SNIPPET_RE = re.compile(r"`([^`\n]{4,80})`")

# This function filters a list of text rows
def carried_rows(source: str, rows: list[str]) -> list[str]:
    out = []
    for row in rows:
        found = ROW_SNIPPET_RE.search(row)
        if not found:
            continue
        snippet = found.group(1).strip()
        if snippet and snippet in (source or ""):
            out.append(row)
    return out

MUTATION_SCAN_MAX_CHARS = 1_000_000

# This function inspects a Python function's AST node and returns the names of parameters whose default value is explicitly `None`
def optional_none_params(node: ast.AST) -> set[str]:
    args = getattr(node, "args", None)
    if args is None:
        return set()
    optional = set()
    positional = args.posonlyargs + args.args
    paired = zip(positional[len(positional) - len(args.defaults):], args.defaults)
    for argument, default in list(paired) + list(zip(args.kwonlyargs, args.kw_defaults)):
        if isinstance(default, ast.Constant) and default.value is None:
            optional.add(argument.arg)
    return optional

# This function scans a function's AST for a very specific suspicious pattern involving parameters that default to None.
def dead_conjuncts(node: ast.AST) -> list[tuple[int, str, str]]:
    optional = optional_none_params(node)
    if not optional:
        return []
    out = []
    pending = list(ast.iter_child_nodes(node))
    while pending:
        inner = pending.pop()
        if isinstance(inner, (ast.AsyncFunctionDef, ast.ClassDef,
                              ast.FunctionDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(inner))
        if not (isinstance(inner, ast.If) and len(inner.body) == 1
                and isinstance(inner.body[0], (ast.Continue, ast.Pass, ast.Return))):
            continue
        if not (isinstance(inner.test, ast.BoolOp) and isinstance(inner.test.op, ast.And)):
            continue
        for part in inner.test.values:
            if not (isinstance(part, ast.Name) and part.id in optional):
                continue
            rest = [other for other in inner.test.values if other is not part]
            named = {n.id for other in rest for n in ast.walk(other) if isinstance(n, ast.Name)}
            if part.id not in named:
                out.append((inner.lineno, part.id, inner.test))
    return out

# This function is a static-analysis warning generator
# It opens a Python file, parses its AST, looks for several patterns that might indicate the patch changed behavior incorrectly, ranks those suspicions, and returns at most about 10 human-readdable warnings
def mutation_suspects(path: str, statement: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read(MUTATION_SCAN_MAX_CHARS + 1)
        if len(source) > MUTATION_SCAN_MAX_CHARS:
            return []
        tree = ast.parse(source)
    except (MemoryError, OSError, RecursionError, SyntaxError, ValueError):
        return []

    def rendered(node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "<expression>"
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in ("sort", "sorted") and SCAN_ORDER:
                for keyword in node.keywords:
                    if keyword.arg == "reverse":
                        found.append(("order", node.lineno, "line %d: %s(reverse=%s) -- is that "
                                      "the direction the statement asks for?"
                                      % (node.lineno, name, rendered(keyword.value)), name))
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            for index, first in enumerate(node.body[:3] if SCAN_ABSENT else []):
                if (isinstance(first, ast.If) and len(first.body) == 1
                        and isinstance(first.body[0], (ast.Return, ast.Continue))):
                    found.append(("guard", first.lineno, "line %d: %s() returns early on `%s` "
                                  "before statement %d -- does every case that must "
                                  "still run reach past it?"
                                  % (first.lineno, node.name,
                                     rendered(first.test)[:60], index + 1), node.name))
            for lineno, name, test in (dead_conjuncts(node) if DEAD_CONJUNCT else []):
                found.append(("dead", lineno, "line %d: %s() skips on `%s`, and `%s` "
                              "defaults to None -- so on any call that leaves it "
                              "out this guard never fires. Is that intended?"
                              % (lineno, node.name, rendered(test)[:60], name), node.name))
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))}
    quoted = {word for word in re.findall(r"`([^`\n]{1,60})`", statement or "")
              if word.isidentifier()}
    scope = defined & quoted
    if scope:
        found = [row for row in found if row[0] != "guard" or row[3] in scope]
    branched = set()
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.Compare, ast.BoolOp, ast.Call)):
            expression = node.test if isinstance(node, (ast.If, ast.While)) else node
            if id(expression) in seen:
                continue
            for part in ast.walk(expression):
                seen.add(id(part))
                if isinstance(part, ast.Constant):
                    branched.add(str(part.value))
                elif isinstance(part, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
                    branched.add(rendered(part))
    for value in named_literals(statement) if SCAN_ABSENT else []:
        if value not in branched:
            found.append(("absent", 0, "the statement names `%s`, and no branch in this file "
                          "tests for it" % value, value))
    grouped: list[list[str]] = []
    for kind, cap in (("absent", None), ("dead", None), ("order", None), ("guard", 5)):
        rows = [text for tag, _, text, _name in sorted(found, key=lambda row: row[1])
                if tag == kind]
        grouped.append(rows if cap is None else rows[:cap])
    ranked: list[str] = []
    for index, rows in enumerate(grouped):
        reserve = sum(bool(later) for later in grouped[index + 1:])
        ranked.extend(rows[:max(0, 10 - len(ranked) - reserve)])
    return ranked

TEST_MODULE_RE = re.compile(r"^(?:test_.*|.*_test)\.py$")

# This function reads a Python file and returns only the top-level modules imported directly at module scope
def _module_level_imports(path: str) -> set[str]:
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

# This function checks whether a directoy base appears to provide/import a Python module named name
def _provides(base: str, name: str) -> bool:
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

# This function checks whether a filename in base is a compiled Python extension module for the given module name
def _is_extension_of(base: str, entry: str, name: str) -> bool:
    for suffix in (".so", ".pyd", ".dylib"):
        if not entry.endswith(suffix) or not entry.startswith(name):
            continue
        tag = entry[len(name):len(entry) - len(suffix)]
        if tag and not (tag.startswith(".") and any(c.isdigit() for c in tag[1:])
                        and "." not in tag[1:]):
            continue
        return os.path.isfile(os.path.join(base, entry))
    return False

# This function takes a set of module names and returns the ones that appear to be truly unavailable: not built into Python, not provided by the repository itself, and not installed/importable from the environment
def _absent(root: str, names: set[str]) -> set[str]:
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

# This function takes a test scope and tries to turn broad test directories into a smaller set of test files that the current environement can cactually collect/run
def readable_scope(root: str, scope: list[str]) -> tuple[list[str], str]:
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

# This function summarizes how many Ruff-check commands and Python file targets are explicitly mentioned in the task statement, and how many of those files are real non-test files in the repository
def declared_trace(statement: str, root: str) -> str:
    argvs = command_lines(statement)
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

# This block defines the rules for recognizing pytest commands written inside the task statement and then extracts the test files/directories that the statement explicitly asks to run
def stated_test_scope(statement: str, root: str) -> list[str]:
    found: list[str] = []
    for block in (FENCED_BLOCK.findall(statement or "")
                  + CODE_SPAN.findall(statement or "")):
        for line in (block or "").split("\n"):
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

# This function takes one token from a pytest command and decides whether it represents a valid test path inside the repository
def _stated_path(token: str, root: str) -> tuple[str, str]:
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

# _stated_targets() parses a recognized pytest invocation, skips known flags and their values, validates positional test paths, and returns those targets only when the command can be interpreted unambiguously; otherwise it conservatively returns an empty list
def _stated_targets(argv: list, root: str) -> list[str]:
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

# This function tries to infer the most likely test file or test directory corresponding to one declared source file
def suite_scope(root: str, declared: str | None) -> list[str]:
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
    for cut in range(len(inner)):
        every_top(lambda d, c=cut: (os.path.join(d, *inner[c:]), os.path.isdir))
    every_top(lambda d: (os.path.join(d, "test_%s" % base), os.path.isdir))
    every_top(lambda d: (d, os.path.isdir))
    for rel, is_right in tries:
        if is_right(os.path.join(root, rel)):
            return [rel]
    return []

DIFF_TRIVIAL = re.compile(r"^[\s)\]}:,]*$")

# This function gives a compact summary of the shape of a Git patch: how many files and hunks it changes, how many lines are added/removed, and how many removed lines were effectively re-added somewhere else.
def patch_shape(patch: str) -> str:
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

LINE_SCAN_CAP = 400
FOLD_SHAPES = (
    ("membership-test-gone",
     re.compile(r"\b(?:if|elif|while|assert|and|or|not)\b[^\n]*\bin\b")),
    ("none-vs-falsy", re.compile(r"\bis\s+(?:not\s+)?None\b|[!=]=\s*None\b")),
    ("absent-vs-star", re.compile(r"""["']\*["']""")),
    ("ordering", re.compile(r"\bsorted\(|\.sort\(|\breversed\(|OrderedDict")),
    ("spelling", re.compile(r"\.encode\(|\.decode\(|\bbytes\(|\bint\([^)]*,\s*\d+\)")),
    ("swallowed-failure",
     re.compile(r"\braise\b|\bexcept\s+\w|\bassert\b|\.error\(|\.warning\(")),
)

# This function summarizes whether certain important code-pattern "shapes" disppeared from Git diff hunks
def fold_report(patch: str) -> str:
    hunks = 0
    dropped: dict = {}
    removed: list = []
    added: list = []

    def close() -> None:
        for name, shape in FOLD_SHAPES:
            if any(shape.search(line) for line in removed) and not any(
                    shape.search(line) for line in added):
                dropped[name] = dropped.get(name, 0) + 1
    inside = False
    for line in (patch or "").split("\n"):
        if line.startswith("diff --git ") or line.startswith("@@ "):
            if inside:
                close()
            removed, added = [], []
            inside = line.startswith("@@ ")
            hunks += line.startswith("@@ ")
        elif inside and line.startswith("+"):
            added.append(line[1:LINE_SCAN_CAP])
        elif inside and line.startswith("-"):
            removed.append(line[1:LINE_SCAN_CAP])
    if inside:
        close()
    return "hunks=%d dropped: %s" % (
        hunks, " ".join("%s=%d" % kv for kv in sorted(dropped.items())) or "none")

# This function parses Python source code and returns a sorted list of the visible class/function definitions, while intentionally excluding functions that are nested inside another function.
def visible_definitions(source: str) -> list[str]:
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

# This function builds a compact outline of all classes and functions in a Python source file, including nested ones, with their line ranges and indentation showing nesting depth.
def outline_source(source: str) -> list[str]:
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
FENCE_PATH = re.compile(
    r"`((?:/[A-Za-z0-9_./-]*|[A-Za-z0-9_.][A-Za-z0-9_./-]*)"
    r"\.(?:py|pyi|ts|tsx|js|jsx|go|rb|rs|java|sql))`")
FENCE_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)
FENCE_IMPORT_LINE = re.compile(r"^\s*(?:import|from)\s")
FENCE_FORBIDS = re.compile(
    r"\b(?:do not|don't|never|must not|should not|without)\b", re.I)
FENCE_SENTENCE = re.compile(r"(?<=[.;!?])\s+")

# This block extracts method/function names explicitly mentioned in a task statement, while ignoring sentences that contain prohibition language such as “do not” or “never”.
def stated_methods(statement: str) -> list[tuple[str, str]]:
    text = " ".join(part for part in FENCE_SENTENCE.split(statement or "")
                    if not FENCE_FORBIDS.search(part))
    qualified = [(holder, name) for holder, name in FENCE_METHOD.findall(text)]
    qualified += [(holder, name) for name, holder in FENCE_METHOD_OF.findall(text)]
    if qualified:
        return sorted(set(qualified))
    return sorted({("", name) for name in FENCE_CALL.findall(text)})

# This function normalizes a file path into a consistent repository-relative style, especially for paths coming from Git diffs.
def fence_path(path: str, root: str = "") -> str:
    path = path.replace("\\", "/")
    root = os.path.normpath(root).replace("\\", "/") if root else ""
    if root and path.startswith(root + "/"):
        path = path[len(root) + 1:]
    while path.startswith("./"):
        path = path[2:]
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return os.path.normpath(path).replace("\\", "/")

# This function tries to identify one specific file path explicitly mentioned in backticks in the task statement.
def stated_file(statement: str, root: str = "") -> str | None:
    paths = {fence_path(path, root)
             for path in FENCE_PATH.findall(statement or "")}
    return paths.pop() if len(paths) == 1 else None

# This function finds the line range of one specific method/function inside Python source code.
def method_bounds(source: str, holder: str, name: str) -> tuple[int, int] | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    found: list[tuple[int, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
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

# This function tries to identify exactly one method/function region in the source from a list of requested method names.
def fenced_region(source: str, wanted: list) -> tuple[str, int, int] | None:
    hits = []
    for holder, name in wanted:
        found = method_bounds(source, holder, name)
        if found:
            hits.append(("%s.%s" % (holder, name) if holder else name,
                         found[0], found[1]))
    return hits[0] if len(hits) == 1 else None

# This function checks whether the task statement explicitly says the work must be limited to a narrow scope, such as one method, one function, or one file.
def statement_bounded(statement: str) -> bool:
    text = " ".join((statement or "").split())
    if re.search(r"\b(?:one|single)\s+(?:method|function|file)\b|"
                 r"\bonly\s+(?:(?:edit|modify|change|repair|touch)\s+)?"
                 r"(?:(?:that|this|the|one|single)\s+)?(?:`[^`]+`\s+)?(?:method|function|file)\b|"
                 r"\buse only names the file already imports\b|"
                 r"\brest of (?:the|its) file unchanged\b|"
                 r"\bbounded to\b", text, re.I):
        return True
    return bool(re.search(
        r"\b(?:(?:edit|modify|change|repair)\s+only|only\s+(?:edit|modify|change|repair))\s+`[^`]+`|"
        r"\b(?:limit|restrict|confine)\s+(?:production\s+)?"
        r"(?:changes|edits|work)\s+to\s+`[^`]+`", text, re.I))

# This function reads the task statement and determines which Python AST constructs the task explicitly says must not be used.
def statement_refused_nodes(statement: str) -> set:
    text = " ".join((statement or "").split())
    refused = set()
    if re.search(r"\bonly names the file already imports\b", text, re.I):
        refused.update(("Import", "ImportFrom"))
    categories = (
        (r"loops?", ("For", "AsyncFor", "While")),
        (r"comprehensions?", ("ListComp", "SetComp", "DictComp", "GeneratorExp")),
        (r"lambdas?", ("Lambda",)),
        (r"exception handling", ("Try", "TryStar", "Raise")),
        (r"context managers?", ("With", "AsyncWith")),
    )
    item = r"(?:Python\s+)?(?:" + "|".join(label for label, _ in categories) + ")"
    separator = r"(?:\s*,\s*(?:(?:or|and)\s+)?|\s+(?:or|and)\s+)"
    lead = (r"(?:(?:^|[.!?;:])\s*(?:[-*]\s+|\d+[.)]\s+)?|\b(?:write|implement)\b[^.!?;:]*?\s+)"
            r"(?:no\s+|without\s+|(?:(?:it|the method|the function|you)\s+)?"
            r"must not\s+(?:use|contain|include|have)\s+)")
    tail = r"(?:\s+(?:inside (?:it|the method|the function)|in (?:it|the method|the function)))?\s*(?=[.!?;:]|$)"
    for match in re.finditer(lead + "(" + item + "(?:" + separator + item + r")*)\b" + tail,
                             text, re.I):
        for label, kinds in categories:
            if re.search(r"\b(?:" + label + r")\b", match[1], re.I):
                refused.update(kinds)
    return refused

# This function tries to determine the project language and its default test command from two sources:
# Commands explicitly mentioned in the task statement.
# Project marker files such as go.mod, Cargo.toml, pom.xml, package.json, etc.
def project_contract(root: str, statement: str = "", path: str = "") -> tuple[str, str]:
    declared_languages = set()
    for block in FENCED_BLOCK.findall(statement) + CODE_SPAN.findall(statement):
        try:
            argv = shlex.split(block)
        except ValueError:
            continue
        if not argv:
            continue
        tool = os.path.basename(argv[0])
        language = {"psql": "sql", "sqlite3": "sql", "mysql": "sql",
                    "go": "go", "node": "node", "npm": "node",
                    "cargo": "rust", "mvn": "java", "javac": "java",
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
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return (declared_languages.pop() if len(declared_languages) == 1 else "unknown"), ""

STATED_RUNNER_TOOLS = frozenset((
    "go", "cargo", "mvn", "gradle", "gradlew", "npm", "yarn", "pnpm",
    "make", "dotnet", "ctest", "rake", "bundle", "rspec", "phpunit"))
STATED_RUNNER_WORDS = frozenset(("test", "tests", "rspec", "phpunit", "ctest"))

# This block defines a conservative shell-command parser used to recognize test/build commands from text.
# Its philosophy is:
# If the command contains shell behavior that is difficult to interpret safely, return [] instead of guessing.
def stated_command_words(line: str) -> list:
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

# This function decides whether a parsed command argv is actually a test/check command for one of the recognized runner tools.
def stated_check(argv: list) -> bool:
    if not argv:
        return False
    tool = os.path.basename(argv[0])
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

# This function looks through code snippets in the task statement and returns the first command that it recognizes as a test/check runner command.
def stated_runner(statement: str) -> str:
    for block in (FENCED_BLOCK.findall(statement or "")
                  + CODE_SPAN.findall(statement or "")):
        for line in (block or "").split("\n"):
            argv = stated_command_words(line)
            if stated_check(argv):
                return " ".join(shlex.quote(word) for word in argv)
    return ""

# This function finds the line numbers occupied by native import statements in Go, Node/JavaScript, Java, or Rust source code.
def native_import_lines(source: str, language: str) -> set:
    starts = {
        "go": r'^\s*import\s+(?:\(|(?:[\w.]+\s+)?["`])',
        "node": r'^\s*import\s+(?!\()[\w*{\'\"]',
        "java": r'^\s*import\s+(?:static\s+)?[\w.]',
        "rust": r'^\s*(?:pub(?:\([^)]*\))?\s+)?use\s+[\w:{]',
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
            if not re.match(pattern, line) or not re.search(r'\b(?:import|use)\b', clean):
                continue
            active = True
        found.add(number)
        depth += sum(clean.count(c) for c in '({[') - sum(clean.count(c) for c in ')}]')
        if depth <= 0 and (language == "go" or ';' in clean or
                           (language == "node" and re.search(r'[\'\"][^\'\"]+[\'\"]\s*;?\s*$', line))):
            active, depth = False, 0
    return found

# This function returns the line numbers that belong to import statements. It supports Python directly via AST, and can delegate to the earlier native_import_lines() for non-Python languages.
def import_lines(source: str, language: str | None = None) -> set:
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

# This function parses Git diff hunk headers and returns the old/new line ranges in a structured numeric form.
def hunk_spans(patch: str) -> list[tuple[int, int, int, int]]:
    spans = []
    for old, old_count, new, new_count in FENCE_HUNK.findall(patch or ""):
        spans.append((int(old), 1 if old_count == "" else int(old_count),
                      int(new), 1 if new_count == "" else int(new_count)))
    return spans

# This helper checks whether a line range overlaps with at least one line number in a given set.
def touches(span: tuple[int, int], numbers: set) -> bool:
    start, count = span
    return any(number in numbers for number in range(start, start + count))

# This Warden class is basically the final submission gatekeeper for the agent.
# Its job is to answer:
# “Did the agent make a valid patch, stay inside the allowed scope, preserve the project structure/API, and avoid breaking tests?”
class Warden:
    def __init__(self, tree: Tree, pool: ShellPool, allowance: Allowance,
                 statement: str = "") -> None:
        self.tree = tree
        self.pool = pool
        self.allowance = allowance
        self.declared = declared_file(statement, tree.root)
        self.statement = statement
        self.native_runner = project_contract(tree.root)[1] if RIDGES_SCOPE_FOLLOWS_STATEMENT else ""
        self.runner = stated_runner(statement) if self.native_runner else ""
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

    def suite_command(self, root: str | None = None) -> str:
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
        return (
            "cd %s && PYTHONPATH=%s PYTHONHASHSEED=0 %s %s -m pytest -q "
            "--no-header --tb=line -rfE -p no:cacheprovider -o addopts= "
            "--continue-on-collection-errors -W ignore::DeprecationWarning%s%s"
            % (shlex.quote(root), shlex.quote(path), cache,
               shlex.quote(repo_python()), SUITE_TIERS[self.tier],
               (" " + where) if where else "")
        )

    def pristine(self) -> str | None:
        where = os.path.join(tempfile.mkdtemp(prefix="start"), "tree")
        if inside(where, self.tree.root):
            shutil.rmtree(os.path.dirname(where), ignore_errors=True)
            self.beacon.skipped("the only place for a separate checkout is "
                                "inside the tree being handed in")
            return None
        code, out = git(["worktree", "add", "--detach", where, self.tree.base or "HEAD"],
                        self.tree.root, 60)
        if code != 0:
            self.beacon.skipped("no separate checkout to read: %s" % out.strip()[:120])
            return None
        return where

    def arm(self) -> None:
        if not SUBMISSION_WARDEN:
            self.beacon.skipped("not switched on for this run")
            return
        self.beacon.reached(0, self.allowance.spent, self.allowance.clock_left())
        if self.native_runner and not self.runner:
            self.report_no_baseline()
            return
        try:
            self.where = self.pristine()
            self.scope = suite_scope(self.tree.root, self.declared) if SUITE_SCOPE else []
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
            say("[" + Beacon.tag("warden") + "] baseline scope: %s"
                % (", ".join(self.scope) if self.scope else "the whole repository"))
            if not self.scope:
                if not SUITE_SCOPE:
                    say("[" + Beacon.tag("warden") + "] wide baseline: not switched on for this run")
                elif self.declared is None:
                    say("[" + Beacon.tag("warden") + "] the statement gave: %s"
                        % declared_trace(self.statement, self.tree.root))
                else:
                    say("[" + Beacon.tag("warden") + "] wide baseline: nothing matched")
            self.job = self.pool.start(self.suite_command(self.where), pack_venv=False,
                                       hard_timeout=SUITE_BASELINE_SEC)
        except Exception as error:
            self.beacon.skipped("could not start the baseline reading: %s" % error)

    def escalate(self) -> bool:
        if RIDGES_SCOPE_FOLLOWS_STATEMENT and getattr(self, "native_runner", ""):
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
            self.job = self.pool.start(self.suite_command(self.where), pack_venv=False,
                                       hard_timeout=SUITE_BASELINE_SEC)
        except Exception as error:
            self.beacon.skipped("could not start the second reading: %s" % error)
            return False
        return True

    def stand_in(self, out: str) -> bool:
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
            self.job = self.pool.start(self.suite_command(self.where), pack_venv=False,
                                       hard_timeout=SUITE_BASELINE_SEC)
        except Exception as error:
            self.beacon.skipped("could not start the reading again: %s" % error)
            return False
        self.stood_in += 1
        self.shims.extend(made)
        self.records.extend(kept)
        return True

    def settle(self) -> None:
        for _ in range((1 + SUITE_SHIM_LIMIT) * len(SUITE_TIERS)):
            self.collect()
            if self.before is not None or self.job is None:
                return
            room = self.allowance.clock_left() - WARDEN_RELEASE_SEC
            window = min(SUITE_SETTLE_SEC, room)
            if window < 5.0:
                break
            self.job.wait(window)
            if not self.job.finished():
                break
        self.collect()

    def collect(self) -> None:
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
        done, out = self.job.wait(0.5)
        self.pool.jobs.pop(self.job.name, None)
        self.job = None
        reading = self.read_suite(out)
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
            green = reading[1] / float(reading[1] + len(reading[0]) or 1)
            if green < SUITE_SHIM_MIN_GREEN:
                self.counted = False
                self.beacon.fired(
                    "read over %d stand-in(s) and only %.0f%% of it passes; the "
                    "count is not evidence here, the names still are"
                    % (propped, 100 * green))
        self.before = reading
        self.armed_after = time.time() - self.started
        say("[" + Beacon.tag("warden") + "] the project's tests at the start: %d failing, %d passing "
            "(tier %d, scope %s, %.0fs)"
            % (len(self.before[0]), self.before[1], self.tier + 1,
               ",".join(self.scope) or "repository", self.armed_after))

    @staticmethod
    def tally(out: str) -> str:
        found = SUITE_TALLY.findall(out or "")
        return found[-1].strip() if found else "no tally"

    @staticmethod
    def reason(out: str) -> str:
        faults = SUITE_FAULT.findall(out or "")
        if faults:
            kinds = len({name for name, _ in faults})
            said = ("%s%s" % faults[0]).strip()[:140]
            return said if kinds == 1 else "%s (+%d other kinds)" % (said, kinds - 1)
        found = SUITE_REASON.findall(out or "")
        return found[0].strip()[:160] if found else "no reason given"

    @staticmethod
    def read_suite(out: str) -> tuple[set, int]:
        if RIDGES_SCOPE_FOLLOWS_STATEMENT:
            status = re.findall(r"^RIDGES_NATIVE_STATUS=(\d+)$", out or "", re.M)
            if status:
                return (set(), 1) if status[-1] == "0" else ({"native-suite"}, 0)
        counts = PASSED_COUNT.findall(out or "")
        return set(FAILED_TEST.findall(out or "")), int(counts[-1]) if counts else 0

    def report_no_baseline(self) -> None:
        if (RIDGES_SCOPE_FOLLOWS_STATEMENT and getattr(self, "before", None) is None
                and not getattr(self, "no_baseline_reported", False)):
            self.beacon.skipped("no baseline: watched=0")
            self.no_baseline_reported = True

    def suite_faults(self) -> list[str]:
        if self.before is None:
            self.report_no_baseline()
            return []
        room = self.allowance.clock_left() - WARDEN_RELEASE_SEC
        if room < 30.0:
            return []
        job = self.pool.start(self.suite_command(), pack_venv=False)
        done, out = job.wait(min(SUITE_RECHECK_SEC, room))
        self.pool.jobs.pop(job.name, None)
        if not done:
            job.stop()
            self.beacon.skipped("the project's tests did not finish in the time left")
            return []
        broke, passing = self.read_suite(out)
        fresh = self.confirm(sorted(broke - self.before[0]))
        if fresh:
            say("[" + Beacon.tag("warden") + "] the refusal %s the question"
                % ("carries" if WARDEN_ASK else "does not carry"))
            return ["The project's own tests were passing when this run started "
                    "and are failing now: %s. The task requires the project's tests to "
                    "keep passing, so this answer does not satisfy it as it "
                    "stands. Fix the behaviour rather than the test.%s"
                    % (", ".join(fresh[:6]), WARDEN_QUESTION if WARDEN_ASK else "")]
        if not passing:
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

    def confirm(self, names: list[str]) -> list[str]:
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
        if not (RIDGES_SCOPE_FOLLOWS_STATEMENT and getattr(self, "native_runner", "")):
            command = "%s %s" % (command, " ".join(shlex.quote(n) for n in scope))
        job = self.pool.start(command, pack_venv=False)
        done, out = job.wait(min(SUITE_RECHECK_SEC, room))
        self.pool.jobs.pop(job.name, None)
        if not done:
            job.stop()
            return asked
        again, count = self.read_suite(out)
        if not again and not count:
            return asked
        settled = [n for n in asked if n in again]
        if len(settled) != len(asked):
            self.beacon.fired("%d of %d only failed once and were let go"
                              % (len(asked) - len(settled), len(asked)))
        return settled

    def changed_paths(self) -> list[str]:
        code, out = git(["diff", "--name-only", self.tree.base or "HEAD"],
                        self.tree.root, self.read_room(30.0))
        paths = [p for p in out.splitlines() if p.strip()] if code == 0 else []
        return paths + sorted((self.tree._untracked() or set())
                              - self.tree.untracked_at_start)

    def read_room(self, want: float) -> float:
        until = getattr(self, "read_until", None)
        if until is None:
            return want
        left = until - time.monotonic()
        if left <= 0:
            raise ReadExpired("the shared reading window is spent")
        return min(want, left)

    def original(self, path: str) -> str | None:
        code, out = git(["show", "%s:%s" % (self.tree.base, path)],
                        self.tree.root, self.read_room(30.0))
        return out if code == 0 else None

    def contract_read(self) -> tuple[list, list]:
        self.contract_did_read = False
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
        if not hard and not soft:
            self.contract.skipped("%d changed file(s), %d read, nothing to say"
                                  % (len(changed), read))
            return [], []
        self.contract.fired("%d changed, %d read: %d refusing, %d listed: %s"
                            % (len(changed), read, len(hard), len(soft),
                               (hard or soft)[0][:160]))
        return hard[:2], soft[:3]

    def contract_one(self, path: str, hard: list, soft: list) -> int:
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
        faults: list[str] = []
        for path in self.changed_paths():
            if (TEST_PATH.search(path) and
                    (not RIDGES_SCOPE_FOLLOWS_STATEMENT or statement_bounded(self.statement))):
                faults.append(
                    "This run edited %s. The task is to change the code under "
                    "repair, not the tests that check it; editing a test file "
                    "or a conftest reads as tampering. Put it back exactly as "
                    "it was and make the source satisfy the test instead." % path)
                continue
            faults.extend(self.file_faults(path))
        return faults

    def file_faults(self, path: str) -> list[str]:
        before = self.original(path)
        if before is None:
            return []
        if not importable(path):
            return []
        carried = visible_definitions(before)
        try:
            after = self.tree.read(path)
        except ToolFault:
            if not carried:
                return []
            return ["This run deleted %s, and callers of %s lose it with the "
                    "file. Keep the file and the names in it." % (path, carried[0])]
        out: list[str] = []
        gone = sorted((collections.Counter(carried)
                       - collections.Counter(visible_definitions(after))).elements())
        if carried and gone:
            out.append(
                "These definitions were in %s when the run started and are not "
                "there now: %s. A refactor may not drop a name callers can "
                "use, and renaming or moving a method out of its class reads "
                "as dropping it. Keep the original name and delegate from it."
                % (path, ", ".join(gone[:6])))
        added = len(NOQA_DIRECTIVE.findall(after)) - len(NOQA_DIRECTIVE.findall(before))
        if added > 0:
            out.append(
                "This run added %d suppression comment(s) to %s. A suppression "
                "hides what a check reports without changing what it reports "
                "on, so the count may not go up." % (added, path))
        return out

    def change_shapes(self) -> list[tuple[str, str, str, str]]:
        rows: list[tuple[str, str, str, str]] = []
        code, out = git(["diff", "--raw", "-M", self.tree.base or "HEAD"],
                        self.tree.root, self.read_room(30.0))
        if code == 0:
            for line in out.splitlines():
                if not line.startswith(":") or "\t" not in line:
                    continue
                meta, _, tail = line.partition("\t")
                parts = meta[1:].split()
                if len(parts) < 5:
                    continue
                rows.append((parts[4][:1], parts[0], parts[1],
                             tail.split("\t")[-1]))
        room = self.read_room(30.0)
        if room < 1.0:
            raise ReadExpired("the shared reading window is spent")
        for path in sorted((self.tree._untracked(room) or set())
                           - self.tree.untracked_at_start):
            rows.append(("A", "000000", "100644", path))
        return rows

    def fenced_diff(self, path: str) -> str:
        code, out = git(["diff", "-U0", self.tree.base or "HEAD", "--", path],
                        self.tree.root, self.read_room(30.0))
        return out if code == 0 else ""

    def scope_faults(self, budgeted: bool = True) -> list[str]:
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
        if RIDGES_SCOPE_FOLLOWS_STATEMENT and not statement_bounded(self.statement):
            return []
        out: list[tuple[str, str]] = []
        moved = sorted({path for status, old, new, path in shapes
                        if status in ("A", "C", "D", "R")
                        or (old != new and "000000" not in (old, new))})
        if moved:
            out.append((
                "tree %s" % moved[0],
                "This change adds, deletes, renames or re-permissions %s. "
                "The task opened one file for editing; every other path in "
                "the project -- contents, mode and link target -- has to be "
                "exactly as it was before this run, so a path that appears, "
                "disappears or changes its bits makes the answer wrong "
                "however good the edit is. Put "
                "those back and keep the answer to edits inside files that "
                "were already there." % ", ".join(moved[:4])))
        paths = sorted({fence_path(path, self.tree.root)
                        for _, _, _, path in shapes})
        if named and (len(paths) != 1 or paths[0] != named):
            out.append((
                "files %d" % len(paths),
                "The statement opens one file for editing, %s, and this change "
                "touches %d: %s. Everything outside that file is compared "
                "against the original before the tests are run at all. Put the "
                "others back exactly as they were and keep the whole answer in "
                "the one file." % (named, len(paths), ", ".join(paths[:4]))))
        return out

    def line_faults(self, path: str, before: str, after: str,
                    spans: list, wanted: list) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        language = None
        if RIDGES_SCOPE_FOLLOWS_STATEMENT:
            language, _ = project_contract(self.tree.root, self.statement, path)
            if not statement_bounded(self.statement):
                language = "unknown"
            elif language == "unknown":
                language = "python"
        old_imports = import_lines(before, language)
        new_imports = import_lines(after, language)
        for old_start, old_count, new_start, new_count in spans:
            if (touches((old_start, old_count), old_imports)
                    or touches((new_start, new_count), new_imports)):
                out.append((
                    "import %s:%d" % (path, old_start),
                    "A hunk of the change to %s "
                    "edits an import statement. That door is closed from both "
                    "sides: every line of the file outside the method under "
                    "repair has to match the original character for character, "
                    "and the file still has to pass the project's linter. So "
                    "if your rewrite stopped using a "
                    "name the module imports, you can neither delete that "
                    "import nor mark it to be ignored -- both edits are outside "
                    "the method, and either one changes a line out there. The way "
                    "out is the method body: write it so it still uses the "
                    "names the file already imports. If this task truly cannot "
                    "be done with the names already there, say which name it "
                    "needs and why, then call submit again." % path))
                break
        region = fenced_region(before, wanted)
        if not region:
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
                "exactly once as %s in %s. The task asks for that same method "
                "in place, so renaming, removing, duplicating it or making it "
                "`async` is not the requested change. Restore its name, its "
                "`def` and its containing class."
                % (name, path))]
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
                "The statement limits the "
                "change to %s, which is lines %d-%d of %s as this run found "
                "it, and your diff reaches %s. Everything before that "
                "method's `def` line and everything after its last line has "
                "to match the original character for character -- a "
                "decorator, a blank line, an import or a comment out there "
                "makes the answer wrong even when the method itself is "
                "exactly right. Put those lines back the "
                "way they were and keep the whole change inside the method "
                "body." % (name, low, high, path, reach)))
        return out

    def ledger_faults(self) -> list[str]:
        if not LEDGER_READBACK:
            self.ledger.skipped("not switched on for this run")
            return []
        items = stated_requirements(self.statement)
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
        try:
            return reader()
        except BaseException as error:
            beacon.skipped("the answer could not be checked: %s"
                           % type(error).__name__)
            return empty

    def final_faults(self, window: float | None = None) -> tuple[int, list[str]]:
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
                  or self.suite_faults())
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
    pass

WORK_METER_WORDS = (
    "scale", "scales", "scaled", "scaling", "scalability", "grow", "grows",
    "growing", "growth", "row", "rows", "query", "queries", "statement",
    "statements", "memory", "index", "indexes", "indices", "indexed",
    "indexing", "scan", "scans", "scanning", "scanned", "prune", "prunes",
    "pruned", "pruning", "slow", "slower", "slowly", "slowdown", "n+1",
    "performance", "performant", "optimize", "optimized", "optimization",
    "optimise", "optimised", "optimisation",
)
WORK_METER_REQUEST = """

REQUIRED work measurement before handing in: use the repository's own fixtures
on its live database to measure the statements your changed code actually
executes, both with the original code and with your change. Preserve your patch
while obtaining the original-code baseline; restore your change afterwards.
For PostgreSQL, run EXPLAIN (ANALYZE, BUFFERS) on those statements with their
actual parameters, through the application's own database client and
configuration (for a Django project, manage.py shell with the repository's
settings; otherwise the client and settings the application itself uses) or
psql with the same connection settings. In Django you can use
connection.execute_wrapper to capture executed SQL and parameters while
exercising the changed path, then connection.cursor().execute('EXPLAIN
(ANALYZE, BUFFERS) ' + sql, params). Count Rows Removed by Filter as rows
visited, not just rows returned, and account for loops and separate statements.
For ClickHouse, first run the repository's own test that exercises the
changed query, then read system.query_log for that query's QueryFinish entry:
SELECT query_id, query, read_rows, memory_usage, result_rows FROM system.query_log
WHERE type = 'QueryFinish' ORDER BY event_time_microseconds DESC LIMIT 30.
Use the repository's configured client or its environment connection settings;
for an HTTP client, send SQL by POST to the configured host and port with its
configured credentials. Allow the query log to flush (SYSTEM FLUSH LOGS if
permitted, otherwise wait and reread), and match the actual query, excluding
the query_log read itself. Do this before and after, using the same fixtures
and request, and compare narrower and wider requests where size is relevant.
Report the before/after numbers and commands. If the quantity still follows
the table size, or is not below the baseline, keep working.
"""

# This block adds a performance/scalability trigger to the system.
# Its purpose is roughly:
# If the task talks about scaling, queries, indexes, scans, performance, N+1, etc., detect that and require real before/after database measurements before accepting the solution.
def work_meter_hits(statement: str) -> list[str]:
    return [word for word in WORK_METER_WORDS
            if re.search(r"(?<![\w])" + re.escape(word) + r"(?![\w])",
                         statement or "", re.I)]

# WorkMeter is a small controller that watches whether a performance-related task actually performs the required before/after measurement.
class WorkMeter:
    def __init__(self):
        self.beacon = Beacon("meter")
        self.state = "armed"
        self.command = ""
        self.before = None
        self.calls = self.spent = 0

    def extend(self, kit, note):
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
                hits = work_meter_hits(statement_of(kit.warden))
                if not hits:
                    reason = "no vocabulary hit"
                elif not note or not kit.selfreview_asked:
                    share = kit.selfreview_share()
                    reason = ("share used" if share is None or
                              share > SELFREVIEW_MAX_SHARE else "pauses spent")
            if reason:
                self.beacon.skipped(reason)
                return note
            self.before = kit.tree.diff(SELFREVIEW_DIFF_SEC)
            self.calls, self.spent = kit.allowance.calls, kit.allowance.spent
            result = note + WORK_METER_REQUEST
            self.beacon.fired(", ".join(hits))
            self.state = "asked"
            return result
        except BaseException as error:
            self.skip_error(error)
            return note

    def skip_error(self, error):
        try:
            self.beacon.skipped("reading ended on %s" % type(error).__name__)
        except BaseException:
            pass

    def record(self, command):
        try:
            if self.state == "asked" and not self.command and re.search(
                    r"\bEXPLAIN\b|\bsystem\.query_log\b", command, re.I):
                self.command = " ".join(command.split())
        except BaseException as error:
            self.skip_error(error)

    def close(self, kit):
        if self.state != "asked":
            return
        self.state = "done"
        try:
            import hashlib
            say("[%s] measured: %s %s" % (
                self.beacon.slug, "yes" if self.command else "no",
                self.command or "(no command seen)"))
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

# Kit is the toolbox and workflow controller used by the coding agent.
# If Warden is the gatekeeper that judges the final patch, Kit is the object the agent actually uses while working:
class Kit:
    def __init__(self, tree: Tree, pool: ShellPool, allowance: Allowance,
                 warden: Warden | None = None, label: str = "",
                 findings: "FindingMap | None" = None) -> None:
        self.tree = tree
        self.pool = pool
        self.allowance = allowance
        self.warden = warden
        self.findings = findings
        self.seen: dict[str, int] = {}
        self.edit_all = Beacon("editall")
        self.bg = Beacon("bgshell")
        self.conform = Beacon("conform")
        self.fence = Beacon("fence")
        self.label = label
        self.conform_state = "armed" if SUBMIT_CONFORM else "off"
        self.conform_edits = 0
        self.scan_rows: list[str] = []
        self.ledger_state = "off"
        self.ledger = Beacon("scanledger")
        self.untouched_state = "off"
        self.untouched = Beacon("untouched")
        self.scan_file = ""
        self.work_meter = WorkMeter()
        self.selfreview_state = "armed" if HIDDEN_SELFREVIEW else "off"
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
        self.pauses = 0

    def note_findings(self, command: str, out: str) -> None:
        if self.findings is None:
            return
        try:
            self.findings.observe(command, out)
        except BaseException:
            self.findings.beacon.skipped("the output could not be read")

    def note_read(self, what: str) -> None:
        say("[READ]%s %s" % (" " + self.label if self.label else "", what))

    def run(self, name: str, args: dict) -> str:
        handler = getattr(self, "do_" + name, None)
        if handler is None:
            raise ToolFault("no tool named %s" % name)
        return handler(args)

    def guard_repeat(self, key: str) -> None:
        self.seen[key] = self.seen.get(key, 0) + 1
        if self.seen[key] > REPEAT_READ_CEILING:
            raise ToolFault(
                "this exact call was already answered %d times and nothing has changed "
                "since. Scroll up and use the earlier result." % (self.seen[key] - 1)
            )

    def do_read_file(self, args: dict) -> str:
        path = str(args.get("path") or "")
        start = args.get("start")
        count = args.get("count")
        self.guard_repeat("read:%s:%s:%s" % (path, start, count))
        text = self.tree.read(path)
        lines = text.splitlines()
        first = max(1, int(start or 1))
        wanted = int(count) if count and int(count) > 0 else 0
        asked = min(len(lines), first + wanted - 1) if wanted else len(lines)
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
        except subprocess.TimeoutExpired:
            raise ToolFault("search timed out; narrow the pattern or the path")
        out = done.stdout or ""
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
        served = clip(out, SEARCH_OUTPUT_CAP, "matches")
        self.note_read("search_text %r %s -> %dc" % (pattern, mode, len(served)))
        return served

    def do_find_files(self, args: dict) -> str:
        import fnmatch
        pattern = str(args.get("pattern") or "*")
        self.guard_repeat("glob:%s" % pattern)
        code, out = git(["ls-files"], self.tree.root, 30)
        if code != 0:
            raise ToolFault("could not list tracked files")
        hits = [p for p in out.splitlines() if fnmatch.fnmatch(p, pattern)]
        if not hits:
            loose = pattern if pattern.startswith("*") else "*" + pattern
            hits = [p for p in out.splitlines() if fnmatch.fnmatch(p, loose)]
        if not hits:
            return "no tracked file matches %s" % pattern
        return clip("\n".join(hits[:400]), SEARCH_OUTPUT_CAP, "paths")

    def do_edit(self, args: dict) -> str:
        path = str(args.get("path") or "")
        old = str(args.get("old") or "")
        new = str(args.get("new") or "")
        every = bool(args.get("replace_all")) and REPLACE_ALL
        if not old:
            raise ToolFault("old must not be empty; use create_file to write a whole file")
        text = self.tree.read(path)
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
        self.tree.write(path, updated)
        self.allowance.edits += 1
        if before:
            self.edit_all.outcome(before, updated)
        note = self.compile_check(path)
        return "edited %s (%d occurrence%s)%s" % (path, hits if every else 1, "" if hits == 1 else "s", note)

    def do_create_file(self, args: dict) -> str:
        path = str(args.get("path") or "")
        self.tree.write(path, str(args.get("content") or ""))
        self.allowance.edits += 1
        return "wrote %s%s" % (path, self.compile_check(path))

    def compile_check(self, path: str) -> str:
        argv = SYNTAX_CHECKS.get(os.path.splitext(path)[1].lower())
        if not argv or not shutil.which(argv[0]):
            return ""
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
            return ""
        if done.returncode == 0:
            return ""
        return "\n\nWARNING: the file no longer parses:\n" + clip(done.stderr or "", 1200, "error")

    def do_bash(self, args: dict) -> str:
        command = str(args.get("command") or "")
        if not command.strip():
            raise ToolFault("command must not be empty")
        outward = NETWORK_COMMAND.search(command) if NETWORK_FENCE else None
        if outward:
            self.fence.fired("refused %r" % outward.group(1))
            raise ToolFault(
                "%s is not available here. Everything this task needs is "
                "already in the tree: the repository, its history, and its "
                "tests. Nothing outside it is reachable, and the change you "
                "are asked to make is not published anywhere -- work from the "
                "code in front of you." % outward.group(1)
            )
        blocked = HISTORY_GIT.search(command)
        if blocked:
            raise ToolFault(
                "git %s is not available here. Your changes are collected from the working "
                "tree as it stands, so moving or discarding them loses the work. Read-only "
                "git (status, diff, log, grep, show, ls-files) is fine." % blocked.group(1)
            )
        want_bg = bool(args.get("background")) and ASYNC_SHELL
        asked = float(args.get("timeout") or 120)
        room = self.allowance.clock_left() - FINISH_BUDGET_SEC
        if room < 5.0:
            raise ToolFault(
                "not enough of the run left to wait on a command; make the "
                "change you already have evidence for, or submit")
        budget = max(5.0, min(asked, SHELL_BUDGET_CEILING_SEC, room))
        if want_bg:
            job = self.pool.start(command, hard_timeout=room)
            self.bg.fired("started %s: %s" % (job.name, command[:120]))
            return "started in the background as %s; collect it with bash_poll" % job.name
        job = self.pool.start(command, hard_timeout=room)
        done, out = job.wait(budget)
        if done:
            self.pool.jobs.pop(job.name, None)
            report_shell(job, out)
            self.consult_record(command, out)
            self.note_findings(command, out)
            return clip(out, SHELL_OUTPUT_CAP, "shell output") or "(no output)"
        self.bg.fired("kept %s alive past %.0fs: %s" % (job.name, budget, command[:120]))
        return (
            clip(out, SHELL_OUTPUT_CAP, "partial output")
            + "\n\n[still running after %.0fs, moved to the background as %s; "
            "keep working and collect it later with bash_poll]" % (budget, job.name)
        )

    def do_bash_poll(self, args: dict) -> str:
        job = self.pool.get(str(args.get("job") or ""))
        out = job.drain()
        if not out.endswith(STILL_RUNNING):
            self.pool.jobs.pop(job.name, None)
            report_shell(job, out)
            self.consult_record(job.command, out)
            self.note_findings(job.command, out)
        return clip(out, SHELL_OUTPUT_CAP, "shell output") or "(no output)"

    def do_submit(self, args: dict) -> str:
        faults = self.warden.verdict() if self.warden else []
        if not faults:
            if self.warden is None or not self.warden.stood_down:
                note = self.handin_pause()
                if note:
                    return note
            else:
                self.selfreview_close("the answer was released unchecked")
            raise Finished(str(args.get("summary") or ""))
        return ("Not handed in. The task states conditions this run can "
                "check itself, and they are not met yet:\n\n"
                + "\n\n".join(faults[:3])
                + "\n\nThere is budget left. Fix this and call submit again.")

    def handin_pause(self) -> str | None:
        self.work_meter.close(self)
        note = self._handin_pause()
        return self.work_meter.extend(self, note)

    def _handin_pause(self) -> str | None:
        if not HIDDEN_SELFREVIEW:
            return (self.selfreview_note() or self.untouched_note()
                    or self.ledger_note() or self.conform_note())
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

    def consult_record(self, command: str, out: str) -> None:
        self.work_meter.record(command)
        if self.consult_state != "armed":
            return
        tail = ""
        for line in reversed((out or "").splitlines()):
            if line.strip() and line.strip() != STILL_RUNNING:
                tail = line.strip()
                break
        self.consult_log.append("$ %s\n    %s"
                                % (" ".join((command or "").split())[:SHELL_REPORT_CAP],
                                   tail[:SHELL_REPORT_CAP]))
        del self.consult_log[:-CONSULT_HISTORY_LINES]

    def consult_ask(self, statement: str, diff: str) -> str:
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

    def selfreview_note(self) -> str | None:
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
        if self.allowance.clock_left() < CONFORM_MIN_WALL_SEC:
            self.selfreview_state = "done"
            self.selfreview.skipped("too little of the run left to act on the answer")
            return None
        used = self.selfreview_share()
        if used is None:
            self.selfreview_state = "done"
            self.selfreview.skipped("the allowance does not say how much is gone")
            return None
        if used > SELFREVIEW_MAX_SHARE:
            self.selfreview_state = "done"
            self.selfreview.skipped("the run is past the early share of its allowance")
            return None
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
            "Not handed in yet - one review before it goes, and it happens only "
            "once. What this has to satisfy is wider than the behaviour the "
            "problem statement describes: work of this kind is also held to "
            "properties the statement never names, and a change that returns "
            "the right answer for every input can still be wrong on one of "
            "them.\n\n"
            "List the ones that could apply to what you changed. Among them: "
            "how much work is done to produce the answer, and whether that "
            "grows with the size of what is asked for; how many separate round "
            "trips it takes; whether the result is computed when it is asked "
            "for or when it is defined; whether it goes through the structures "
            "that already exist to make a lookup cheap, or past them; the order "
            "of what comes back; which end points of a range are included.\n\n"
            "For each one, either name the command you ran that shows it holds "
            "and what its output said, or say in one line why that property "
            "does not apply to this change. Reading your own edit is not a "
            "check: a property nothing was run against is unverified. Anything "
            "left unverified that you can settle from here, settle it now, and "
            "fix what the result shows. Then call submit again."
        ), diff)

    def selfreview_outcome(self) -> None:
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
        try:
            budget = self.allowance.deadline - self.allowance.started
            if not finite_number(budget) or budget <= 0:
                return None
            return max(0.0, self.allowance.elapsed()) / budget
        except BaseException:
            return None

    def untouched_note(self) -> str | None:
        if self.untouched_state != "armed" or not self.scan_rows:
            return None
        if self.allowance.clock_left() < CONFORM_MIN_WALL_SEC:
            self.untouched_state = "done"
            self.untouched.skipped("too little of the run left to act on the answer")
            return None
        self.untouched_state = "done"
        try:
            with open(os.path.join(self.tree.root, self.scan_file), "r",
                      encoding="utf-8", errors="replace") as handle:
                source = handle.read(MUTATION_SCAN_MAX_CHARS)
        except OSError:
            self.untouched.skipped("the declared file could not be read back")
            return None
        carried = carried_rows(source, self.scan_rows)
        if not carried:
            self.untouched.fired("no flagged expression survives in the file")
            return None
        self.untouched.fired("%d of %d flagged expression(s) carried through: %s"
                             % (len(carried), len(self.scan_rows),
                                "; ".join(r.split(" --")[0][:40] for r in carried[:3])))
        return ("Not handed in yet - one reading of the file before it goes, and "
                "it happens only once. These flagged expressions are still in the "
                "file, character for character, so they were carried through your "
                "rewrite rather than settled -- whatever you concluded about them "
                "is not what the tree now says:\n  - %s"
                "\n\nThat is a fact about your diff, not a claim that any of them "
                "is wrong. For each, either change it or state what in the problem "
                "statement makes it correct exactly as written. Then call submit "
                "again." % "\n  - ".join(carried))

    def ledger_note(self) -> str | None:
        if self.ledger_state == "asked":
            self.ledger_state = "done"
            self.ledger.fired("resubmitted after %d further edit(s)"
                              % (self.allowance.edits - self.ledger_edits))
            return None
        if self.ledger_state != "armed" or not self.scan_rows:
            return None
        if self.allowance.clock_left() < CONFORM_MIN_WALL_SEC:
            self.ledger_state = "done"
            self.ledger.skipped("too little of the run left to act on the answer")
            return None
        self.ledger_state = "asked"
        self.ledger_edits = self.allowance.edits
        self.ledger.fired("hand-in paused for %d row(s)" % len(self.scan_rows))
        return ("Not handed in yet - one check before it goes, and it happens only "
                "once. The opening listed places in this file where an injected "
                "reversal could sit. Take them ONE AT A TIME, in this order, and "
                "do not answer two of them with one look: each line is its own "
                "question, and an answer read off a neighbouring line is not an "
                "answer. For each, say either which edit of yours settles it, or "
                "what in the statement makes the line correct as it stands.\n  - %s"
                "\n\nAny row you cannot settle from the statement is a row to go "
                "back and read the code for. Then call submit again."
                % "\n  - ".join(self.scan_rows))

    def conform_note(self) -> str | None:
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

# This block defines two regexes, a set of common words to ignore, and a helper that breaks dotted quoted names into useful search terms.
def quoted_parts(name: str, called: bool = False) -> list[str]:
    parts = name.split(".")
    if len(parts) == 1:
        return parts
    segments = [p for p in parts if p and p not in COMMON_WORDS]
    return segments if called else [name] + segments

# This function sorts search terms into a deliberate priority order.
def sweep_order(terms: set, quoted: set) -> list[str]:
    return sorted(terms, key=lambda t: (t not in quoted, -len(t), t))

# candidate_files() tries to guess which repository files are most likely relevant to the task statement.
# It does this by extracting useful words/identifiers from the statement, searching the Git-tracked repository for those terms, scoring matching files, and returning the top candidates.
def candidate_files(tree: Tree, statement: str, beacon: Beacon,
                    limit: int = 12) -> list[str]:
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

# repo_sketch() builds a small textual summary of the repository structure.
# It does not inspect file contents deeply. It answers questions like:
# How many tracked files are there? Which top-level folders are biggest? Which file extensions dominate? Which project/config marker files exist?
def repo_sketch(tree: Tree) -> str:
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

# BRIEF is the main system instruction given to the coding agent. It defines how the agent should work inside the repository, how it should spend model turns, how it should investigate defects, and what must happen before submission.
# It is essentially the agent’s operating manual.
BRIEF = """You are fixing a defect in a checked-out repository. You have shell access
and file tools. When you are done, the working tree is the answer: your changes are read
straight off it, so leave the fix in place and call submit.

Work inside the repository as it is. Do not add dependencies and do not rewrite
unrelated code. Fix the cause, never silence the check: suppressing a warning,
deleting an assertion, loosening a test or special-casing the checker is not a fix,
and a change that only makes the report go quiet leaves the defect in place.

CRITICAL - spend turns carefully. Every reply costs one exchange with the model, and
exchanges are the scarcest thing you have. Put every tool call that does not depend on another one into the
SAME reply. Reading four files is four calls in one reply, not four replies. Searching for
three patterns is three calls in one reply. Only wait for a result when the next thing you
do genuinely depends on it.

Do not sit idle while a slow command runs. Start a test suite with background=true, keep
reading code, and collect it with bash_poll when you need the answer.

HOW TO READ THE PROBLEM

1. If the problem names a value, a flag or an attribute, search for the symbol whose name
   literally contains that word. Do not substitute a concept you inferred from context.
   Near-identical names often live side by side, and the one that is read may not be the
   one that is written; check which of them the problem is actually about.

2. If the problem forbids a side effect, remember that a high-level call such as refresh()
   or reload() often re-runs the forbidden step because this class overrides it. Read the
   subclass override before you decide where the fix goes.

3. Before you change a call site, ask whether the honest fix belongs one level down, in the
   base class or the root method. If you move behaviour, delete the old path: a leftover
   still runs, still mutates counters and still shifts control flow.

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
task had repeated sites. Then read your own diff and ask whether it changes anything the
problem did not ask for. Call submit with a one-line summary of what you changed."""

# This function removes the entire “WHEN THE SAME DEFECT…” section from the BRIEF string.
# The intent is:
# Keep the normal coding instructions, but remove the special workflow for repeated/multi-site defects.
def without_sweep(brief: str) -> str:
    start = brief.find("WHEN THE SAME DEFECT")
    end = brief.find("BEFORE YOU SUBMIT")
    return brief[:start] + brief[end:] if 0 <= start < end else brief

# This block defines three pieces of prompt policy text used to guide the coding agent when it is restructuring existing working code.
# The main idea is:
# Refactoring is allowed, but behavior preservation is strict. However, if the existing code itself may already contain a defect, the system allows carefully repairing that defect instead of blindly preserving it.
RESTRUCTURE = """WHEN THE TASK ASKS YOU TO RESTRUCTURE WORKING CODE

The behaviours the statement asks to change are the only behaviours that may change.
Every other visible behaviour must survive your edit, whether or not any test names
it. Structure is yours: add helpers, adjust call sites, fix at the layer the defect
actually lives - but through all of it the code keeps doing what it did.

1. Restructure by MOVING lines, never by re-deriving them from memory. A block lifted
   into a helper keeps its text apart from what the move itself requires - a name now
   qualified, a value now passed in: the same names, the same exception tuples, the
   same operators and boundaries, the same short-circuits, the same order of cases.
   If two places handled failure differently before, they still differ after; do not
   merge branches that were merely similar.
2. Where the statement spells out a specific detail - an ordering, a boundary, a
   default, which failures are tolerated - that detail is a requirement, not colour.
   Implement each one literally and at full strength.
3. Before you finish, read your own diff hunk by hunk. Every changed line should be a
   move, a mechanical consequence of one (a name now qualified, a parameter now
   passed), or a change the statement asked for. A line that is none of these is a
   defect you introduced."""
TAINT_JOIN = "but through all of it the code keeps doing what it did."
TAINT_TEXT = """

That premise - that the file does what it is supposed to do - is not always
true here. The file may have been altered before it reached you, and the check
you are asked to clear can be reporting the alteration. A line does not become
correct by already being in the tree, and preserving it is not fidelity; it is
copying a defect forward.

Where that happens, rule 1 below does not apply to the altered line: moving it
into a helper carries the defect with it, and a line you repair here is repaired
from what the file itself shows, not re-derived wholesale from memory.

So as you read the region the check names, look for the marks an alteration
leaves. Each is a place to check, not a defect: only when the statement or an
existing test confirms the current line is wrong is changing it a repair:
   - a comment or docstring that says one thing while the line beneath it does
     another: an order, a boundary, a default, which errors escape.
   - a name that disagrees with what the line does - a branch named for one
     direction that calls the helper for the other.
   - a guard placed where it cannot be right: ahead of a case it was never
     meant to cover, so that case can no longer run at all.
   - a comparison against a length or an index that admits one value too many.
   - a condition that was plainly narrower once: a test for a particular prefix,
     scheme or shape that is gone while the branch it protected is still there.

Repair only what you find inside the region the check names, and say in your
submit summary which of these you changed. Everything the file gets right still
has to survive: this licenses reading a line as wrong, not rewriting the file to
taste."""
FOLD_JOIN = "merge branches that were merely similar."
FOLD_PAIRS = """ A condition you rewrite must still tell
   apart everything the original told apart. These are the pairs that get folded
   together, each of which has broken a working module:
   - a key absent from a mapping, and a key present with a falsy value:
     `d.get(k)` is not `k in d`, and dropping keys the caller did not mention is
     not the same as keeping them.
   - `None` meaning not-given, and a value that is empty, zero or False:
     `if x is None` is not `if not x`. If `None` cleared something and omitting
     it left it alone, that stays true.
   - an empty selection and a selection of everything. Replacing nothing is not
     replacing all.
   - an absent pattern and one that matches everything. `None` is not `*`.
   - a path that stops one level short of a target, and one that reaches it.
   - which of two shapes an ambiguous name is read as.
   - the order results come back in.
   - the spelling a value carries: one base against another, bytes against text.
   - a failure that was visible to the caller, and one that is swallowed. If a
     path raised, logged, or ended a session before, it still does; a `try` you
     widen or an error you turn into a default is a behaviour you removed."""

# compose_brief() builds the final instruction prompt that will be given to the coding agent.
# It starts from the base BRIEF, optionally removes the repeated-defect sweep section, optionally injects the restructuring rules, optionally injects the fold-preservation rules, optionally injects the tainted-baseline rules, logs what happened, and returns the final prompt string.
def compose_brief(statement: str, root: str) -> str:
    text = BRIEF if SWEEP_WORKFLOW else without_sweep(BRIEF)
    joins = spliced = tainted = 0
    taint = "2"
    if MOVE_VERBATIM:
        section = RESTRUCTURE
        joins = section.count(FOLD_JOIN)
        if FOLD_ANCHORS and joins == 1:
            section = section.replace(FOLD_JOIN, FOLD_JOIN + FOLD_PAIRS, 1)
            spliced = 1
        if not TAINTED_BASELINE:
            taint = "1"
        elif not command_lines(statement):
            taint = "3"
        elif not declared_file(statement, root):
            taint = "5"
        elif section.count(TAINT_JOIN) != 1:
            taint = "4:%d" % section.count(TAINT_JOIN)
        else:
            section = section.replace(TAINT_JOIN, TAINT_JOIN + TAINT_TEXT, 1)
            tainted, taint = 1, "0"
        marker = "BEFORE YOU SUBMIT"
        text = (text.replace(marker, section + "\n\n" + marker, 1)
                if marker in text else text + "\n\n" + section)
    say("[FOLD] anchors=%d verbatim=%d joins=%d spliced=%d tainted=%d taint=%s"
        % (FOLD_ANCHORS, MOVE_VERBATIM, joins, spliced, tainted, taint))
    return text

# opening_message() builds the initial message shown to the coding agent before it starts working.
# It combines three things:
# The problem statement.
# A compact repository summary.
# Optional candidate-file hints.
def opening_message(statement: str, tree: Tree, hints: list[str]) -> str:
    blocks = ["Problem to fix:\n\n" + statement.strip(), "\nRepository at a glance:\n" + repo_sketch(tree)]
    if hints:
        blocks.append(
            "\nFiles whose contents overlap the rare terms in the problem, most overlap first. "
            "This is a starting point produced by text matching, not an answer:\n"
            + "\n".join("  " + p for p in hints)
        )
    return "\n".join(blocks)

# shrink_transcript() reduces the size of an existing chat transcript when its stored message content exceeds a configured limit.
# Its strategy is deliberately conservative:
# Keep the beginning and recent messages, but replace large older tool outputs with short placeholders until the transcript becomes much smaller.
def shrink_transcript(messages: list[dict], cap: int, beacon: Beacon) -> bool:
    total = sum(len(str(m.get("content") or "")) for m in messages)
    if total <= cap:
        return False
    beacon.fired("transcript %dB over cap %dB" % (total, cap))
    freed = 0
    for message in messages[2 : max(2, len(messages) - 12)]:
        if message.get("role") != "tool":
            continue
        body = str(message.get("content") or "")
        if len(body) <= 400:
            continue
        message["content"] = "[%d characters of earlier tool output dropped to fit the context]" % len(body)
        freed += len(body)
        total -= len(body)
        if total <= cap * 0.7:
            break
    if not freed:
        beacon.skipped("nothing bulky enough to drop")
        return False
    say("[" + Beacon.tag("trim") + "] freed %dB, transcript now ~%dB" % (freed, total))
    return True

# This block defines a read-only planning mode for a second agent.
# Its purpose is:
# If the main coding agent has spent several turns without making progress, launch a planning-only agent that can inspect the repository, figure out where the fix belongs, and hand back a concrete implementation plan.

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

# plan_tools() builds the exact tool list that the read-only planning agent is allowed to use.
def plan_tools() -> list[dict]:
    return [s for s in TOOL_SCHEMAS
            if s["function"]["name"] in PLAN_TOOL_NAMES] + PLAN_EXTRA_TOOLS

# run_plan() launches the read-only planning agent you just examined, lets it investigate the repository for a limited number of turns and tool-output characters, collects its latest set_plan() note, records telemetry, and returns that plan to the main coding workflow.
def run_plan(statement: str, tree: Tree, pool: ShellPool, allowance: Allowance,
             beacon: Beacon, turn: int) -> str:
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
    try:
        for step in range(1, PLAN_TURN_CAP + 1):
            if read >= PLAN_READ_BUDGET:
                stop = "budget"
                break
            if allowance.spent >= ceiling or allowance.money_left() <= 0:
                stop = "spend"
                break
            reply = seat.ask(messages, plan_tools())
            raw = reply.get("tool_calls")
            calls = [c for c in raw if isinstance(c, dict)] if isinstance(raw, list) else []
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

# drive() is the main agent loop. It creates the coding model, starts the Warden, prepares the repository context, optionally scans for likely files and suspicious code shapes, then repeatedly lets the model call tools until it submits, runs out of budget, or hits the turn limit.
def drive(statement: str, tree: Tree, pool: ShellPool, allowance: Allowance,
          findings: "FindingMap | None" = None) -> None:
    seat = Seat(allowance)
    warden = Warden(tree, pool, allowance, statement)
    warden.arm()
    kit = Kit(tree, pool, allowance, warden, findings=findings)
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
    verbatim = Beacon("verbatim")
    verbatim.reached(0, allowance.spent, allowance.clock_left())
    if MOVE_VERBATIM:
        verbatim.fired("clause present")
    else:
        verbatim.skipped("not switched on for this run")
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
    scan = Beacon("scan")
    scan.reached(0, allowance.spent, allowance.clock_left())
    opening = opening_message(statement, tree, hints)
    if not (SCAN_ORDER or SCAN_ABSENT or DEAD_CONJUNCT):
        scan.skipped("no shape switched on for this run")
    else:
        declared = declared_file(statement, tree.root)
        rows = mutation_suspects(os.path.join(tree.root, declared), statement) if declared else []
        if not rows:
            scan.skipped("no declared file" if not declared
                         else "no reversal-prone shape in %s" % declared)
        else:
            opening += ("\n\nInjected reversals hide in a small number of shapes, and "
                        "these are where they could sit in %s. They are separate "
                        "places, and settling one says nothing about the others: "
                        "account for every line below before you hand in -- either "
                        "you changed it, or you can say from the statement why it is "
                        "already right. None of them is a defect on sight; a correct "
                        "file carries these shapes too.\n  - %s"
                        % (declared, "\n  - ".join(rows[:8])))
            scan.fired("%d suspect(s) in %s: %s" % (len(rows), declared,
                       "; ".join(r.split(" --")[0] for r in rows[:4])))
            if SCAN_LEDGER or UNTOUCHED_ROWS:
                kit.scan_rows = list(rows[:8])
                kit.ledger_state = "armed" if SCAN_LEDGER else "off"
                kit.untouched_state = "armed" if UNTOUCHED_ROWS else "off"
                kit.scan_file = declared
    messages: list[dict] = [
        {"role": "system", "content": compose_brief(statement, tree.root)},
        {"role": "user", "content": opening},
    ]
    cap = transcript_cap_chars(seat.current(), allowance.ceiling_usd)
    say("[LOOP] transcript cap set for %s" % seat.current())
    blanks = 0
    ending = "the loop reached its last turn"
    try:
        pressed = 0
        wrapped_up = False
        for turn in range(1, TURN_CEILING + 1):
            warden.collect()
            halt = allowance.halt_reason()
            if halt:
                say("[LOOP] stopping on %s at turn %d" % (halt, turn))
                ending = "the run stopped on %s" % halt
                return
            if (allowance.edits == 0 and turn > FIRST_EDIT_DEADLINE_TURN
                    and pressed < EDIT_PRESSES_MAX):
                pressed += 1
                say("[LOOP] %d turns without an edit; pressing for one (#%d)"
                    % (turn - 1, pressed))
                press = (
                    "No edit yet. Narrow down and change something now; "
                    "an imperfect fix in the tree beats a perfect one you never wrote. "
                    "Reading more before the first edit cannot help: nothing you have "
                    "learned so far is in the answer until it is in the file. Do not "
                    "submit while the tree is unchanged."
                )
                if pressed == 1:
                    note = run_plan(statement, tree, pool, allowance, plan, turn)
                    if note:
                        press += (
                            "\n\nA second agent read the repository and left this note. "
                            "It did not run anything and may be wrong -- check it against "
                            "the file before you act on it.\n\n" + note
                        )
                messages.append({"role": "user", "content": press})
            if TRANSCRIPT_CAP:
                shrink_transcript(messages, cap, trim)
            answering = seat.current()
            reply = seat.ask(messages, TOOL_SCHEMAS)
            if seat.current() != answering:
                cap = transcript_cap_chars(seat.current(), allowance.ceiling_usd)
                say("[LOOP] seat changed under us; transcript cap set for %s"
                    % seat.current())
            raw = reply.get("tool_calls")
            calls = [c for c in raw if isinstance(c, dict)] if isinstance(raw, list) else []
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
                blanks += 1
                if not text.strip() and blanks >= BLANK_REPLY_CEILING:
                    if seat.retire(seat.current()):
                        say("[LOOP] %d blank replies; changed seats" % blanks)
                        cap = transcript_cap_chars(seat.current(), allowance.ceiling_usd)
                        say("[LOOP] transcript cap set for %s"
                            % seat.current())
                        blanks = 0
                        continue
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
                    try:
                        result = kit.run(name, args)
                    except Finished as done:
                        say("[LOOP] submit at turn %d: %s" % (turn, str(done)[:200]))
                        ending = "the answer was handed in"
                        return
                    except ToolFault as fault:
                        result = "error: %s" % fault
                    except Exception as error:
                        result = "error: %s: %s" % (type(error).__name__, error)
                messages.append(
                    {"role": "tool", "tool_call_id": call_ident(call, index),
                     "content": clip(str(result), READ_OUTPUT_CAP)}
                )
            if (not wrapped_up
                    and (turn >= WRAPUP_TURN
                         or allowance.money_left() < allowance.soft_usd * 0.15)):
                wrapped_up = True
                messages.append(
                    {
                        "role": "user",
                        "content": "You are near the end of the run. Finish the change you are on, "
                        "re-run the command that lists the remaining sites, and call submit.",
                    }
                )
        say("[LOOP] hit the turn ceiling")
    except BaseException as error:
        ending = "the run ended on %s" % type(error).__name__
        raise
    finally:
        kit.selfreview_close(ending)

# echo_statement() prints the task statement to the log in a bounded, safe-to-read form.
STATEMENT_ECHO_LINES = 200
STATEMENT_ECHO_WIDTH = 400

def echo_statement(statement: str) -> None:
    lines = statement.splitlines()
    say("[TASK] %d line(s), %d char(s)" % (len(lines), len(statement)))
    for line in lines[:STATEMENT_ECHO_LINES]:
        say("[TASK] | %s" % line[:STATEMENT_ECHO_WIDTH])
    if len(lines) > STATEMENT_ECHO_LINES:
        say("[TASK] | ... %d more line(s)" % (len(lines) - STATEMENT_ECHO_LINES))

# agent_main() is the top-level entry point for the entire agent program.
# Everything you have been reading eventually comes together here.
# Its job is to:
# - initialize the repository/runtime,
# - read the problem statement,
# - run the main coding loop through drive(),
# - collect the resulting Git patch,
# - restore the working tree,
# - sanitize/salvage the patch if needed,
# - report diagnostics,
# - return the final patch string
def agent_main(input: dict) -> str:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    allowance = Allowance()
    root = os.getcwd()
    tree = Tree(root)
    pool = ShellPool(root)
    statement = str((input or {}).get("problem_statement") or "").strip()
    say("[RUN] budget=$%.3f clock=%.0fs build=97f100"
        % (allowance.ceiling_usd, allowance.clock_left()))
    try:
        echo_statement(statement)
    except BaseException:
        pass
    findings = FindingMap(root) if FINDING_MAP else None
    try:
        drive(statement, tree, pool, allowance, findings)
    except Spent as stop:
        say("[RUN] out of allowance: %s" % stop)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        say("[RUN] crashed: %s: %s" % (type(error).__name__, error))
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
    vouched = False
    try:
        vouched = tree.restore(finish_room(60.0))
    except BaseException:
        pass
    if PATCH_ENVELOPE and patch.strip():
        try:
            patch = envelope_trim(patch, Beacon("envelope"), statement)
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
    return patch
