"""Typed configuration for the lab: file parsing, env overrides, secrets policy.

What it does:
    - :class:`Config` holds every lab setting with typed defaults.
    - :meth:`Config.load` reads an optional YAML file (a strict subset, see
      below), then applies environment-variable overrides, which always win.
    - :meth:`Config.get` resolves dotted keys like ``"budgets.max_steps"``.

YAML subset supported (anything else raises :class:`ConfigError`):
    - ``#`` comments: full-line, or trailing a value when the ``#`` is
      preceded by whitespace and is outside a double-quoted string.
    - ``key: value`` pairs; keys match ``[A-Za-z_][A-Za-z0-9_]*``.
    - Nesting via 2-space indentation (tabs are rejected).
    - Lists via ``- `` items containing scalars only (no nested mappings,
      no inline ``[a, b]`` flow collections, no ``|``/``>`` block scalars).
    - Scalars: ints (``42``), floats (``1.5``, ``2e3``), ``true``/``false``
      (case-insensitive), bare strings, and ``"double-quoted"`` strings with
      ``\\n \\t \\r \\\\ \\" \\0`` escapes. Single-quoted strings are
      rejected.

Secrets policy:
    - Secrets are NEVER read from the YAML file. Provider credentials come
      only from the environment (``OLLAMA_BASE_URL``, ``OPENAI_API_KEY``,
      ``OPENAI_BASE_URL``, ``GEMINI_API_KEY``, ``GEMINI_MODEL``,
      ``OLLAMA_MODEL``, ``OPENAI_MODEL``) and are consumed directly by the
      provider adapters, never stored on :class:`Config`.
    - Non-secret overrides come from ``ACL_DB_PATH``, ``ACL_PROVIDER``,
      ``ACL_PROVIDER_MODEL``, ``ACL_LOG_LEVEL``, ``ACL_AUTONOMOUS_ENABLED``,
      ``ACL_SEED``, ``ACL_RESULTS_DIR``, ``ACL_WORKSPACE_CAPACITY``,
      ``ACL_MAX_STEPS``, ``ACL_MAX_TOKENS``, ``ACL_MAX_WALL_TIME_S``,
      ``ACL_MAX_TOOL_CALLS``. Empty-string env values are treated as unset.

Limits:
    - This is a hand-written subset parser, not a YAML implementation: it
      does not support anchors, tags, multi-document files, or flow syntax.
    - Unknown top-level keys are kept in :attr:`Config.extra` rather than
      rejected, so experiment files can carry extra metadata.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, is_dataclass
from typing import Any

from .errors import ConfigError

__all__ = ["BudgetConfig", "Config"]


# ---------------------------------------------------------------------------
# YAML subset parser
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?$|^[+-]?\d+[eE][+-]?\d+$")
_DQ_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "0": "\0"}
_BOOL_WORDS = {"true": True, "false": False}


def _strip_inline_comment(line: str) -> str:
    """Remove a ``#`` comment, honouring double-quoted strings.

    A ``#`` starts a comment only when it is outside a double-quoted string
    and is the first character or is preceded by whitespace.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in line:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
                out.append(ch)
            elif ch == "#" and (not out or out[-1] in (" ", "\t")):
                break
            else:
                out.append(ch)
    return "".join(out)


def _parse_double_quoted(inner: str, lineno: int) -> str:
    out: list[str] = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == "\\":
            nxt = inner[i + 1] if i + 1 < len(inner) else ""
            if nxt not in _DQ_ESCAPES:
                raise ConfigError(f"line {lineno}: unsupported escape '\\{nxt}' in string")
            out.append(_DQ_ESCAPES[nxt])
            i += 2
        elif ch == '"':
            raise ConfigError(f"line {lineno}: unescaped '\"' inside double-quoted string")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _parse_scalar(text: str, lineno: int) -> Any:
    if not text:
        raise ConfigError(f"line {lineno}: expected a value")
    if text.startswith('"'):
        if len(text) < 2 or not text.endswith('"'):
            raise ConfigError(f"line {lineno}: unterminated double-quoted string")
        return _parse_double_quoted(text[1:-1], lineno)
    if text.startswith("'"):
        raise ConfigError(f"line {lineno}: single-quoted strings are not supported; use \"...\"")
    if text[0] in "[{":
        raise ConfigError(f"line {lineno}: flow collections are not supported; use '- ' list items")
    if text.startswith("- "):
        raise ConfigError(f"line {lineno}: inline lists are not supported; use '- ' list items")
    if text in ("|", ">", "|-", ">-", "|+", ">+"):
        raise ConfigError(f"line {lineno}: block scalars are not supported")
    # Reject YAML anchors (&name) and aliases (*name) appearing as whole tokens.
    # (An ampersand inside a word like "R&D" is fine; only &-/*-prefixed tokens
    # are anchor/alias syntax.)
    for token in text.split():
        if re.match(r"^[&*][A-Za-z0-9_-]+$", token):
            raise ConfigError(
                f"line {lineno}: anchors and aliases are not supported "
                f"(got {token!r})"
            )
    lowered = text.lower()
    if lowered in _BOOL_WORDS:
        return _BOOL_WORDS[lowered]
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _split_key_value(content: str, lineno: int) -> tuple[str, str]:
    if ":" not in content:
        raise ConfigError(f"line {lineno}: expected 'key: value', got {content!r}")
    key, _, rest = content.partition(":")
    key = key.strip()
    if not _KEY_RE.match(key):
        raise ConfigError(f"line {lineno}: invalid key {key!r} (use [A-Za-z_][A-Za-z0-9_]*)")
    return key, rest.strip()


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    """Parse the documented YAML subset into nested dicts/lists/scalars."""
    items: list[tuple[int, int, str]] = []  # (lineno, indent, content)
    for lineno, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw:
            raise ConfigError(f"line {lineno}: tab characters are not allowed; use 2-space indent")
        code = _strip_inline_comment(raw)
        if not code.strip():
            continue
        stripped = code.lstrip(" ")
        indent = len(code) - len(stripped)
        if indent % 2 != 0:
            raise ConfigError(f"line {lineno}: indentation must be a multiple of 2 spaces")
        items.append((lineno, indent, stripped))

    pos = 0
    total = len(items)

    def parse_mapping(base_indent: int) -> dict[str, Any]:
        nonlocal pos
        mapping: dict[str, Any] = {}
        while pos < total:
            lineno, indent, content = items[pos]
            if indent < base_indent:
                break
            if indent > base_indent:
                raise ConfigError(f"line {lineno}: unexpected indentation")
            if content == "-" or content.startswith("- "):
                raise ConfigError(f"line {lineno}: list item is not attached to a key")
            key, rest = _split_key_value(content, lineno)
            if key in mapping:
                raise ConfigError(f"line {lineno}: duplicate key {key!r}")
            pos += 1
            if rest == "":
                if pos < total and items[pos][1] == base_indent + 2:
                    nxt = items[pos][2]
                    if nxt == "-" or nxt.startswith("- "):
                        mapping[key] = parse_list(base_indent + 2)
                    else:
                        mapping[key] = parse_mapping(base_indent + 2)
                elif pos < total and items[pos][1] > base_indent:
                    raise ConfigError(
                        f"line {items[pos][0]}: child of {key!r} must be indented "
                        f"exactly 2 spaces relative to its parent"
                    )
                else:
                    mapping[key] = {}
            else:
                mapping[key] = _parse_scalar(rest, lineno)
        return mapping

    def parse_list(list_indent: int) -> list[Any]:
        nonlocal pos
        result: list[Any] = []
        while pos < total:
            lineno, indent, content = items[pos]
            if indent < list_indent:
                break
            if indent > list_indent:
                raise ConfigError(f"line {lineno}: unexpected indentation inside list")
            if not (content == "-" or content.startswith("- ")):
                raise ConfigError(f"line {lineno}: expected a '- ' list item")
            item_text = content[1:].strip()
            pos += 1
            if not item_text:
                raise ConfigError(f"line {lineno}: empty list items are not supported")
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:(?:\s|$)", item_text):
                raise ConfigError(f"line {lineno}: nested mappings inside lists are not supported")
            result.append(_parse_scalar(item_text, lineno))
        return result

    mapping = parse_mapping(0)
    if pos != total:
        raise ConfigError(f"line {items[pos][0]}: could not parse line {items[pos][2]!r}")
    return mapping


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

def _to_int(value: Any, what: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{what}: expected int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and _INT_RE.match(value.strip()):
        return int(value.strip())
    raise ConfigError(f"{what}: expected int, got {value!r}")


def _to_float(value: Any, what: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{what}: expected float, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise ConfigError(f"{what}: expected float, got {value!r}")


def _to_bool(value: Any, what: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y", "on"):
            return True
        if lowered in ("false", "0", "no", "n", "off"):
            return False
    raise ConfigError(f"{what}: expected true/false, got {value!r}")


def _to_str(value: Any, what: str) -> str:
    if isinstance(value, str):
        return value
    raise ConfigError(f"{what}: expected string, got {value!r}")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Execution budgets for a run. All are hard caps enforced by safety."""

    max_steps: int = 20
    max_tokens: int = 8000
    max_wall_time_s: float = 120.0
    max_tool_calls: int = 10


@dataclass
class Config:
    """Lab configuration.

    Attributes mirror the documented YAML keys; ``components`` toggles the
    optional cognitive modules (a ``False`` value makes :func:`build_loop`
    pass ``None`` for that component so the loop degrades gracefully);
    ``extra`` keeps any unknown top-level keys from the file.
    """

    db_path: str = "./data/acl.sqlite3"
    provider: str = "mock"
    provider_model: str = ""
    provider_params: dict = field(default_factory=dict)
    workspace_capacity: int = 7
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    autonomous_enabled: bool = True
    approval_required_default: bool = False
    log_level: str = "INFO"
    seed: int = 0
    results_dir: str = "experiments/results"
    components: dict = field(
        default_factory=lambda: {
            "memory": True,
            "self_model": True,
            "workspace": True,
            "metacognition": True,
            "attention": True,
            "learning": True,
        }
    )
    extra: dict = field(default_factory=dict)

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        """Load configuration from a YAML file and environment overrides.

        Args:
            path: Optional path to a YAML file in the documented subset.
                When ``None``, only defaults plus environment overrides apply.

        Raises:
            ConfigError: On unreadable files, YAML-subset violations, or
                invalid values (file or environment).
        """
        data: dict[str, Any] = {}
        if path is not None:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except OSError as exc:
                raise ConfigError(f"cannot read config file {path!r}: {exc}") from exc
            data = _parse_yaml_subset(text)
        cfg = cls()
        _apply_mapping(cfg, data, source=f"file {path!r}" if path else "defaults")
        _apply_env(cfg)
        return cfg

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Resolve a dotted key like ``"budgets.max_steps"``.

        Walks dataclass attributes and dict entries; returns ``default``
        when any segment is missing.
        """
        current: Any = self
        for part in dotted_key.split("."):
            if isinstance(current, dict):
                if part not in current:
                    return default
                current = current[part]
            elif is_dataclass(current):
                if not hasattr(current, part):
                    return default
                current = getattr(current, part)
            else:
                return default
        return current


def _apply_mapping(cfg: Config, data: dict[str, Any], source: str) -> None:
    """Apply a parsed file mapping onto ``cfg`` with type coercion."""
    for key, value in data.items():
        where = f"{source}: key {key!r}"
        if key == "budgets":
            if not isinstance(value, dict):
                raise ConfigError(f"{where}: expected a mapping")
            for sub, subval in value.items():
                if not hasattr(cfg.budgets, sub):
                    raise ConfigError(f"{where}: unknown budgets key {sub!r}")
                if sub in ("max_steps", "max_tokens", "max_tool_calls"):
                    setattr(cfg.budgets, sub, _to_int(subval, f"{where}.{sub}"))
                else:  # max_wall_time_s
                    setattr(cfg.budgets, sub, _to_float(subval, f"{where}.{sub}"))
        elif key == "components":
            if not isinstance(value, dict):
                raise ConfigError(f"{where}: expected a mapping")
            merged = dict(cfg.components)
            for sub, subval in value.items():
                merged[sub] = _to_bool(subval, f"{where}.{sub}")
            cfg.components = merged
        elif key == "provider_params":
            if not isinstance(value, dict):
                raise ConfigError(f"{where}: expected a mapping")
            cfg.provider_params = dict(value)
        elif key in ("db_path", "provider", "provider_model", "log_level", "results_dir"):
            setattr(cfg, key, _to_str(value, where))
        elif key in ("workspace_capacity", "seed"):
            setattr(cfg, key, _to_int(value, where))
        elif key in ("autonomous_enabled", "approval_required_default"):
            setattr(cfg, key, _to_bool(value, where))
        else:
            cfg.extra[key] = value


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def _apply_env(cfg: Config) -> None:
    """Apply ``ACL_*`` environment overrides; these always win over the file."""
    value = _env("ACL_DB_PATH")
    if value is not None:
        cfg.db_path = value
    value = _env("ACL_PROVIDER")
    if value is not None:
        cfg.provider = value
    value = _env("ACL_PROVIDER_MODEL")
    if value is not None:
        cfg.provider_model = value
    value = _env("ACL_LOG_LEVEL")
    if value is not None:
        cfg.log_level = value
    value = _env("ACL_AUTONOMOUS_ENABLED")
    if value is not None:
        cfg.autonomous_enabled = _to_bool(value, "env ACL_AUTONOMOUS_ENABLED")
    value = _env("ACL_SEED")
    if value is not None:
        cfg.seed = _to_int(value, "env ACL_SEED")
    value = _env("ACL_RESULTS_DIR")
    if value is not None:
        cfg.results_dir = value
    value = _env("ACL_WORKSPACE_CAPACITY")
    if value is not None:
        cfg.workspace_capacity = _to_int(value, "env ACL_WORKSPACE_CAPACITY")
    value = _env("ACL_MAX_STEPS")
    if value is not None:
        cfg.budgets.max_steps = _to_int(value, "env ACL_MAX_STEPS")
    value = _env("ACL_MAX_TOKENS")
    if value is not None:
        cfg.budgets.max_tokens = _to_int(value, "env ACL_MAX_TOKENS")
    value = _env("ACL_MAX_WALL_TIME_S")
    if value is not None:
        cfg.budgets.max_wall_time_s = _to_float(value, "env ACL_MAX_WALL_TIME_S")
    value = _env("ACL_MAX_TOOL_CALLS")
    if value is not None:
        cfg.budgets.max_tool_calls = _to_int(value, "env ACL_MAX_TOOL_CALLS")
