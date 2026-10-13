"""REAL TESTS — config YAML validation with an independent subset parser.

No src imports (Config may not exist yet): this module embeds a minimal parser
for exactly the YAML subset the contract supports (2-space indent,
``key: value``, ``- `` lists, int/float/true/false/bare or "quoted" scalars,
``#`` comments) and validates structure, flag semantics, and subset
compliance of all 7 config files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

EXPECTED_FILES = [
    "default.yaml",
    "mock.yaml",
    "ablation_memory.yaml",
    "ablation_selfmodel.yaml",
    "ablation_workspace.yaml",
    "ablation_metacognition.yaml",
    "ablation_attention.yaml",
]
# filename -> component flag that must be False (None = none must be False)
ABLATED_COMPONENT = {
    "default.yaml": None,
    "mock.yaml": None,
    "ablation_memory.yaml": "memory",
    "ablation_selfmodel.yaml": "self_model",
    "ablation_workspace.yaml": "workspace",
    "ablation_metacognition.yaml": "metacognition",
    "ablation_attention.yaml": "attention",
}
ALL_COMPONENTS = ["memory", "self_model", "workspace", "metacognition", "attention"]


def parse_scalar(token: str):
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    if token == "true":
        return True
    if token == "false":
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    if not token:
        raise ValueError("empty scalar")
    if any(ch in token for ch in "{}[]&*!|>@`'\""):
        raise ValueError(f"unsupported scalar characters: {token!r}")
    return token  # bare string


def parse_subset(text: str):
    """Parse the contract's YAML subset; raise ValueError on anything else."""
    lines = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw:
            raise ValueError(f"line {lineno}: tabs are not allowed")
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"line {lineno}: indent must be a multiple of 2 spaces")
        # strip trailing comments outside double quotes
        buf, in_quotes = [], False
        for ch in stripped:
            if ch == '"':
                in_quotes = not in_quotes
            if ch == "#" and not in_quotes and buf and buf[-1] == " ":
                break
            buf.append(ch)
        content = "".join(buf).rstrip()
        if not content:
            continue
        lines.append((indent, content, lineno))

    pos = 0

    def parse_block(indent: int):
        nonlocal pos
        if pos < len(lines) and lines[pos][1].startswith("- "):
            items = []
            while (pos < len(lines) and lines[pos][0] == indent
                   and lines[pos][1].startswith("- ")):
                _, content, lineno = lines[pos]
                items.append(parse_scalar(content[2:].strip()))
                pos += 1
            return items
        mapping: dict = {}
        while pos < len(lines) and lines[pos][0] == indent:
            _, content, lineno = lines[pos]
            if content.startswith("- "):
                raise ValueError(f"line {lineno}: unexpected list item in mapping")
            key, sep, value = content.partition(":")
            if not sep:
                raise ValueError(f"line {lineno}: expected 'key: value', got {content!r}")
            key, value = key.strip(), value.strip()
            if not key or " " in key:
                raise ValueError(f"line {lineno}: bad key {key!r}")
            pos += 1
            if value == "":
                if pos < len(lines) and lines[pos][0] > indent:
                    mapping[key] = parse_block(lines[pos][0])
                else:
                    raise ValueError(f"line {lineno}: empty value for {key!r}")
            else:
                mapping[key] = parse_scalar(value)
        return mapping

    result = parse_block(0)
    if pos != len(lines):
        _, _, lineno = lines[pos]
        raise ValueError(f"line {lineno}: could not consume input")
    return result


def json_like_strings(node) -> str:
    """Lower-cased concatenation of all keys and scalar values in a doc."""
    parts = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                parts.append(str(k))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            parts.append(str(x))

    walk(node)
    return "\n".join(parts).lower()


@pytest.fixture(params=EXPECTED_FILES)
def config_doc(request):
    path = CONFIG_DIR / request.param
    assert path.exists(), f"missing config file: {request.param}"
    return request.param, parse_subset(path.read_text(encoding="utf-8"))


def test_all_configs_parse_as_subset(config_doc):
    _, doc = config_doc
    assert isinstance(doc, dict) and doc


def test_config_has_required_keys(config_doc):
    name, doc = config_doc
    for key in ("provider", "seed", "components", "budgets", "workspace_capacity"):
        assert key in doc, f"{name}: missing key {key!r}"
    assert isinstance(doc["provider"], str) and doc["provider"]
    assert isinstance(doc["seed"], int)
    assert isinstance(doc["workspace_capacity"], int) and doc["workspace_capacity"] > 0
    budgets = doc["budgets"]
    for key in ("max_steps", "max_tokens", "max_wall_time_s", "max_tool_calls"):
        assert key in budgets, f"{name}: budgets missing {key!r}"
    assert isinstance(budgets["max_steps"], int)
    assert isinstance(budgets["max_tokens"], int)
    assert isinstance(budgets["max_wall_time_s"], float)
    assert isinstance(budgets["max_tool_calls"], int)


def test_config_component_flags(config_doc):
    name, doc = config_doc
    components = doc["components"]
    for comp in ALL_COMPONENTS:
        assert comp in components, f"{name}: components missing {comp!r}"
        assert isinstance(components[comp], bool), f"{name}: {comp} must be a bool"
    ablated = ABLATED_COMPONENT[name]
    for comp in ALL_COMPONENTS:
        expected = comp != ablated
        assert components[comp] is expected, (
            f"{name}: component {comp!r} should be {expected}"
        )


def test_mock_config_is_explicit():
    doc = parse_subset((CONFIG_DIR / "mock.yaml").read_text(encoding="utf-8"))
    assert doc["provider"] == "mock"
    assert doc["seed"] == 42


def test_default_config_all_components_on():
    doc = parse_subset((CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))
    assert all(doc["components"][c] is True for c in ALL_COMPONENTS)
    assert doc["seed"] == 0


def test_configs_use_no_secrets_or_unsupported_syntax():
    for name in EXPECTED_FILES:
        text = (CONFIG_DIR / name).read_text(encoding="utf-8")
        for bad in ("&", "*", "{", "}", "[", "]", "\t"):
            assert bad not in text, f"{name}: unsupported YAML syntax {bad!r}"
        doc = parse_subset(text)
        # secrets must never appear as config keys/values; comments that
        # document the env-var mechanism are fine, so check parsed content
        flat = json_like_strings(doc)
        for banned in ("api_key", "secret", "password"):
            assert banned not in flat, (
                f"{name}: config content must not contain {banned!r}"
            )
