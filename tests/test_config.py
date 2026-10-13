"""MOCKED TESTS — Config loading, env overrides, and the YAML subset parser.

Skips if consciousness_lab.core.config is not implemented yet.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import REPO_ROOT, require

Config = require("consciousness_lab.core.config").Config
ConfigError = require("consciousness_lab.core.errors").ConfigError

CONFIG_DIR = REPO_ROOT / "configs"


def test_load_default_yaml():
    cfg = Config.load(str(CONFIG_DIR / "default.yaml"))
    assert cfg.provider == "mock"
    assert cfg.workspace_capacity == 7
    assert cfg.budgets.max_steps == 20
    assert cfg.budgets.max_tokens == 8000
    assert cfg.budgets.max_wall_time_s == pytest.approx(120.0)
    assert cfg.budgets.max_tool_calls == 10
    assert cfg.seed == 0
    assert cfg.autonomous_enabled is True
    assert cfg.components["memory"] is True


def test_load_mock_yaml_has_fixed_seed():
    cfg = Config.load(str(CONFIG_DIR / "mock.yaml"))
    assert cfg.provider == "mock"
    assert cfg.seed == 42


@pytest.mark.parametrize(
    "filename,component",
    [
        ("ablation_memory.yaml", "memory"),
        ("ablation_selfmodel.yaml", "self_model"),
        ("ablation_workspace.yaml", "workspace"),
        ("ablation_metacognition.yaml", "metacognition"),
        ("ablation_attention.yaml", "attention"),
    ],
)
def test_ablation_configs_disable_exactly_one_component(filename, component):
    cfg = Config.load(str(CONFIG_DIR / filename))
    for name, flag in cfg.components.items():
        if name == component:
            assert flag is False, f"{filename}: {name} should be False"
        elif name in ("memory", "self_model", "workspace", "metacognition", "attention"):
            assert flag is True, f"{filename}: {name} should stay True"


def test_env_overrides_file(monkeypatch):
    monkeypatch.setenv("ACL_SEED", "99")
    monkeypatch.setenv("ACL_WORKSPACE_CAPACITY", "5")
    monkeypatch.setenv("ACL_MAX_STEPS", "3")
    monkeypatch.setenv("ACL_AUTONOMOUS_ENABLED", "false")
    monkeypatch.setenv("ACL_LOG_LEVEL", "DEBUG")
    cfg = Config.load(str(CONFIG_DIR / "default.yaml"))
    assert cfg.seed == 99
    assert cfg.workspace_capacity == 5
    assert cfg.budgets.max_steps == 3
    assert cfg.autonomous_enabled is False
    assert cfg.log_level == "DEBUG"


def test_env_overrides_apply_without_file(monkeypatch):
    monkeypatch.setenv("ACL_PROVIDER", "mock")
    monkeypatch.setenv("ACL_SEED", "7")
    cfg = Config.load(None)
    assert cfg.provider == "mock"
    assert cfg.seed == 7


def test_invalid_yaml_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("components:\n\tmemory: true\n", encoding="utf-8")  # tab indent
    with pytest.raises(ConfigError):
        Config.load(str(bad))


def test_unsupported_yaml_syntax_raises_config_error(tmp_path):
    bad = tmp_path / "anchor.yaml"
    bad.write_text("a: &x 1\nb: *x\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        Config.load(str(bad))


def test_get_dotted_key():
    cfg = Config.load(str(CONFIG_DIR / "default.yaml"))
    assert cfg.get("budgets.max_steps") == 20
    assert cfg.get("components.memory") is True
    assert cfg.get("no.such.key", "fallback") == "fallback"


def test_secrets_never_come_from_yaml(tmp_path, monkeypatch):
    cfg_file = tmp_path / "s.yaml"
    cfg_file.write_text("provider: openai_compat\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Config.load(str(cfg_file))
    assert "OPENAI_API_KEY" not in str(cfg.provider_params)
    assert getattr(cfg, "openai_api_key", None) is None
