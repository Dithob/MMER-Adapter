import ast
import subprocess
import sys
from pathlib import Path

import pytest

from config.cli_args import parse_args


ROOT = Path(__file__).resolve().parents[1]


def test_yaml_nested_config_rejects_duplicate_keys(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        """
training:
  batch_size: 8
model:
  batch_size: 16
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate config key"):
        parse_args(["--config", str(cfg)])


def test_run_dry_run_does_not_require_training_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--dry_run",
            "--print_config",
            "--dataset",
            "meld",
            "--model_type",
            "chatglm3",
            "--use_context",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert '"datasetName": "meld"' in result.stdout
    assert '"use_context": true' in result.stdout


def test_run_py_no_longer_defines_legacy_parse_args():
    tree = ast.parse((ROOT / "run.py").read_text(encoding="utf-8"))

    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "legacy_parse_args" not in function_names


def test_select_most_free_gpu_scans_all_visible_devices():
    from utils.gpu_utils import select_most_free_gpu

    memory_by_gpu = {0: 900, 1: 100, 2: 500}

    selected = select_most_free_gpu(
        cuda_device_count=lambda: 3,
        nvml_init=lambda: None,
        nvml_get_handle=lambda idx: idx,
        nvml_get_memory=lambda handle: type("Mem", (), {"used": memory_by_gpu[handle]})(),
    )

    assert selected == 1
