from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = TESTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


ROOT = Path(__file__).resolve().parents[2]


def test_config_exists_and_parseable() -> None:
    cfg_path = ROOT / "experiments" / "configs" / "validation_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "hm_th_01" in cfg
    assert "hm_th_02" in cfg
    assert "hm_th_03" in cfg


def test_hm_th_01_a2_off_hits_counterexample_boundary() -> None:
    from src.method.contract_semantics import TransitionCase, unsafe_commit_indicator

    case = TransitionCase(
        claim_id="HM-TH-01",
        baseline="BL-SAGALLM-TX-ROLLBACK",
        strictness="strict",
        assumption_profile="A2_off",
        seed=11,
    )
    assert unsafe_commit_indicator(case) == 1


def test_manifest_paths_are_declared() -> None:
    manifest = json.loads((ROOT / "experiments" / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["entrypoint_path"] == "experiments/run_experiments.py"
    assert manifest["package_root"] == "experiments/src"
