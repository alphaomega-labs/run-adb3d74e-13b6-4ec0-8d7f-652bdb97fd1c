from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize_dataset_resolution_log(
    root: Path,
    run_dir: Path,
    dataset_plan: list[dict[str, Any]],
) -> Path:
    log_path = run_dir / "dataset_resolution_log.json"
    rows: list[dict[str, Any]] = []
    for entry in dataset_plan:
        item = dict(entry)
        local_path = item.get("local_path")
        if local_path:
            target = root / local_path
            item["local_exists"] = target.exists()
        rows.append(item)
    log_path.write_text(json.dumps({"datasets": rows}, indent=2) + "\n", encoding="utf-8")
    return log_path


def scan_quarks_substrate(root: Path) -> dict[str, Any]:
    substrate = root / "resources" / "codebases" / "quarks-e2c886f43f88" / "quarks"
    graph_py = substrate / "backend" / "src" / "quarks" / "orchestrator" / "graph.py"
    coordinator_py = substrate / "backend" / "src" / "quarks" / "orchestrator" / "coordinator.py"
    runner_py = substrate / "backend" / "src" / "quarks" / "orchestrator" / "runner.py"
    return {
        "substrate_root": str(substrate),
        "files": {
            "graph.py": str(graph_py),
            "coordinator.py": str(coordinator_py),
            "runner.py": str(runner_py),
        },
        "checksums": {
            "graph.py": _sha256_of_file(graph_py) if graph_py.exists() else None,
            "coordinator.py": _sha256_of_file(coordinator_py) if coordinator_py.exists() else None,
            "runner.py": _sha256_of_file(runner_py) if runner_py.exists() else None,
        },
    }


def create_trace_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
