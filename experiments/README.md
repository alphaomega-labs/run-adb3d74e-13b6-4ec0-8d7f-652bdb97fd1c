# Validation Simulation Package

This package executes proof-oriented validation for three theorem-linked claims:

- `HM-TH-01` committed-prefix soundness under semantic contract composition.
- `HM-TH-02` bounded recovery and non-oscillation constraints.
- `HM-TH-03` adversarial guard composition and provenance integrity.

## Entrypoint

- `experiments/run_experiments.py`
- Supported stages:
  - `--stage smoke`
  - `--stage full`

## Setup

Use the workspace virtual environment:

```bash
uv pip install --python experiments/.venv/bin/python sympy numpy pandas matplotlib seaborn pytest ruff mypy
```

## Commands

```bash
experiments/.venv/bin/python experiments/run_experiments.py --stage smoke
experiments/.venv/bin/python experiments/run_experiments.py --stage full
experiments/.venv/bin/python -m pytest experiments/tests
experiments/.venv/bin/ruff check experiments
experiments/.venv/bin/mypy experiments/run_experiments.py experiments/src
```

## Artifacts

- Data and logs: `experiments/contract_graph_orchestration_validation/`
- Figures (PDF): `paper/figures/iter_19/`
- Tables: `paper/tables/iter_19/`
- Appendix bundles: `paper/appendix/`
- Experiment manifest: `experiments/experiment_manifest.json`
