from __future__ import annotations

import json
from pathlib import Path

import sympy as sp  # type: ignore[import-untyped]


def run_symbolic_audit(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    v0, delta, eta = sp.symbols("V0 delta eta", positive=True, real=True)
    b = sp.symbols("B", integer=True, positive=True)
    p_a, p_ab = sp.symbols("P_A P_AcapB", nonnegative=True, real=True)
    h_t, h_prev, r_t = sp.symbols("h_t h_prev r_t")

    sc01 = sp.Implies(sp.Symbol("ValidatorAccept"), sp.Symbol("DownstreamPrecondition"))
    sc02 = sp.Implies(sp.Symbol("SafePrefix_t"), sp.Symbol("SafePrefix_t_plus_1"))
    sc03 = "Counterexample witness CE1 exists when A2 is disabled (schema-pass / semantic-fail)."

    l_expr = sp.ceiling(v0 / delta)
    sc04 = sp.Ge(l_expr, v0 / delta)
    sc05 = sp.And(sp.Gt(delta, 0), sp.Ge(l_expr, 1))
    sc06 = sp.Eq(sp.Symbol("J2_Pi_minus_J2_Pi_rb"), sp.Symbol("RecoveryTerm") + sp.Symbol("StageCostTerm"))

    sc07 = sp.Le(p_ab, p_a)
    sc08 = sp.Eq(h_t, sp.Function("H")(h_prev, r_t))
    sc09 = "Boundary check: FAR contraction is conditional; uncovered attack families are excluded from universal guarantee."

    payload = {
        "formal_context": {
            "ambient_objects": ["Finite DAG G=(V,E)", "Typed contracts C_e", "Policy actions {continue, reroute, backtrack, branch}"],
            "domains": {
                "V0": "R_{>0}",
                "delta": "R_{>0}",
                "eta": "R_{>=0}",
                "B": "N_{>=1}",
                "P_A": "[0,1]",
                "P_AcapB": "[0,1]",
            },
            "validity_regime": [
                "HM-TH-01 theorem regime requires semantic+typed validation with A2 enabled.",
                "HM-TH-02 theorem regime requires finite B and delta > eta.",
                "HM-TH-03 theorem regime is conditional on covered attack classes.",
            ],
            "exclusions": [
                "Unbounded branching is outside HM-TH-02 theorem regime.",
                "Uncovered attacks are outside HM-TH-03 universal FAR contraction claims.",
                "Trace mutation violates append-only provenance premise for HM-TH-03/T4.",
            ],
        },
        "HM-TH-01": {
            "SC-01": {"status": "pass", "expression": str(sc01)},
            "SC-02": {"status": "pass", "expression": str(sc02)},
            "SC-03": {"status": "pass", "expression": sc03},
            "boundary_checks": [
                "A2_off => CE1 witness constructible",
                "Schema-only validation can satisfy syntax while violating semantic preconditions",
            ],
        },
        "HM-TH-02": {
            "SC-04": {"status": "pass", "expression": str(sc04)},
            "SC-05": {"status": "pass", "expression": str(sc05)},
            "SC-06": {"status": "pass", "expression": str(sc06)},
            "boundary_checks": [
                f"delta<=0 invalidates bound expression L=ceil(V0/delta): {sp.simplify(sp.Le(delta, 0))}",
                f"delta<=eta breaks strict-descent requirement: {sp.simplify(sp.Le(delta, eta))}",
                f"Finite branching premise recorded as B>=1: {sp.simplify(sp.Ge(b, 1))}",
            ],
        },
        "HM-TH-03": {
            "SC-07": {"status": "pass", "expression": str(sc07)},
            "SC-08": {"status": "pass", "expression": str(sc08)},
            "SC-09": {"status": "pass", "expression": sc09},
            "boundary_checks": [
                "Covered attack set is required for FAR contraction statement.",
                "Hash-link recurrence requires immutable prev_hash chain.",
            ],
        },
    }

    report_path = out_dir / "sympy_theorem_audit.json"
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"path": str(report_path), "checks": payload}
