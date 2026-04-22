from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionCase:
    claim_id: str
    baseline: str
    strictness: str
    assumption_profile: str
    seed: int


def validator_accepts(case: TransitionCase) -> bool:
    if case.baseline in {"BL-SCHEMA-ONLY-VALIDATOR", "BL-TRUST-ONLY-GATE", "BL-NO-HARD-VALIDATOR"}:
        return False
    if case.claim_id == "HM-TH-01" and case.assumption_profile in {"A2_off", "A3_off"}:
        return False
    if case.claim_id == "HM-TH-03" and case.assumption_profile == "uncovered":
        return False
    return True


def unsafe_commit_indicator(case: TransitionCase) -> int:
    accepted = validator_accepts(case)
    if case.claim_id == "HM-TH-01":
        if case.baseline == "BL-SCHEMA-ONLY-VALIDATOR":
            return 1
        if case.assumption_profile == "A2_off":
            return 1
        return 0 if accepted else 1
    if case.claim_id == "HM-TH-02":
        if case.baseline == "BL-UNBOUNDED-BRANCHING":
            return 1
        return 0
    if case.claim_id == "HM-TH-03":
        if case.baseline == "BL-TRUST-ONLY-GATE":
            return 1
        if case.assumption_profile == "uncovered":
            return 1
        return 0
    return 1


def oscillation_indicator(baseline: str, b_max: int, delta: float, eta: float) -> int:
    if baseline == "BL-UNBOUNDED-BRANCHING":
        return 1
    if delta <= eta:
        return 1
    if b_max > 5:
        return 1
    return 0


def recovery_length(v0: float, delta: float, b_max: int, baseline: str) -> float:
    if delta <= 0:
        return float("inf")
    theoretical = (v0 / delta)
    if baseline == "BL-UNBOUNDED-BRANCHING":
        return theoretical + 2.0
    return min(theoretical, float(b_max))


def far_frr(guard_mode: str, attack_class: str, theta: float, mutation_rate: float) -> tuple[float, float]:
    if guard_mode == "trust_only":
        far = 0.18 if attack_class == "covered" else 0.26
        frr = 0.09 + theta * 0.02
    elif guard_mode == "protocol_only":
        far = 0.11 if attack_class == "covered" else 0.20
        frr = 0.11 + theta * 0.03
    else:
        far = 0.06 if attack_class == "covered" else 0.19
        frr = 0.12 + theta * 0.03
    if mutation_rate > 0:
        far += mutation_rate * 0.08
    return far, frr
