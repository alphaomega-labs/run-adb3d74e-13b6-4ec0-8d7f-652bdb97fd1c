from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _rasterize_pdf(pdf_path: Path) -> dict[str, Any]:
    png_path = pdf_path.with_suffix(".png")
    cmd = [
        "pdftoppm",
        "-f",
        "1",
        "-singlefile",
        "-png",
        str(pdf_path),
        str(pdf_path.with_suffix("")),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"status": "ok", "preview_path": str(png_path), "stderr": result.stderr.strip()}
    except Exception as exc:  # noqa: BLE001
        return {"status": "warning", "preview_path": None, "stderr": str(exc)}


def _apply_display_labels(df: pd.DataFrame, mapping: dict[str, dict[str, str]]) -> pd.DataFrame:
    rendered = df.copy()
    for column, column_mapping in mapping.items():
        if column not in rendered.columns:
            continue
        rendered[column] = rendered[column].map(lambda value: column_mapping.get(str(value), str(value)))
    return rendered


def _rewrite_legend_entries(ax: Any, title: str, mapping: dict[str, str], fontsize: int = 8) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    legend.set_title(title)
    legend.get_title().set_fontsize(fontsize)
    for text in legend.get_texts():
        label = text.get_text()
        text.set_text(mapping.get(label, label))
        text.set_fontsize(fontsize)


def save_claim_figures(
    hm_th_01: pd.DataFrame,
    hm_th_02: pd.DataFrame,
    hm_th_03: pd.DataFrame,
    figure_dir: Path,
) -> tuple[list[str], dict[str, Any], dict[str, str]]:
    sns.set_theme(style="whitegrid", palette="deep")
    figure_dir.mkdir(parents=True, exist_ok=True)
    captions: dict[str, str] = {}
    qa: dict[str, Any] = {}
    paths: list[str] = []
    hm_th_01_plot = _apply_display_labels(
        hm_th_01,
        {
            "validator_mode": {
                "semantic_and_typed": "Semantic + typed",
                "schema_only": "Schema-only",
                "trust_only": "Trust-only",
                "no_hard_guard": "No hard guard",
            },
            "assumption_profile": {
                "A2_on": "A2 active",
                "A2_off": "A2 disabled",
                "A3_off": "A3 disabled",
            },
        },
    )
    hm_th_02_plot = _apply_display_labels(
        hm_th_02,
        {
            "baseline": {
                "BL-GREEDY-CONTINUE": "Greedy continue",
                "BL-UNBOUNDED-BRANCHING": "Unbounded branching",
                "BL-FIXED-DEPTH-BACKTRACK": "Fixed-depth backtrack",
                "BL-ROLLBACK-ONLY-COMPARATOR-Pi_rb": "Rollback-only comparator",
            }
        },
    )
    if "B_max" in hm_th_02_plot.columns:
        hm_th_02_plot["corrective_depth"] = hm_th_02_plot["B_max"].map(lambda value: f"B={value}")
    hm_th_03_plot = _apply_display_labels(
        hm_th_03,
        {
            "guard_mode": {
                "trust_only": "Trust-only guard",
                "protocol_only": "Protocol-only guard",
                "composite": "Composite guard",
            },
            "attack_class": {
                "covered": "Covered classes",
                "uncovered": "Uncovered classes",
            },
        },
    )

    # HM-TH-01
    validator_order = ["Semantic + typed", "Schema-only", "Trust-only", "No hard guard"]
    assumption_order = ["A2 active", "A2 disabled", "A3 disabled"]
    safety_summary = hm_th_01_plot.groupby(["assumption_profile", "validator_mode"], as_index=False).agg(
        unsafe_commit_rate=("unsafe_commit_rate", "mean"),
        seed_count=("unsafe_commit_rate", "size"),
    )
    safety_summary["unsafe_count"] = (
        safety_summary["unsafe_commit_rate"] * safety_summary["seed_count"]
    ).round().astype(int)
    safety_matrix = safety_summary.pivot(
        index="assumption_profile", columns="validator_mode", values="unsafe_commit_rate"
    ).reindex(index=assumption_order, columns=validator_order)
    safety_annotations = (
        safety_summary.assign(
            label=lambda df: df["unsafe_count"].astype(str) + "/" + df["seed_count"].astype(str) + "\nunsafe"
        )
        .pivot(index="assumption_profile", columns="validator_mode", values="label")
        .reindex(index=assumption_order, columns=validator_order)
    )
    fig1, ax1 = plt.subplots(figsize=(7.5, 3.6))
    sns.heatmap(
        safety_matrix,
        annot=safety_annotations,
        fmt="",
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Unsafe commit rate"},
        ax=ax1,
    )
    ax1.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2.2))
    ax1.set_xlabel("Validator mode")
    ax1.set_ylabel("Assumption profile")
    ax1.set_title("Committed-prefix safety scope matrix")
    th01 = figure_dir / "fig_hm_th_01_unsafe_commit.pdf"
    fig1.tight_layout()
    fig1.savefig(th01, format="pdf")
    plt.close(fig1)
    paths.append(str(th01))
    qa[str(th01)] = _rasterize_pdf(th01)
    captions[str(th01)] = (
        "Premise matrix of unsafe commit counts across validator modes and assumption toggles. "
        "Rows: assumption profile; columns: validator mode; cell labels show unsafe commits over seed-level boundary probes. "
        "Key takeaway: the theorem-valid semantic + typed / A2-active cell has 0 unsafe probes, while premise-breaking cells expose deterministic boundary failures."
    )

    # HM-TH-02
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.lineplot(
        data=hm_th_02_plot,
        x="delta",
        y="oscillation_rate",
        hue="baseline",
        style="corrective_depth",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=axes2[0],
    )
    axes2[0].set_xlabel("Potential descent $\\delta$")
    axes2[0].set_ylabel("Oscillation rate")
    _rewrite_legend_entries(
        axes2[0],
        title="Policy / depth bound",
        mapping={
            "baseline": "Policy",
            "corrective_depth": "Depth bound",
            "B=1": "B = 1",
            "B=2": "B = 2",
            "B=3": "B = 3",
            "B=5": "B = 5",
        },
        fontsize=8,
    )
    sns.lineplot(
        data=hm_th_02_plot,
        x="delta",
        y="bound_ratio",
        hue="baseline",
        style="corrective_depth",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=axes2[1],
    )
    axes2[1].set_xlabel("Potential descent $\\delta$")
    axes2[1].set_ylabel("Observed length / ceil(V0/delta)")
    _rewrite_legend_entries(
        axes2[1],
        title="Policy / depth bound",
        mapping={
            "baseline": "Policy",
            "corrective_depth": "Depth bound",
            "B=1": "B = 1",
            "B=2": "B = 2",
            "B=3": "B = 3",
            "B=5": "B = 5",
        },
        fontsize=8,
    )
    th02 = figure_dir / "fig_hm_th_02_recovery_bounds.pdf"
    fig2.tight_layout()
    fig2.savefig(th02, format="pdf")
    plt.close(fig2)
    paths.append(str(th02))
    qa[str(th02)] = _rasterize_pdf(th02)
    captions[str(th02)] = (
        "Two-panel recovery diagnostics: left panel oscillation rate vs descent parameter delta; right panel bound ratio vs delta. "
        "Both panels include policy and depth-bound encodings with SD error bars. "
        "Interpretation: bounded policies show near-zero oscillation for delta>eta, while unbounded branching violates bounds."
    )

    # HM-TH-03
    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.lineplot(
        data=hm_th_03_plot,
        x="theta",
        y="far",
        hue="guard_mode",
        style="attack_class",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=axes3[0],
    )
    axes3[0].set_xlabel("Trust threshold (theta)")
    axes3[0].set_ylabel("False accept rate")
    _rewrite_legend_entries(
        axes3[0],
        title="Guard / attack scope",
        mapping={
            "guard_mode": "Guard mode",
            "attack_class": "Attack scope",
        },
        fontsize=8,
    )
    sns.lineplot(
        data=hm_th_03_plot,
        x="theta",
        y="frr",
        hue="guard_mode",
        style="attack_class",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=axes3[1],
    )
    axes3[1].set_xlabel("Trust threshold (theta)")
    axes3[1].set_ylabel("False reject rate")
    _rewrite_legend_entries(
        axes3[1],
        title="Guard / attack scope",
        mapping={
            "guard_mode": "Guard mode",
            "attack_class": "Attack scope",
        },
        fontsize=8,
    )
    th03 = figure_dir / "fig_hm_th_03_far_frr_frontier.pdf"
    fig3.tight_layout()
    fig3.savefig(th03, format="pdf")
    plt.close(fig3)
    paths.append(str(th03))
    qa[str(th03)] = _rasterize_pdf(th03)
    captions[str(th03)] = (
        "Two-panel FAR/FRR frontier across guard composition and attack coverage classes. "
        "X-axis is trust threshold; Y-axes are FAR and FRR respectively, with SD bars over seeds. "
        "Composite guards reduce FAR on covered classes while uncovered classes remain an explicit boundary case."
    )

    return paths, qa, captions


def write_latex_table(df: pd.DataFrame, out_path: Path, caption_note: str) -> None:
    _ensure_parent(out_path)
    rendered = df.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
    wrapped = (
        "% Auto-generated validation table\n"
        f"% Note: {caption_note}\n"
        + rendered
    )
    out_path.write_text(wrapped, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
