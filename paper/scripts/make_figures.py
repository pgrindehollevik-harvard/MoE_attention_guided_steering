"""Render paper figures and headline tables from the sweep CSVs.

Intended call order:

    python3 paper/scripts/aggregate_results.py
    python3 paper/scripts/make_figures.py

Inputs are produced by ``aggregate_results.py``:

    paper/notebooks/sweep_results.csv      -- one row per (run, response)
    paper/notebooks/expert_selection.csv   -- one row per (run, layer, expert)

Outputs are written to ``paper/docs/paper/figures/`` and
``paper/docs/paper/tables/``. The script is deliberately lightweight: it
relies on pandas + matplotlib and skips a figure cleanly if the upstream
data is missing rather than aborting the whole render.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_CSV = REPO_ROOT / "paper" / "notebooks" / "sweep_results.csv"
EXPERT_CSV = REPO_ROOT / "paper" / "notebooks" / "expert_selection.csv"
FIG_DIR = REPO_ROOT / "paper" / "docs" / "paper" / "figures"
TBL_DIR = REPO_ROOT / "paper" / "docs" / "paper" / "tables"


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep = pd.read_csv(SWEEP_CSV) if SWEEP_CSV.exists() else pd.DataFrame()
    experts = pd.read_csv(EXPERT_CSV) if EXPERT_CSV.exists() else pd.DataFrame()
    return sweep, experts


def _question_only_runs(sweep: pd.DataFrame) -> pd.DataFrame:
    if sweep.empty:
        return sweep
    keep = sweep["run_kind"].eq("question_only") & sweep["condition_kind"].isin(
        ["mixtral_baseline", "mixtral_steermoe"]
    )
    return sweep.loc[keep].copy()


def headline_response_shape(sweep: pd.DataFrame) -> pd.DataFrame:
    """Aggregate response-shape diagnostics per (run, condition)."""

    if sweep.empty:
        return pd.DataFrame()

    df = sweep.copy()
    df["concept_token_hit"] = df["concept_token_hit"].astype(float)
    df["exceeds_20_words"] = df["exceeds_20_words"].astype(float)
    grouped = df.groupby(
        [
            "run_name",
            "run_kind",
            "prompt_mode",
            "steering_coefficient",
            "top_positive_experts",
            "top_negative_experts",
            "condition_kind",
        ],
        dropna=False,
    ).agg(
        n_responses=("response", "size"),
        mean_word_count=("word_count", "mean"),
        concept_token_hit_rate=("concept_token_hit", "mean"),
        exceeds_20_words_rate=("exceeds_20_words", "mean"),
    )
    return grouped.reset_index()


def steermoe_vs_baseline_delta(table: pd.DataFrame) -> pd.DataFrame:
    """Same-run paired delta between Mixtral baseline and Mixtral + SteerMoE."""

    if table.empty:
        return table

    keep = table["condition_kind"].isin(["mixtral_baseline", "mixtral_steermoe"])
    sub = table.loc[keep].copy()
    if sub.empty:
        return sub

    pivot_cols = ["mean_word_count", "concept_token_hit_rate", "exceeds_20_words_rate"]
    keys = [
        "run_name",
        "prompt_mode",
        "steering_coefficient",
        "top_positive_experts",
        "top_negative_experts",
    ]
    pivot = sub.pivot_table(
        index=keys,
        columns="condition_kind",
        values=pivot_cols,
        aggfunc="first",
    )
    pivot.columns = [f"{cond}__{stat}" for stat, cond in pivot.columns]
    pivot = pivot.reset_index()

    for stat in pivot_cols:
        col_steered = f"mixtral_steermoe__{stat}"
        col_baseline = f"mixtral_baseline__{stat}"
        if col_steered in pivot and col_baseline in pivot:
            pivot[f"delta__{stat}"] = pivot[col_steered] - pivot[col_baseline]
    return pivot


def expert_concentration(experts: pd.DataFrame) -> pd.DataFrame:
    """How concentrated is the risk-difference signal per (run, concept)."""

    if experts.empty:
        return experts

    df = experts.copy()
    df["abs_risk_difference"] = df["abs_risk_difference"].astype(float)
    grouped = (
        df.groupby(["run_name", "concept"], dropna=False)["abs_risk_difference"]
        .agg(
            n_layer_expert_cells="size",
            max_abs_risk="max",
            mean_abs_risk="mean",
        )
        .reset_index()
    )

    def _topk_share(group: pd.DataFrame, k: int) -> float:
        sorted_vals = group.sort_values("abs_risk_difference", ascending=False)
        head = sorted_vals.head(k)["abs_risk_difference"].sum()
        total = group["abs_risk_difference"].sum()
        return float(head / total) if total > 0 else float("nan")

    top20 = (
        df.groupby(["run_name", "concept"])
        .apply(lambda g: _topk_share(g, 20), include_groups=False)
        .rename("top20_share")
        .reset_index()
    )
    return grouped.merge(top20, on=["run_name", "concept"], how="left")


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _escape_latex(value: object) -> object:
    if not isinstance(value, str):
        return value
    return value.replace("\\", "\\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def _save_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = df.copy()
    safe.columns = [_escape_latex(str(c)) for c in safe.columns]
    for col in safe.columns:
        if safe[col].dtype == object:
            safe[col] = safe[col].map(_escape_latex)
    text = safe.to_latex(index=False, float_format="%.3f", na_rep="--")
    wrapped = dedent(
        f"""\
        \\begin{{table}}[ht]
          \\centering
          \\caption{{{caption}}}
          \\label{{{label}}}
        """
    )
    wrapped += text + "\n\\end{table}\n"
    path.write_text(wrapped)


def fig_steering_strength_sweep(table: pd.DataFrame) -> Path | None:
    """Plot mean response length vs steering coefficient for question-only ablations."""

    if table.empty:
        return None

    sub = table[
        table["run_kind"].eq("question_only")
        & table["condition_kind"].isin(["mixtral_baseline", "mixtral_steermoe"])
    ].copy()
    if sub.empty or sub["steering_coefficient"].isna().all():
        return None

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for kind, marker in [("mixtral_baseline", "o"), ("mixtral_steermoe", "s")]:
        chunk = sub[sub["condition_kind"].eq(kind)]
        if chunk.empty:
            continue
        ax.scatter(
            chunk["steering_coefficient"].fillna(0.0),
            chunk["mean_word_count"],
            label=kind.replace("_", " "),
            marker=marker,
        )
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("Mean response length (words)")
    ax.set_title("Question-only response length vs steering strength")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "steering_strength_sweep.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_expert_concentration(table: pd.DataFrame) -> Path | None:
    """Histogram of top-20 risk-difference share per (run, concept)."""

    if table.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.hist(table["top20_share"].dropna(), bins=20)
    ax.set_xlabel("Share of total |risk difference| in top 20 experts")
    ax.set_ylabel("Count of (run, concept)")
    ax.set_title("Concentration of router-bias signal")
    fig.tight_layout()
    out = FIG_DIR / "expert_concentration.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true", help="emit only CSV/LaTeX tables")
    args = parser.parse_args()

    sweep, experts = _load()

    shape_table = headline_response_shape(sweep)
    delta_table = steermoe_vs_baseline_delta(shape_table)
    concentration_table = expert_concentration(experts)

    _save_csv(shape_table, TBL_DIR / "response_shape.csv")
    _save_csv(delta_table, TBL_DIR / "steermoe_vs_baseline_delta.csv")
    _save_csv(concentration_table, TBL_DIR / "expert_concentration.csv")

    headline_cols = [
        "run_name",
        "steering_coefficient",
        "top_positive_experts",
        "top_negative_experts",
        "delta__mean_word_count",
        "delta__concept_token_hit_rate",
        "delta__exceeds_20_words_rate",
    ]
    if not delta_table.empty and all(c in delta_table.columns for c in headline_cols):
        _save_latex(
            delta_table[headline_cols].sort_values("run_name"),
            TBL_DIR / "headline_delta.tex",
            caption=(
                "Same-run paired deltas between Mixtral baseline and "
                "Mixtral + SteerMoE on question-only generations."
            ),
            label="tab:headline_delta",
        )

    fig_paths: list[Path] = []
    if not args.no_figures:
        for fig_fn in (fig_steering_strength_sweep, fig_expert_concentration):
            target = (
                fig_fn(shape_table)
                if fig_fn is fig_steering_strength_sweep
                else fig_fn(concentration_table)
            )
            if target is not None:
                fig_paths.append(target)

    print("tables written:", TBL_DIR)
    print("figures written:")
    for p in fig_paths:
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
