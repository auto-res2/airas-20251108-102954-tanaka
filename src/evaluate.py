"""src/evaluate.py
Independent evaluation and visualisation script.

CLI (MANDATORY):
    uv run python -m src.evaluate results_dir=PATH run_ids='["run-1", "run-2"]'

This script purposefully does NOT use Hydra; instead it parses the
`key=value` style arguments required by the specification.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy import stats

# -------------------------------------------------------------------------------------
# Make repository importable regardless of CWD ----------------------------------------
# -------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -------------------------------------------------------------------------------------
# CLI Parsing (key=value style) --------------------------------------------------------
# -------------------------------------------------------------------------------------

def parse_cli(argv: List[str]) -> dict[str, str]:
    """Parses `key=value` arguments into a dictionary.

    Parameters
    ----------
    argv : List[str]
        sys.argv list (including program name).

    Returns
    -------
    dict[str, str]
        Mapping from argument names (str) to their raw string values.
    """
    if len(argv) <= 1:
        raise SystemExit(
            "No arguments supplied. Expected usage: python -m src.evaluate "
            "results_dir=PATH run_ids='[\"run-1\"]'"
        )

    parsed: dict[str, str] = {}
    for arg in argv[1:]:
        if "=" not in arg:
            raise SystemExit(
                f"Invalid argument '{arg}'. Expected key=value style with '=' present."
            )
        key, value = arg.split("=", 1)
        if not key:
            raise SystemExit(f"Malformed argument '{arg}' (empty key before '=').")
        parsed[key] = value
    return parsed


# -------------------------------------------------------------------------------------
# Helper utilities --------------------------------------------------------------------
# -------------------------------------------------------------------------------------

def load_wandb_entity_project() -> tuple[str, str]:
    """Reads entity/project from config/config.yaml (root of repo)."""
    import yaml

    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot locate {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    return cfg["wandb"]["entity"], cfg["wandb"]["project"]


def plot_learning_curve(history_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=history_df, x=history_df.index, y=metric)
    best_idx = history_df[metric].idxmax()
    best_val = history_df.loc[best_idx, metric]
    plt.scatter([best_idx], [best_val], color="red")
    plt.annotate(f"{best_val:.3f}", (best_idx, best_val))
    plt.title(metric)
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_confusion_matrix(cm: list[list[int]], out_path: Path) -> None:
    cm_arr = np.array(cm)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def bar_chart(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(x="run_id", y=metric, data=df)
    plt.xticks(rotation=45, ha="right")
    for p in plt.gca().patches:
        height = p.get_height()
        plt.text(
            p.get_x() + p.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel(metric)
    plt.title(f"Comparison – {metric}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def box_plot(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=metric, data=df)
    sns.stripplot(y=metric, data=df, color="red", jitter=0.2, size=4)
    plt.title(f"Distribution of {metric}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------------------------------------
# Main --------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    # ------------------------------ Parse CLI ----------------------------------------
    cli_args = parse_cli(sys.argv)
    if "results_dir" not in cli_args or "run_ids" not in cli_args:
        raise SystemExit(
            "Both 'results_dir' and 'run_ids' must be specified. Example: "
            "python -m src.evaluate results_dir=PATH run_ids='[\"run-1\"]'"
        )

    results_dir = Path(cli_args["results_dir"]).expanduser()
    run_ids: List[str] = json.loads(cli_args["run_ids"])

    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------ WandB API ----------------------------------------
    entity, project = load_wandb_entity_project()
    api = wandb.Api()

    per_run_records: list[dict] = []
    generated_files: list[Path] = []

    # ------------------------------ Per-run processing ------------------------------
    for run_id in run_ids:
        run_out_dir = results_dir / run_id
        run_out_dir.mkdir(parents=True, exist_ok=True)

        run = api.run(f"{entity}/{project}/{run_id}")
        history_df = run.history()  # pandas DataFrame with step index
        summary = dict(run.summary._json_dict)
        config = dict(run.config)

        # --------- Save metrics.json -------------------------------------------------
        metrics_path = run_out_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as fp:
            json.dump({"summary": summary, "config": config}, fp, indent=2)
        generated_files.append(metrics_path)

        # --------- Learning curves ---------------------------------------------------
        for metric in [m for m in ["train_acc", "val_acc"] if m in history_df.columns]:
            fig_path = run_out_dir / f"{run_id}_learning_curve_{metric}.pdf"
            plot_learning_curve(history_df, metric, fig_path)
            generated_files.append(fig_path)

        # --------- Confusion matrix (if present) -------------------------------------
        if "test_confusion_matrix" in summary:
            cm_fig_path = run_out_dir / f"{run_id}_confusion_matrix.pdf"
            save_confusion_matrix(summary["test_confusion_matrix"], cm_fig_path)
            generated_files.append(cm_fig_path)

        per_run_record = {"run_id": run_id}
        per_run_record.update({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        per_run_records.append(per_run_record)

    # ------------------------------ Aggregated analysis -----------------------------
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    comp_df = pd.DataFrame(per_run_records)
    comp_json_path = comp_dir / "aggregated_metrics.json"
    comp_df.to_json(comp_json_path, orient="records", indent=2)
    generated_files.append(comp_json_path)

    # ------------------------------ Improvement rates -------------------------------
    if len(comp_df) >= 2 and "val_acc" in comp_df.columns:
        baseline_val = comp_df.iloc[0]["val_acc"]
        comp_df["improvement_rate"] = (comp_df["val_acc"] - baseline_val) / baseline_val

    # ------------------------------ Figures -----------------------------------------
    metric_for_bar = "val_acc" if "val_acc" in comp_df.columns else comp_df.columns[1]
    bar_path = comp_dir / f"comparison_{metric_for_bar}_bar_chart.pdf"
    bar_chart(comp_df, metric_for_bar, bar_path)
    generated_files.append(bar_path)

    box_path = comp_dir / f"comparison_{metric_for_bar}_box_plot.pdf"
    box_plot(comp_df, metric_for_bar, box_path)
    generated_files.append(box_path)

    # ------------------------------ Statistical test --------------------------------
    if len(comp_df) >= 2 and metric_for_bar in comp_df.columns:
        a, b = comp_df.iloc[0][metric_for_bar], comp_df.iloc[1][metric_for_bar]
        t_stat, p_val = stats.ttest_ind([a], [b], equal_var=False)
        stats_path = comp_dir / "significance_tests.json"
        with open(stats_path, "w", encoding="utf-8") as fp:
            json.dump({"metric": metric_for_bar, "t_stat": t_stat, "p_value": p_val}, fp, indent=2)
        generated_files.append(stats_path)

    # ------------------------------ Print generated files ---------------------------
    for path in generated_files:
        print(path)


if __name__ == "__main__":
    # Respect WANDB_API_KEY if provided in environment; if not, log-in-less.
    os.environ.setdefault("WANDB_SILENT", "true")
    main()