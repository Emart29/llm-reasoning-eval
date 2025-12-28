# src/analysis.py
"""Statistical analysis and visualisation utilities.
Exports a single function `run_analysis` that reads the aggregated CSV
(`results/summary/results.csv`) and produces:
* Accuracy bar‑plot per model/strategy
* Error‑type heat‑map
* McNemar paired test between strategies for the same model
* Wilcoxon signed‑rank test on self‑consistency vote confidence (if available)
All figures are saved under `results/summary/figures/`.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mcnemar, wilcoxon

RESULTS_CSV = Path(__file__).resolve().parents[2] / "results" / "summary" / "results.csv"
FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "summary" / "figures"

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _plot_accuracy(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="model", y="accuracy", hue="strategy")
    plt.ylim(0, 1)
    plt.title("Accuracy per Model & Prompting Strategy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    _ensure_dir(FIG_DIR)
    plt.savefig(FIG_DIR / "accuracy_bar.png")
    plt.close()

def _plot_error_heatmap(df: pd.DataFrame):
    err_counts = df.groupby(["model", "strategy", "error_type"]).size().unstack(fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(err_counts, annot=True, fmt="d", cmap="Reds")
    plt.title("Error‑type frequencies per Model & Strategy")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "error_heatmap.png")
    plt.close()

def _run_mcnemar(df: pd.DataFrame):
    # Perform McNemar between zero_shot and cot for each model (binary outcomes per sample)
    results = []
    for model in df["model"].unique():
        zs = df[(df["model"] == model) & (df["strategy"] == "zero_shot")]
        cot = df[(df["model"] == model) & (df["strategy"] == "cot")]
        # Align on sample id
        merged = pd.merge(zs[["sample_id", "correct"]], cot[["sample_id", "correct"]], on="sample_id", suffixes=("_zs", "_cot"))
        table = pd.crosstab(merged["correct_zs"], merged["correct_cot"])
        if table.shape == (2, 2):
            stat, p = mcnemar(table, exact=True)
            results.append({"model": model, "statistic": stat, "p_value": p})
    return pd.DataFrame(results)

def _run_wilcoxon(df: pd.DataFrame):
    # For self_consistency we may have a column `vote_agreement` (fraction of samples where majority vote matched). If not present, skip.
    if "vote_agreement" not in df.columns:
        return pd.DataFrame()
    results = []
    for model in df["model"].unique():
        sc = df[(df["model"] == model) & (df["strategy"] == "self_consistency")]
        if not sc.empty:
            # Compare vote_agreement against a baseline of 0.5 (random)
            stat, p = wilcoxon(sc["vote_agreement"] - 0.5)
            results.append({"model": model, "statistic": stat, "p_value": p})
    return pd.DataFrame(results)

def run_analysis():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results CSV not found at {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)
    # Basic accuracy aggregation
    acc = df.groupby(["model", "strategy"]).agg(accuracy=("correct", "mean")).reset_index()
    _plot_accuracy(acc)
    _plot_error_heatmap(df)
    # Statistical tests
    mcnemar_df = _run_mcnemar(df)
    wilcoxon_df = _run_wilcoxon(df)
    # Save test tables
    _ensure_dir(FIG_DIR)
    mcnemar_df.to_csv(FIG_DIR / "mcnemar_results.csv", index=False)
    wilcoxon_df.to_csv(FIG_DIR / "wilcoxon_results.csv", index=False)
    print("Analysis complete. Figures and test tables saved to", FIG_DIR)

if __name__ == "__main__":
    run_analysis()
