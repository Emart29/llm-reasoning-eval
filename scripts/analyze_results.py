# scripts/analyze_results.py
"""Enhanced analysis script for LLM reasoning evaluation results.

Integrates the ResearchReportGenerator to produce:
- Publication-quality visualizations (accuracy by category, difficulty, error heatmaps, etc.)
- Markdown research report with methodology, results, findings, and statistical analysis
- Reproducibility manifest for experiment verification

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 8.2
"""
import argparse
import ast
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT
from src.metrics import MetricsCalculator, ReasoningMetrics
from src.report_generator import ResearchReportGenerator
from src.reproducibility import generate_manifest, save_manifest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results"
ENHANCED_DIR = RESULTS_DIR / "enhanced"
SUMMARY_DIR = RESULTS_DIR / "summary"
FIGURES_DIR = SUMMARY_DIR / "figures"
REPORTS_DIR = SUMMARY_DIR / "reports"
MANIFESTS_DIR = RESULTS_DIR / "manifests"

# Legacy results path for backward compatibility
LEGACY_RESULTS_CSV = SUMMARY_DIR / "results.csv"


def _ensure_dirs():
    """Create all required output directories."""
    for d in [SUMMARY_DIR, FIGURES_DIR, REPORTS_DIR, MANIFESTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_enhanced_results(results_path: Optional[Path] = None) -> pd.DataFrame:
    """Load enhanced results from JSONL file.
    
    Args:
        results_path: Path to the JSONL results file. If None, searches
                     for the most recent results file in ENHANCED_DIR.
    
    Returns:
        DataFrame with evaluation results.
    """
    if results_path is None:
        # Find the most recent results file
        jsonl_files = list(ENHANCED_DIR.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No JSONL results files found in {ENHANCED_DIR}. "
                "Run experiments first with scripts/run_experiments.py"
            )
        results_path = max(jsonl_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading results from: {results_path}")
    
    # Load JSONL file
    records = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        raise ValueError(f"No records found in {results_path}")
    
    return pd.DataFrame(records)


def load_legacy_results() -> pd.DataFrame:
    """Load legacy CSV results for backward compatibility.
    
    Returns:
        DataFrame with evaluation results.
    """
    if not LEGACY_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Legacy results CSV not found at {LEGACY_RESULTS_CSV}"
        )
    return pd.read_csv(LEGACY_RESULTS_CSV)


def normalize_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize results DataFrame to expected format.
    
    Handles both enhanced JSONL format and legacy CSV format.
    
    Args:
        df: Raw results DataFrame
        
    Returns:
        Normalized DataFrame with consistent column names
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Normalize column names
    column_mapping = {
        "correct": "is_correct",
        "problem_id": "problem_id",
        "id": "problem_id",
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Ensure is_correct is boolean
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(bool)
    
    # Parse step_scores if it's a string
    if "step_scores" in df.columns:
        df["step_scores"] = df["step_scores"].apply(_parse_json_field)
    
    # Parse error_categories if it's a string
    if "error_categories" in df.columns:
        df["error_categories"] = df["error_categories"].apply(_parse_json_field)
    elif "error_classification" in df.columns:
        # Extract categories from error_classification
        df["error_categories"] = df["error_classification"].apply(
            lambda x: _extract_error_categories(x)
        )
    
    return df


def _parse_json_field(value: Any) -> Any:
    """Parse a JSON field that might be a string."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
    return value


def _extract_error_categories(classification: Any) -> List[str]:
    """Extract error categories from classification object."""
    if classification is None:
        return []
    if isinstance(classification, str):
        classification = _parse_json_field(classification)
    if isinstance(classification, dict):
        return classification.get("categories", [])
    return []


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics_by_model(df: pd.DataFrame) -> Dict[str, ReasoningMetrics]:
    """Compute metrics for each model in the results.
    
    Args:
        df: Results DataFrame
        
    Returns:
        Dictionary mapping model names to ReasoningMetrics objects
    """
    calculator = MetricsCalculator()
    metrics_by_model = {}
    
    if "model" not in df.columns:
        # Single model case - use "default" as model name
        models = ["default"]
        df = df.copy()
        df["model"] = "default"
    else:
        models = df["model"].unique()
    
    for model in models:
        model_df = df[df["model"] == model]
        
        # Compute basic accuracy
        accuracy = model_df["is_correct"].mean() if "is_correct" in model_df.columns else 0.0
        
        # Compute reasoning depth from step_scores
        reasoning_depth = _compute_reasoning_depth_from_df(model_df)
        
        # Compute recovery rate from propagation data
        recovery_rate = _compute_recovery_rate_from_df(model_df)
        
        # Compute consistency score
        consistency_score = _compute_consistency_from_df(model_df)
        
        # Compute step efficiency
        step_efficiency = _compute_step_efficiency_from_df(model_df)
        
        # Compute error propagation rate
        error_propagation_rate = _compute_propagation_rate_from_df(model_df)
        
        metrics_by_model[model] = ReasoningMetrics(
            accuracy=accuracy,
            reasoning_depth=reasoning_depth,
            recovery_rate=recovery_rate,
            consistency_score=consistency_score,
            step_efficiency=step_efficiency,
            error_propagation_rate=error_propagation_rate,
        )
    
    return metrics_by_model


def _compute_reasoning_depth_from_df(df: pd.DataFrame) -> float:
    """Compute average reasoning depth from DataFrame."""
    if "step_scores" not in df.columns:
        if "reasoning_depth" in df.columns:
            return df["reasoning_depth"].mean()
        return 0.0
    
    depths = []
    for step_scores in df["step_scores"]:
        if not step_scores:
            continue
        
        # Count consecutive correct steps from beginning
        depth = 0
        sorted_scores = sorted(step_scores, key=lambda s: s.get("step_index", 0) if isinstance(s, dict) else 0)
        for score in sorted_scores:
            is_correct = score.get("is_correct", False) if isinstance(score, dict) else False
            if is_correct:
                depth += 1
            else:
                break
        depths.append(depth)
    
    return sum(depths) / len(depths) if depths else 0.0


def _compute_recovery_rate_from_df(df: pd.DataFrame) -> float:
    """Compute recovery rate from DataFrame."""
    if "propagation_analysis" not in df.columns:
        if "recovery_rate" in df.columns:
            return df["recovery_rate"].mean()
        return 0.0
    
    recoverable_count = 0
    total_count = 0
    
    for analysis in df["propagation_analysis"]:
        if analysis is None:
            continue
        if isinstance(analysis, str):
            analysis = _parse_json_field(analysis)
        if isinstance(analysis, dict):
            total_count += 1
            if analysis.get("propagation_type") == "recoverable":
                recoverable_count += 1
    
    return recoverable_count / total_count if total_count > 0 else 0.0


def _compute_consistency_from_df(df: pd.DataFrame) -> float:
    """Compute consistency score from DataFrame."""
    if "consistency_score" in df.columns:
        return df["consistency_score"].mean()
    
    # Group by problem_id and compute consistency
    if "problem_id" not in df.columns or "final_answer" not in df.columns:
        return 1.0  # Default to perfect consistency if no data
    
    from collections import Counter
    
    consistency_scores = []
    for problem_id, group in df.groupby("problem_id"):
        if len(group) < 2:
            continue
        answers = group["final_answer"].tolist()
        answer_counts = Counter(answers)
        most_common_count = answer_counts.most_common(1)[0][1]
        consistency_scores.append(most_common_count / len(answers))
    
    return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0


def _compute_step_efficiency_from_df(df: pd.DataFrame) -> float:
    """Compute average step efficiency from DataFrame."""
    if "step_efficiency" in df.columns:
        return df["step_efficiency"].mean()
    
    # Compute from step counts if available
    efficiencies = []
    for _, row in df.iterrows():
        model_steps = len(row.get("step_scores", [])) if "step_scores" in row else 0
        optimal_steps = row.get("optimal_steps", model_steps)
        
        if model_steps > 0:
            efficiency = min(1.0, optimal_steps / model_steps)
            efficiencies.append(efficiency)
    
    return sum(efficiencies) / len(efficiencies) if efficiencies else 1.0


def _compute_propagation_rate_from_df(df: pd.DataFrame) -> float:
    """Compute error propagation rate from DataFrame."""
    if "propagation_analysis" not in df.columns:
        if "error_propagation_rate" in df.columns:
            return df["error_propagation_rate"].mean()
        return 0.0
    
    cascading_count = 0
    total_count = 0
    
    for analysis in df["propagation_analysis"]:
        if analysis is None:
            continue
        if isinstance(analysis, str):
            analysis = _parse_json_field(analysis)
        if isinstance(analysis, dict):
            total_count += 1
            prop_type = analysis.get("propagation_type", "")
            if prop_type in ("cascading", "terminal"):
                cascading_count += 1
    
    return cascading_count / total_count if total_count > 0 else 0.0


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_all_visualizations(
    df: pd.DataFrame,
    metrics: Dict[str, ReasoningMetrics],
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate all visualizations and save to files.
    
    Args:
        df: Results DataFrame
        metrics: Metrics by model
        output_dir: Directory to save figures
        
    Returns:
        Dictionary mapping figure names to file paths
    """
    generator = ResearchReportGenerator()
    figure_paths = {}
    
    # 1. Accuracy by category
    if "category" in df.columns:
        try:
            fig = generator.generate_accuracy_by_category(df)
            path = output_dir / "accuracy_by_category.png"
            generator.save_figure(fig, str(path))
            figure_paths["accuracy_by_category"] = path
            print(f"  Generated: {path.name}")
        except Exception as e:
            print(f"  Warning: Could not generate accuracy_by_category: {e}")
    
    # 2. Accuracy by difficulty
    if "difficulty" in df.columns:
        try:
            fig = generator.generate_accuracy_by_difficulty(df)
            path = output_dir / "accuracy_by_difficulty.png"
            generator.save_figure(fig, str(path))
            figure_paths["accuracy_by_difficulty"] = path
            print(f"  Generated: {path.name}")
        except Exception as e:
            print(f"  Warning: Could not generate accuracy_by_difficulty: {e}")
    
    # 3. Error heatmap
    if "error_categories" in df.columns or "error_classification" in df.columns:
        try:
            fig = generator.generate_error_heatmap(df)
            path = output_dir / "error_heatmap.png"
            generator.save_figure(fig, str(path))
            figure_paths["error_heatmap"] = path
            print(f"  Generated: {path.name}")
        except Exception as e:
            print(f"  Warning: Could not generate error_heatmap: {e}")
    
    # 4. Step accuracy curves
    if "step_scores" in df.columns:
        try:
            fig = generator.generate_step_accuracy_curves(df)
            path = output_dir / "step_accuracy_curves.png"
            generator.save_figure(fig, str(path))
            figure_paths["step_accuracy_curves"] = path
            print(f"  Generated: {path.name}")
        except Exception as e:
            print(f"  Warning: Could not generate step_accuracy_curves: {e}")
    
    # 5. Model comparison radar
    if metrics:
        try:
            fig = generator.generate_model_comparison_radar(metrics)
            path = output_dir / "model_comparison_radar.png"
            generator.save_figure(fig, str(path))
            figure_paths["model_comparison_radar"] = path
            print(f"  Generated: {path.name}")
        except Exception as e:
            print(f"  Warning: Could not generate model_comparison_radar: {e}")
    
    return figure_paths


def generate_markdown_report(
    df: pd.DataFrame,
    metrics: Dict[str, ReasoningMetrics],
    figure_paths: Dict[str, Path],
    output_path: Path,
) -> str:
    """Generate and save markdown research report.
    
    Args:
        df: Results DataFrame
        metrics: Metrics by model
        figure_paths: Paths to generated figures
        output_path: Path to save the report
        
    Returns:
        The generated markdown report string
    """
    generator = ResearchReportGenerator()
    
    # Generate the report
    report = generator.generate_markdown_report(df, metrics)
    
    # Add figure references
    if figure_paths:
        report += "\n## Figures\n\n"
        for name, path in figure_paths.items():
            # Use relative path from reports directory
            rel_path = f"../figures/{path.name}"
            report += f"### {name.replace('_', ' ').title()}\n\n"
            report += f"![{name}]({rel_path})\n\n"
    
    # Save the report
    generator.save_report(report, str(output_path))
    
    return report


def generate_reproducibility_manifest(
    df: pd.DataFrame,
    results_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate and save reproducibility manifest.
    
    Args:
        df: Results DataFrame
        results_path: Path to the results file (for hash computation)
        output_dir: Directory to save the manifest
        
    Returns:
        Path to the saved manifest file
    """
    if output_dir is None:
        output_dir = MANIFESTS_DIR
    
    # Generate experiment ID from timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"analysis_{timestamp}"
    
    # Custom config with analysis metadata
    custom_config = {
        "analysis_type": "full_report",
        "num_records": len(df),
        "models_analyzed": df["model"].unique().tolist() if "model" in df.columns else ["unknown"],
        "categories_analyzed": df["category"].unique().tolist() if "category" in df.columns else [],
    }
    
    # Generate manifest
    manifest = generate_manifest(
        experiment_id=experiment_id,
        dataset_path=results_path,
        custom_config=custom_config,
        notes=f"Analysis report generated at {timestamp}",
    )
    
    # Save manifest
    manifest_path = save_manifest(manifest, output_dir)
    
    return manifest_path


# ---------------------------------------------------------------------------
# Main Analysis Function
# ---------------------------------------------------------------------------

def run_analysis(
    results_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    use_legacy: bool = False,
) -> Dict[str, Any]:
    """Run complete analysis pipeline.
    
    Generates all visualizations, markdown report, and reproducibility manifest.
    
    Args:
        results_path: Path to results file. If None, uses most recent.
        output_dir: Base output directory. If None, uses default.
        use_legacy: If True, use legacy CSV format instead of JSONL.
        
    Returns:
        Dictionary with paths to all generated outputs.
    """
    print("=" * 60)
    print("LLM Reasoning Evaluation - Analysis Report Generator")
    print("=" * 60)
    
    # Ensure output directories exist
    _ensure_dirs()
    
    if output_dir is None:
        output_dir = SUMMARY_DIR
    
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\n1. Loading results...")
    try:
        if use_legacy:
            df = load_legacy_results()
            print(f"   Loaded {len(df)} records from legacy CSV")
        else:
            df = load_enhanced_results(results_path)
            print(f"   Loaded {len(df)} records from JSONL")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        print("   Attempting to load legacy results...")
        try:
            df = load_legacy_results()
            print(f"   Loaded {len(df)} records from legacy CSV")
        except FileNotFoundError:
            print("   No results found. Run experiments first.")
            return {}
    
    # Normalize DataFrame
    print("\n2. Normalizing data...")
    df = normalize_results_dataframe(df)
    print(f"   Columns: {list(df.columns)}")
    
    # Compute metrics
    print("\n3. Computing metrics...")
    metrics = compute_metrics_by_model(df)
    for model, m in metrics.items():
        print(f"   {model}:")
        print(f"     - Accuracy: {m.accuracy:.1%}")
        print(f"     - Reasoning Depth: {m.reasoning_depth:.2f}")
        print(f"     - Recovery Rate: {m.recovery_rate:.1%}")
        print(f"     - Consistency: {m.consistency_score:.1%}")
        print(f"     - Step Efficiency: {m.step_efficiency:.1%}")
    
    # Generate visualizations
    print("\n4. Generating visualizations...")
    figure_paths = generate_all_visualizations(df, metrics, figures_dir)
    
    # Generate markdown report
    print("\n5. Generating markdown report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"research_report_{timestamp}.md"
    report = generate_markdown_report(df, metrics, figure_paths, report_path)
    print(f"   Saved: {report_path.name}")
    
    # Also save a "latest" version
    latest_report_path = reports_dir / "research_report_latest.md"
    with open(latest_report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"   Saved: {latest_report_path.name}")
    
    # Generate reproducibility manifest
    print("\n6. Generating reproducibility manifest...")
    manifest_path = generate_reproducibility_manifest(df, results_path)
    print(f"   Saved: {manifest_path.name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Figures: {figures_dir}")
    print(f"  Report:  {report_path}")
    print(f"  Manifest: {manifest_path}")
    
    return {
        "figures": figure_paths,
        "report": report_path,
        "manifest": manifest_path,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate analysis report for LLM reasoning evaluation results"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to results JSONL file (default: most recent in results/enhanced/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/summary/)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy CSV format instead of JSONL",
    )
    
    args = parser.parse_args()
    
    run_analysis(
        results_path=args.results,
        output_dir=args.output,
        use_legacy=args.legacy,
    )


if __name__ == "__main__":
    main()
