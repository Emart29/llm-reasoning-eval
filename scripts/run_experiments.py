# scripts/run_experiments.py
"""Enhanced experiment driver for LLM reasoning evaluation.

Integrates all analysis components:
- Chain-of-Thought Analyzer for step parsing
- Step Accuracy Scorer for per-step evaluation
- Error Taxonomy Engine for error classification
- Error Propagation Tracker for cascade analysis
- Metrics Calculator for novel research metrics

Stores detailed results in enhanced JSONL format with full analysis data.

Requirements: All
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

# Core modules
from src.config import DATA_PATH, MODELS, RANDOM_SEED, REPO_ROOT, STRATEGIES
from src.dataset import get_test_split, load_dataset
from src.evaluation import evaluate_prediction, extract_final_answer
from src.inference import get_responses, APIError

# Analysis components
from src.cot_analyzer import ChainOfThoughtAnalyzer
from src.step_scorer import StepAccuracyScorer
from src.error_taxonomy import ErrorTaxonomyEngine
from src.error_propagation import ErrorPropagationTracker
from src.metrics import MetricsCalculator

# Data models
from src.models import (
    EvaluationResult,
    ParsedChainOfThought,
    ReasoningMetrics,
    StepScore,
    ErrorClassification,
    PropagationAnalysis,
)

# Reproducibility
from src.reproducibility import generate_manifest, save_manifest, set_random_seeds

# API logging
from src.api_logger import get_api_logger, set_api_logger, APILogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default models - Groq models for fast inference
DEFAULT_MODELS = ["groq_qwen3_32b", "groq_gpt_oss_120b", "groq_gpt_oss_20b", "groq_llama3_70b", "groq_llama3_8b"]
MODEL_KEYS = list(MODELS.keys())
STRATEGY_KEYS = list(STRATEGIES.keys())

# Output directories
RAW_DIR = REPO_ROOT / "results" / "raw"
ENHANCED_DIR = REPO_ROOT / "results" / "enhanced"
SUMMARY_DIR = REPO_ROOT / "results" / "summary"
LOGS_DIR = REPO_ROOT / "results" / "logs"
MANIFESTS_DIR = REPO_ROOT / "results" / "manifests"


def _ensure_dirs():
    """Create all required output directories."""
    for d in [RAW_DIR, ENHANCED_DIR, SUMMARY_DIR, LOGS_DIR, MANIFESTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _load_problem_data(problem_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Load problem data including solution steps if available."""
    row = df[df["id"] == problem_id].iloc[0]
    
    # Handle solution_steps - could be a string (JSON) or already a list
    solution_steps = row.get("solution_steps", [])
    if isinstance(solution_steps, str):
        try:
            solution_steps = json.loads(solution_steps)
        except (json.JSONDecodeError, TypeError):
            solution_steps = []
    elif not isinstance(solution_steps, list):
        solution_steps = []
    
    return {
        "id": row["id"],
        "category": row.get("category", "unknown"),
        "subtype": row.get("subtype", "unknown"),
        "difficulty": row.get("difficulty", 3),
        "prompt": row["prompt"],
        "ground_truth": row["ground_truth"],
        "solution_steps": solution_steps,
        "adversarial": row.get("adversarial", False),
    }


def _serialize_step_scores(step_scores: List[StepScore]) -> List[Dict[str, Any]]:
    """Serialize step scores to JSON-compatible format."""
    return [
        {
            "step_index": s.step_index,
            "is_correct": s.is_correct,
            "confidence": s.confidence,
            "error_details": s.error_details,
        }
        for s in step_scores
    ]


def _serialize_error_classification(
    classification: Optional[ErrorClassification],
) -> Optional[Dict[str, Any]]:
    """Serialize error classification to JSON-compatible format."""
    if classification is None:
        return None
    return {
        "categories": [c.value for c in classification.categories],
        "confidence_scores": {
            c.value: score for c, score in classification.confidence_scores.items()
        },
        "primary_category": classification.primary_category.value,
        "explanation": classification.explanation,
        "flagged_for_review": classification.flagged_for_review,
    }


def _serialize_propagation_analysis(
    analysis: Optional[PropagationAnalysis],
) -> Optional[Dict[str, Any]]:
    """Serialize propagation analysis to JSON-compatible format."""
    if analysis is None:
        return None
    return {
        "first_error_step": analysis.first_error_step,
        "propagation_type": analysis.propagation_type.value,
        "affected_steps": analysis.affected_steps,
        "recovery_attempted": analysis.recovery_attempted,
        "recovery_successful": analysis.recovery_successful,
    }


def _serialize_parsed_cot(cot: Optional[ParsedChainOfThought]) -> Optional[Dict[str, Any]]:
    """Serialize parsed chain-of-thought to JSON-compatible format."""
    if cot is None:
        return None
    return {
        "steps": [
            {
                "index": s.index,
                "content": s.content,
                "step_type": s.step_type,
                "extracted_values": s.extracted_values,
            }
            for s in cot.steps
        ],
        "final_answer": cot.final_answer,
        "parse_success": cot.parse_success,
        "parse_errors": cot.parse_errors,
    }



class EnhancedExperimentRunner:
    """Enhanced experiment runner with full analysis pipeline.
    
    Integrates all analysis components:
    - Chain-of-Thought Analyzer for step parsing
    - Step Accuracy Scorer for per-step evaluation
    - Error Taxonomy Engine for error classification
    - Error Propagation Tracker for cascade analysis
    - Metrics Calculator for novel research metrics
    """
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        experiment_id: Optional[str] = None,
    ):
        """Initialize the experiment runner.
        
        Args:
            models: List of model keys to evaluate. Defaults to all models.
            strategies: List of strategy keys to use. Defaults to all strategies.
            experiment_id: Unique identifier for this experiment run.
        """
        self.models = models or DEFAULT_MODELS
        self.strategies = strategies or STRATEGY_KEYS
        
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = experiment_id or f"exp_{timestamp}"
        
        # Initialize analysis components
        self.cot_analyzer = ChainOfThoughtAnalyzer()
        self.step_scorer = StepAccuracyScorer()
        self.error_taxonomy = ErrorTaxonomyEngine()
        self.propagation_tracker = ErrorPropagationTracker()
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize API logger
        self.api_logger = APILogger(
            log_dir=LOGS_DIR,
            experiment_id=self.experiment_id,
        )
        set_api_logger(self.api_logger)
        
        # Results storage
        self.results: List[EvaluationResult] = []
        
        # Ensure output directories exist
        _ensure_dirs()
    
    def run(
        self,
        dataset_frac: float = 1.0,
        save_results: bool = True,
        verbose: bool = True,
    ) -> List[EvaluationResult]:
        """Run the full experiment pipeline.
        
        Args:
            dataset_frac: Fraction of dataset to use (for testing).
            save_results: Whether to save results to files.
            verbose: Whether to print progress information.
            
        Returns:
            List of EvaluationResult objects.
        """
        # Set random seeds for reproducibility
        set_random_seeds(RANDOM_SEED)
        
        # Load dataset
        df = get_test_split(frac=dataset_frac)
        if verbose:
            print(f"Loaded {len(df)} problems for evaluation")
        
        # Generate reproducibility manifest
        manifest = generate_manifest(
            experiment_id=self.experiment_id,
            dataset_path=DATA_PATH,
            custom_config={
                "models": self.models,
                "strategies": self.strategies,
                "dataset_frac": dataset_frac,
            },
            notes=f"Enhanced experiment run with {len(df)} problems",
        )
        
        # Save manifest
        if save_results:
            save_manifest(manifest, output_dir=MANIFESTS_DIR)
        
        # Run evaluations
        total_evals = len(df) * len(self.models) * len(self.strategies)
        
        with tqdm(total=total_evals, desc="Running evaluations", disable=not verbose) as pbar:
            for _, row in df.iterrows():
                problem_data = _load_problem_data(row["id"], df)
                
                for model_key in self.models:
                    for strategy_key in self.strategies:
                        try:
                            result = self._evaluate_single(
                                problem_data=problem_data,
                                model_key=model_key,
                                strategy_key=strategy_key,
                            )
                            self.results.append(result)
                        except Exception as e:
                            if verbose:
                                print(f"Error evaluating {row['id']} with {model_key}/{strategy_key}: {e}")
                            # Create a failed result
                            result = self._create_failed_result(
                                problem_data=problem_data,
                                model_key=model_key,
                                strategy_key=strategy_key,
                                error=str(e),
                            )
                            self.results.append(result)
                        
                        pbar.update(1)
        
        # Save results
        if save_results:
            self._save_results()
        
        return self.results
    
    def _evaluate_single(
        self,
        problem_data: Dict[str, Any],
        model_key: str,
        strategy_key: str,
    ) -> EvaluationResult:
        """Evaluate a single problem with a specific model and strategy.
        
        Args:
            problem_data: Problem data dictionary.
            model_key: Model configuration key.
            strategy_key: Strategy configuration key.
            
        Returns:
            EvaluationResult with full analysis.
        """
        start_time = time.time()
        
        # Get model response
        strategy_cfg = STRATEGIES[strategy_key]
        responses = get_responses(
            model_key=model_key,
            strategy_key=strategy_key,
            prompt=problem_data["prompt"],
            strategy_cfg=strategy_cfg,
            normalize=True,
            api_logger=self.api_logger,
        )
        
        # Use first response (or aggregate for self-consistency)
        raw_output = responses[0] if responses else ""
        
        # Calculate API latency
        api_latency_ms = int((time.time() - start_time) * 1000)
        
        # Parse chain-of-thought
        parsed_cot = self.cot_analyzer.parse_output(raw_output)
        
        # Extract final answer
        final_answer = extract_final_answer(raw_output)
        
        # Evaluate correctness
        is_correct, _ = evaluate_prediction(raw_output, problem_data["ground_truth"])
        
        # Score steps
        ground_truth_steps = problem_data.get("solution_steps", [])
        step_scores = self.step_scorer.score_all_steps(parsed_cot, ground_truth_steps)
        
        # Classify errors (only if incorrect)
        error_classification = None
        if not is_correct and step_scores:
            error_classification = self.error_taxonomy.classify_error(
                model_output=raw_output,
                ground_truth=problem_data["ground_truth"],
                step_scores=step_scores,
            )
        
        # Analyze error propagation
        propagation_analysis = None
        if step_scores:
            propagation_analysis = self.propagation_tracker.analyze_propagation(
                step_scores=step_scores,
                final_correct=is_correct,
            )
        
        # Compute per-problem metrics
        metrics = self._compute_problem_metrics(
            step_scores=step_scores,
            parsed_cot=parsed_cot,
            ground_truth_steps=ground_truth_steps,
            is_correct=is_correct,
        )
        
        return EvaluationResult(
            problem_id=problem_data["id"],
            model=model_key,
            strategy=strategy_key,
            raw_output=raw_output,
            parsed_cot=parsed_cot,
            step_scores=step_scores,
            error_classification=error_classification,
            propagation_analysis=propagation_analysis,
            final_answer=final_answer,
            is_correct=is_correct,
            metrics=metrics,
            timestamp=datetime.now(),
            api_latency_ms=api_latency_ms,
        )
    
    def _create_failed_result(
        self,
        problem_data: Dict[str, Any],
        model_key: str,
        strategy_key: str,
        error: str,
    ) -> EvaluationResult:
        """Create a result for a failed evaluation.
        
        Args:
            problem_data: Problem data dictionary.
            model_key: Model configuration key.
            strategy_key: Strategy configuration key.
            error: Error message.
            
        Returns:
            EvaluationResult with error information.
        """
        return EvaluationResult(
            problem_id=problem_data["id"],
            model=model_key,
            strategy=strategy_key,
            raw_output=f"ERROR: {error}",
            parsed_cot=None,
            step_scores=[],
            error_classification=None,
            propagation_analysis=None,
            final_answer="",
            is_correct=False,
            metrics={"error": 1.0},
            timestamp=datetime.now(),
            api_latency_ms=0,
        )
    
    def _compute_problem_metrics(
        self,
        step_scores: List[StepScore],
        parsed_cot: ParsedChainOfThought,
        ground_truth_steps: List[str],
        is_correct: bool,
    ) -> Dict[str, float]:
        """Compute per-problem metrics.
        
        Args:
            step_scores: List of step scores.
            parsed_cot: Parsed chain-of-thought.
            ground_truth_steps: Ground truth solution steps.
            is_correct: Whether the final answer is correct.
            
        Returns:
            Dictionary of metric values.
        """
        metrics = {}
        
        # Reasoning depth (consecutive correct steps from start)
        depth = 0
        for score in sorted(step_scores, key=lambda s: s.step_index):
            if score.is_correct:
                depth += 1
            else:
                break
        metrics["reasoning_depth"] = float(depth)
        
        # Step accuracy
        if step_scores:
            correct_steps = sum(1 for s in step_scores if s.is_correct)
            metrics["step_accuracy"] = correct_steps / len(step_scores)
        else:
            metrics["step_accuracy"] = 0.0
        
        # Step efficiency
        model_steps = len(parsed_cot.steps) if parsed_cot else 0
        optimal_steps = len(ground_truth_steps)
        metrics["model_steps"] = float(model_steps)
        metrics["optimal_steps"] = float(optimal_steps)
        
        if model_steps > 0 and optimal_steps > 0:
            metrics["step_efficiency"] = min(1.0, optimal_steps / model_steps)
        else:
            metrics["step_efficiency"] = 1.0 if model_steps == 0 and optimal_steps == 0 else 0.0
        
        # Final correctness
        metrics["is_correct"] = 1.0 if is_correct else 0.0
        
        return metrics

    
    def _save_results(self) -> None:
        """Save results to files in enhanced JSONL format."""
        # Save enhanced results (full analysis)
        enhanced_path = ENHANCED_DIR / f"results_{self.experiment_id}.jsonl"
        with open(enhanced_path, "w", encoding="utf-8") as f:
            for result in self.results:
                record = self._serialize_result(result)
                f.write(json.dumps(record, default=str) + "\n")
        
        print(f"Enhanced results saved to: {enhanced_path}")
        
        # Save summary metrics
        summary = self._compute_summary()
        summary_path = SUMMARY_DIR / f"summary_{self.experiment_id}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary saved to: {summary_path}")
        
        # Save API logger summary
        self.api_logger.save_summary()
    
    def _serialize_result(self, result: EvaluationResult) -> Dict[str, Any]:
        """Serialize an EvaluationResult to JSON-compatible format.
        
        Args:
            result: The evaluation result to serialize.
            
        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "problem_id": result.problem_id,
            "model": result.model,
            "strategy": result.strategy,
            "raw_output": result.raw_output,
            "parsed_cot": _serialize_parsed_cot(result.parsed_cot),
            "step_scores": _serialize_step_scores(result.step_scores),
            "error_classification": _serialize_error_classification(result.error_classification),
            "propagation_analysis": _serialize_propagation_analysis(result.propagation_analysis),
            "final_answer": result.final_answer,
            "is_correct": result.is_correct,
            "metrics": result.metrics,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "api_latency_ms": result.api_latency_ms,
        }
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics from all results.
        
        Returns:
            Dictionary of summary statistics.
        """
        if not self.results:
            return {"error": "No results to summarize"}
        
        # Compute aggregate metrics
        aggregate_metrics = self.metrics_calculator.compute_all_metrics(self.results)
        
        # Compute per-model metrics
        model_metrics = {}
        for model_key in self.models:
            model_results = [r for r in self.results if r.model == model_key]
            if model_results:
                model_metrics[model_key] = {
                    "accuracy": sum(1 for r in model_results if r.is_correct) / len(model_results),
                    "count": len(model_results),
                    "avg_latency_ms": sum(r.api_latency_ms for r in model_results) / len(model_results),
                }
        
        # Compute per-strategy metrics
        strategy_metrics = {}
        for strategy_key in self.strategies:
            strategy_results = [r for r in self.results if r.strategy == strategy_key]
            if strategy_results:
                strategy_metrics[strategy_key] = {
                    "accuracy": sum(1 for r in strategy_results if r.is_correct) / len(strategy_results),
                    "count": len(strategy_results),
                }
        
        # Compute per-category metrics (if category info available)
        category_metrics = self._compute_category_metrics()
        
        # Error type distribution
        error_distribution = self._compute_error_distribution()
        
        return {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(self.results),
            "aggregate_metrics": {
                "accuracy": aggregate_metrics.accuracy,
                "reasoning_depth": aggregate_metrics.reasoning_depth,
                "recovery_rate": aggregate_metrics.recovery_rate,
                "consistency_score": aggregate_metrics.consistency_score,
                "step_efficiency": aggregate_metrics.step_efficiency,
                "error_propagation_rate": aggregate_metrics.error_propagation_rate,
            },
            "model_metrics": model_metrics,
            "strategy_metrics": strategy_metrics,
            "category_metrics": category_metrics,
            "error_distribution": error_distribution,
            "api_summary": self.api_logger.get_summary(),
        }
    
    def _compute_category_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics grouped by problem category.
        
        Returns:
            Dictionary mapping category to metrics.
        """
        # Load dataset to get category info
        try:
            df = load_dataset()
            category_map = dict(zip(df["id"], df["category"]))
        except Exception:
            return {}
        
        category_results: Dict[str, List[EvaluationResult]] = {}
        for result in self.results:
            category = category_map.get(result.problem_id, "unknown")
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        category_metrics = {}
        for category, results in category_results.items():
            if results:
                category_metrics[category] = {
                    "accuracy": sum(1 for r in results if r.is_correct) / len(results),
                    "count": len(results),
                    "avg_reasoning_depth": sum(
                        r.metrics.get("reasoning_depth", 0) for r in results
                    ) / len(results),
                }
        
        return category_metrics
    
    def _compute_error_distribution(self) -> Dict[str, int]:
        """Compute distribution of error types.
        
        Returns:
            Dictionary mapping error category to count.
        """
        error_counts: Dict[str, int] = {}
        
        for result in self.results:
            if result.error_classification:
                primary = result.error_classification.primary_category.value
                error_counts[primary] = error_counts.get(primary, 0) + 1
        
        return error_counts
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis.
        
        Returns:
            DataFrame with one row per evaluation.
        """
        records = []
        for result in self.results:
            record = {
                "problem_id": result.problem_id,
                "model": result.model,
                "strategy": result.strategy,
                "final_answer": result.final_answer,
                "is_correct": result.is_correct,
                "api_latency_ms": result.api_latency_ms,
                "num_steps": len(result.step_scores),
                "reasoning_depth": result.metrics.get("reasoning_depth", 0),
                "step_accuracy": result.metrics.get("step_accuracy", 0),
                "step_efficiency": result.metrics.get("step_efficiency", 0),
            }
            
            # Add error classification info
            if result.error_classification:
                record["primary_error"] = result.error_classification.primary_category.value
                record["error_flagged"] = result.error_classification.flagged_for_review
            else:
                record["primary_error"] = None
                record["error_flagged"] = False
            
            # Add propagation info
            if result.propagation_analysis:
                record["propagation_type"] = result.propagation_analysis.propagation_type.value
                record["first_error_step"] = result.propagation_analysis.first_error_step
                record["recovery_attempted"] = result.propagation_analysis.recovery_attempted
                record["recovery_successful"] = result.propagation_analysis.recovery_successful
            else:
                record["propagation_type"] = None
                record["first_error_step"] = None
                record["recovery_attempted"] = False
                record["recovery_successful"] = False
            
            records.append(record)
        
        return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for running experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run enhanced LLM reasoning evaluation experiments"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys to evaluate (default: all)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategy keys to use (default: all)",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (default: 1.0)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Custom experiment ID",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Validate model keys
    if args.models:
        invalid_models = set(args.models) - set(MODEL_KEYS)
        if invalid_models:
            print(f"Invalid model keys: {invalid_models}")
            print(f"Available models: {MODEL_KEYS}")
            return
    
    # Validate strategy keys
    if args.strategies:
        invalid_strategies = set(args.strategies) - set(STRATEGY_KEYS)
        if invalid_strategies:
            print(f"Invalid strategy keys: {invalid_strategies}")
            print(f"Available strategies: {STRATEGY_KEYS}")
            return
    
    # Create and run experiment
    runner = EnhancedExperimentRunner(
        models=args.models,
        strategies=args.strategies,
        experiment_id=args.experiment_id,
    )
    
    results = runner.run(
        dataset_frac=args.frac,
        save_results=not args.no_save,
        verbose=not args.quiet,
    )
    
    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Total evaluations: {len(results)}")
        print(f"Correct: {sum(1 for r in results if r.is_correct)}")
        print(f"Accuracy: {sum(1 for r in results if r.is_correct) / len(results):.2%}")
        print("=" * 60)


if __name__ == "__main__":
    main()
