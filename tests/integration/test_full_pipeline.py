"""Integration test for full pipeline validation.

This test validates the end-to-end flow of the LLM reasoning evaluation system
without requiring actual API calls.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset_generator import DatasetGenerator
from src.cot_analyzer import ChainOfThoughtAnalyzer
from src.step_scorer import StepAccuracyScorer
from src.error_taxonomy import ErrorTaxonomyEngine
from src.error_propagation import ErrorPropagationTracker
from src.metrics import MetricsCalculator, ReasoningMetrics
from src.report_generator import ResearchReportGenerator
from src.models import StepScore, PropagationType, PropagationAnalysis


class TestFullPipeline:
    """Integration tests for the full evaluation pipeline."""
    
    def test_dataset_generation_to_analysis(self):
        """Test generating a dataset and running analysis components."""
        # 1. Generate a small dataset
        generator = DatasetGenerator()
        problems = []
        problems.extend(generator.generate_math_problems(5, (1, 3)))
        problems.extend(generator.generate_logic_problems(5, (1, 3)))
        
        assert len(problems) >= 10
        
        # 2. Simulate model outputs for each problem
        cot_analyzer = ChainOfThoughtAnalyzer()
        step_scorer = StepAccuracyScorer()
        error_engine = ErrorTaxonomyEngine()
        propagation_tracker = ErrorPropagationTracker()
        
        results = []
        for problem in problems:
            # Simulate a model output with steps
            simulated_output = self._simulate_model_output(problem)
            
            # Parse the output
            parsed_cot = cot_analyzer.parse_output(simulated_output)
            
            # Score steps
            step_scores = step_scorer.score_all_steps(
                parsed_cot, 
                problem.solution_steps
            )
            
            # Classify errors if any
            error_classification = None
            if any(not s.is_correct for s in step_scores):
                error_classification = error_engine.classify_error(
                    simulated_output,
                    problem.ground_truth,
                    step_scores
                )
            
            # Analyze propagation
            propagation = propagation_tracker.analyze_propagation(
                step_scores,
                parsed_cot.final_answer == problem.ground_truth
            )
            
            results.append({
                "problem_id": problem.id,
                "category": problem.category,
                "difficulty": problem.difficulty,
                "is_correct": parsed_cot.final_answer == problem.ground_truth,
                "step_scores": [
                    {"step_index": s.step_index, "is_correct": s.is_correct, "confidence": s.confidence}
                    for s in step_scores
                ],
                "error_categories": [c.value for c in error_classification.categories] if error_classification else [],
                "propagation_type": propagation.propagation_type.value if propagation else None,
            })
        
        # 3. Compute metrics
        df = pd.DataFrame(results)
        assert len(df) == len(problems)
        assert "is_correct" in df.columns
        assert "category" in df.columns
    
    def test_report_generation(self):
        """Test generating reports from sample data."""
        # Create sample results DataFrame
        sample_data = [
            {
                "problem_id": f"P{i}",
                "model": "test-model",
                "category": ["math", "logic", "causal"][i % 3],
                "difficulty": (i % 5) + 1,
                "is_correct": i % 2 == 0,
                "step_scores": [
                    {"step_index": 0, "is_correct": True, "confidence": 0.9},
                    {"step_index": 1, "is_correct": i % 2 == 0, "confidence": 0.8},
                ],
                "error_categories": [] if i % 2 == 0 else ["arithmetic"],
            }
            for i in range(20)
        ]
        
        df = pd.DataFrame(sample_data)
        
        # Create sample metrics
        metrics = {
            "test-model": ReasoningMetrics(
                accuracy=0.5,
                reasoning_depth=1.5,
                recovery_rate=0.2,
                consistency_score=0.8,
                step_efficiency=0.9,
                error_propagation_rate=0.3,
            )
        }
        
        # Generate report
        generator = ResearchReportGenerator()
        
        # Test accuracy by category
        fig = generator.generate_accuracy_by_category(df)
        assert fig is not None
        
        # Test accuracy by difficulty
        fig = generator.generate_accuracy_by_difficulty(df)
        assert fig is not None
        
        # Test model comparison radar
        fig = generator.generate_model_comparison_radar(metrics)
        assert fig is not None
        
        # Test markdown report
        report = generator.generate_markdown_report(df, metrics)
        assert "# LLM Reasoning Evaluation" in report
        assert "Methodology" in report
        assert "Results" in report
        
        # Test abstract
        abstract = generator.generate_abstract(metrics)
        assert len(abstract) > 0
    
    def test_metrics_calculation(self):
        """Test metrics calculation from step scores."""
        calculator = MetricsCalculator()
        
        # Test reasoning depth
        step_scores_list = [
            [
                StepScore(step_index=0, is_correct=True, confidence=0.9),
                StepScore(step_index=1, is_correct=True, confidence=0.8),
                StepScore(step_index=2, is_correct=False, confidence=0.5),
            ],
            [
                StepScore(step_index=0, is_correct=True, confidence=0.9),
                StepScore(step_index=1, is_correct=False, confidence=0.4),
            ],
        ]
        
        depth = calculator.compute_reasoning_depth(step_scores_list)
        assert depth == 1.5  # (2 + 1) / 2
        
        # Test recovery rate
        analyses = [
            PropagationAnalysis(
                first_error_step=1,
                propagation_type=PropagationType.RECOVERABLE,
                affected_steps=[1],
                recovery_attempted=True,
                recovery_successful=True,
            ),
            PropagationAnalysis(
                first_error_step=0,
                propagation_type=PropagationType.CASCADING,
                affected_steps=[0, 1, 2],
                recovery_attempted=False,
                recovery_successful=False,
            ),
        ]
        
        recovery_rate = calculator.compute_recovery_rate(analyses)
        assert recovery_rate == 0.5  # 1 recoverable out of 2
        
        # Test consistency score
        multi_samples = [
            ["42", "42", "42"],  # All same
            ["42", "43", "42"],  # 2/3 same
        ]
        
        consistency = calculator.compute_consistency_score(multi_samples)
        assert 0.8 <= consistency <= 0.9  # Average of 1.0 and ~0.67
        
        # Test step efficiency
        efficiency = calculator.compute_step_efficiency(3, 3)
        assert efficiency == 1.0
        
        efficiency = calculator.compute_step_efficiency(6, 3)
        assert efficiency == 0.5
    
    def test_error_taxonomy_classification(self):
        """Test error classification."""
        engine = ErrorTaxonomyEngine()
        
        # Test arithmetic error
        step_scores = [
            StepScore(step_index=0, is_correct=True, confidence=0.9),
            StepScore(step_index=1, is_correct=False, confidence=0.3, error_details="calculation error"),
        ]
        
        classification = engine.classify_error(
            "Step 1: 5 + 3 = 9",  # Wrong calculation
            "8",
            step_scores
        )
        
        assert classification is not None
        assert len(classification.categories) > 0
        assert all(0 <= score <= 1 for score in classification.confidence_scores.values())
        
        # Test round-trip serialization
        json_str = engine.to_json(classification)
        restored = engine.from_json(json_str)
        
        assert restored.primary_category == classification.primary_category
        assert len(restored.categories) == len(classification.categories)
    
    def test_cot_parsing_and_alignment(self):
        """Test chain-of-thought parsing and step alignment."""
        analyzer = ChainOfThoughtAnalyzer()
        
        # Test Step N: format
        output = """Step 1: First, we identify the numbers: 5 and 3
Step 2: Then we add them: 5 + 3 = 8
Step 3: Therefore, the answer is 8
Answer: 8"""
        
        parsed = analyzer.parse_output(output)
        
        assert parsed.parse_success
        assert len(parsed.steps) == 3
        # The final answer should contain "8"
        assert "8" in parsed.final_answer
        
        # Test alignment
        ground_truth = [
            "Identify numbers 5 and 3",
            "Add: 5 + 3 = 8",
            "Answer is 8"
        ]
        
        alignments = analyzer.align_steps(parsed, ground_truth)
        
        assert len(alignments) == len(parsed.steps)
        for alignment in alignments:
            assert 0 <= alignment.alignment_score <= 1
    
    def _simulate_model_output(self, problem) -> str:
        """Simulate a model output for a problem."""
        steps = problem.solution_steps
        if not steps:
            return f"The answer is {problem.ground_truth}"
        
        output_lines = []
        for i, step in enumerate(steps):
            output_lines.append(f"Step {i+1}: {step}")
        output_lines.append(f"Answer: {problem.ground_truth}")
        
        return "\n".join(output_lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
