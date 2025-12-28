"""Metrics Calculator module.

Computes novel research metrics beyond simple accuracy.
"""

from collections import Counter
from typing import Dict, List, Optional

from src.models import (
    ReasoningMetrics,
    StepScore,
    PropagationAnalysis,
    PropagationType,
    EvaluationResult,
)


class MetricsCalculator:
    """Calculates research metrics for reasoning evaluation.
    
    This class computes novel metrics that go beyond simple accuracy,
    including reasoning depth, recovery rate, consistency score,
    and step efficiency.
    """
    
    def compute_reasoning_depth(
        self, 
        step_scores_list: List[List[StepScore]]
    ) -> float:
        """Compute average reasoning depth across evaluations.
        
        Reasoning depth is the count of consecutive correct steps from
        the beginning before the first error (or total steps if all correct).
        
        Args:
            step_scores_list: List of step score lists, one per evaluation
            
        Returns:
            Average reasoning depth across all evaluations.
            Returns 0.0 if no evaluations are provided.
        """
        if not step_scores_list:
            return 0.0
        
        depths = []
        for step_scores in step_scores_list:
            depth = self._compute_single_reasoning_depth(step_scores)
            depths.append(depth)
        
        return sum(depths) / len(depths)

    def _compute_single_reasoning_depth(
        self, 
        step_scores: List[StepScore]
    ) -> int:
        """Compute reasoning depth for a single evaluation.
        
        Counts consecutive correct steps from the beginning before
        the first error.
        
        Args:
            step_scores: List of step scores for one evaluation
            
        Returns:
            Number of consecutive correct steps from the beginning
        """
        if not step_scores:
            return 0
        
        # Sort by step index to ensure correct order
        sorted_scores = sorted(step_scores, key=lambda s: s.step_index)
        
        depth = 0
        for score in sorted_scores:
            if score.is_correct:
                depth += 1
            else:
                break
        
        return depth
    
    def compute_recovery_rate(
        self, 
        propagation_analyses: List[PropagationAnalysis]
    ) -> float:
        """Compute recovery rate from propagation analyses.
        
        The recovery rate is the fraction of propagation analyses where
        the propagation type is RECOVERABLE.
        
        Args:
            propagation_analyses: List of propagation analyses
            
        Returns:
            Float between 0.0 and 1.0 representing the recovery rate.
            Returns 0.0 if no analyses are provided.
        """
        if not propagation_analyses:
            return 0.0
        
        recoverable_count = sum(
            1 for a in propagation_analyses 
            if a.propagation_type == PropagationType.RECOVERABLE
        )
        
        return recoverable_count / len(propagation_analyses)

    def compute_consistency_score(
        self, 
        multi_sample_results: List[List[str]]
    ) -> float:
        """Compute consistency across multiple samples.
        
        Consistency score measures agreement across multiple samples
        for the same problem. A score of 1.0 means all samples produced
        identical final answers.
        
        For each problem, we compute the fraction of samples that agree
        with the most common answer, then average across all problems.
        
        Args:
            multi_sample_results: List of sample result lists, where each
                inner list contains final answers from multiple samples
                of the same problem
            
        Returns:
            Float between 0.0 and 1.0 representing the consistency score.
            Returns 0.0 if no results are provided.
        """
        if not multi_sample_results:
            return 0.0
        
        consistency_scores = []
        for samples in multi_sample_results:
            if not samples:
                continue
            
            # Count occurrences of each answer
            answer_counts = Counter(samples)
            
            # Find the most common answer count
            most_common_count = answer_counts.most_common(1)[0][1]
            
            # Consistency is the fraction that agree with most common
            consistency = most_common_count / len(samples)
            consistency_scores.append(consistency)
        
        if not consistency_scores:
            return 0.0
        
        return sum(consistency_scores) / len(consistency_scores)
    
    def compute_step_efficiency(
        self, 
        model_steps: int, 
        optimal_steps: int
    ) -> float:
        """Compute step efficiency ratio.
        
        Step efficiency is the ratio of optimal steps to actual model steps,
        capped at 1.0. A value of 1.0 means the model used exactly the
        optimal number of steps. Values less than 1.0 indicate the model
        used more steps than necessary.
        
        Args:
            model_steps: Number of steps the model used
            optimal_steps: Number of steps in the optimal solution
            
        Returns:
            Float between 0.0 and 1.0 representing step efficiency.
            Returns 1.0 if model_steps is 0 (to avoid division by zero).
        """
        if model_steps <= 0:
            # If model produced no steps, return 1.0 if optimal is also 0
            # otherwise return 0.0 (model failed to produce steps)
            return 1.0 if optimal_steps <= 0 else 0.0
        
        if optimal_steps <= 0:
            # If optimal is 0 but model has steps, efficiency is 0
            return 0.0
        
        # Efficiency = optimal / actual, capped at 1.0
        efficiency = optimal_steps / model_steps
        return min(1.0, efficiency)

    def compute_all_metrics(
        self, 
        results: List[EvaluationResult]
    ) -> ReasoningMetrics:
        """Compute all metrics from evaluation results.
        
        Aggregates all individual metrics into a ReasoningMetrics object.
        
        Args:
            results: List of evaluation results
            
        Returns:
            ReasoningMetrics object with all computed metrics
        """
        if not results:
            return ReasoningMetrics()
        
        # Compute accuracy
        correct_count = sum(1 for r in results if r.is_correct)
        accuracy = correct_count / len(results)
        
        # Collect step scores for reasoning depth
        step_scores_list = [r.step_scores for r in results if r.step_scores]
        reasoning_depth = self.compute_reasoning_depth(step_scores_list)
        
        # Collect propagation analyses for recovery rate and propagation rate
        propagation_analyses = [
            r.propagation_analysis for r in results 
            if r.propagation_analysis is not None
        ]
        recovery_rate = self.compute_recovery_rate(propagation_analyses)
        
        # Compute error propagation rate (fraction of errors that cascade)
        error_propagation_rate = self._compute_error_propagation_rate(
            propagation_analyses
        )
        
        # Consistency score requires multiple samples per problem
        # Group results by problem_id
        consistency_score = self._compute_consistency_from_results(results)
        
        # Compute average step efficiency
        step_efficiency = self._compute_average_step_efficiency(results)
        
        return ReasoningMetrics(
            accuracy=accuracy,
            reasoning_depth=reasoning_depth,
            recovery_rate=recovery_rate,
            consistency_score=consistency_score,
            step_efficiency=step_efficiency,
            error_propagation_rate=error_propagation_rate,
        )

    def _compute_error_propagation_rate(
        self, 
        propagation_analyses: List[PropagationAnalysis]
    ) -> float:
        """Compute the rate at which errors cascade.
        
        Args:
            propagation_analyses: List of propagation analyses
            
        Returns:
            Float between 0.0 and 1.0 representing the propagation rate.
        """
        if not propagation_analyses:
            return 0.0
        
        cascading_count = sum(
            1 for a in propagation_analyses 
            if a.propagation_type in (PropagationType.CASCADING, PropagationType.TERMINAL)
        )
        
        return cascading_count / len(propagation_analyses)
    
    def _compute_consistency_from_results(
        self, 
        results: List[EvaluationResult]
    ) -> float:
        """Compute consistency score from evaluation results.
        
        Groups results by problem_id and computes consistency for each group.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Float between 0.0 and 1.0 representing the consistency score.
        """
        # Group results by problem_id
        problem_results: Dict[str, List[str]] = {}
        for r in results:
            if r.problem_id not in problem_results:
                problem_results[r.problem_id] = []
            problem_results[r.problem_id].append(r.final_answer)
        
        # Only consider problems with multiple samples
        multi_sample_results = [
            answers for answers in problem_results.values() 
            if len(answers) > 1
        ]
        
        if not multi_sample_results:
            # If no problems have multiple samples, return 1.0 (perfect consistency)
            return 1.0
        
        return self.compute_consistency_score(multi_sample_results)
    
    def _compute_average_step_efficiency(
        self, 
        results: List[EvaluationResult]
    ) -> float:
        """Compute average step efficiency from evaluation results.
        
        Uses the 'optimal_steps' and 'model_steps' from result metrics
        if available.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Float between 0.0 and 1.0 representing average step efficiency.
        """
        efficiencies = []
        for r in results:
            # Get step counts from metrics or step_scores
            model_steps = r.metrics.get('model_steps', len(r.step_scores))
            optimal_steps = r.metrics.get('optimal_steps', model_steps)
            
            if model_steps > 0 or optimal_steps > 0:
                efficiency = self.compute_step_efficiency(model_steps, optimal_steps)
                efficiencies.append(efficiency)
        
        if not efficiencies:
            return 1.0  # Default to perfect efficiency if no data
        
        return sum(efficiencies) / len(efficiencies)
