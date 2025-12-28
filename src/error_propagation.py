"""Error Propagation Tracker module.

Tracks how errors cascade through reasoning chains.
"""

from typing import List, Optional

from src.models import (
    PropagationType,
    PropagationAnalysis,
    StepScore,
)


class ErrorPropagationTracker:
    """Tracks error propagation through reasoning chains.
    
    This class analyzes how errors in reasoning steps propagate through
    the chain, classifying them as recoverable, cascading, or terminal.
    It also computes aggregate statistics like propagation rate and
    recovery rate.
    """
    
    def analyze_propagation(
        self, 
        step_scores: List[StepScore], 
        final_correct: bool
    ) -> Optional[PropagationAnalysis]:
        """Analyze how errors propagate through the reasoning chain.
        
        Examines the step scores to find the first error, determine which
        subsequent steps are affected, and classify the propagation pattern.
        
        Args:
            step_scores: List of step scores with correctness assessments
            final_correct: Whether the final answer was correct
            
        Returns:
            PropagationAnalysis if errors exist, None if all steps are correct
        """
        if not step_scores:
            return None
        
        # Find the first error step
        first_error_idx = self._find_first_error(step_scores)
        
        if first_error_idx is None:
            # No errors found - no propagation to analyze
            return None
        
        # Identify affected steps (incorrect steps at or after first error)
        affected_steps = self._find_affected_steps(step_scores, first_error_idx)
        
        # Detect recovery attempts
        recovery_attempted, recovery_successful = self._detect_recovery(
            step_scores, first_error_idx
        )
        
        # Classify propagation type
        propagation_type = self._classify_propagation(
            step_scores,
            first_error_idx,
            final_correct,
            recovery_successful
        )
        
        return PropagationAnalysis(
            first_error_step=first_error_idx,
            propagation_type=propagation_type,
            affected_steps=affected_steps,
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful,
        )
    
    def compute_propagation_rate(
        self, 
        analyses: List[PropagationAnalysis]
    ) -> float:
        """Compute the rate at which errors cascade.
        
        The propagation rate is the fraction of errors that cascade
        (i.e., are not recoverable).
        
        Args:
            analyses: List of propagation analyses
            
        Returns:
            Float between 0.0 and 1.0 representing the propagation rate.
            Returns 0.0 if no analyses are provided.
        """
        if not analyses:
            return 0.0
        
        cascading_count = sum(
            1 for a in analyses 
            if a.propagation_type in (PropagationType.CASCADING, PropagationType.TERMINAL)
        )
        
        return cascading_count / len(analyses)
    
    def compute_recovery_rate(
        self, 
        analyses: List[PropagationAnalysis]
    ) -> float:
        """Compute the rate at which models recover from errors.
        
        The recovery rate is the fraction of propagation analyses where
        the propagation type is RECOVERABLE.
        
        Args:
            analyses: List of propagation analyses
            
        Returns:
            Float between 0.0 and 1.0 representing the recovery rate.
            Returns 0.0 if no analyses are provided.
        """
        if not analyses:
            return 0.0
        
        recoverable_count = sum(
            1 for a in analyses 
            if a.propagation_type == PropagationType.RECOVERABLE
        )
        
        return recoverable_count / len(analyses)
    
    def _find_first_error(self, step_scores: List[StepScore]) -> Optional[int]:
        """Find the index of the first incorrect step.
        
        Args:
            step_scores: List of step scores
            
        Returns:
            Index of first incorrect step, or None if all correct
        """
        for score in step_scores:
            if not score.is_correct:
                return score.step_index
        return None
    
    def _find_affected_steps(
        self, 
        step_scores: List[StepScore], 
        first_error_idx: int
    ) -> List[int]:
        """Find all steps affected by the error.
        
        Affected steps are incorrect steps at or after the first error.
        
        Args:
            step_scores: List of step scores
            first_error_idx: Index of the first error
            
        Returns:
            List of affected step indices (sorted)
        """
        affected = []
        for score in step_scores:
            if score.step_index >= first_error_idx and not score.is_correct:
                affected.append(score.step_index)
        return sorted(affected)
    
    def _detect_recovery(
        self, 
        step_scores: List[StepScore], 
        first_error_idx: int
    ) -> tuple:
        """Detect if recovery was attempted and successful.
        
        Recovery is attempted if there are correct steps after an error.
        Recovery is successful if the model returns to correct reasoning
        after an error.
        
        Args:
            step_scores: List of step scores
            first_error_idx: Index of the first error
            
        Returns:
            Tuple of (recovery_attempted, recovery_successful)
        """
        # Get steps after the first error
        steps_after_error = [
            s for s in step_scores if s.step_index > first_error_idx
        ]
        
        if not steps_after_error:
            # No steps after error - no recovery possible
            return False, False
        
        # Check if any step after the error is correct
        has_correct_after_error = any(s.is_correct for s in steps_after_error)
        
        if not has_correct_after_error:
            # No correct steps after error - no recovery attempted
            return False, False
        
        # Recovery was attempted (there's at least one correct step after error)
        recovery_attempted = True
        
        # Check if recovery was successful
        # Recovery is successful if the last step is correct
        last_step = max(step_scores, key=lambda s: s.step_index)
        recovery_successful = last_step.is_correct
        
        return recovery_attempted, recovery_successful
    
    def _classify_propagation(
        self,
        step_scores: List[StepScore],
        first_error_idx: int,
        final_correct: bool,
        recovery_successful: bool
    ) -> PropagationType:
        """Classify the propagation pattern.
        
        - RECOVERABLE: Model self-corrects and gets final answer right
        - CASCADING: Error affects subsequent steps but may not be terminal
        - TERMINAL: Error leads to wrong final answer
        
        Args:
            step_scores: List of step scores
            first_error_idx: Index of the first error
            final_correct: Whether the final answer was correct
            recovery_successful: Whether recovery was successful
            
        Returns:
            PropagationType classification
        """
        if final_correct and recovery_successful:
            return PropagationType.RECOVERABLE
        
        if not final_correct:
            return PropagationType.TERMINAL
        
        # Final is correct but recovery wasn't marked as successful
        # This can happen if the error didn't affect the final answer
        # Check if there are multiple errors after the first
        errors_after_first = sum(
            1 for s in step_scores 
            if s.step_index > first_error_idx and not s.is_correct
        )
        
        if errors_after_first > 0:
            return PropagationType.CASCADING
        
        # Single error that didn't cascade
        return PropagationType.RECOVERABLE
