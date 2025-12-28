"""Step Accuracy Scorer module.

Evaluates correctness of individual reasoning steps.
"""

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from src.models import (
    ReasoningStep,
    ParsedChainOfThought,
    StepScore,
)


class StepAccuracyScorer:
    """Scores the accuracy of reasoning steps.
    
    Uses fuzzy matching for text comparison and numeric tolerance
    for floating point comparisons.
    """
    
    # Default tolerance for floating point comparisons
    DEFAULT_NUMERIC_TOLERANCE = 1e-6
    # Default threshold for text similarity to consider a match
    DEFAULT_SIMILARITY_THRESHOLD = 0.6
    
    def __init__(
        self,
        numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ):
        """Initialize the scorer.
        
        Args:
            numeric_tolerance: Tolerance for floating point comparisons
            similarity_threshold: Minimum similarity score to consider text matching
        """
        self.numeric_tolerance = numeric_tolerance
        self.similarity_threshold = similarity_threshold
        # Pattern to extract numbers from text
        self._number_pattern = re.compile(r'-?\d+\.?\d*')
    
    def score_step(
        self, 
        model_step: ReasoningStep, 
        ground_truth_step: str
    ) -> StepScore:
        """Score a single model step against ground truth.
        
        Compares the model's reasoning step to the ground truth step using
        both text similarity and numeric value matching.
        
        Args:
            model_step: The model's reasoning step
            ground_truth_step: The ground truth step text
            
        Returns:
            StepScore with correctness assessment and confidence
        """
        if not model_step.content.strip():
            return StepScore(
                step_index=model_step.index,
                is_correct=False,
                confidence=1.0,
                error_details="Empty model step content"
            )
        
        if not ground_truth_step.strip():
            return StepScore(
                step_index=model_step.index,
                is_correct=False,
                confidence=0.5,
                error_details="Empty ground truth step"
            )
        
        # Compute text similarity
        text_similarity = self._compute_text_similarity(
            model_step.content, 
            ground_truth_step
        )
        
        # Check numeric values
        numeric_match, numeric_confidence = self._check_numeric_match(
            model_step.content,
            ground_truth_step
        )
        
        # Combine scores: prioritize numeric match if numbers are present
        if numeric_confidence > 0:
            # Numbers were found - weight numeric match more heavily
            combined_score = 0.6 * numeric_confidence + 0.4 * text_similarity
            is_correct = numeric_match and text_similarity >= self.similarity_threshold * 0.5
        else:
            # No numbers - rely on text similarity
            combined_score = text_similarity
            is_correct = text_similarity >= self.similarity_threshold
        
        error_details = None
        if not is_correct:
            error_details = self._generate_error_details(
                model_step.content,
                ground_truth_step,
                text_similarity,
                numeric_match
            )
        
        return StepScore(
            step_index=model_step.index,
            is_correct=is_correct,
            confidence=min(1.0, max(0.0, combined_score)),
            error_details=error_details
        )
    
    def score_all_steps(
        self, 
        model_cot: ParsedChainOfThought, 
        ground_truth_steps: List[str]
    ) -> List[StepScore]:
        """Score all steps in a chain-of-thought.
        
        Aligns model steps with ground truth steps and scores each one.
        If there are more model steps than ground truth steps, extra model
        steps are scored against the last ground truth step or marked as
        potentially incorrect.
        
        Args:
            model_cot: Parsed chain-of-thought from the model
            ground_truth_steps: List of ground truth solution steps
            
        Returns:
            List of StepScore objects, one for each model step
        """
        scores = []
        
        if not model_cot.steps:
            return scores
        
        if not ground_truth_steps:
            # No ground truth - mark all steps as uncertain
            for step in model_cot.steps:
                scores.append(StepScore(
                    step_index=step.index,
                    is_correct=False,
                    confidence=0.0,
                    error_details="No ground truth steps available for comparison"
                ))
            return scores
        
        # Score each model step against corresponding or best-matching ground truth
        for i, model_step in enumerate(model_cot.steps):
            if i < len(ground_truth_steps):
                # Direct alignment by position
                gt_step = ground_truth_steps[i]
            else:
                # Extra model steps - find best match or use last
                gt_step = self._find_best_matching_gt_step(
                    model_step, 
                    ground_truth_steps
                )
            
            score = self.score_step(model_step, gt_step)
            scores.append(score)
        
        return scores
    
    def compute_step_accuracy_curve(self, scores: List[StepScore]) -> List[float]:
        """Compute cumulative accuracy at each step for visualization.
        
        Returns a list where each element represents the fraction of steps
        that are correct up to and including that step index.
        
        Args:
            scores: List of step scores
            
        Returns:
            List of cumulative accuracy values (0.0 to 1.0) at each step
        """
        if not scores:
            return []
        
        curve = []
        correct_count = 0
        
        for i, score in enumerate(scores):
            if score.is_correct:
                correct_count += 1
            cumulative_accuracy = correct_count / (i + 1)
            curve.append(cumulative_accuracy)
        
        return curve
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text strings.
        
        Uses SequenceMatcher for fuzzy string matching.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Normalize texts
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.
        
        Converts to lowercase, removes extra whitespace, and standardizes
        common mathematical notation.
        """
        text = text.lower().strip()
        # Collapse multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        # Standardize some common variations
        text = text.replace('×', '*').replace('÷', '/')
        return text
    
    def _check_numeric_match(
        self, 
        model_text: str, 
        gt_text: str
    ) -> Tuple[bool, float]:
        """Check if numeric values in model output match ground truth.
        
        Args:
            model_text: Model's step text
            gt_text: Ground truth step text
            
        Returns:
            Tuple of (all_match, confidence) where confidence indicates
            how many numbers were compared (0 if no numbers found)
        """
        model_numbers = self._extract_numbers(model_text)
        gt_numbers = self._extract_numbers(gt_text)
        
        if not gt_numbers:
            # No numbers in ground truth to compare
            return True, 0.0
        
        if not model_numbers:
            # Ground truth has numbers but model doesn't
            return False, 0.5
        
        # Check if all ground truth numbers appear in model output
        matched = 0
        for gt_num in gt_numbers:
            for model_num in model_numbers:
                if self._numbers_match(model_num, gt_num):
                    matched += 1
                    break
        
        match_ratio = matched / len(gt_numbers)
        all_match = match_ratio >= 0.8  # Allow some tolerance
        
        return all_match, match_ratio
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numeric values from text.
        
        Args:
            text: Text to extract numbers from
            
        Returns:
            List of extracted numbers as floats
        """
        matches = self._number_pattern.findall(text)
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        return numbers
    
    def _numbers_match(self, num1: float, num2: float) -> bool:
        """Check if two numbers match within tolerance.
        
        Args:
            num1: First number
            num2: Second number
            
        Returns:
            True if numbers match within tolerance
        """
        if num2 == 0:
            return abs(num1) <= self.numeric_tolerance
        
        # Use relative tolerance for larger numbers
        relative_diff = abs(num1 - num2) / max(abs(num1), abs(num2), 1e-10)
        absolute_diff = abs(num1 - num2)
        
        return (relative_diff <= self.numeric_tolerance or 
                absolute_diff <= self.numeric_tolerance)
    
    def _find_best_matching_gt_step(
        self, 
        model_step: ReasoningStep, 
        ground_truth_steps: List[str]
    ) -> str:
        """Find the best matching ground truth step for a model step.
        
        Args:
            model_step: The model's reasoning step
            ground_truth_steps: List of ground truth steps
            
        Returns:
            The best matching ground truth step
        """
        best_score = -1.0
        best_step = ground_truth_steps[-1]  # Default to last step
        
        for gt_step in ground_truth_steps:
            similarity = self._compute_text_similarity(
                model_step.content, 
                gt_step
            )
            if similarity > best_score:
                best_score = similarity
                best_step = gt_step
        
        return best_step
    
    def _generate_error_details(
        self,
        model_content: str,
        gt_content: str,
        text_similarity: float,
        numeric_match: bool
    ) -> str:
        """Generate human-readable error details.
        
        Args:
            model_content: Model's step content
            gt_content: Ground truth step content
            text_similarity: Computed text similarity
            numeric_match: Whether numeric values matched
            
        Returns:
            Error description string
        """
        details = []
        
        if text_similarity < self.similarity_threshold:
            details.append(
                f"Text similarity ({text_similarity:.2f}) below threshold "
                f"({self.similarity_threshold})"
            )
        
        if not numeric_match:
            model_nums = self._extract_numbers(model_content)
            gt_nums = self._extract_numbers(gt_content)
            if gt_nums:
                details.append(
                    f"Numeric mismatch: model={model_nums}, expected={gt_nums}"
                )
        
        return "; ".join(details) if details else "Step does not match ground truth"
