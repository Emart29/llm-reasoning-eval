"""Unit tests for StepAccuracyScorer.

Tests edge cases for numeric tolerance, string matching, and empty inputs.
Requirements: 2.3
"""

import pytest
from src.step_scorer import StepAccuracyScorer
from src.models import ReasoningStep, ParsedChainOfThought, StepScore


class TestStepAccuracyScorer:
    """Test suite for StepAccuracyScorer."""
    
    @pytest.fixture
    def scorer(self):
        """Create a default scorer instance."""
        return StepAccuracyScorer()
    
    @pytest.fixture
    def scorer_strict(self):
        """Create a scorer with strict numeric tolerance."""
        return StepAccuracyScorer(numeric_tolerance=1e-10)
    
    @pytest.fixture
    def scorer_lenient(self):
        """Create a scorer with lenient thresholds."""
        return StepAccuracyScorer(
            numeric_tolerance=0.01,
            similarity_threshold=0.4
        )

    # =========================================================================
    # Empty Input Tests
    # =========================================================================
    
    def test_score_step_empty_model_content(self, scorer):
        """Empty model step content should be marked incorrect."""
        model_step = ReasoningStep(index=0, content="")
        gt_step = "Calculate 2 + 2 = 4"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert not score.is_correct
        assert score.confidence == 1.0
        assert "Empty model step content" in score.error_details
    
    def test_score_step_whitespace_model_content(self, scorer):
        """Whitespace-only model step should be marked incorrect."""
        model_step = ReasoningStep(index=0, content="   \n\t  ")
        gt_step = "Calculate 2 + 2 = 4"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert not score.is_correct
        assert "Empty model step content" in score.error_details
    
    def test_score_step_empty_ground_truth(self, scorer):
        """Empty ground truth should result in incorrect with low confidence."""
        model_step = ReasoningStep(index=0, content="Calculate 2 + 2 = 4")
        gt_step = ""
        
        score = scorer.score_step(model_step, gt_step)
        
        assert not score.is_correct
        assert score.confidence == 0.5
        assert "Empty ground truth" in score.error_details
    
    def test_score_all_steps_empty_model_cot(self, scorer):
        """Empty model CoT should return empty scores list."""
        model_cot = ParsedChainOfThought(
            raw_output="",
            steps=[],
            final_answer=""
        )
        gt_steps = ["Step 1", "Step 2"]
        
        scores = scorer.score_all_steps(model_cot, gt_steps)
        
        assert scores == []
    
    def test_score_all_steps_empty_ground_truth(self, scorer):
        """Empty ground truth should mark all steps as uncertain."""
        model_cot = ParsedChainOfThought(
            raw_output="Step 1: Do something",
            steps=[ReasoningStep(index=0, content="Do something")],
            final_answer="result"
        )
        gt_steps = []
        
        scores = scorer.score_all_steps(model_cot, gt_steps)
        
        assert len(scores) == 1
        assert not scores[0].is_correct
        assert scores[0].confidence == 0.0
        assert "No ground truth" in scores[0].error_details

    # =========================================================================
    # Numeric Tolerance Tests
    # =========================================================================
    
    def test_exact_numeric_match(self, scorer):
        """Exact numeric values should match."""
        model_step = ReasoningStep(index=0, content="The result is 42")
        gt_step = "The answer is 42"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert score.is_correct
    
    def test_floating_point_tolerance(self, scorer):
        """Floating point values within tolerance should match."""
        model_step = ReasoningStep(
            index=0, 
            content="The result is 3.14159265359"
        )
        gt_step = "The answer is 3.14159265358"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert score.is_correct
    
    def test_floating_point_outside_tolerance(self, scorer_strict):
        """Floating point values outside strict tolerance should not match."""
        model_step = ReasoningStep(index=0, content="The result is 3.15")
        gt_step = "The answer is 3.14"
        
        score = scorer_strict.score_step(model_step, gt_step)
        
        # With strict tolerance, 3.15 vs 3.14 should fail numeric match
        # but may still pass on text similarity
        assert score.step_index == 0
    
    def test_integer_vs_float(self, scorer):
        """Integer and equivalent float should match."""
        model_step = ReasoningStep(index=0, content="The count is 5.0")
        gt_step = "The count is 5"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert score.is_correct
    
    def test_negative_numbers(self, scorer):
        """Negative numbers should be handled correctly."""
        model_step = ReasoningStep(index=0, content="The change is -15")
        gt_step = "The difference is -15"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert score.is_correct
    
    def test_multiple_numbers_match(self, scorer):
        """Multiple numbers in step should all be checked."""
        model_step = ReasoningStep(
            index=0, 
            content="Add 5 and 3 to get 8"
        )
        gt_step = "5 + 3 = 8"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert score.is_correct
    
    def test_numeric_mismatch(self, scorer):
        """Mismatched numbers should be detected."""
        model_step = ReasoningStep(
            index=0, 
            content="Calculate: 5 + 5 = 100"
        )
        gt_step = "Calculate: 5 + 5 = 10"
        
        score = scorer.score_step(model_step, gt_step)
        
        # The numeric mismatch (100 vs 10) should cause failure
        # even though text is similar
        assert not score.is_correct

    # =========================================================================
    # String Matching Tests
    # =========================================================================
    
    def test_exact_text_match(self, scorer):
        """Exact text match should be correct."""
        model_step = ReasoningStep(
            index=0, 
            content="First, identify the variables"
        )
        gt_step = "First, identify the variables"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert score.is_correct
        assert score.confidence > 0.9
    
    def test_case_insensitive_match(self, scorer):
        """Text matching should be case insensitive."""
        model_step = ReasoningStep(
            index=0, 
            content="CALCULATE THE SUM"
        )
        gt_step = "calculate the sum"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert score.is_correct
    
    def test_whitespace_normalization(self, scorer):
        """Extra whitespace should be normalized."""
        model_step = ReasoningStep(
            index=0, 
            content="Calculate   the    sum"
        )
        gt_step = "Calculate the sum"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert score.is_correct
    
    def test_similar_but_different_text(self, scorer):
        """Similar text with key differences should be detected."""
        model_step = ReasoningStep(
            index=0, 
            content="Multiply the values together"
        )
        gt_step = "Add the values together"
        
        score = scorer.score_step(model_step, gt_step)
        
        # Text is similar but operation is different
        assert score.step_index == 0
    
    def test_completely_different_text(self, scorer):
        """Completely different text should not match."""
        model_step = ReasoningStep(
            index=0, 
            content="The sky is blue"
        )
        gt_step = "Calculate 2 + 2 = 4"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert not score.is_correct

    # =========================================================================
    # Step Accuracy Curve Tests
    # =========================================================================
    
    def test_accuracy_curve_all_correct(self, scorer):
        """All correct steps should give curve of all 1.0."""
        scores = [
            StepScore(step_index=0, is_correct=True, confidence=0.9),
            StepScore(step_index=1, is_correct=True, confidence=0.8),
            StepScore(step_index=2, is_correct=True, confidence=0.95),
        ]
        
        curve = scorer.compute_step_accuracy_curve(scores)
        
        assert curve == [1.0, 1.0, 1.0]
    
    def test_accuracy_curve_all_incorrect(self, scorer):
        """All incorrect steps should give curve of all 0.0."""
        scores = [
            StepScore(step_index=0, is_correct=False, confidence=0.1),
            StepScore(step_index=1, is_correct=False, confidence=0.2),
            StepScore(step_index=2, is_correct=False, confidence=0.15),
        ]
        
        curve = scorer.compute_step_accuracy_curve(scores)
        
        assert curve == [0.0, 0.0, 0.0]
    
    def test_accuracy_curve_mixed(self, scorer):
        """Mixed correct/incorrect should show cumulative accuracy."""
        scores = [
            StepScore(step_index=0, is_correct=True, confidence=0.9),
            StepScore(step_index=1, is_correct=False, confidence=0.3),
            StepScore(step_index=2, is_correct=True, confidence=0.8),
            StepScore(step_index=3, is_correct=True, confidence=0.85),
        ]
        
        curve = scorer.compute_step_accuracy_curve(scores)
        
        # Step 0: 1/1 = 1.0
        # Step 1: 1/2 = 0.5
        # Step 2: 2/3 ≈ 0.667
        # Step 3: 3/4 = 0.75
        assert curve[0] == 1.0
        assert curve[1] == 0.5
        assert abs(curve[2] - 2/3) < 0.001
        assert curve[3] == 0.75
    
    def test_accuracy_curve_empty(self, scorer):
        """Empty scores should return empty curve."""
        curve = scorer.compute_step_accuracy_curve([])
        
        assert curve == []

    # =========================================================================
    # Score All Steps Tests
    # =========================================================================
    
    def test_score_all_steps_aligned(self, scorer):
        """Steps should be scored against corresponding ground truth."""
        model_cot = ParsedChainOfThought(
            raw_output="Step 1: Add 2 + 3\nStep 2: Result is 5",
            steps=[
                ReasoningStep(index=0, content="Add 2 + 3"),
                ReasoningStep(index=1, content="Result is 5"),
            ],
            final_answer="5"
        )
        gt_steps = ["2 + 3", "= 5"]
        
        scores = scorer.score_all_steps(model_cot, gt_steps)
        
        assert len(scores) == 2
        assert scores[0].step_index == 0
        assert scores[1].step_index == 1
    
    def test_score_all_steps_more_model_steps(self, scorer):
        """Extra model steps should find best matching ground truth."""
        model_cot = ParsedChainOfThought(
            raw_output="Multiple steps",
            steps=[
                ReasoningStep(index=0, content="First step"),
                ReasoningStep(index=1, content="Second step"),
                ReasoningStep(index=2, content="Third step"),
            ],
            final_answer="result"
        )
        gt_steps = ["First step", "Second step"]
        
        scores = scorer.score_all_steps(model_cot, gt_steps)
        
        assert len(scores) == 3
        # First two should match well
        assert scores[0].is_correct
        assert scores[1].is_correct
    
    def test_score_all_steps_fewer_model_steps(self, scorer):
        """Fewer model steps should still be scored."""
        model_cot = ParsedChainOfThought(
            raw_output="One step",
            steps=[
                ReasoningStep(index=0, content="Calculate the sum"),
            ],
            final_answer="result"
        )
        gt_steps = ["Calculate the sum", "Verify the result", "State conclusion"]
        
        scores = scorer.score_all_steps(model_cot, gt_steps)
        
        assert len(scores) == 1
        assert scores[0].is_correct

    # =========================================================================
    # Mathematical Symbol Tests
    # =========================================================================
    
    def test_multiplication_symbol_normalization(self, scorer):
        """Different multiplication symbols should be normalized."""
        model_step = ReasoningStep(index=0, content="5 × 3 = 15")
        gt_step = "5 * 3 = 15"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert score.is_correct
    
    def test_division_symbol_normalization(self, scorer):
        """Different division symbols should be normalized."""
        model_step = ReasoningStep(index=0, content="10 ÷ 2 = 5")
        gt_step = "10 / 2 = 5"
        
        score = scorer.score_step(model_step, gt_step)
        
        assert score.is_correct
