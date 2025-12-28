"""Property tests for Metrics Calculator.

Feature: reasoning-eval-enhancement
Properties 11, 12, and 13: Reasoning Depth Calculation, Consistency Score Bounds,
and Step Efficiency Calculation
Validates: Requirements 7.1, 7.3, 7.4
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from src.models import StepScore
from src.metrics import MetricsCalculator

from tests.property.conftest import step_scores_list_strategy


@st.composite
def step_scores_with_known_depth(draw):
    """Generate step scores with a known reasoning depth."""
    num_scores = draw(st.integers(min_value=1, max_value=10))
    num_correct_at_start = draw(st.integers(min_value=0, max_value=num_scores))
    
    scores = []
    for i in range(num_scores):
        if i < num_correct_at_start:
            is_correct = True
        elif i == num_correct_at_start:
            is_correct = False
        else:
            is_correct = draw(st.booleans())
        
        score = StepScore(
            step_index=i,
            is_correct=is_correct,
            confidence=0.9 if is_correct else 0.3,
            error_details=None if is_correct else "Error detected",
        )
        scores.append(score)
    
    expected_depth = num_correct_at_start
    return scores, expected_depth



# =============================================================================
# Property 11: Reasoning Depth Calculation
# Validates: Requirements 7.1
# =============================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(data=step_scores_with_known_depth())
def test_reasoning_depth_equals_consecutive_correct_steps(data):
    """
    Feature: reasoning-eval-enhancement, Property 11: Reasoning Depth Calculation
    
    *For any* list of step scores, reasoning_depth SHALL equal the count of
    consecutive correct steps from the beginning before the first error
    (or total steps if all correct).
    
    Validates: Requirements 7.1
    """
    step_scores, expected_depth = data
    
    calculator = MetricsCalculator()
    actual_depth = calculator.compute_reasoning_depth([step_scores])
    
    assert actual_depth == expected_depth, (
        f"Expected reasoning depth {expected_depth}, got {actual_depth}. "
        f"Step correctness: {[s.is_correct for s in step_scores]}"
    )


@settings(max_examples=100)
@given(num_steps=st.integers(min_value=1, max_value=10))
def test_reasoning_depth_all_correct_equals_total_steps(num_steps: int):
    """
    Feature: reasoning-eval-enhancement, Property 11: Reasoning Depth Calculation
    
    *For any* list of step scores where all steps are correct,
    reasoning_depth SHALL equal the total number of steps.
    
    Validates: Requirements 7.1
    """
    step_scores = [
        StepScore(step_index=i, is_correct=True, confidence=1.0)
        for i in range(num_steps)
    ]
    
    calculator = MetricsCalculator()
    depth = calculator.compute_reasoning_depth([step_scores])
    
    assert depth == num_steps, (
        f"Expected depth {num_steps} for all-correct steps, got {depth}"
    )


def test_reasoning_depth_empty_list():
    """
    Feature: reasoning-eval-enhancement, Property 11: Reasoning Depth Calculation
    
    For an empty list of step scores, compute_reasoning_depth SHALL return 0.0.
    
    Validates: Requirements 7.1
    """
    calculator = MetricsCalculator()
    depth = calculator.compute_reasoning_depth([])
    
    assert depth == 0.0, f"Expected depth 0.0 for empty list, got {depth}"



# =============================================================================
# Property 12: Consistency Score Bounds
# Validates: Requirements 7.3
# =============================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    multi_sample_results=st.lists(
        st.lists(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))), min_size=1, max_size=5),
        min_size=0,
        max_size=5,
    ),
)
def test_consistency_score_bounds(multi_sample_results: list):
    """
    Feature: reasoning-eval-enhancement, Property 12: Consistency Score Bounds
    
    *For any* set of multiple samples for the same problem, consistency_score
    SHALL be in range [0.0, 1.0].
    
    Validates: Requirements 7.3
    """
    calculator = MetricsCalculator()
    score = calculator.compute_consistency_score(multi_sample_results)
    
    assert 0.0 <= score <= 1.0, (
        f"Consistency score {score} is not in range [0.0, 1.0]"
    )


@settings(max_examples=100)
@given(
    num_problems=st.integers(min_value=1, max_value=5),
    num_samples=st.integers(min_value=2, max_value=10),
    answer=st.text(min_size=1, max_size=20),
)
def test_consistency_score_all_identical_equals_one(
    num_problems: int, 
    num_samples: int, 
    answer: str
):
    """
    Feature: reasoning-eval-enhancement, Property 12: Consistency Score Bounds
    
    *For any* set of samples where all samples produce identical final answers,
    consistency_score SHALL be 1.0.
    
    Validates: Requirements 7.3
    """
    multi_sample_results = [
        [answer] * num_samples
        for _ in range(num_problems)
    ]
    
    calculator = MetricsCalculator()
    score = calculator.compute_consistency_score(multi_sample_results)
    
    assert score == 1.0, (
        f"Expected consistency score 1.0 for all identical answers, got {score}"
    )


@settings(max_examples=100)
@given(num_samples=st.integers(min_value=2, max_value=10))
def test_consistency_score_all_different(num_samples: int):
    """
    Feature: reasoning-eval-enhancement, Property 12: Consistency Score Bounds
    
    *For any* set of samples where all samples produce different final answers,
    consistency_score SHALL be 1/num_samples (minimum possible).
    
    Validates: Requirements 7.3
    """
    multi_sample_results = [
        [f"answer_{i}" for i in range(num_samples)]
    ]
    
    calculator = MetricsCalculator()
    score = calculator.compute_consistency_score(multi_sample_results)
    
    expected_score = 1.0 / num_samples
    assert abs(score - expected_score) < 1e-9, (
        f"Expected consistency score {expected_score} for all different answers, "
        f"got {score}"
    )


def test_consistency_score_empty_list():
    """
    Feature: reasoning-eval-enhancement, Property 12: Consistency Score Bounds
    
    For an empty list of samples, compute_consistency_score SHALL return 0.0.
    
    Validates: Requirements 7.3
    """
    calculator = MetricsCalculator()
    score = calculator.compute_consistency_score([])
    
    assert score == 0.0, f"Expected score 0.0 for empty list, got {score}"



# =============================================================================
# Property 13: Step Efficiency Calculation
# Validates: Requirements 7.4
# =============================================================================

@settings(max_examples=100)
@given(
    model_steps=st.integers(min_value=1, max_value=100),
    optimal_steps=st.integers(min_value=1, max_value=100),
)
def test_step_efficiency_bounds(model_steps: int, optimal_steps: int):
    """
    Feature: reasoning-eval-enhancement, Property 13: Step Efficiency Calculation
    
    *For any* model output with M steps and optimal solution with N steps,
    step_efficiency SHALL be in range [0.0, 1.0].
    
    Validates: Requirements 7.4
    """
    calculator = MetricsCalculator()
    efficiency = calculator.compute_step_efficiency(model_steps, optimal_steps)
    
    assert 0.0 <= efficiency <= 1.0, (
        f"Step efficiency {efficiency} is not in range [0.0, 1.0]"
    )


@settings(max_examples=100)
@given(
    model_steps=st.integers(min_value=1, max_value=100),
    optimal_steps=st.integers(min_value=1, max_value=100),
)
def test_step_efficiency_equals_optimal_over_actual_capped(
    model_steps: int, 
    optimal_steps: int
):
    """
    Feature: reasoning-eval-enhancement, Property 13: Step Efficiency Calculation
    
    *For any* model output with M steps and optimal solution with N steps,
    step_efficiency SHALL equal N/M (optimal divided by actual), capped at 1.0.
    
    Validates: Requirements 7.4
    """
    calculator = MetricsCalculator()
    efficiency = calculator.compute_step_efficiency(model_steps, optimal_steps)
    
    expected = min(1.0, optimal_steps / model_steps)
    assert abs(efficiency - expected) < 1e-9, (
        f"Expected efficiency {expected}, got {efficiency}"
    )


@settings(max_examples=100)
@given(steps=st.integers(min_value=1, max_value=100))
def test_step_efficiency_perfect_when_equal(steps: int):
    """
    Feature: reasoning-eval-enhancement, Property 13: Step Efficiency Calculation
    
    *For any* model output where model_steps equals optimal_steps,
    step_efficiency SHALL be 1.0.
    
    Validates: Requirements 7.4
    """
    calculator = MetricsCalculator()
    efficiency = calculator.compute_step_efficiency(steps, steps)
    
    assert efficiency == 1.0, (
        f"Expected efficiency 1.0 when model_steps == optimal_steps, got {efficiency}"
    )


@settings(max_examples=100)
@given(
    optimal_steps=st.integers(min_value=1, max_value=50),
    extra_steps=st.integers(min_value=1, max_value=50),
)
def test_step_efficiency_less_than_one_when_model_uses_more(
    optimal_steps: int, 
    extra_steps: int
):
    """
    Feature: reasoning-eval-enhancement, Property 13: Step Efficiency Calculation
    
    *For any* model output where model_steps > optimal_steps,
    step_efficiency SHALL be less than 1.0.
    
    Validates: Requirements 7.4
    """
    model_steps = optimal_steps + extra_steps
    
    calculator = MetricsCalculator()
    efficiency = calculator.compute_step_efficiency(model_steps, optimal_steps)
    
    assert efficiency < 1.0, (
        f"Expected efficiency < 1.0 when model uses more steps, got {efficiency}"
    )


def test_step_efficiency_zero_model_steps():
    """
    Feature: reasoning-eval-enhancement, Property 13: Step Efficiency Calculation
    
    When model_steps is 0 and optimal_steps > 0, step_efficiency SHALL be 0.0.
    
    Validates: Requirements 7.4
    """
    calculator = MetricsCalculator()
    efficiency = calculator.compute_step_efficiency(0, 5)
    
    assert efficiency == 0.0, (
        f"Expected efficiency 0.0 when model_steps is 0, got {efficiency}"
    )


def test_step_efficiency_both_zero():
    """
    Feature: reasoning-eval-enhancement, Property 13: Step Efficiency Calculation
    
    When both model_steps and optimal_steps are 0, step_efficiency SHALL be 1.0.
    
    Validates: Requirements 7.4
    """
    calculator = MetricsCalculator()
    efficiency = calculator.compute_step_efficiency(0, 0)
    
    assert efficiency == 1.0, (
        f"Expected efficiency 1.0 when both are 0, got {efficiency}"
    )
