"""Property tests for Error Propagation Tracker.

Feature: reasoning-eval-enhancement
Properties 7, 8, and 9: Propagation Analysis Validity, Propagation Rate Bounds,
and Recovery Rate Calculation
Validates: Requirements 4.1, 4.2, 4.3, 4.4, 7.2
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from src.models import (
    PropagationType,
    PropagationAnalysis,
    StepScore,
)
from src.error_propagation import ErrorPropagationTracker

from tests.property.conftest import (
    step_scores_with_error_strategy,
    propagation_analysis_strategy,
)


# Custom strategy for step scores with at least one error (simplified for performance)
@st.composite
def simple_step_scores_with_error(draw):
    """Generate step scores where at least one step is incorrect."""
    num_scores = draw(st.integers(min_value=2, max_value=8))
    error_idx = draw(st.integers(min_value=0, max_value=num_scores - 1))
    
    scores = []
    for i in range(num_scores):
        is_correct = i != error_idx
        score = StepScore(
            step_index=i,
            is_correct=is_correct,
            confidence=0.8 if is_correct else 0.2,
            error_details=None if is_correct else "Error detected",
        )
        scores.append(score)
    return scores


# =============================================================================
# Property 7: Propagation Analysis Validity
# Validates: Requirements 4.1, 4.2
# =============================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    step_scores=simple_step_scores_with_error(),
    final_correct=st.booleans(),
)
def test_propagation_analysis_validity(
    step_scores: list,
    final_correct: bool
):
    """
    Feature: reasoning-eval-enhancement, Property 7: Propagation Analysis Validity
    
    *For any* list of step scores containing at least one error,
    PropagationAnalysis SHALL have:
    (a) first_error_step matching the index of the first incorrect step,
    (b) propagation_type in {RECOVERABLE, CASCADING, TERMINAL}, and
    (c) affected_steps containing only indices >= first_error_step.
    
    Validates: Requirements 4.1, 4.2
    """
    tracker = ErrorPropagationTracker()
    analysis = tracker.analyze_propagation(step_scores, final_correct)
    
    # Analysis should not be None since we have at least one error
    assert analysis is not None, "Analysis should not be None when errors exist"
    
    # (a) first_error_step should match the index of the first incorrect step
    first_incorrect_idx = None
    for score in step_scores:
        if not score.is_correct:
            first_incorrect_idx = score.step_index
            break
    
    assert analysis.first_error_step == first_incorrect_idx, (
        f"first_error_step {analysis.first_error_step} does not match "
        f"first incorrect step index {first_incorrect_idx}"
    )
    
    # (b) propagation_type should be one of the valid types
    valid_types = {PropagationType.RECOVERABLE, PropagationType.CASCADING, PropagationType.TERMINAL}
    assert analysis.propagation_type in valid_types, (
        f"propagation_type {analysis.propagation_type} is not valid"
    )
    
    # (c) affected_steps should only contain indices >= first_error_step
    for affected_idx in analysis.affected_steps:
        assert affected_idx >= analysis.first_error_step, (
            f"affected_step {affected_idx} is before first_error_step {analysis.first_error_step}"
        )


@settings(max_examples=100)
@given(
    num_steps=st.integers(min_value=1, max_value=10),
    final_correct=st.booleans(),
)
def test_propagation_analysis_returns_none_for_all_correct(
    num_steps: int,
    final_correct: bool
):
    """
    Feature: reasoning-eval-enhancement, Property 7: Propagation Analysis Validity (edge case)
    
    *For any* list of step scores where all steps are correct,
    analyze_propagation SHALL return None.
    
    Validates: Requirements 4.1
    """
    # Create all-correct step scores
    step_scores = [
        StepScore(step_index=i, is_correct=True, confidence=1.0)
        for i in range(num_steps)
    ]
    
    tracker = ErrorPropagationTracker()
    analysis = tracker.analyze_propagation(step_scores, final_correct)
    
    assert analysis is None, "Analysis should be None when all steps are correct"


# =============================================================================
# Property 8: Propagation Rate Bounds
# Validates: Requirements 4.3
# =============================================================================

@settings(max_examples=100)
@given(
    analyses=st.lists(propagation_analysis_strategy(), min_size=0, max_size=20),
)
def test_propagation_rate_bounds(analyses: list):
    """
    Feature: reasoning-eval-enhancement, Property 8: Propagation Rate Bounds
    
    *For any* list of PropagationAnalysis objects, compute_propagation_rate
    SHALL return a value in range [0.0, 1.0], representing the fraction
    of errors that cascade.
    
    Validates: Requirements 4.3
    """
    tracker = ErrorPropagationTracker()
    rate = tracker.compute_propagation_rate(analyses)
    
    # Rate should be in [0.0, 1.0]
    assert 0.0 <= rate <= 1.0, (
        f"Propagation rate {rate} is not in range [0.0, 1.0]"
    )


@settings(max_examples=100)
@given(
    num_analyses=st.integers(min_value=1, max_value=20),
)
def test_propagation_rate_all_cascading(num_analyses: int):
    """
    Feature: reasoning-eval-enhancement, Property 8: Propagation Rate Bounds (edge case)
    
    *For any* list of PropagationAnalysis objects where all are CASCADING or TERMINAL,
    compute_propagation_rate SHALL return 1.0.
    
    Validates: Requirements 4.3
    """
    # Create all cascading/terminal analyses
    analyses = [
        PropagationAnalysis(
            first_error_step=0,
            propagation_type=PropagationType.CASCADING,
            affected_steps=[0],
            recovery_attempted=False,
            recovery_successful=False,
        )
        for _ in range(num_analyses)
    ]
    
    tracker = ErrorPropagationTracker()
    rate = tracker.compute_propagation_rate(analyses)
    
    assert rate == 1.0, f"Expected rate 1.0 for all cascading, got {rate}"


@settings(max_examples=100)
@given(
    num_analyses=st.integers(min_value=1, max_value=20),
)
def test_propagation_rate_all_recoverable(num_analyses: int):
    """
    Feature: reasoning-eval-enhancement, Property 8: Propagation Rate Bounds (edge case)
    
    *For any* list of PropagationAnalysis objects where all are RECOVERABLE,
    compute_propagation_rate SHALL return 0.0.
    
    Validates: Requirements 4.3
    """
    # Create all recoverable analyses
    analyses = [
        PropagationAnalysis(
            first_error_step=0,
            propagation_type=PropagationType.RECOVERABLE,
            affected_steps=[0],
            recovery_attempted=True,
            recovery_successful=True,
        )
        for _ in range(num_analyses)
    ]
    
    tracker = ErrorPropagationTracker()
    rate = tracker.compute_propagation_rate(analyses)
    
    assert rate == 0.0, f"Expected rate 0.0 for all recoverable, got {rate}"


# =============================================================================
# Property 9: Recovery Rate Calculation
# Validates: Requirements 4.4, 7.2
# =============================================================================

@settings(max_examples=100)
@given(
    analyses=st.lists(propagation_analysis_strategy(), min_size=0, max_size=20),
)
def test_recovery_rate_bounds(analyses: list):
    """
    Feature: reasoning-eval-enhancement, Property 9: Recovery Rate Calculation
    
    *For any* list of PropagationAnalysis objects, compute_recovery_rate
    SHALL return a value in range [0.0, 1.0].
    
    Validates: Requirements 4.4, 7.2
    """
    tracker = ErrorPropagationTracker()
    rate = tracker.compute_recovery_rate(analyses)
    
    # Rate should be in [0.0, 1.0]
    assert 0.0 <= rate <= 1.0, (
        f"Recovery rate {rate} is not in range [0.0, 1.0]"
    )


@settings(max_examples=100)
@given(
    analyses=st.lists(propagation_analysis_strategy(), min_size=1, max_size=20),
)
def test_recovery_rate_equals_recoverable_fraction(analyses: list):
    """
    Feature: reasoning-eval-enhancement, Property 9: Recovery Rate Calculation
    
    *For any* list of PropagationAnalysis objects, compute_recovery_rate
    SHALL equal the count of RECOVERABLE propagations divided by total propagations.
    
    Validates: Requirements 4.4, 7.2
    """
    tracker = ErrorPropagationTracker()
    rate = tracker.compute_recovery_rate(analyses)
    
    # Manually compute expected rate
    recoverable_count = sum(
        1 for a in analyses if a.propagation_type == PropagationType.RECOVERABLE
    )
    expected_rate = recoverable_count / len(analyses)
    
    assert abs(rate - expected_rate) < 1e-9, (
        f"Recovery rate {rate} does not match expected {expected_rate}"
    )


@settings(max_examples=100)
@given(
    num_analyses=st.integers(min_value=1, max_value=20),
)
def test_recovery_rate_all_recoverable(num_analyses: int):
    """
    Feature: reasoning-eval-enhancement, Property 9: Recovery Rate Calculation (edge case)
    
    *For any* list of PropagationAnalysis objects where all are RECOVERABLE,
    compute_recovery_rate SHALL return 1.0.
    
    Validates: Requirements 4.4, 7.2
    """
    # Create all recoverable analyses
    analyses = [
        PropagationAnalysis(
            first_error_step=0,
            propagation_type=PropagationType.RECOVERABLE,
            affected_steps=[0],
            recovery_attempted=True,
            recovery_successful=True,
        )
        for _ in range(num_analyses)
    ]
    
    tracker = ErrorPropagationTracker()
    rate = tracker.compute_recovery_rate(analyses)
    
    assert rate == 1.0, f"Expected rate 1.0 for all recoverable, got {rate}"


def test_recovery_rate_empty_list():
    """
    Feature: reasoning-eval-enhancement, Property 9: Recovery Rate Calculation (edge case)
    
    For an empty list of analyses, compute_recovery_rate SHALL return 0.0.
    
    Validates: Requirements 4.4, 7.2
    """
    tracker = ErrorPropagationTracker()
    rate = tracker.compute_recovery_rate([])
    
    assert rate == 0.0, f"Expected rate 0.0 for empty list, got {rate}"
