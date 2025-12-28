"""Property tests for Error Taxonomy Engine.

Feature: reasoning-eval-enhancement
Properties 5 and 6: Error Classification Completeness and Round-Trip
Validates: Requirements 3.2, 3.5
"""

import pytest
from hypothesis import given, settings, strategies as st

from src.models import (
    ErrorCategory,
    ErrorClassification,
    StepScore,
)
from src.error_taxonomy import ErrorTaxonomyEngine

from tests.property.conftest import (
    error_classification_strategy,
    step_score_strategy,
)


# =============================================================================
# Property 5: Error Classification Completeness
# Validates: Requirements 3.2
# =============================================================================

@settings(max_examples=100)
@given(
    model_output=st.text(min_size=0, max_size=500),
    ground_truth=st.text(min_size=0, max_size=200),
    step_scores=st.lists(step_score_strategy, min_size=0, max_size=10),
)
def test_error_classification_completeness(
    model_output: str,
    ground_truth: str,
    step_scores: list
):
    """
    Feature: reasoning-eval-enhancement, Property 5: Error Classification Completeness
    
    *For any* error classification, confidence_scores SHALL contain entries
    for ALL ErrorCategory enum values, and all confidence values SHALL be
    in range [0.0, 1.0].
    
    Validates: Requirements 3.2
    """
    engine = ErrorTaxonomyEngine()
    classification = engine.classify_error(model_output, ground_truth, step_scores)
    
    # Verify all ErrorCategory enum values are present in confidence_scores
    all_categories = set(ErrorCategory)
    present_categories = set(classification.confidence_scores.keys())
    
    assert all_categories == present_categories, (
        f"Missing categories: {all_categories - present_categories}"
    )
    
    # Verify all confidence values are in range [0.0, 1.0]
    for category, confidence in classification.confidence_scores.items():
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence for {category} is {confidence}, expected [0.0, 1.0]"
        )


# =============================================================================
# Property 6: Error Classification Round-Trip
# Validates: Requirements 3.5
# =============================================================================

@settings(max_examples=100)
@given(classification=error_classification_strategy())
def test_error_classification_round_trip(classification: ErrorClassification):
    """
    Feature: reasoning-eval-enhancement, Property 6: Error Classification Round-Trip
    
    *For any* valid ErrorClassification object, serializing to JSON and
    deserializing back SHALL produce an equivalent object (to_json then
    from_json is identity).
    
    Validates: Requirements 3.5
    """
    engine = ErrorTaxonomyEngine()
    
    # Serialize to JSON
    json_str = engine.to_json(classification)
    
    # Deserialize back
    restored = engine.from_json(json_str)
    
    # Verify equivalence
    assert set(restored.categories) == set(classification.categories), (
        f"Categories mismatch: {restored.categories} != {classification.categories}"
    )
    
    assert restored.confidence_scores == classification.confidence_scores, (
        f"Confidence scores mismatch"
    )
    
    assert restored.primary_category == classification.primary_category, (
        f"Primary category mismatch: {restored.primary_category} != {classification.primary_category}"
    )
    
    assert restored.explanation == classification.explanation, (
        f"Explanation mismatch"
    )
    
    assert restored.flagged_for_review == classification.flagged_for_review, (
        f"Flagged for review mismatch"
    )
