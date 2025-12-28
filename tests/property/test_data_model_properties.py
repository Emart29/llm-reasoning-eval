"""Property tests for data model invariants.

Feature: reasoning-eval-enhancement, Property 1: Dataset Validity (partial)
Validates: Requirements 1.2 - difficulty levels from 1-5
"""

import pytest
from hypothesis import given, settings, strategies as st

from src.models import (
    ReasoningProblem,
    VALID_CATEGORIES,
)


# =============================================================================
# Property 1: Dataset Validity (partial - difficulty range validation)
# Validates: Requirements 1.2
# =============================================================================

@settings(max_examples=100)
@given(
    difficulty=st.integers(min_value=1, max_value=5),
    category=st.sampled_from(list(VALID_CATEGORIES)),
)
def test_valid_difficulty_range_accepted(difficulty: int, category: str):
    """
    Feature: reasoning-eval-enhancement, Property 1: Dataset Validity
    
    *For any* difficulty value in range [1, 5], creating a ReasoningProblem
    SHALL succeed and the difficulty SHALL be preserved.
    
    Validates: Requirements 1.2
    """
    problem = ReasoningProblem(
        id="test_problem",
        category=category,
        subtype="test_subtype",
        difficulty=difficulty,
        prompt="Test prompt for reasoning",
        ground_truth="Test answer",
        solution_steps=["Step 1", "Step 2"],
    )
    
    # Verify difficulty is preserved
    assert problem.difficulty == difficulty
    # Verify difficulty is in valid range
    assert 1 <= problem.difficulty <= 5


@settings(max_examples=100)
@given(
    difficulty=st.integers().filter(lambda x: x < 1 or x > 5),
)
def test_invalid_difficulty_rejected(difficulty: int):
    """
    Feature: reasoning-eval-enhancement, Property 1: Dataset Validity
    
    *For any* difficulty value outside range [1, 5], creating a ReasoningProblem
    SHALL raise a ValueError.
    
    Validates: Requirements 1.2
    """
    with pytest.raises(ValueError) as exc_info:
        ReasoningProblem(
            id="test_problem",
            category="math",
            subtype="test_subtype",
            difficulty=difficulty,
            prompt="Test prompt for reasoning",
            ground_truth="Test answer",
        )
    
    assert "difficulty" in str(exc_info.value).lower()
    assert "1" in str(exc_info.value) and "5" in str(exc_info.value)
