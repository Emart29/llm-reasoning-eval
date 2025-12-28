"""Property tests for Chain-of-Thought parsing.

Feature: reasoning-eval-enhancement
Properties 2, 3, 4: CoT Parsing, Step Alignment, Parse Failure Handling
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from src.cot_analyzer import ChainOfThoughtAnalyzer
from src.models import ParsedChainOfThought, StepAlignment
from tests.property.conftest import (
    cot_output_with_steps,
    cot_output_numbered_list,
    cot_output_bullet_points,
    cot_output_strategy,
    unparseable_output_strategy,
    parsed_cot_strategy,
)


# =============================================================================
# Property 2: CoT Parsing Produces Steps
# Validates: Requirements 2.1
# =============================================================================

@settings(max_examples=100)
@given(cot_output=cot_output_with_steps())
def test_cot_parsing_step_n_format_produces_steps(cot_output: str):
    """
    Feature: reasoning-eval-enhancement, Property 2: CoT Parsing Produces Steps
    
    *For any* valid chain-of-thought output string containing "Step N:" markers,
    parsing SHALL produce a ParsedChainOfThought with at least one ReasoningStep.
    
    Validates: Requirements 2.1
    """
    analyzer = ChainOfThoughtAnalyzer()
    result = analyzer.parse_output(cot_output)
    
    assert isinstance(result, ParsedChainOfThought)
    assert result.parse_success is True
    assert len(result.steps) >= 1
    assert result.raw_output == cot_output



@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(cot_output=cot_output_numbered_list())
def test_cot_parsing_numbered_list_produces_steps(cot_output: str):
    """
    Feature: reasoning-eval-enhancement, Property 2: CoT Parsing Produces Steps
    
    *For any* valid chain-of-thought output string containing numbered lists,
    parsing SHALL produce a ParsedChainOfThought with at least one ReasoningStep.
    
    Validates: Requirements 2.1
    """
    analyzer = ChainOfThoughtAnalyzer()
    result = analyzer.parse_output(cot_output)
    
    assert isinstance(result, ParsedChainOfThought)
    assert result.parse_success is True
    assert len(result.steps) >= 1


@settings(max_examples=100)
@given(cot_output=cot_output_bullet_points())
def test_cot_parsing_bullet_points_produces_steps(cot_output: str):
    """
    Feature: reasoning-eval-enhancement, Property 2: CoT Parsing Produces Steps
    
    *For any* valid chain-of-thought output string containing bullet points,
    parsing SHALL produce a ParsedChainOfThought with at least one ReasoningStep.
    
    Validates: Requirements 2.1
    """
    analyzer = ChainOfThoughtAnalyzer()
    result = analyzer.parse_output(cot_output)
    
    assert isinstance(result, ParsedChainOfThought)
    assert result.parse_success is True
    assert len(result.steps) >= 1



# =============================================================================
# Property 3: Step Alignment Consistency
# Validates: Requirements 2.2, 2.4
# =============================================================================

@settings(max_examples=100)
@given(
    num_model_steps=st.integers(min_value=1, max_value=5),
    num_gt_steps=st.integers(min_value=1, max_value=5),
)
def test_step_alignment_produces_alignment_for_each_model_step(
    num_model_steps: int,
    num_gt_steps: int
):
    """
    Feature: reasoning-eval-enhancement, Property 3: Step Alignment Consistency
    
    *For any* parsed chain-of-thought and ground-truth steps, alignment SHALL
    produce a StepAlignment for each model step.
    
    Validates: Requirements 2.2
    """
    analyzer = ChainOfThoughtAnalyzer()
    
    # Create a CoT output with the specified number of steps
    steps_text = "\n".join([f"Step {i+1}: This is step {i+1} content" for i in range(num_model_steps)])
    cot_output = steps_text + "\nAnswer: 42"
    
    parsed_cot = analyzer.parse_output(cot_output)
    ground_truth_steps = [f"Ground truth step {i+1}" for i in range(num_gt_steps)]
    
    alignments = analyzer.align_steps(parsed_cot, ground_truth_steps)
    
    # Should have one alignment for each model step
    assert len(alignments) == len(parsed_cot.steps)
    
    # Each alignment should be a StepAlignment
    for alignment in alignments:
        assert isinstance(alignment, StepAlignment)
        assert 0.0 <= alignment.alignment_score <= 1.0



@settings(max_examples=100)
@given(
    num_steps=st.integers(min_value=1, max_value=5),
)
def test_find_first_error_returns_correct_index(num_steps: int):
    """
    Feature: reasoning-eval-enhancement, Property 3: Step Alignment Consistency
    
    *For any* list of alignments, find_first_error SHALL return the index of
    the first alignment where is_correct is False (or None if all correct).
    
    Validates: Requirements 2.4
    """
    analyzer = ChainOfThoughtAnalyzer()
    
    # Create alignments where all are correct
    all_correct = [
        StepAlignment(
            model_step_idx=i,
            ground_truth_step_idx=i,
            alignment_score=0.9,
            is_correct=True
        )
        for i in range(num_steps)
    ]
    
    # All correct should return None
    assert analyzer.find_first_error(all_correct) is None
    
    # Create alignments with first error at index 0
    first_error_at_0 = [
        StepAlignment(
            model_step_idx=i,
            ground_truth_step_idx=i,
            alignment_score=0.3 if i == 0 else 0.9,
            is_correct=(i != 0)
        )
        for i in range(num_steps)
    ]
    
    assert analyzer.find_first_error(first_error_at_0) == 0


@settings(max_examples=100)
@given(
    error_idx=st.integers(min_value=0, max_value=4),
    num_steps=st.integers(min_value=1, max_value=5),
)
def test_find_first_error_finds_earliest_error(error_idx: int, num_steps: int):
    """
    Feature: reasoning-eval-enhancement, Property 3: Step Alignment Consistency
    
    *For any* list of alignments with an error at a specific index,
    find_first_error SHALL return that index.
    
    Validates: Requirements 2.4
    """
    assume(error_idx < num_steps)
    
    analyzer = ChainOfThoughtAnalyzer()
    
    alignments = [
        StepAlignment(
            model_step_idx=i,
            ground_truth_step_idx=i,
            alignment_score=0.3 if i >= error_idx else 0.9,
            is_correct=(i < error_idx)
        )
        for i in range(num_steps)
    ]
    
    result = analyzer.find_first_error(alignments)
    assert result == error_idx



# =============================================================================
# Property 4: Parse Failure Handling
# Validates: Requirements 2.5
# =============================================================================

@settings(max_examples=100)
@given(unparseable=unparseable_output_strategy)
def test_parse_failure_returns_structured_error(unparseable: str):
    """
    Feature: reasoning-eval-enhancement, Property 4: Parse Failure Handling
    
    *For any* unparseable input (empty string, random bytes, malformed text),
    the Chain_of_Thought_Analyzer SHALL return a ParsedChainOfThought with
    parse_success=False and non-empty parse_errors containing the problematic content.
    
    Validates: Requirements 2.5
    """
    analyzer = ChainOfThoughtAnalyzer()
    result = analyzer.parse_output(unparseable)
    
    assert isinstance(result, ParsedChainOfThought)
    assert result.parse_success is False
    assert len(result.parse_errors) > 0
    assert result.raw_output == unparseable


def test_empty_string_parse_failure():
    """
    Feature: reasoning-eval-enhancement, Property 4: Parse Failure Handling
    
    Empty string input SHALL return parse_success=False with appropriate error.
    
    Validates: Requirements 2.5
    """
    analyzer = ChainOfThoughtAnalyzer()
    result = analyzer.parse_output("")
    
    assert result.parse_success is False
    assert len(result.parse_errors) > 0
    assert "empty" in result.parse_errors[0].lower() or "whitespace" in result.parse_errors[0].lower()


def test_whitespace_only_parse_failure():
    """
    Feature: reasoning-eval-enhancement, Property 4: Parse Failure Handling
    
    Whitespace-only input SHALL return parse_success=False with appropriate error.
    
    Validates: Requirements 2.5
    """
    analyzer = ChainOfThoughtAnalyzer()
    result = analyzer.parse_output("   \n\t  \n  ")
    
    assert result.parse_success is False
    assert len(result.parse_errors) > 0


def test_no_step_markers_parse_failure():
    """
    Feature: reasoning-eval-enhancement, Property 4: Parse Failure Handling
    
    Input without step markers SHALL return parse_success=False with error
    containing the problematic content.
    
    Validates: Requirements 2.5
    """
    analyzer = ChainOfThoughtAnalyzer()
    malformed = "This is just some text without any step markers or structure."
    result = analyzer.parse_output(malformed)
    
    assert result.parse_success is False
    assert len(result.parse_errors) > 0
    # Error should contain reference to the problematic content
    assert "step markers" in result.parse_errors[0].lower() or malformed[:20] in result.parse_errors[0]
