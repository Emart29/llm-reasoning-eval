"""Hypothesis strategies for property-based testing.

This module provides custom generators for property tests as specified
in the design document.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypothesis import strategies as st, settings

from src.models import (
    ReasoningProblem,
    ReasoningStep,
    ParsedChainOfThought,
    StepAlignment,
    StepScore,
    ErrorCategory,
    ErrorClassification,
    PropagationType,
    PropagationAnalysis,
    VALID_CATEGORIES,
)


# Configure Hypothesis settings for minimum 100 iterations
settings.register_profile("default", max_examples=100)
settings.load_profile("default")


# =============================================================================
# Dataset Generator Strategies
# =============================================================================

# Valid categories for reasoning problems (imported from models)
# VALID_CATEGORIES is already imported from src.models

# Strategy for generating valid reasoning problems
reasoning_problem_strategy = st.builds(
    ReasoningProblem,
    id=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=10
    ),
    category=st.sampled_from(list(VALID_CATEGORIES)),
    subtype=st.text(min_size=1, max_size=20),
    difficulty=st.integers(min_value=1, max_value=5),
    prompt=st.text(min_size=10, max_size=500),
    ground_truth=st.text(min_size=1, max_size=100),
    solution_steps=st.lists(
        st.text(min_size=5, max_size=200),
        min_size=1,
        max_size=10
    ),
    metadata=st.fixed_dictionaries({}),
    adversarial=st.booleans(),
)


# =============================================================================
# Chain-of-Thought Analyzer Strategies
# =============================================================================

# Strategy for generating valid CoT outputs with step markers
@st.composite
def cot_output_with_steps(draw):
    """Generate a chain-of-thought output string with step markers."""
    num_steps = draw(st.integers(min_value=1, max_value=5))
    steps = []
    for i in range(num_steps):
        step_content = draw(st.text(min_size=5, max_size=100))
        steps.append(f"Step {i + 1}: {step_content}")
    
    answer = draw(st.text(min_size=1, max_size=50))
    return "\n".join(steps) + f"\nAnswer: {answer}"


# Strategy for generating numbered list CoT outputs
@st.composite
def cot_output_numbered_list(draw):
    """Generate a chain-of-thought output with numbered list format."""
    num_steps = draw(st.integers(min_value=1, max_value=5))
    steps = []
    for i in range(num_steps):
        # Use text without newlines to avoid breaking the numbered list pattern
        # Also ensure min_size is at least MIN_STEP_LENGTH (3) from cot_analyzer
        step_content = draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P"), whitelist_characters=" "),
            min_size=5,
            max_size=100
        ))
        steps.append(f"{i + 1}. {step_content}")
    
    answer = draw(st.text(min_size=1, max_size=50))
    return "\n".join(steps) + f"\nFinal answer: {answer}"


# Strategy for generating bullet point CoT outputs
@st.composite
def cot_output_bullet_points(draw):
    """Generate a chain-of-thought output with bullet points."""
    num_steps = draw(st.integers(min_value=1, max_value=5))
    steps = []
    for i in range(num_steps):
        step_content = draw(st.text(min_size=5, max_size=100))
        steps.append(f"- {step_content}")
    
    answer = draw(st.text(min_size=1, max_size=50))
    return "\n".join(steps) + f"\nAnswer: {answer}"


# Combined strategy for any valid CoT format
cot_output_strategy = st.one_of(
    cot_output_with_steps(),
    cot_output_numbered_list(),
    cot_output_bullet_points(),
)


# Strategy for generating malformed text without step markers
@st.composite
def malformed_text_without_markers(draw):
    """Generate text that doesn't contain any step markers.
    
    Ensures the text doesn't match patterns like:
    - "Step N:" format
    - Numbered lists (N. text)
    - Bullet points (- text or * text)
    """
    # Generate text that avoids step marker patterns
    # Use only letters and spaces to avoid accidental pattern matches
    text = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L",), whitelist_characters=" "),
        min_size=10,
        max_size=200
    ))
    # Ensure no accidental "Step" word at start of lines
    lines = text.split("\n")
    safe_lines = []
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("step"):
            line = "text " + line
        safe_lines.append(line)
    return "\n".join(safe_lines)


# Strategy for generating unparseable/malformed outputs
unparseable_output_strategy = st.one_of(
    st.just(""),  # Empty string
    st.just("   \n\t  \n  "),  # Whitespace only
    st.binary().map(lambda b: b.decode("utf-8", errors="replace")),  # Random bytes
    st.text(max_size=2),  # Too short to have meaningful steps
    malformed_text_without_markers(),  # Malformed text without step markers
)


# Strategy for reasoning steps
reasoning_step_strategy = st.builds(
    ReasoningStep,
    index=st.integers(min_value=0, max_value=20),
    content=st.text(min_size=1, max_size=200),
    step_type=st.sampled_from(["calculation", "inference", "conclusion", "setup"]),
    extracted_values=st.lists(st.text(min_size=1, max_size=20), max_size=5),
)


# Strategy for parsed chain-of-thought
@st.composite
def parsed_cot_strategy(draw):
    """Generate a ParsedChainOfThought object."""
    raw_output = draw(cot_output_strategy)
    num_steps = draw(st.integers(min_value=1, max_value=5))
    steps = [draw(reasoning_step_strategy) for _ in range(num_steps)]
    # Ensure indices are sequential
    for i, step in enumerate(steps):
        step.index = i
    
    return ParsedChainOfThought(
        raw_output=raw_output,
        steps=steps,
        final_answer=draw(st.text(min_size=1, max_size=50)),
        parse_success=True,
        parse_errors=[],
    )


# =============================================================================
# Step Scorer Strategies
# =============================================================================

# Strategy for step scores
step_score_strategy = st.builds(
    StepScore,
    step_index=st.integers(min_value=0, max_value=20),
    is_correct=st.booleans(),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    error_details=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
)


# Strategy for lists of step scores (with sequential indices)
@st.composite
def step_scores_list_strategy(draw, min_size=1, max_size=10):
    """Generate a list of step scores with sequential indices."""
    num_scores = draw(st.integers(min_value=min_size, max_value=max_size))
    scores = []
    for i in range(num_scores):
        score = StepScore(
            step_index=i,
            is_correct=draw(st.booleans()),
            confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            error_details=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        )
        scores.append(score)
    return scores


# Strategy for step scores with at least one error
@st.composite
def step_scores_with_error_strategy(draw, min_size=2, max_size=10):
    """Generate step scores where at least one step is incorrect."""
    num_scores = draw(st.integers(min_value=min_size, max_value=max_size))
    error_idx = draw(st.integers(min_value=0, max_value=num_scores - 1))
    
    scores = []
    for i in range(num_scores):
        is_correct = i != error_idx and draw(st.booleans())
        score = StepScore(
            step_index=i,
            is_correct=is_correct,
            confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            error_details=None if is_correct else "Error detected",
        )
        scores.append(score)
    return scores


# =============================================================================
# Error Taxonomy Strategies
# =============================================================================

# Strategy for error categories
error_category_strategy = st.sampled_from(list(ErrorCategory))


# Strategy for confidence scores (all categories must be present)
@st.composite
def confidence_scores_strategy(draw):
    """Generate confidence scores for all error categories."""
    return {
        cat: draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        for cat in ErrorCategory
    }


# Strategy for error classifications
@st.composite
def error_classification_strategy(draw):
    """Generate a valid ErrorClassification object."""
    categories = draw(st.lists(error_category_strategy, min_size=1, max_size=5, unique=True))
    confidence_scores = draw(confidence_scores_strategy())
    primary = draw(st.sampled_from(categories))
    
    return ErrorClassification(
        categories=categories,
        confidence_scores=confidence_scores,
        primary_category=primary,
        explanation=draw(st.text(min_size=0, max_size=200)),
        flagged_for_review=draw(st.booleans()),
    )


# =============================================================================
# Error Propagation Strategies
# =============================================================================

# Strategy for propagation types
propagation_type_strategy = st.sampled_from(list(PropagationType))


# Strategy for propagation analysis
@st.composite
def propagation_analysis_strategy(draw):
    """Generate a valid PropagationAnalysis object."""
    first_error = draw(st.integers(min_value=0, max_value=10))
    total_steps = draw(st.integers(min_value=first_error + 1, max_value=15))
    
    # Affected steps must be >= first_error_step
    affected = draw(st.lists(
        st.integers(min_value=first_error, max_value=total_steps - 1),
        min_size=0,
        max_size=total_steps - first_error,
        unique=True,
    ))
    
    prop_type = draw(propagation_type_strategy)
    recovery_attempted = draw(st.booleans())
    recovery_successful = recovery_attempted and prop_type == PropagationType.RECOVERABLE
    
    return PropagationAnalysis(
        first_error_step=first_error,
        propagation_type=prop_type,
        affected_steps=sorted(affected),
        recovery_attempted=recovery_attempted,
        recovery_successful=recovery_successful,
    )


# =============================================================================
# Metrics Strategies
# =============================================================================

# Strategy for multi-sample results (for consistency testing)
@st.composite
def multi_sample_results_strategy(draw, num_samples=5):
    """Generate multiple sample results for the same problem."""
    # Generate a base answer and some variations
    base_answer = draw(st.text(min_size=1, max_size=50))
    results = []
    for _ in range(num_samples):
        if draw(st.booleans()):  # Sometimes use base answer
            results.append(base_answer)
        else:  # Sometimes use different answer
            results.append(draw(st.text(min_size=1, max_size=50)))
    return results


# Strategy for step alignment
step_alignment_strategy = st.builds(
    StepAlignment,
    model_step_idx=st.integers(min_value=0, max_value=20),
    ground_truth_step_idx=st.one_of(st.none(), st.integers(min_value=0, max_value=20)),
    alignment_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    is_correct=st.booleans(),
)
