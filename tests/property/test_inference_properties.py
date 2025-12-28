"""Property tests for Inference Engine.

Feature: reasoning-eval-enhancement
Property 10: Output Normalization Idempotence
Validates: Requirements 5.3
"""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from src.normalization import normalize_output


# =============================================================================
# Strategies for generating model outputs
# =============================================================================

@st.composite
def raw_model_output_strategy(draw):
    """Generate raw model outputs that may contain provider-specific artifacts."""
    content = draw(st.text(min_size=0, max_size=500))
    prefix = draw(st.sampled_from(["", "Assistant: ", "AI: ", "Bot: "]))
    suffix = draw(st.sampled_from(["", "\nHuman:", "\nUser:"]))
    leading_ws = draw(st.sampled_from(["", " ", "\n", "\t"]))
    trailing_ws = draw(st.sampled_from(["", " ", "\n", "\t"]))
    return leading_ws + prefix + content + suffix + trailing_ws


@st.composite
def output_with_line_endings_strategy(draw):
    """Generate outputs with mixed line ending styles."""
    lines = draw(st.lists(st.text(min_size=0, max_size=50), min_size=0, max_size=5))
    endings = [draw(st.sampled_from(["\n", "\r\n", "\r"])) for _ in lines]
    result = ""
    for i, line in enumerate(lines):
        result += line + (endings[i] if i < len(endings) else "")
    return result


@st.composite
def output_with_blank_lines_strategy(draw):
    """Generate outputs with multiple consecutive blank lines."""
    parts = draw(st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=5))
    result = parts[0]
    for part in parts[1:]:
        num_blanks = draw(st.integers(min_value=1, max_value=5))
        result += "\n" * num_blanks + part
    return result


any_raw_output_strategy = st.one_of(
    raw_model_output_strategy(),
    output_with_line_endings_strategy(),
    output_with_blank_lines_strategy(),
    st.text(min_size=0, max_size=500),
    st.just(""),
)


# =============================================================================
# Property 10: Output Normalization Idempotence
# Validates: Requirements 5.3
# =============================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(raw_output=any_raw_output_strategy)
def test_normalize_output_idempotence(raw_output: str):
    """
    Feature: reasoning-eval-enhancement, Property 10: Output Normalization Idempotence
    
    *For any* model output, normalizing twice SHALL produce the same result
    as normalizing once (normalization is idempotent).
    
    Validates: Requirements 5.3
    """
    normalized_once = normalize_output(raw_output)
    normalized_twice = normalize_output(normalized_once)
    
    assert normalized_once == normalized_twice, (
        f"Normalization is not idempotent!\n"
        f"Original: {repr(raw_output)}\n"
        f"Normalized once: {repr(normalized_once)}\n"
        f"Normalized twice: {repr(normalized_twice)}"
    )


@settings(max_examples=100)
@given(raw_output=raw_model_output_strategy())
def test_normalize_output_idempotence_with_artifacts(raw_output: str):
    """
    Feature: reasoning-eval-enhancement, Property 10: Output Normalization Idempotence
    
    *For any* model output with provider-specific artifacts, normalizing twice
    SHALL produce the same result as normalizing once.
    
    Validates: Requirements 5.3
    """
    normalized_once = normalize_output(raw_output)
    normalized_twice = normalize_output(normalized_once)
    
    assert normalized_once == normalized_twice


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(raw_output=output_with_line_endings_strategy())
def test_normalize_output_idempotence_line_endings(raw_output: str):
    """
    Feature: reasoning-eval-enhancement, Property 10: Output Normalization Idempotence
    
    *For any* model output with mixed line endings, normalizing twice
    SHALL produce the same result as normalizing once.
    
    Validates: Requirements 5.3
    """
    normalized_once = normalize_output(raw_output)
    normalized_twice = normalize_output(normalized_once)
    
    assert normalized_once == normalized_twice


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(raw_output=output_with_blank_lines_strategy())
def test_normalize_output_idempotence_blank_lines(raw_output: str):
    """
    Feature: reasoning-eval-enhancement, Property 10: Output Normalization Idempotence
    
    *For any* model output with multiple blank lines, normalizing twice
    SHALL produce the same result as normalizing once.
    
    Validates: Requirements 5.3
    """
    normalized_once = normalize_output(raw_output)
    normalized_twice = normalize_output(normalized_once)
    
    assert normalized_once == normalized_twice


@settings(max_examples=100)
@given(raw_output=st.text(min_size=0, max_size=500))
def test_normalize_output_removes_leading_trailing_whitespace(raw_output: str):
    """
    Feature: reasoning-eval-enhancement, Property 10: Output Normalization Idempotence
    
    *For any* model output, the normalized result SHALL have no leading
    or trailing whitespace.
    
    Validates: Requirements 5.3
    """
    normalized = normalize_output(raw_output)
    if normalized:
        assert normalized == normalized.strip()


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(raw_output=output_with_line_endings_strategy())
def test_normalize_output_uses_unix_line_endings(raw_output: str):
    """
    Feature: reasoning-eval-enhancement, Property 10: Output Normalization Idempotence
    
    *For any* model output, the normalized result SHALL use only Unix-style
    line endings.
    
    Validates: Requirements 5.3
    """
    normalized = normalize_output(raw_output)
    assert '\r' not in normalized


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(raw_output=output_with_blank_lines_strategy())
def test_normalize_output_collapses_blank_lines(raw_output: str):
    """
    Feature: reasoning-eval-enhancement, Property 10: Output Normalization Idempotence
    
    *For any* model output, the normalized result SHALL have at most two
    consecutive newlines.
    
    Validates: Requirements 5.3
    """
    normalized = normalize_output(raw_output)
    assert '\n\n\n' not in normalized


def test_normalize_output_empty_string():
    """Empty string normalization returns empty string."""
    assert normalize_output("") == ""


def test_normalize_output_whitespace_only():
    """Whitespace-only input normalizes to empty string."""
    assert normalize_output("   \n\t  ") == ""
