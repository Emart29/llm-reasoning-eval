# src/normalization.py
"""Output normalization utilities for model outputs.

This module provides functions to normalize model outputs across different
providers, ensuring consistent formatting regardless of the source.
"""
import re


def normalize_output(raw_output: str) -> str:
    """Normalize model output across different providers.
    
    This function standardizes outputs by:
    1. Stripping leading/trailing whitespace
    2. Normalizing line endings to \\n
    3. Collapsing multiple consecutive blank lines to single blank lines
    4. Removing common provider-specific artifacts
    
    The normalization is idempotent: normalizing twice produces the same
    result as normalizing once.
    
    Args:
        raw_output: Raw output string from any model provider
        
    Returns:
        Normalized output string
    """
    if not raw_output:
        return ""
    
    # Strip leading/trailing whitespace
    output = raw_output.strip()
    
    # Normalize line endings (CRLF -> LF, CR -> LF)
    output = output.replace('\r\n', '\n').replace('\r', '\n')
    
    # Collapse multiple consecutive blank lines to single blank line
    output = re.sub(r'\n{3,}', '\n\n', output)
    
    # Remove common provider-specific artifacts
    # Remove "Assistant:" or "AI:" prefixes that some models add
    output = re.sub(r'^(Assistant|AI|Bot):\s*', '', output, flags=re.IGNORECASE)
    
    # Remove trailing "Human:" or similar that might be added
    output = re.sub(r'\n(Human|User):\s*$', '', output, flags=re.IGNORECASE)
    
    # Normalize multiple spaces to single space (but preserve newlines)
    lines = output.split('\n')
    lines = [re.sub(r' {2,}', ' ', line) for line in lines]
    output = '\n'.join(lines)
    
    # Final strip to ensure clean output
    output = output.strip()
    
    return output
