# src/evaluation.py
"""Evaluation utilities for the reasoning benchmark.
Provides answer extraction, correctness check, and error‑type taxonomy.
"""
import re
from typing import Tuple
import pandas as pd

# Regex to capture a numeric answer after the word "Answer" (case‑insensitive)
ANSWER_REGEX = re.compile(r"(?i)(?:answer\s*[:\-]?\s*)([-+]?[0-9]*\.?[0-9]+)")

def extract_final_answer(text: str) -> str:
    """Extract the final numeric answer from a model's raw output.
    Falls back to the last number in the string if the explicit pattern fails.
    """
    if not isinstance(text, str):
        return ""
    m = ANSWER_REGEX.search(text)
    if m:
        return m.group(1).strip()
    # fallback – grab last number token
    numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    return numbers[-1] if numbers else ""

def evaluate_prediction(prediction: str, ground_truth: str) -> Tuple[bool, str]:
    """Return (is_correct, error_type).
    Error taxonomy is deliberately simple but extensible.
    """
    pred_ans = extract_final_answer(prediction)
    gold_ans = extract_final_answer(ground_truth)
    if pred_ans == gold_ans:
        return True, "none"
    # Heuristic error categories – feel free to expand
    lowered = prediction.lower()
    if "ignore" in lowered or "disregard" in lowered or "do not" in lowered:
        return False, "instruction_misinterpret"
    if any(op in lowered for op in ["add", "subtract", "multiply", "divide"]):
        return False, "arithmetic_error"
    if any(k in lowered for k in ["if", "then", "else", "case"]):
        return False, "logic_error"
    return False, "other"

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add `correct` and `error_type` columns to a DataFrame containing
    `prediction` and `ground_truth` columns.
    """
    correct, err = [], []
    for _, row in df.iterrows():
        is_corr, e_type = evaluate_prediction(row["prediction"], row["ground_truth"])
        correct.append(is_corr)
        err.append(e_type)
    out = df.copy()
    out["correct"] = correct
    out["error_type"] = err
    return out
