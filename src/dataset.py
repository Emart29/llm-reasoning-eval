# src/dataset.py
"""Dataset utilities for the reasoning evaluation project.
Loads the JSONL file, validates schema, and provides a test split.
"""
import json
from pathlib import Path
from typing import List
import pandas as pd
from .config import DATA_PATH, RANDOM_SEED

def load_dataset() -> pd.DataFrame:
    """Read the JSONL dataset and return a tidy DataFrame.
    Raises ValueError if required fields are missing.
    """
    records: List[dict] = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    required = {"id", "category", "prompt", "ground_truth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df

def get_test_split(frac: float = 1.0) -> pd.DataFrame:
    """Return a random subset (default the whole set) for evaluation.
    The split is deterministic thanks to the global RANDOM_SEED.
    """
    df = load_dataset()
    return df.sample(frac=frac, random_state=RANDOM_SEED).reset_index(drop=True)
