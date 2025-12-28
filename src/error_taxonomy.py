"""Error Taxonomy Engine module.

Classifies reasoning errors into detailed categories using keyword matching
and pattern detection.
"""

import json
import re
from typing import Dict, List

from src.models import (
    ErrorCategory,
    ErrorClassification,
    StepScore,
)


# Keywords and patterns associated with each error category
ERROR_PATTERNS: Dict[ErrorCategory, Dict[str, list]] = {
    ErrorCategory.ARITHMETIC: {
        "keywords": [
            "calculation", "compute", "add", "subtract", "multiply", "divide",
            "sum", "product", "difference", "quotient", "total", "equals",
            "plus", "minus", "times", "divided by", "result", "answer"
        ],
        "patterns": [r"=\s*\d+", r"\d+\s*%"],
        "error_indicators": [
            "wrong calculation", "arithmetic error", "math error",
            "incorrect sum", "wrong total", "miscalculation"
        ],
    },
    ErrorCategory.LOGICAL_FALLACY: {
        "keywords": [
            "therefore", "thus", "hence", "because", "since", "implies",
            "if", "then", "conclude", "follows", "reason", "logic"
        ],
        "patterns": [r"therefore\s+.+", r"because\s+.+"],
        "error_indicators": [
            "invalid inference", "logical error", "fallacy", "non sequitur",
            "false premise", "circular reasoning", "invalid conclusion"
        ],
    },
    ErrorCategory.PREMISE_MISUNDERSTANDING: {
        "keywords": [
            "given", "stated", "problem says", "according to", "based on",
            "the question", "we know", "it says", "mentioned", "provided"
        ],
        "patterns": [r"given\s+that\s+.+"],
        "error_indicators": [
            "misread", "misunderstood", "wrong interpretation",
            "incorrect premise", "misinterpreted"
        ],
    },
    ErrorCategory.STEP_OMISSION: {
        "keywords": [
            "skip", "omit", "missing", "forgot", "directly", "immediately",
            "jump", "shortcut", "without", "bypassing"
        ],
        "patterns": [],
        "error_indicators": [
            "missing step", "skipped", "omitted", "incomplete reasoning",
            "jumped to conclusion"
        ],
    },
    ErrorCategory.HALLUCINATION: {
        "keywords": [
            "assume", "suppose", "imagine", "pretend", "hypothetically",
            "invented", "made up", "fabricated"
        ],
        "patterns": [r"assume\s+.+"],
        "error_indicators": [
            "hallucinated", "invented", "fabricated", "not in problem",
            "made up", "fictional"
        ],
    },
    ErrorCategory.INSTRUCTION_OVERRIDE: {
        "keywords": [
            "instead", "rather", "different", "alternative", "ignore",
            "disregard", "override", "change", "modify"
        ],
        "patterns": [r"instead\s+of\s+.+"],
        "error_indicators": [
            "ignored instruction", "override", "changed the question"
        ],
    },
    ErrorCategory.CONTEXT_LOSS: {
        "keywords": [
            "earlier", "previous", "before", "mentioned", "stated",
            "above", "prior", "already", "previously"
        ],
        "patterns": [r"as\s+mentioned\s+earlier"],
        "error_indicators": [
            "lost context", "forgot earlier", "inconsistent with previous",
            "contradicts earlier"
        ],
    },
    ErrorCategory.UNIT_ERROR: {
        "keywords": [
            "unit", "meter", "kilogram", "second", "dollar", "percent",
            "feet", "inches", "miles", "hours", "minutes"
        ],
        "patterns": [],
        "error_indicators": [
            "wrong unit", "unit error", "unit mismatch", "conversion error"
        ],
    },
    ErrorCategory.OFF_BY_ONE: {
        "keywords": [
            "count", "index", "position", "first", "last", "boundary",
            "fence", "inclusive", "exclusive", "starting", "ending"
        ],
        "patterns": [],
        "error_indicators": [
            "off by one", "fence post", "boundary error", "index error",
            "one too many", "one too few"
        ],
    },
    ErrorCategory.SIGN_ERROR: {
        "keywords": [
            "negative", "positive", "minus", "plus", "sign", "opposite",
            "subtract", "add", "direction"
        ],
        "patterns": [r"-\d+", r"\+\d+"],
        "error_indicators": [
            "sign error", "wrong sign", "should be negative",
            "should be positive", "opposite sign"
        ],
    },
    ErrorCategory.OTHER: {
        "keywords": [],
        "patterns": [],
        "error_indicators": [],
    },
}


class ErrorTaxonomyEngine:
    """Classifies reasoning errors into taxonomy categories.
    
    Uses keyword matching and pattern detection to classify errors
    into predefined categories with confidence scores.
    """
    
    def __init__(self):
        """Initialize the error taxonomy engine."""
        self._compiled_patterns: Dict[ErrorCategory, List[re.Pattern]] = {}
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        for category, config in ERROR_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in config.get("patterns", [])
            ]
    
    def classify_error(
        self, 
        model_output: str, 
        ground_truth: str, 
        step_scores: List[StepScore]
    ) -> ErrorClassification:
        """Classify an error based on model output and ground truth.
        
        Args:
            model_output: The model's raw output string
            ground_truth: The expected correct answer/output
            step_scores: List of step scores from step accuracy scoring
            
        Returns:
            ErrorClassification with categories, confidence scores, and explanation
        """
        confidence_scores = self._compute_all_confidence_scores(
            model_output, ground_truth, step_scores
        )
        
        threshold = 0.3
        detected_categories = [
            cat for cat, score in confidence_scores.items()
            if score >= threshold and cat != ErrorCategory.OTHER
        ]
        
        if not detected_categories:
            detected_categories = [ErrorCategory.OTHER]
            confidence_scores[ErrorCategory.OTHER] = 0.5
        
        primary_category = max(
            detected_categories,
            key=lambda c: confidence_scores[c]
        )
        
        explanation = self._generate_explanation(
            detected_categories, confidence_scores, step_scores
        )
        
        flagged = (
            confidence_scores[primary_category] < 0.5 or
            len(detected_categories) > 2 or
            primary_category == ErrorCategory.OTHER
        )
        
        return ErrorClassification(
            categories=detected_categories,
            confidence_scores=confidence_scores,
            primary_category=primary_category,
            explanation=explanation,
            flagged_for_review=flagged,
        )

    def _compute_all_confidence_scores(
        self,
        model_output: str,
        ground_truth: str,
        step_scores: List[StepScore]
    ) -> Dict[ErrorCategory, float]:
        """Compute confidence scores for all error categories."""
        scores: Dict[ErrorCategory, float] = {}
        error_text = self._extract_error_context(model_output, step_scores)
        combined_text = f"{model_output} {ground_truth} {error_text}"
        
        for category in ErrorCategory:
            scores[category] = self._compute_category_confidence(
                category, combined_text
            )
        
        return scores
    
    def _compute_category_confidence(
        self,
        category: ErrorCategory,
        combined_text: str
    ) -> float:
        """Compute confidence score for a single category."""
        config = ERROR_PATTERNS.get(category, {})
        keywords = config.get("keywords", [])
        error_indicators = config.get("error_indicators", [])
        patterns = self._compiled_patterns.get(category, [])
        
        if not keywords and not patterns and not error_indicators:
            return 0.0
        
        score = 0.0
        text_lower = combined_text.lower()
        
        if keywords:
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            keyword_score = min(1.0, keyword_matches / max(3, len(keywords) * 0.3))
            score += 0.3 * keyword_score
        
        if error_indicators:
            indicator_matches = sum(1 for ind in error_indicators if ind.lower() in text_lower)
            indicator_score = min(1.0, indicator_matches / max(1, len(error_indicators) * 0.2))
            score += 0.4 * indicator_score
        
        if patterns:
            pattern_matches = sum(1 for pattern in patterns if pattern.search(combined_text))
            pattern_score = min(1.0, pattern_matches / max(1, len(patterns) * 0.5))
            score += 0.3 * pattern_score
        
        return min(1.0, max(0.0, score))
    
    def _extract_error_context(
        self,
        model_output: str,
        step_scores: List[StepScore]
    ) -> str:
        """Extract error context from step scores."""
        error_details = []
        for score in step_scores:
            if not score.is_correct and score.error_details:
                error_details.append(score.error_details)
        return " ".join(error_details)

    def _generate_explanation(
        self,
        categories: List[ErrorCategory],
        confidence_scores: Dict[ErrorCategory, float],
        step_scores: List[StepScore]
    ) -> str:
        """Generate human-readable explanation of the classification."""
        if not categories:
            return "No specific error pattern detected."
        
        first_error_idx = None
        for score in step_scores:
            if not score.is_correct:
                first_error_idx = score.step_index
                break
        
        parts = []
        cat_descriptions = []
        for cat in categories:
            conf = confidence_scores.get(cat, 0.0)
            cat_descriptions.append(f"{cat.value} ({conf:.0%} confidence)")
        
        parts.append(f"Detected error types: {', '.join(cat_descriptions)}")
        
        if first_error_idx is not None:
            parts.append(f"First error at step {first_error_idx}")
        
        return ". ".join(parts) + "."
    
    def to_json(self, classification: ErrorClassification) -> str:
        """Serialize error classification to JSON.
        
        Args:
            classification: The ErrorClassification to serialize
            
        Returns:
            JSON string representation
        """
        data = {
            "categories": [cat.value for cat in classification.categories],
            "confidence_scores": {
                cat.value: score 
                for cat, score in classification.confidence_scores.items()
            },
            "primary_category": classification.primary_category.value,
            "explanation": classification.explanation,
            "flagged_for_review": classification.flagged_for_review,
        }
        return json.dumps(data, indent=2)
    
    def from_json(self, json_str: str) -> ErrorClassification:
        """Deserialize error classification from JSON.
        
        Args:
            json_str: JSON string to deserialize
            
        Returns:
            ErrorClassification object
        """
        data = json.loads(json_str)
        
        categories = [
            ErrorCategory(cat_str) for cat_str in data["categories"]
        ]
        
        confidence_scores = {
            ErrorCategory(cat_str): score
            for cat_str, score in data["confidence_scores"].items()
        }
        
        primary_category = ErrorCategory(data["primary_category"])
        
        return ErrorClassification(
            categories=categories,
            confidence_scores=confidence_scores,
            primary_category=primary_category,
            explanation=data["explanation"],
            flagged_for_review=data["flagged_for_review"],
        )
