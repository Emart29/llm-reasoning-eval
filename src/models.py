"""Core data models for LLM Reasoning Evaluation.

This module contains all dataclasses used throughout the evaluation system.
These models are defined centrally to ensure consistency and enable proper
validation across all components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums
# =============================================================================

class ErrorCategory(Enum):
    """Categories of reasoning errors."""
    ARITHMETIC = "arithmetic"
    LOGICAL_FALLACY = "logical_fallacy"
    PREMISE_MISUNDERSTANDING = "premise_misunderstanding"
    STEP_OMISSION = "step_omission"
    HALLUCINATION = "hallucination"
    INSTRUCTION_OVERRIDE = "instruction_override"
    CONTEXT_LOSS = "context_loss"
    UNIT_ERROR = "unit_error"
    OFF_BY_ONE = "off_by_one"
    SIGN_ERROR = "sign_error"
    OTHER = "other"


class PropagationType(Enum):
    """Types of error propagation patterns."""
    RECOVERABLE = "recoverable"  # Model self-corrects
    CASCADING = "cascading"      # Error affects subsequent steps
    TERMINAL = "terminal"        # Error leads to wrong final answer


# =============================================================================
# Dataset Models
# =============================================================================

VALID_CATEGORIES = {"math", "logic", "causal", "instruction", "multi_hop"}


@dataclass
class ReasoningProblem:
    """A reasoning problem with ground truth solution.
    
    Attributes:
        id: Unique identifier for the problem
        category: Problem category (math, logic, causal, instruction, multi_hop)
        subtype: Specific subtype within the category
        difficulty: Difficulty level from 1-5
        prompt: The problem prompt/question
        ground_truth: The correct final answer
        solution_steps: List of intermediate solution steps
        metadata: Additional problem metadata
        adversarial: Whether this is an adversarial variant
    """
    id: str
    category: str
    subtype: str
    difficulty: int
    prompt: str
    ground_truth: str
    solution_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    adversarial: bool = False
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category: {self.category}. "
                f"Must be one of {VALID_CATEGORIES}"
            )
        if not 1 <= self.difficulty <= 5:
            raise ValueError(
                f"Invalid difficulty: {self.difficulty}. "
                f"Must be between 1 and 5"
            )


# =============================================================================
# Chain-of-Thought Analysis Models
# =============================================================================

@dataclass
class ReasoningStep:
    """Represents a single reasoning step extracted from model output.
    
    Attributes:
        index: Position of this step in the reasoning chain
        content: The text content of the step
        step_type: Type of reasoning (calculation, inference, conclusion, etc.)
        extracted_values: Numeric or key values extracted from the step
    """
    index: int
    content: str
    step_type: str = "inference"
    extracted_values: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if self.index < 0:
            raise ValueError(f"Step index must be non-negative, got {self.index}")


@dataclass
class ParsedChainOfThought:
    """Result of parsing a chain-of-thought output.
    
    Attributes:
        raw_output: The original model output string
        steps: List of parsed reasoning steps
        final_answer: The extracted final answer
        parse_success: Whether parsing succeeded
        parse_errors: List of parsing error messages
    """
    raw_output: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    parse_success: bool = True
    parse_errors: List[str] = field(default_factory=list)


@dataclass
class StepAlignment:
    """Alignment between a model step and ground truth step.
    
    Attributes:
        model_step_idx: Index of the model's reasoning step
        ground_truth_step_idx: Index of aligned ground truth step (None if no match)
        alignment_score: Similarity score between 0.0 and 1.0
        is_correct: Whether the model step is considered correct
    """
    model_step_idx: int
    ground_truth_step_idx: Optional[int]
    alignment_score: float
    is_correct: bool
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if not 0.0 <= self.alignment_score <= 1.0:
            raise ValueError(
                f"Alignment score must be between 0.0 and 1.0, "
                f"got {self.alignment_score}"
            )


# =============================================================================
# Step Scoring Models
# =============================================================================

@dataclass
class StepScore:
    """Score for a single reasoning step.
    
    Attributes:
        step_index: Index of the step being scored
        is_correct: Whether the step is correct
        confidence: Confidence in the correctness assessment (0.0 to 1.0)
        error_details: Description of the error if incorrect
    """
    step_index: int
    is_correct: bool
    confidence: float
    error_details: Optional[str] = None
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, "
                f"got {self.confidence}"
            )


# =============================================================================
# Error Classification Models
# =============================================================================

@dataclass
class ErrorClassification:
    """Classification result for a reasoning error.
    
    Attributes:
        categories: List of error categories that apply
        confidence_scores: Confidence score for each category
        primary_category: The main error category
        explanation: Human-readable explanation of the error
        flagged_for_review: Whether this needs manual review
    """
    categories: List[ErrorCategory] = field(default_factory=list)
    confidence_scores: Dict[ErrorCategory, float] = field(default_factory=dict)
    primary_category: ErrorCategory = ErrorCategory.OTHER
    explanation: str = ""
    flagged_for_review: bool = False
    
    def __post_init__(self):
        """Validate fields after initialization."""
        # Validate confidence scores are in valid range
        for category, score in self.confidence_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Confidence score for {category} must be between 0.0 and 1.0, "
                    f"got {score}"
                )


# =============================================================================
# Error Propagation Models
# =============================================================================

@dataclass
class PropagationAnalysis:
    """Analysis of how an error propagates through reasoning.
    
    Attributes:
        first_error_step: Index of the first incorrect step
        propagation_type: How the error propagates (recoverable, cascading, terminal)
        affected_steps: List of step indices affected by the error
        recovery_attempted: Whether the model attempted to recover
        recovery_successful: Whether recovery was successful
    """
    first_error_step: int
    propagation_type: PropagationType
    affected_steps: List[int] = field(default_factory=list)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if self.first_error_step < 0:
            raise ValueError(
                f"first_error_step must be non-negative, "
                f"got {self.first_error_step}"
            )
        # Affected steps must be >= first_error_step
        for step in self.affected_steps:
            if step < self.first_error_step:
                raise ValueError(
                    f"Affected step {step} cannot be before "
                    f"first_error_step {self.first_error_step}"
                )


# =============================================================================
# Metrics Models
# =============================================================================

@dataclass
class ReasoningMetrics:
    """Aggregated metrics for reasoning evaluation.
    
    Attributes:
        accuracy: Overall accuracy (fraction of correct final answers)
        reasoning_depth: Average steps completed before failure
        recovery_rate: Fraction of errors that are recovered from
        consistency_score: Agreement across multiple samples (0.0 to 1.0)
        step_efficiency: Ratio of optimal steps to actual steps
        error_propagation_rate: Fraction of errors that cascade
    """
    accuracy: float = 0.0
    reasoning_depth: float = 0.0
    recovery_rate: float = 0.0
    consistency_score: float = 0.0
    step_efficiency: float = 0.0
    error_propagation_rate: float = 0.0
    
    def __post_init__(self):
        """Validate fields after initialization."""
        for field_name in ['accuracy', 'recovery_rate', 'consistency_score', 
                          'step_efficiency', 'error_propagation_rate']:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0.0 and 1.0, got {value}"
                )
        if self.reasoning_depth < 0:
            raise ValueError(
                f"reasoning_depth must be non-negative, got {self.reasoning_depth}"
            )


# =============================================================================
# Evaluation Result Model
# =============================================================================

@dataclass
class EvaluationResult:
    """Complete evaluation result for a single problem.
    
    Attributes:
        problem_id: ID of the evaluated problem
        model: Name of the model used
        strategy: Prompting strategy used (e.g., 'cot', 'direct')
        raw_output: Raw model output string
        parsed_cot: Parsed chain-of-thought result
        step_scores: Scores for each reasoning step
        error_classification: Classification of any errors
        propagation_analysis: Analysis of error propagation
        final_answer: The model's final answer
        is_correct: Whether the final answer is correct
        metrics: Per-problem metrics
        timestamp: When the evaluation was performed
        api_latency_ms: API call latency in milliseconds
    """
    problem_id: str
    model: str
    strategy: str
    raw_output: str
    parsed_cot: Optional[ParsedChainOfThought] = None
    step_scores: List[StepScore] = field(default_factory=list)
    error_classification: Optional[ErrorClassification] = None
    propagation_analysis: Optional[PropagationAnalysis] = None
    final_answer: str = ""
    is_correct: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    api_latency_ms: int = 0
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if self.api_latency_ms < 0:
            raise ValueError(
                f"api_latency_ms must be non-negative, got {self.api_latency_ms}"
            )
