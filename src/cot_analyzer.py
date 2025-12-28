"""Chain-of-Thought Analyzer module."""

import re
from difflib import SequenceMatcher
from typing import List, Optional

from src.models import ReasoningStep, ParsedChainOfThought, StepAlignment


class ChainOfThoughtAnalyzer:
    """Analyzes chain-of-thought reasoning outputs."""
    
    MIN_STEP_LENGTH = 3
    
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
        self._step_n_pattern = re.compile(
            r'Step\s*(\d+)\s*[:\.]?\s*(.*?)(?=Step\s*\d+|Answer:|Final answer:|$)',
            re.IGNORECASE | re.DOTALL
        )
        self._numbered_pattern = re.compile(
            r'^(\d+)\.\s+(.+?)(?=^\d+\.|^Answer:|^Final answer:|$)',
            re.MULTILINE | re.DOTALL
        )
        self._bullet_pattern = re.compile(
            r'^[-*]\s+(.+?)(?=^[-*]|^Answer:|^Final answer:|$)',
            re.MULTILINE | re.DOTALL
        )
        self._answer_pattern = re.compile(
            r'(?:Answer|Final answer|Therefore|Thus|So|Hence)[:\s]+(.+?)$',
            re.IGNORECASE | re.DOTALL
        )


    def parse_output(self, raw_output: str) -> ParsedChainOfThought:
        """Parse raw model output into discrete reasoning steps."""
        if not raw_output or not raw_output.strip():
            return ParsedChainOfThought(
                raw_output=raw_output or "",
                steps=[],
                final_answer="",
                parse_success=False,
                parse_errors=["Empty or whitespace-only output"]
            )
        
        steps = self._parse_step_n_format(raw_output)
        if not steps:
            steps = self._parse_numbered_list_format(raw_output)
        if not steps:
            steps = self._parse_bullet_format(raw_output)
        
        final_answer = self._extract_final_answer(raw_output)
        parse_success = len(steps) > 0
        parse_errors = []
        
        if not parse_success:
            truncated = raw_output[:200]
            if len(raw_output) > 200:
                truncated += "..."
            parse_errors.append("No step markers found in output: " + truncated)
        
        return ParsedChainOfThought(
            raw_output=raw_output,
            steps=steps,
            final_answer=final_answer,
            parse_success=parse_success,
            parse_errors=parse_errors
        )


    def _parse_step_n_format(self, text: str) -> List[ReasoningStep]:
        """Parse steps in Step N: format."""
        steps = []
        matches = self._step_n_pattern.findall(text)
        for i, (_, content) in enumerate(matches):
            content = content.strip()
            if len(content) >= self.MIN_STEP_LENGTH:
                steps.append(ReasoningStep(
                    index=i,
                    content=content,
                    step_type=self._infer_step_type(content),
                    extracted_values=self._extract_values(content)
                ))
        return steps

    def _parse_numbered_list_format(self, text: str) -> List[ReasoningStep]:
        """Parse steps in numbered list format."""
        steps = []
        matches = self._numbered_pattern.findall(text)
        for i, (_, content) in enumerate(matches):
            content = content.strip()
            if len(content) >= self.MIN_STEP_LENGTH:
                steps.append(ReasoningStep(
                    index=i,
                    content=content,
                    step_type=self._infer_step_type(content),
                    extracted_values=self._extract_values(content)
                ))
        return steps

    def _parse_bullet_format(self, text: str) -> List[ReasoningStep]:
        """Parse steps in bullet point format."""
        steps = []
        matches = self._bullet_pattern.findall(text)
        for i, content in enumerate(matches):
            content = content.strip()
            if len(content) >= self.MIN_STEP_LENGTH:
                steps.append(ReasoningStep(
                    index=i,
                    content=content,
                    step_type=self._infer_step_type(content),
                    extracted_values=self._extract_values(content)
                ))
        return steps


    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from the output."""
        match = self._answer_pattern.search(text)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        return lines[-1] if lines else ""

    def _infer_step_type(self, content: str) -> str:
        """Infer the type of reasoning step from its content."""
        lower = content.lower()
        calc_ops = ['+', '-', '*', '/', '=']
        calc_words = ['calculate', 'compute', 'equals', 'sum', 'multiply']
        if any(op in content for op in calc_ops) or any(w in lower for w in calc_words):
            return "calculation"
        conclusion_words = ['therefore', 'thus', 'hence', 'conclude', 'answer']
        if any(w in lower for w in conclusion_words):
            return "conclusion"
        setup_words = ['given', 'assume', 'let', 'suppose']
        if any(w in lower for w in setup_words):
            return "setup"
        return "inference"

    def _extract_values(self, content: str) -> List[str]:
        """Extract numeric values from step content."""
        return re.findall(r'-?\d+\.?\d*', content)


    def align_steps(
        self,
        model_cot: ParsedChainOfThought,
        ground_truth_steps: List[str]
    ) -> List[StepAlignment]:
        """Align model steps with ground truth solution steps."""
        alignments = []
        for model_step in model_cot.steps:
            best_idx = None
            best_score = 0.0
            for gt_idx, gt_step in enumerate(ground_truth_steps):
                score = self._compute_similarity(model_step.content, gt_step)
                if score > best_score:
                    best_score = score
                    best_idx = gt_idx
            is_correct = best_score >= self.similarity_threshold
            alignments.append(StepAlignment(
                model_step_idx=model_step.index,
                ground_truth_step_idx=best_idx,
                alignment_score=best_score,
                is_correct=is_correct
            ))
        return alignments

    def _compute_similarity(self, s1: str, s2: str) -> float:
        """Compute similarity between two strings using SequenceMatcher."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def find_first_error(self, alignments: List[StepAlignment]) -> Optional[int]:
        """Find the index of the first incorrect step."""
        for alignment in alignments:
            if not alignment.is_correct:
                return alignment.model_step_idx
        return None
