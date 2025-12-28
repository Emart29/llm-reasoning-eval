"""Property tests for dataset generation.

Feature: reasoning-eval-enhancement, Property 1: Dataset Validity (full)
Validates: Requirements 1.1, 1.2, 1.3, 1.4
"""

import pytest
from hypothesis import given, settings, strategies as st, assume

from src.dataset_generator import DatasetGenerator
from src.models import VALID_CATEGORIES


# =============================================================================
# Property 1: Dataset Validity (full)
# Validates: Requirements 1.1, 1.2, 1.3, 1.4
# =============================================================================

class TestDatasetValidityProperty:
    """
    Feature: reasoning-eval-enhancement, Property 1: Dataset Validity
    
    *For any* generated dataset, all problems SHALL have:
    (a) category in {math, logic, causal, instruction, multi_hop}
    (b) difficulty in range [1,5]
    (c) non-empty solution_steps for multi-step problems
    (d) minimum 50 problems per category
    
    Validates: Requirements 1.1, 1.2, 1.3, 1.4
    """

    @settings(max_examples=100)
    @given(
        seed=st.integers(min_value=0, max_value=10000),
        count=st.integers(min_value=1, max_value=20),
        min_diff=st.integers(min_value=1, max_value=3),
        max_diff=st.integers(min_value=3, max_value=5),
    )
    def test_math_problems_have_valid_categories_and_difficulty(
        self, seed: int, count: int, min_diff: int, max_diff: int
    ):
        """
        *For any* generated math problems, all SHALL have:
        - category == 'math'
        - difficulty in [min_diff, max_diff]
        - non-empty solution_steps
        
        Validates: Requirements 1.1, 1.2, 1.4
        """
        assume(min_diff <= max_diff)
        generator = DatasetGenerator(seed=seed)
        problems = generator.generate_math_problems(count, (min_diff, max_diff))
        
        assert len(problems) == count
        for problem in problems:
            assert problem.category == "math"
            assert problem.category in VALID_CATEGORIES
            assert min_diff <= problem.difficulty <= max_diff
            assert 1 <= problem.difficulty <= 5
            assert len(problem.solution_steps) > 0

    @settings(max_examples=100)
    @given(
        seed=st.integers(min_value=0, max_value=10000),
        count=st.integers(min_value=1, max_value=20),
        min_diff=st.integers(min_value=1, max_value=3),
        max_diff=st.integers(min_value=3, max_value=5),
    )
    def test_logic_problems_have_valid_categories_and_difficulty(
        self, seed: int, count: int, min_diff: int, max_diff: int
    ):
        """
        *For any* generated logic problems, all SHALL have:
        - category == 'logic'
        - difficulty in [min_diff, max_diff]
        - non-empty solution_steps
        
        Validates: Requirements 1.1, 1.2, 1.4
        """
        assume(min_diff <= max_diff)
        generator = DatasetGenerator(seed=seed)
        problems = generator.generate_logic_problems(count, (min_diff, max_diff))
        
        assert len(problems) == count
        for problem in problems:
            assert problem.category == "logic"
            assert problem.category in VALID_CATEGORIES
            assert min_diff <= problem.difficulty <= max_diff
            assert 1 <= problem.difficulty <= 5
            assert len(problem.solution_steps) > 0

    @settings(max_examples=100)
    @given(
        seed=st.integers(min_value=0, max_value=10000),
        count=st.integers(min_value=1, max_value=20),
        min_diff=st.integers(min_value=1, max_value=3),
        max_diff=st.integers(min_value=3, max_value=5),
    )
    def test_causal_problems_have_valid_categories_and_difficulty(
        self, seed: int, count: int, min_diff: int, max_diff: int
    ):
        """
        *For any* generated causal problems, all SHALL have:
        - category == 'causal'
        - difficulty in [min_diff, max_diff]
        - non-empty solution_steps
        
        Validates: Requirements 1.1, 1.2, 1.4
        """
        assume(min_diff <= max_diff)
        generator = DatasetGenerator(seed=seed)
        problems = generator.generate_causal_problems(count, (min_diff, max_diff))
        
        assert len(problems) == count
        for problem in problems:
            assert problem.category == "causal"
            assert problem.category in VALID_CATEGORIES
            assert min_diff <= problem.difficulty <= max_diff
            assert 1 <= problem.difficulty <= 5
            assert len(problem.solution_steps) > 0

    @settings(max_examples=100)
    @given(
        seed=st.integers(min_value=0, max_value=10000),
        count=st.integers(min_value=1, max_value=20),
        min_diff=st.integers(min_value=1, max_value=3),
        max_diff=st.integers(min_value=3, max_value=5),
    )
    def test_instruction_problems_have_valid_categories_and_difficulty(
        self, seed: int, count: int, min_diff: int, max_diff: int
    ):
        """
        *For any* generated instruction problems, all SHALL have:
        - category == 'instruction'
        - difficulty in [min_diff, max_diff]
        - non-empty solution_steps
        
        Validates: Requirements 1.1, 1.2, 1.4
        """
        assume(min_diff <= max_diff)
        generator = DatasetGenerator(seed=seed)
        problems = generator.generate_instruction_problems(count, (min_diff, max_diff))
        
        assert len(problems) == count
        for problem in problems:
            assert problem.category == "instruction"
            assert problem.category in VALID_CATEGORIES
            assert min_diff <= problem.difficulty <= max_diff
            assert 1 <= problem.difficulty <= 5
            assert len(problem.solution_steps) > 0

    @settings(max_examples=100)
    @given(
        seed=st.integers(min_value=0, max_value=10000),
        count=st.integers(min_value=1, max_value=20),
        min_diff=st.integers(min_value=1, max_value=3),
        max_diff=st.integers(min_value=3, max_value=5),
    )
    def test_multihop_problems_have_valid_categories_and_difficulty(
        self, seed: int, count: int, min_diff: int, max_diff: int
    ):
        """
        *For any* generated multi-hop problems, all SHALL have:
        - category == 'multi_hop'
        - difficulty in [min_diff, max_diff]
        - non-empty solution_steps with 3+ steps
        
        Validates: Requirements 1.1, 1.2, 1.4
        """
        assume(min_diff <= max_diff)
        generator = DatasetGenerator(seed=seed)
        problems = generator.generate_multihop_problems(count, (min_diff, max_diff))
        
        assert len(problems) == count
        for problem in problems:
            assert problem.category == "multi_hop"
            assert problem.category in VALID_CATEGORIES
            assert min_diff <= problem.difficulty <= max_diff
            assert 1 <= problem.difficulty <= 5
            # Multi-hop problems should have 3+ steps
            assert len(problem.solution_steps) >= 3

    @settings(max_examples=50)
    @given(
        seed=st.integers(min_value=0, max_value=10000),
        problems_per_category=st.integers(min_value=50, max_value=60),
    )
    def test_full_dataset_has_minimum_problems_per_category(
        self, seed: int, problems_per_category: int
    ):
        """
        *For any* full dataset generation, there SHALL be at least
        50 problems per category.
        
        Validates: Requirements 1.3
        """
        generator = DatasetGenerator(seed=seed)
        problems = generator.generate_full_dataset(problems_per_category)
        
        # Count problems per category
        category_counts = {}
        for problem in problems:
            category_counts[problem.category] = category_counts.get(problem.category, 0) + 1
        
        # Verify all 5 categories are present
        assert set(category_counts.keys()) >= VALID_CATEGORIES
        
        # Verify minimum 50 per category (from base generation, before adversarial)
        for category in VALID_CATEGORIES:
            assert category_counts.get(category, 0) >= problems_per_category

    @settings(max_examples=100)
    @given(
        seed=st.integers(min_value=0, max_value=10000),
        count=st.integers(min_value=1, max_value=10),
    )
    def test_adversarial_variants_preserve_category_and_increase_difficulty(
        self, seed: int, count: int
    ):
        """
        *For any* adversarial variant generation, the variants SHALL:
        - Preserve the original category
        - Have difficulty <= 5 (capped)
        - Be marked as adversarial
        
        Validates: Requirements 1.5
        """
        generator = DatasetGenerator(seed=seed)
        original_problems = generator.generate_math_problems(count)
        variants = generator.generate_adversarial_variants(original_problems)
        
        assert len(variants) == len(original_problems)
        for variant, original in zip(variants, original_problems):
            assert variant.category == original.category
            assert variant.adversarial is True
            assert 1 <= variant.difficulty <= 5
            # Difficulty should be increased by 1 (capped at 5)
            expected_diff = min(original.difficulty + 1, 5)
            assert variant.difficulty == expected_diff

    @settings(max_examples=100)
    @given(
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_all_generated_problems_have_valid_structure(self, seed: int):
        """
        *For any* generated dataset, all problems SHALL have:
        - Non-empty id
        - Non-empty prompt
        - Non-empty ground_truth
        - Valid category
        - Valid difficulty
        
        Validates: Requirements 1.1, 1.2, 1.4
        """
        generator = DatasetGenerator(seed=seed)
        problems = generator.generate_full_dataset(10)  # Small dataset for speed
        
        for problem in problems:
            assert problem.id and len(problem.id) > 0
            assert problem.prompt and len(problem.prompt) > 0
            assert problem.ground_truth and len(problem.ground_truth) > 0
            assert problem.category in VALID_CATEGORIES
            assert 1 <= problem.difficulty <= 5
            assert len(problem.solution_steps) > 0
