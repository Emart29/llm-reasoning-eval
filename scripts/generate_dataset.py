#!/usr/bin/env python
"""Generate expanded reasoning dataset with 300+ problems.

This script generates a comprehensive dataset of reasoning problems across
five categories (math, logic, causal, instruction, multi_hop) with difficulty
levels 1-5 and minimum 50 problems per category.

Usage:
    python scripts/generate_dataset.py [--output PATH] [--per-category N] [--seed S]
"""
import argparse
import sys
from pathlib import Path

# Add project root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_generator import DatasetGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate expanded reasoning dataset with 300+ problems"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reasoning_dataset.jsonl",
        help="Output path for the dataset (default: data/reasoning_dataset.jsonl)"
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=60,
        help="Number of problems per category (default: 60, minimum 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--include-adversarial",
        action="store_true",
        default=True,
        help="Include adversarial variants (default: True)"
    )
    parser.add_argument(
        "--adversarial-count",
        type=int,
        default=50,
        help="Number of adversarial variants to generate (default: 50)"
    )
    return parser.parse_args()


def main():
    """Generate the expanded dataset."""
    args = parse_args()
    
    # Ensure minimum 50 per category
    per_category = max(args.per_category, 50)
    
    print(f"Generating reasoning dataset...")
    print(f"  Problems per category: {per_category}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output: {args.output}")
    print()
    
    # Initialize generator
    generator = DatasetGenerator(seed=args.seed)
    
    all_problems = []
    categories = ["math", "logic", "causal", "instruction", "multi_hop"]
    
    # Generate problems for each category
    print("Generating problems by category:")
    
    # Math problems
    math_problems = generator.generate_math_problems(per_category, difficulty_range=(1, 5))
    all_problems.extend(math_problems)
    print(f"  math: {len(math_problems)} problems")
    
    # Logic problems
    logic_problems = generator.generate_logic_problems(per_category, difficulty_range=(1, 5))
    all_problems.extend(logic_problems)
    print(f"  logic: {len(logic_problems)} problems")
    
    # Causal problems
    causal_problems = generator.generate_causal_problems(per_category, difficulty_range=(1, 5))
    all_problems.extend(causal_problems)
    print(f"  causal: {len(causal_problems)} problems")
    
    # Instruction problems
    instruction_problems = generator.generate_instruction_problems(per_category, difficulty_range=(1, 5))
    all_problems.extend(instruction_problems)
    print(f"  instruction: {len(instruction_problems)} problems")
    
    # Multi-hop problems
    multihop_problems = generator.generate_multihop_problems(per_category, difficulty_range=(1, 5))
    all_problems.extend(multihop_problems)
    print(f"  multi_hop: {len(multihop_problems)} problems")
    
    # Generate adversarial variants
    if args.include_adversarial:
        import random
        random.seed(args.seed)
        sample_size = min(args.adversarial_count, len(all_problems))
        sample_for_adversarial = random.sample(all_problems, sample_size)
        adversarial_problems = generator.generate_adversarial_variants(sample_for_adversarial)
        all_problems.extend(adversarial_problems)
        print(f"  adversarial variants: {len(adversarial_problems)} problems")
    
    print()
    
    # Print statistics
    print("Dataset statistics:")
    print(f"  Total problems: {len(all_problems)}")
    
    # Count by category
    category_counts = {}
    for problem in all_problems:
        cat = problem.category
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("  By category:")
    for cat in categories:
        count = category_counts.get(cat, 0)
        print(f"    {cat}: {count}")
    
    # Count by difficulty
    difficulty_counts = {}
    for problem in all_problems:
        diff = problem.difficulty
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    print("  By difficulty:")
    for diff in sorted(difficulty_counts.keys()):
        count = difficulty_counts[diff]
        print(f"    Level {diff}: {count}")
    
    # Count adversarial
    adversarial_count = sum(1 for p in all_problems if p.adversarial)
    print(f"  Adversarial problems: {adversarial_count}")
    
    print()
    
    # Export dataset
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    
    generator.export_dataset(all_problems, output_path)
    print(f"Dataset exported to: {output_path}")
    
    # Verify minimum requirements
    print()
    print("Verification:")
    all_pass = True
    
    if len(all_problems) >= 300:
        print(f"  ✓ Total problems >= 300: {len(all_problems)}")
    else:
        print(f"  ✗ Total problems >= 300: {len(all_problems)} (FAILED)")
        all_pass = False
    
    for cat in categories:
        count = category_counts.get(cat, 0)
        if count >= 50:
            print(f"  ✓ {cat} >= 50: {count}")
        else:
            print(f"  ✗ {cat} >= 50: {count} (FAILED)")
            all_pass = False
    
    if all(d in difficulty_counts for d in range(1, 6)):
        print(f"  ✓ All difficulty levels 1-5 present")
    else:
        missing = [d for d in range(1, 6) if d not in difficulty_counts]
        print(f"  ✗ Missing difficulty levels: {missing} (FAILED)")
        all_pass = False
    
    if all_pass:
        print()
        print("✅ Dataset generation complete and verified!")
        return 0
    else:
        print()
        print("⚠️ Dataset generated but some requirements not met.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
