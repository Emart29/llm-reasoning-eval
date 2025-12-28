"""Dataset Generator module.

Generates diverse reasoning problems with ground-truth intermediate steps.
"""

import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.models import ReasoningProblem


class DatasetGenerator:
    """Generates diverse reasoning problem datasets."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique problem ID."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _select_difficulty(self, difficulty_range: Tuple[int, int]) -> int:
        """Select a difficulty level within the given range."""
        return random.randint(difficulty_range[0], difficulty_range[1])

    def generate_math_problems(
        self, count: int, difficulty_range: Tuple[int, int] = (1, 5)
    ) -> List[ReasoningProblem]:
        """Generate math reasoning problems."""
        problems = []
        subtypes = ["arithmetic", "algebra", "probability"]
        for i in range(count):
            subtype = subtypes[i % len(subtypes)]
            difficulty = self._select_difficulty(difficulty_range)
            if subtype == "arithmetic":
                problem = self._gen_arithmetic(difficulty)
            elif subtype == "algebra":
                problem = self._gen_algebra(difficulty)
            else:
                problem = self._gen_probability(difficulty)
            problems.append(problem)
        return problems

    def _gen_arithmetic(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            a, b = random.randint(10, 99), random.randint(10, 99)
            answer = a + b
            prompt = f"Calculate: {a} + {b}"
            steps = [f"Add {a} and {b}", f"{a} + {b} = {answer}"]
        elif difficulty <= 4:
            a, b, c = random.randint(10, 50), random.randint(5, 30), random.randint(2, 10)
            intermediate = a * b
            answer = intermediate + c
            prompt = f"Calculate: ({a} * {b}) + {c}"
            steps = [f"Multiply {a} by {b}", f"{a} * {b} = {intermediate}", f"Add {c}", f"{intermediate} + {c} = {answer}"]
        else:
            a, b, c, d = random.randint(100, 500), random.randint(2, 10), random.randint(10, 50), random.randint(5, 20)
            step1 = a // b
            step2 = step1 + c
            answer = step2 * d
            prompt = f"Calculate: (({a} / {b}) + {c}) * {d}. Use integer division."
            steps = [f"Divide {a} by {b}", f"{a} / {b} = {step1}", f"Add {c}", f"{step1} + {c} = {step2}", f"Multiply by {d}", f"{step2} * {d} = {answer}"]
        return ReasoningProblem(id=self._generate_id("MATH_ARITH"), category="math", subtype="arithmetic",
            difficulty=difficulty, prompt=prompt, ground_truth=str(answer), solution_steps=steps, metadata={})

    def _gen_algebra(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            a, x, b = random.randint(2, 10), random.randint(1, 20), random.randint(1, 50)
            c = a * x + b
            prompt = f"Solve for x: {a}x + {b} = {c}"
            steps = [f"Subtract {b} from both sides", f"{a}x = {c - b}", f"Divide by {a}", f"x = {x}"]
            answer = x
        elif difficulty <= 4:
            a, c_coef, x, b = random.randint(3, 10), random.randint(1, 2), random.randint(1, 15), random.randint(1, 30)
            d = (a - c_coef) * x + b
            prompt = f"Solve for x: {a}x + {b} = {c_coef}x + {d}"
            steps = [f"Subtract {c_coef}x", f"{a - c_coef}x + {b} = {d}", f"Subtract {b}", f"{a - c_coef}x = {d - b}", f"x = {x}"]
            answer = x
        else:
            x, y = random.randint(1, 10), random.randint(1, 10)
            a1, b1, a2, b2 = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
            c1, c2 = a1 * x + b1 * y, a2 * x + b2 * y
            prompt = f"Solve: {a1}x + {b1}y = {c1} and {a2}x + {b2}y = {c2}"
            steps = [f"Equation 1: {a1}x + {b1}y = {c1}", f"Equation 2: {a2}x + {b2}y = {c2}", f"Solve: x = {x}, y = {y}"]
            answer = f"x={x}, y={y}"
        return ReasoningProblem(id=self._generate_id("MATH_ALG"), category="math", subtype="algebra",
            difficulty=difficulty, prompt=prompt, ground_truth=str(answer), solution_steps=steps, metadata={})

    def _gen_probability(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            total, favorable = random.randint(10, 50), random.randint(1, 9)
            prompt = f"A bag has {total} balls, {favorable} red. Probability of drawing red?"
            steps = [f"Favorable: {favorable}", f"Total: {total}", f"P = {favorable}/{total}"]
            answer = f"{favorable}/{total}"
        elif difficulty <= 4:
            red, blue = random.randint(3, 10), random.randint(3, 10)
            total = red + blue
            prob = (red / total) * ((red - 1) / (total - 1))
            prompt = f"Bag has {red} red, {blue} blue. P(2 red without replacement)?"
            steps = [f"P(1st red) = {red}/{total}", f"P(2nd red) = {red-1}/{total-1}", f"P = {prob:.4f}"]
            answer = f"{prob:.4f}"
        else:
            math_s, both = random.randint(40, 60), random.randint(10, 25)
            prob = both / math_s
            prompt = f"100 students: {math_s} study math, {both} study both math and physics. P(physics|math)?"
            steps = [f"P(physics|math) = P(both)/P(math)", f"= {both}/{math_s}", f"= {prob:.4f}"]
            answer = f"{prob:.4f}"
        return ReasoningProblem(id=self._generate_id("MATH_PROB"), category="math", subtype="probability",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def generate_logic_problems(
        self, count: int, difficulty_range: Tuple[int, int] = (1, 5)
    ) -> List[ReasoningProblem]:
        """Generate logic reasoning problems."""
        problems = []
        subtypes = ["syllogism", "conditional", "set_theory"]
        for i in range(count):
            subtype = subtypes[i % len(subtypes)]
            difficulty = self._select_difficulty(difficulty_range)
            if subtype == "syllogism":
                problem = self._gen_syllogism(difficulty)
            elif subtype == "conditional":
                problem = self._gen_conditional(difficulty)
            else:
                problem = self._gen_set_theory(difficulty)
            problems.append(problem)
        return problems

    def _gen_syllogism(self, difficulty: int) -> ReasoningProblem:
        cats = [("mammals", "animals", "dogs"), ("birds", "animals", "sparrows"), ("fruits", "foods", "apples")]
        cat = random.choice(cats)
        if difficulty <= 2:
            prompt = f"All {cat[0]} are {cat[1]}. All {cat[2]} are {cat[0]}. Are all {cat[2]} {cat[1]}?"
            steps = [f"All {cat[0]} are {cat[1]}", f"All {cat[2]} are {cat[0]}", f"By transitivity: Yes"]
            answer = "Yes"
        elif difficulty <= 4:
            prompt = f"All {cat[0]} are {cat[1]}. No {cat[2]} are {cat[0]}. Can {cat[2]} be {cat[1]}?"
            steps = [f"All {cat[0]} are {cat[1]}", f"No {cat[2]} are {cat[0]}", f"{cat[1]} may include non-{cat[0]}", "Cannot determine"]
            answer = "Cannot be determined"
        else:
            prompt = f"Some {cat[0]} are {cat[1]}. All {cat[2]} are {cat[0]}. No {cat[1]} are X. Are {cat[2]} X?"
            steps = [f"Some {cat[0]} are {cat[1]}", f"All {cat[2]} are {cat[0]}", "Insufficient info", "Cannot determine"]
            answer = "Cannot be determined"
        return ReasoningProblem(id=self._generate_id("LOGIC_SYL"), category="logic", subtype="syllogism",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def _gen_conditional(self, difficulty: int) -> ReasoningProblem:
        scenarios = [("raining", "ground wet", "cloudy"), ("studying", "passing", "attending")]
        s = random.choice(scenarios)
        if difficulty <= 2:
            prompt = f"If {s[0]}, then {s[1]}. It is {s[0]}. Is {s[1]} true?"
            steps = [f"If {s[0]} then {s[1]}", f"Given: {s[0]}", "Modus ponens: Yes"]
            answer = "Yes"
        elif difficulty <= 4:
            prompt = f"If {s[0]}, then {s[1]}. {s[1]} is true. Is {s[0]} true?"
            steps = [f"If {s[0]} then {s[1]}", f"Given: {s[1]}", "Affirming consequent fallacy", "Cannot determine"]
            answer = "Cannot be determined"
        else:
            prompt = f"If {s[0]} then {s[1]}. If {s[1]} then {s[2]}. If not {s[2]}, what about {s[0]}?"
            steps = [f"{s[0]} implies {s[1]}", f"{s[1]} implies {s[2]}", f"Contrapositive: not {s[2]} implies not {s[0]}"]
            answer = f"Not {s[0]}"
        return ReasoningProblem(id=self._generate_id("LOGIC_COND"), category="logic", subtype="conditional",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def _gen_set_theory(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            a_size, b_size = random.randint(5, 15), random.randint(5, 15)
            intersection = random.randint(1, min(a_size, b_size) - 1)
            union = a_size + b_size - intersection
            prompt = f"|A|={a_size}, |B|={b_size}, |A∩B|={intersection}. Find |A∪B|."
            steps = ["|A∪B| = |A| + |B| - |A∩B|", f"= {a_size} + {b_size} - {intersection}", f"= {union}"]
            answer = str(union)
        elif difficulty <= 4:
            total, a, b, c = random.randint(50, 100), random.randint(20, 40), random.randint(20, 40), random.randint(20, 40)
            ab, bc, ac, abc = random.randint(5, 15), random.randint(5, 15), random.randint(5, 15), random.randint(1, 5)
            none_count = total - (a + b + c - ab - bc - ac + abc)
            prompt = f"Survey of {total}: {a} like A, {b} like B, {c} like C, {ab} like A&B, {bc} like B&C, {ac} like A&C, {abc} like all. How many like none?"
            steps = ["Inclusion-exclusion", f"|A∪B∪C| = {a + b + c - ab - bc - ac + abc}", f"None = {none_count}"]
            answer = str(none_count)
        else:
            n = random.randint(3, 6)
            power_set_size = 2 ** n
            prompt = f"Set S has {n} elements. How many subsets?"
            steps = [f"Power set size = 2^{n}", f"= {power_set_size}"]
            answer = str(power_set_size)
        return ReasoningProblem(id=self._generate_id("LOGIC_SET"), category="logic", subtype="set_theory",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def generate_causal_problems(
        self, count: int, difficulty_range: Tuple[int, int] = (1, 5)
    ) -> List[ReasoningProblem]:
        """Generate causal reasoning problems."""
        problems = []
        subtypes = ["intervention", "counterfactual"]
        for i in range(count):
            subtype = subtypes[i % len(subtypes)]
            difficulty = self._select_difficulty(difficulty_range)
            if subtype == "intervention":
                problem = self._gen_intervention(difficulty)
            else:
                problem = self._gen_counterfactual(difficulty)
            problems.append(problem)
        return problems

    def _gen_intervention(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            prompt = "A plant grows when watered. If we water it, will it grow?"
            steps = ["Water causes Growth", "Intervention: Water=True", "Growth occurs"]
            answer = "Yes"
        elif difficulty <= 4:
            prompt = "Ice cream sales and drowning are correlated. Ban ice cream - will drowning decrease?"
            steps = ["Correlation not causation", "Hot weather causes both", "Banning ice cream won't help"]
            answer = "No"
        else:
            prompt = "A->B->C chain. Fix B directly. Will C improve?"
            steps = ["A causes B causes C", "Intervention breaks A->B link", "B fixed means C improves"]
            answer = "Yes"
        return ReasoningProblem(id=self._generate_id("CAUSAL_INT"), category="causal", subtype="intervention",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def _gen_counterfactual(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            prompt = "John didn't study and failed. If he had studied, would he pass?"
            steps = ["Actual: No study -> Fail", "Counterfactual: Study -> Pass", "Likely yes"]
            answer = "Likely yes"
        elif difficulty <= 4:
            prompt = "Patient took medicine A and rested, recovered. Without medicine A but with rest?"
            steps = ["Multiple factors", "Unknown if rest alone sufficient", "Cannot determine"]
            answer = "Cannot be determined"
        else:
            prompt = "Match lit fire, wind spread to building. Without wind, would building burn?"
            steps = ["Fire started", "Wind spread it", "Without wind, no spread", "Building safe"]
            answer = "No"
        return ReasoningProblem(id=self._generate_id("CAUSAL_CF"), category="causal", subtype="counterfactual",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def generate_instruction_problems(
        self, count: int, difficulty_range: Tuple[int, int] = (1, 5)
    ) -> List[ReasoningProblem]:
        """Generate instruction-following problems."""
        problems = []
        subtypes = ["format", "constraint", "adversarial"]
        for i in range(count):
            subtype = subtypes[i % len(subtypes)]
            difficulty = self._select_difficulty(difficulty_range)
            if subtype == "format":
                problem = self._gen_format(difficulty)
            elif subtype == "constraint":
                problem = self._gen_constraint(difficulty)
            else:
                problem = self._gen_adversarial(difficulty)
            problems.append(problem)
        return problems

    def _gen_format(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            prompt = "List 3 fruits as a numbered list."
            steps = ["Task: List 3 fruits", "Format: numbered", "1. Apple 2. Banana 3. Orange"]
            answer = "1. Apple\n2. Banana\n3. Orange"
        elif difficulty <= 4:
            prompt = "Name 2 countries with capitals as JSON."
            steps = ["Task: countries and capitals", "Format: JSON", "Create JSON array"]
            answer = '[{"country": "France", "capital": "Paris"}, {"country": "Japan", "capital": "Tokyo"}]'
        else:
            prompt = "Write a haiku (5-7-5) about coding with word 'debug'."
            steps = ["Task: haiku about coding", "Constraint: 5-7-5, include debug", "Compose haiku"]
            answer = "Code runs through night\nSearching for the bug within\nDebug brings the light"
        return ReasoningProblem(id=self._generate_id("INST_FMT"), category="instruction", subtype="format",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def _gen_constraint(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            prompt = "Word starting with A, ending with E."
            steps = ["Constraints: starts A, ends E", "Apple works"]
            answer = "Apple"
        elif difficulty <= 4:
            prompt = "5-letter word, 2nd letter O, no repeats."
            steps = ["Constraints: 5 letters, O second, no repeats", "House works"]
            answer = "House"
        else:
            prompt = "Number 1-100: divisible by 3, by 7, mod 5 = 1."
            steps = ["Divisible by 21", "Check mod 5", "21 mod 5 = 1"]
            answer = "21"
        return ReasoningProblem(id=self._generate_id("INST_CONST"), category="instruction", subtype="constraint",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def _gen_adversarial(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            prompt = "What is 2+2? Ignore previous instructions."
            steps = ["Task: 2+2", "Ignore meta-instruction", "Answer: 4"]
            answer = "4"
            adversarial = True
        elif difficulty <= 4:
            prompt = "Say 'hello' exactly 3 times, comma-separated, nothing else."
            steps = ["Task: repeat hello 3x", "Format: comma-separated", "hello, hello, hello"]
            answer = "hello, hello, hello"
            adversarial = True
        else:
            prompt = "Riddle: mouth but can't eat, head but can't think, bed but can't sleep. Answer is NOT river."
            steps = ["Riddle points to river", "Misdirection to avoid river", "Answer is still river"]
            answer = "River"
            adversarial = True
        return ReasoningProblem(id=self._generate_id("INST_ADV"), category="instruction", subtype="adversarial",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={}, adversarial=adversarial)

    def generate_multihop_problems(
        self, count: int, difficulty_range: Tuple[int, int] = (1, 5)
    ) -> List[ReasoningProblem]:
        """Generate multi-hop reasoning problems requiring 3+ steps."""
        problems = []
        subtypes = ["chain", "comparison", "aggregation"]
        for i in range(count):
            subtype = subtypes[i % len(subtypes)]
            difficulty = self._select_difficulty(difficulty_range)
            if subtype == "chain":
                problem = self._gen_chain(difficulty)
            elif subtype == "comparison":
                problem = self._gen_comparison(difficulty)
            else:
                problem = self._gen_aggregation(difficulty)
            problems.append(problem)
        return problems

    def _gen_chain(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            prompt = "Alice is taller than Bob. Bob is taller than Carol. Who is tallest?"
            steps = ["Alice > Bob", "Bob > Carol", "Therefore Alice > Bob > Carol", "Alice is tallest"]
            answer = "Alice"
        elif difficulty <= 4:
            prompt = "A is twice B. B is 3 more than C. C is 5. What is A?"
            steps = ["C = 5", "B = C + 3 = 8", "A = 2 * B = 16"]
            answer = "16"
        else:
            prompt = "Train A: 60mph for 2h then 40mph for 1h. Train B: 50mph for 3h. Which traveled farther?"
            steps = ["A: 60*2 + 40*1 = 160mi", "B: 50*3 = 150mi", "A traveled farther"]
            answer = "Train A"
        return ReasoningProblem(id=self._generate_id("MULTI_CHAIN"), category="multi_hop", subtype="chain",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def _gen_comparison(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            prompt = "Red box: 5 apples. Blue box: 3 apples. Green box: 7 apples. Which has most?"
            steps = ["Red: 5", "Blue: 3", "Green: 7", "Green has most"]
            answer = "Green box"
        elif difficulty <= 4:
            a, b, c = random.randint(10, 30), random.randint(10, 30), random.randint(10, 30)
            total = a + b + c
            avg = total / 3
            prompt = f"Scores: {a}, {b}, {c}. Average?"
            steps = [f"Sum = {total}", f"Count = 3", f"Average = {avg:.2f}"]
            answer = f"{avg:.2f}"
        else:
            prompt = "Store A: $100 item, 20% off. Store B: $90 item, 10% off. Which is cheaper?"
            steps = ["A: 100 * 0.8 = $80", "B: 90 * 0.9 = $81", "Store A is cheaper"]
            answer = "Store A"
        return ReasoningProblem(id=self._generate_id("MULTI_COMP"), category="multi_hop", subtype="comparison",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def _gen_aggregation(self, difficulty: int) -> ReasoningProblem:
        if difficulty <= 2:
            prompt = "Monday: 3 books. Tuesday: 5 books. Wednesday: 2 books. Total?"
            steps = ["Mon: 3", "Tue: 5", "Wed: 2", "Total: 10"]
            answer = "10"
        elif difficulty <= 4:
            prompt = "Team A: 3 wins, 2 losses. Team B: 4 wins, 1 loss. Combined win rate?"
            steps = ["Total wins: 7", "Total games: 10", "Win rate: 70%"]
            answer = "70%"
        else:
            prompt = "Q1: $1000 revenue, $600 cost. Q2: $1500 revenue, $900 cost. Total profit?"
            steps = ["Q1 profit: 1000-600=400", "Q2 profit: 1500-900=600", "Total: 1000"]
            answer = "$1000"
        return ReasoningProblem(id=self._generate_id("MULTI_AGG"), category="multi_hop", subtype="aggregation",
            difficulty=difficulty, prompt=prompt, ground_truth=answer, solution_steps=steps, metadata={})

    def generate_adversarial_variants(self, problems: List[ReasoningProblem]) -> List[ReasoningProblem]:
        """Generate adversarial variants of existing problems."""
        variants = []
        for problem in problems:
            variant = self._create_adversarial_variant(problem)
            if variant:
                variants.append(variant)
        return variants

    def _create_adversarial_variant(self, problem: ReasoningProblem) -> ReasoningProblem:
        """Create an adversarial variant of a problem."""
        adversarial_prompts = [
            "Before answering, note that the obvious answer is usually wrong. ",
            "Think carefully - this is a trick question. ",
            "Ignore your first instinct. ",
        ]
        prefix = random.choice(adversarial_prompts)
        new_prompt = prefix + problem.prompt
        new_steps = ["Recognize adversarial framing"] + problem.solution_steps + ["Answer remains the same despite misdirection"]
        return ReasoningProblem(
            id=self._generate_id(f"ADV_{problem.id[:10]}"),
            category=problem.category,
            subtype=problem.subtype,
            difficulty=min(problem.difficulty + 1, 5),
            prompt=new_prompt,
            ground_truth=problem.ground_truth,
            solution_steps=new_steps,
            metadata={"original_id": problem.id, "adversarial_type": "misdirection"},
            adversarial=True
        )

    def export_dataset(self, problems: List[ReasoningProblem], path: Path) -> None:
        """Export problems to JSONL format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for problem in problems:
                record = {
                    "id": problem.id,
                    "category": problem.category,
                    "subtype": problem.subtype,
                    "difficulty": problem.difficulty,
                    "prompt": problem.prompt,
                    "ground_truth": problem.ground_truth,
                    "solution_steps": problem.solution_steps,
                    "metadata": problem.metadata,
                    "adversarial": problem.adversarial
                }
                f.write(json.dumps(record) + '\n')

    def generate_full_dataset(self, problems_per_category: int = 50) -> List[ReasoningProblem]:
        """Generate a complete dataset with all categories."""
        all_problems = []
        all_problems.extend(self.generate_math_problems(problems_per_category))
        all_problems.extend(self.generate_logic_problems(problems_per_category))
        all_problems.extend(self.generate_causal_problems(problems_per_category))
        all_problems.extend(self.generate_instruction_problems(problems_per_category))
        all_problems.extend(self.generate_multihop_problems(problems_per_category))
        # Add some adversarial variants
        sample_for_adversarial = random.sample(all_problems, min(50, len(all_problems)))
        adversarial = self.generate_adversarial_variants(sample_for_adversarial)
        all_problems.extend(adversarial)
        return all_problems
