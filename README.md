# LLM Reasoning Evaluation

A research-grade framework for investigating multi-step reasoning failures in large language models. This project goes beyond simple accuracy measurement to provide deep insights into where, why, and how LLMs fail during complex reasoning tasks.

## Research Question

**"Where and why do LLMs fail in multi-step reasoning, and how do different models compare in their failure patterns and recovery capabilities?"**

### Hypotheses

1. **Error Localization**: LLM reasoning failures are not uniformly distributed across reasoning chains—specific step types (e.g., arithmetic operations, logical inferences) are more error-prone.
2. **Error Propagation**: Early errors in reasoning chains cascade predictably, but some models demonstrate better recovery capabilities than others.
3. **Difficulty Scaling**: Model performance degrades non-linearly with problem difficulty, with distinct failure modes emerging at different complexity levels.
4. **Category Specificity**: Different model families exhibit characteristic strengths and weaknesses across reasoning categories (math, logic, causal, instruction-following, multi-hop).

## Novel Metrics

This framework introduces four novel metrics that capture reasoning quality beyond simple accuracy:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Reasoning Depth** | Average number of valid reasoning steps completed before the first error | Higher values indicate models can sustain correct reasoning longer; useful for comparing robustness across problem complexities |
| **Recovery Rate** | Fraction of errors where the model self-corrects in subsequent steps | Higher values suggest better error-awareness and self-correction capabilities; important for understanding model reliability |
| **Consistency Score** | Agreement rate across multiple samples of the same problem | Values near 1.0 indicate deterministic reasoning; lower values reveal stochastic failure modes |
| **Step Efficiency** | Ratio of optimal solution steps to actual model steps (capped at 1.0) | Values near 1.0 indicate concise reasoning; lower values suggest unnecessary verbosity or circular reasoning |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Data Layer                                      │
│  ┌──────────────────┐    ┌─────────────────────┐    ┌──────────────────┐   │
│  │ Dataset Generator │───▶│ Reasoning Dataset   │───▶│   Data Loader    │   │
│  │                   │    │   (300+ problems)   │    │                  │   │
│  └──────────────────┘    └─────────────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Inference Layer                                   │
│  ┌──────────────────┐                                                       │
│  │ Inference Engine │───▶ OpenAI │ Anthropic │ Google │ HuggingFace        │
│  └──────────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Analysis Layer                                    │
│  ┌────────────────┐   ┌─────────────────┐   ┌──────────────────────────┐   │
│  │  CoT Analyzer  │──▶│  Step Scorer    │──▶│  Error Taxonomy Engine   │   │
│  └────────────────┘   └─────────────────┘   └──────────────────────────┘   │
│                                                          │                  │
│                                              ┌───────────▼────────────┐    │
│                                              │ Error Propagation      │    │
│                                              │ Tracker                │    │
│                                              └────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Output Layer                                     │
│  ┌────────────────────┐   ┌────────────────┐   ┌─────────────────────┐     │
│  │ Metrics Calculator │──▶│ Report Generator│──▶│ Visualizations +   │     │
│  │                    │   │                 │   │ Markdown Report    │     │
│  └────────────────────┘   └────────────────┘   └─────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Layout

```
llm-reasoning-eval/
├── data/                          # Dataset files
│   └── reasoning_dataset.jsonl    # 300+ curated reasoning problems
├── src/                           # Core library modules
│   ├── models.py                  # Data models and schemas
│   ├── dataset_generator.py       # Problem generation
│   ├── cot_analyzer.py            # Chain-of-thought parsing
│   ├── step_scorer.py             # Step-level accuracy scoring
│   ├── error_taxonomy.py          # Error classification engine
│   ├── error_propagation.py       # Error cascade tracking
│   ├── metrics.py                 # Novel metrics computation
│   ├── inference.py               # Multi-model API integration
│   ├── report_generator.py        # Visualization and reports
│   └── config.py                  # Configuration management
├── scripts/                       # Entry-point scripts
│   ├── generate_dataset.py        # Generate reasoning problems
│   ├── run_experiments.py         # Execute evaluation pipeline
│   └── analyze_results.py         # Generate reports and visualizations
├── results/                       # Output directory
│   ├── logs/                      # API call logs
│   ├── manifests/                 # Reproducibility manifests
│   └── reports/                   # Generated reports and figures
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── property/                  # Property-based tests (Hypothesis)
│   └── integration/               # Integration tests
└── README.md                      # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-reasoning-eval

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Set environment variables for the model providers you want to use:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

### 3. Generate Dataset (Optional)

Generate a fresh dataset of reasoning problems:

```bash
python scripts/generate_dataset.py
```

This creates `data/reasoning_dataset.jsonl` with 300+ problems across five categories.

### 4. Run Experiments

Execute the full evaluation pipeline:

```bash
python scripts/run_experiments.py
```

Options:
- `--models`: Specify models to evaluate (default: all configured)
- `--categories`: Filter by problem category
- `--max-problems`: Limit number of problems for quick testing

### 5. Analyze Results

Generate visualizations and research report:

```bash
python scripts/analyze_results.py
```

This produces:
- Accuracy breakdowns by category and difficulty
- Error type heatmaps across models
- Step accuracy curves showing failure points
- Markdown research summary with statistical analysis

## Problem Categories

| Category | Description | Example |
|----------|-------------|---------|
| **Math** | Arithmetic, algebra, probability | "If a train travels 60 mph for 2.5 hours, how far does it go?" |
| **Logic** | Syllogisms, conditionals, set theory | "All A are B. Some B are C. What can we conclude about A and C?" |
| **Causal** | Interventions, counterfactuals | "If we had increased the temperature, what would have happened to the reaction rate?" |
| **Instruction** | Following complex directives | "List the items in reverse alphabetical order, excluding any that start with vowels." |
| **Multi-hop** | Chained reasoning (3+ steps) | "Given facts A, B, and C, derive conclusion D through intermediate steps." |

## Error Taxonomy

The framework classifies reasoning errors into 11 categories:

| Category | Description |
|----------|-------------|
| `ARITHMETIC` | Calculation errors (addition, multiplication, etc.) |
| `LOGICAL_FALLACY` | Invalid logical inferences |
| `PREMISE_MISUNDERSTANDING` | Misinterpreting problem setup |
| `STEP_OMISSION` | Skipping necessary reasoning steps |
| `HALLUCINATION` | Introducing facts not in the problem |
| `INSTRUCTION_OVERRIDE` | Ignoring explicit instructions |
| `CONTEXT_LOSS` | Forgetting earlier information |
| `UNIT_ERROR` | Incorrect unit handling |
| `OFF_BY_ONE` | Boundary/counting errors |
| `SIGN_ERROR` | Positive/negative confusion |
| `OTHER` | Uncategorized errors |

## Experiment Results

### Model Comparison Summary

| Model | Accuracy | Reasoning Depth | Recovery Rate | Consistency | Avg Latency |
|-------|----------|-----------------|---------------|-------------|-------------|
| Llama 3.3 70B | 62.9% | 0.47 | 9.7% | 72.9% | 2,728ms |
| Llama 3.1 8B | 61.4% | 0.53 | 17.1% | 72.9% | 2,558ms |
| Qwen3-32B | 55.7% | 0.48 | 7.1% | 74.3% | 4,126ms |

**Aggregate Metrics:**
- Overall Accuracy: 60.0%
- Average Reasoning Depth: 0.49
- Recovery Rate: 11.1%
- Consistency Score: 70.5%
- Error Propagation Rate: 88.9%

### Strategy Comparison

| Strategy | Accuracy | Evaluations |
|----------|----------|-------------|
| Zero-shot | 80.0% | 105 |
| Chain-of-Thought | 40.0% | 105 |

**Key Finding**: Zero-shot prompting significantly outperformed Chain-of-Thought in this evaluation, suggesting that explicit step-by-step reasoning may introduce more opportunities for error in these models.

### Performance by Category

| Category | Accuracy | Count | Avg Reasoning Depth |
|----------|----------|-------|---------------------|
| Math | 80.6% | 36 | 0.39 |
| Instruction | 71.4% | 42 | 0.10 |
| Multi-hop | 59.5% | 42 | 0.62 |
| Causal | 54.2% | 48 | 0.00 |
| Logic | 38.1% | 42 | 0.31 |

**Key Findings:**
1. **Math problems** showed the highest accuracy (80.6%), indicating strong arithmetic capabilities
2. **Logic problems** were most challenging (38.1%), revealing weaknesses in formal reasoning
3. **Multi-hop problems** had the highest reasoning depth (0.62), suggesting models engage more deeply with chained reasoning

### Error Distribution

| Error Type | Count | Percentage |
|------------|-------|------------|
| Logical Fallacy | 27 | 38.6% |
| Other | 17 | 24.3% |
| Arithmetic | 12 | 17.1% |
| Premise Misunderstanding | 10 | 14.3% |
| Hallucination | 2 | 2.9% |
| Sign Error | 2 | 2.9% |

**Key Findings:**
1. **Logical fallacies** are the dominant error type (38.6%), confirming hypothesis about formal reasoning weaknesses
2. **Arithmetic errors** account for only 17.1% of failures despite math being a common benchmark focus
3. **Hallucination rate is low** (2.9%), suggesting models stay grounded in problem context

### Key Insights

1. **Smaller models can recover better**: Llama 3.1 8B showed the highest recovery rate (17.1%) despite being the smallest model, suggesting that model size doesn't directly correlate with self-correction ability.

2. **CoT hurts more than helps**: The 40% accuracy with Chain-of-Thought vs 80% with zero-shot indicates that forcing explicit reasoning steps may expose model weaknesses rather than improve performance.

3. **Error propagation is high**: With an 88.9% error propagation rate, once a model makes a mistake, it rarely recovers—highlighting the importance of getting early reasoning steps correct.

4. **Category-specific weaknesses**: Logic problems (38.1% accuracy) represent a clear area for improvement, while math (80.6%) is a relative strength.

### Generated Visualizations

The analysis generates the following visualizations in `results/summary/figures/`:

- `error_heatmap.png` - Heatmap of error types by model
- `step_accuracy_curves.png` - Curves showing where reasoning fails
- `model_comparison_radar.png` - Radar chart comparing all metrics

## Reproducibility

All experiments are fully reproducible:

- **Deterministic seeding**: Random seed = 42 for all stochastic operations
- **API logging**: Every API call logged with timestamp, parameters, and response
- **Manifest generation**: Each run produces a manifest with exact versions and configurations

Manifests are saved to `results/manifests/` and include:
- Python version and package versions
- Model configurations and API versions
- Dataset checksums
- Random seeds used

## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Property-based tests (Hypothesis)
pytest tests/property/

# Integration tests
pytest tests/integration/
```

## Citation

If you use this code for a publication, please cite:

```bibtex
@software{llm_reasoning_eval,
  title = {LLM Reasoning Evaluation: A Research Framework for Multi-Step Reasoning Analysis},
  year = {2025},
  url = {<https://github.com/Emart29/llm-reasoning-eval.git>}
}
```

## License

MIT License - see LICENSE file for details.
