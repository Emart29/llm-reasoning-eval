"""Research Report Generator module.

Produces publication-quality visualizations and research summaries.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.metrics import ReasoningMetrics


class ResearchReportGenerator:
    """Generates research reports and visualizations.
    
    This class produces publication-quality visualizations and markdown
    research summaries for LLM reasoning evaluation results.
    """
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """Initialize the report generator.
        
        Args:
            style: Matplotlib style to use for visualizations
        """
        self.style = style
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
        })
    
    def generate_accuracy_by_category(
        self, 
        results: pd.DataFrame
    ) -> plt.Figure:
        """Generate bar chart of accuracy by problem category.
        
        Args:
            results: DataFrame with columns 'category', 'is_correct', and optionally 'model'
            
        Returns:
            Matplotlib Figure with bar chart
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate accuracy by category
        if 'model' in results.columns and results['model'].nunique() > 1:
            # Multiple models - grouped bar chart
            accuracy_data = results.groupby(['category', 'model'])['is_correct'].mean().unstack()
            accuracy_data.plot(kind='bar', ax=ax, width=0.8)
            ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            # Single model or no model column - simple bar chart
            accuracy_data = results.groupby('category')['is_correct'].mean()
            colors = sns.color_palette("husl", len(accuracy_data))
            bars = ax.bar(accuracy_data.index, accuracy_data.values, color=colors)
            
            # Add value labels on bars
            for bar, val in zip(bars, accuracy_data.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Problem Category')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Problem Category')
        ax.set_ylim(0, 1.1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def generate_accuracy_by_difficulty(
        self, 
        results: pd.DataFrame
    ) -> plt.Figure:
        """Generate line chart of accuracy by difficulty level.
        
        Args:
            results: DataFrame with columns 'difficulty', 'is_correct', and optionally 'model'
            
        Returns:
            Matplotlib Figure with line chart
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'model' in results.columns and results['model'].nunique() > 1:
            # Multiple models - multiple lines
            for model in results['model'].unique():
                model_data = results[results['model'] == model]
                accuracy_by_diff = model_data.groupby('difficulty')['is_correct'].mean()
                ax.plot(accuracy_by_diff.index, accuracy_by_diff.values, 
                       marker='o', linewidth=2, markersize=8, label=model)
            ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            # Single model
            accuracy_by_diff = results.groupby('difficulty')['is_correct'].mean()
            ax.plot(accuracy_by_diff.index, accuracy_by_diff.values,
                   marker='o', linewidth=2, markersize=8, color='steelblue')
            
            # Add value labels
            for x, y in zip(accuracy_by_diff.index, accuracy_by_diff.values):
                ax.annotate(f'{y:.1%}', (x, y), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=10)
        
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Difficulty Level')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(1, 6))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    
    def generate_error_heatmap(
        self, 
        results: pd.DataFrame
    ) -> plt.Figure:
        """Generate heatmap of error types across models.
        
        Args:
            results: DataFrame with columns 'model', 'error_categories' (list or string)
            
        Returns:
            Matplotlib Figure with heatmap
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Process error categories
        error_counts = self._extract_error_counts(results)
        
        if error_counts.empty:
            ax.text(0.5, 0.5, 'No error data available', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Error Type Distribution')
            return fig
        
        # Create heatmap
        sns.heatmap(error_counts, annot=True, fmt='.0f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': 'Error Count'})
        
        ax.set_xlabel('Error Category')
        ax.set_ylabel('Model')
        ax.set_title('Error Type Distribution Across Models')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def _extract_error_counts(self, results: pd.DataFrame) -> pd.DataFrame:
        """Extract error counts by model and category from results.
        
        Args:
            results: DataFrame with error information
            
        Returns:
            DataFrame with models as rows and error categories as columns
        """
        error_data = []
        
        for _, row in results.iterrows():
            model = row.get('model', 'unknown')
            
            # Handle different formats of error_categories
            categories = row.get('error_categories', [])
            if isinstance(categories, str):
                # Parse string representation
                if categories.startswith('['):
                    try:
                        import ast
                        categories = ast.literal_eval(categories)
                    except (ValueError, SyntaxError):
                        categories = [categories] if categories else []
                else:
                    categories = [categories] if categories else []
            elif categories is None:
                categories = []
            
            for cat in categories:
                error_data.append({'model': model, 'category': cat})
        
        if not error_data:
            return pd.DataFrame()
        
        error_df = pd.DataFrame(error_data)
        return error_df.groupby(['model', 'category']).size().unstack(fill_value=0)
    
    def generate_step_accuracy_curves(
        self, 
        results: pd.DataFrame
    ) -> plt.Figure:
        """Generate step accuracy curves showing failure points.
        
        Args:
            results: DataFrame with 'step_scores' column containing lists of
                    step correctness values, and optionally 'model' column
            
        Returns:
            Matplotlib Figure with step accuracy curves
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract step accuracy data
        step_data = self._extract_step_accuracy_data(results)
        
        if not step_data:
            ax.text(0.5, 0.5, 'No step accuracy data available',
                   ha='center', va='center', fontsize=14)
            ax.set_title('Step Accuracy Curves')
            return fig
        
        # Plot curves for each model
        colors = sns.color_palette("husl", len(step_data))
        for (model, accuracies), color in zip(step_data.items(), colors):
            steps = range(1, len(accuracies) + 1)
            ax.plot(steps, accuracies, marker='o', linewidth=2, 
                   markersize=6, label=model, color=color)
            
            # Add shaded area under curve
            ax.fill_between(steps, accuracies, alpha=0.1, color=color)
        
        ax.set_xlabel('Reasoning Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('Step-by-Step Accuracy (Where Reasoning Fails)')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        if len(step_data) > 1:
            ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def _extract_step_accuracy_data(
        self, 
        results: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Extract step accuracy data from results.
        
        Args:
            results: DataFrame with step_scores column
            
        Returns:
            Dictionary mapping model names to lists of step accuracies
        """
        step_data: Dict[str, List[List[bool]]] = {}
        
        for _, row in results.iterrows():
            model = row.get('model', 'default')
            step_scores = row.get('step_scores', [])
            
            if not step_scores:
                continue
            
            # Handle different formats
            if isinstance(step_scores, str):
                try:
                    import ast
                    step_scores = ast.literal_eval(step_scores)
                except (ValueError, SyntaxError):
                    continue
            
            # Extract correctness values
            correctness = []
            for score in step_scores:
                if isinstance(score, dict):
                    correctness.append(score.get('is_correct', False))
                elif hasattr(score, 'is_correct'):
                    correctness.append(score.is_correct)
                elif isinstance(score, bool):
                    correctness.append(score)
            
            if correctness:
                if model not in step_data:
                    step_data[model] = []
                step_data[model].append(correctness)
        
        # Compute average accuracy at each step position
        result = {}
        for model, all_scores in step_data.items():
            if not all_scores:
                continue
            
            # Find max length
            max_len = max(len(s) for s in all_scores)
            
            # Compute accuracy at each position
            accuracies = []
            for i in range(max_len):
                correct_count = sum(1 for s in all_scores if len(s) > i and s[i])
                total_count = sum(1 for s in all_scores if len(s) > i)
                if total_count > 0:
                    accuracies.append(correct_count / total_count)
            
            if accuracies:
                result[model] = accuracies
        
        return result

    
    def generate_model_comparison_radar(
        self, 
        metrics: Dict[str, ReasoningMetrics]
    ) -> plt.Figure:
        """Generate radar chart comparing models across metrics.
        
        Args:
            metrics: Dictionary mapping model names to ReasoningMetrics objects
            
        Returns:
            Matplotlib Figure with radar chart
        """
        if not metrics:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, 'No metrics data available',
                   ha='center', va='center', fontsize=14)
            ax.set_title('Model Comparison')
            return fig
        
        # Define metrics to compare
        metric_names = [
            'Accuracy',
            'Reasoning Depth',
            'Recovery Rate',
            'Consistency',
            'Step Efficiency',
            'Error Resistance'  # 1 - error_propagation_rate
        ]
        
        num_metrics = len(metric_names)
        
        # Compute angles for radar chart
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = sns.color_palette("husl", len(metrics))
        
        for (model, m), color in zip(metrics.items(), colors):
            # Normalize reasoning_depth to 0-1 scale (assume max 10 steps)
            normalized_depth = min(m.reasoning_depth / 10.0, 1.0)
            
            values = [
                m.accuracy,
                normalized_depth,
                m.recovery_rate,
                m.consistency_score,
                m.step_efficiency,
                1.0 - m.error_propagation_rate  # Error resistance
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
        ax.set_title('Model Comparison Across Metrics', size=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def generate_markdown_report(
        self, 
        results: pd.DataFrame, 
        metrics: Dict[str, ReasoningMetrics]
    ) -> str:
        """Generate markdown research report.
        
        Args:
            results: DataFrame with evaluation results
            metrics: Dictionary mapping model names to ReasoningMetrics
            
        Returns:
            Markdown string with complete research report
        """
        report_parts = []
        
        # Title and metadata
        report_parts.append("# LLM Reasoning Evaluation Research Report\n")
        report_parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Abstract
        report_parts.append("\n## Abstract\n")
        report_parts.append(self.generate_abstract(metrics))
        
        # Methodology
        report_parts.append("\n## Methodology\n")
        report_parts.append(self._generate_methodology_section(results))
        
        # Results
        report_parts.append("\n## Results\n")
        report_parts.append(self._generate_results_section(results, metrics))
        
        # Key Findings
        report_parts.append("\n## Key Findings\n")
        report_parts.append(self._generate_findings_section(results, metrics))
        
        # Statistical Analysis
        report_parts.append("\n## Statistical Analysis\n")
        report_parts.append(self._generate_statistical_section(results))
        
        # Conclusion
        report_parts.append("\n## Conclusion\n")
        report_parts.append(self._generate_conclusion_section(metrics))
        
        return "\n".join(report_parts)
    
    def _generate_methodology_section(self, results: pd.DataFrame) -> str:
        """Generate methodology section of the report."""
        lines = []
        
        # Dataset description
        total_problems = len(results)
        lines.append("### Dataset\n")
        lines.append(f"This evaluation used a dataset of **{total_problems}** reasoning problems ")
        
        if 'category' in results.columns:
            categories = results['category'].unique()
            lines.append(f"across **{len(categories)}** categories: {', '.join(categories)}.\n")
        else:
            lines.append(".\n")
        
        if 'difficulty' in results.columns:
            diff_range = (results['difficulty'].min(), results['difficulty'].max())
            lines.append(f"Problems span difficulty levels {diff_range[0]} to {diff_range[1]}.\n")
        
        # Models evaluated
        lines.append("\n### Models Evaluated\n")
        if 'model' in results.columns:
            models = results['model'].unique()
            for model in models:
                count = len(results[results['model'] == model])
                lines.append(f"- **{model}**: {count} evaluations\n")
        else:
            lines.append("- Single model evaluation\n")
        
        # Metrics description
        lines.append("\n### Metrics\n")
        lines.append("The following metrics were computed:\n")
        lines.append("- **Accuracy**: Fraction of problems with correct final answers\n")
        lines.append("- **Reasoning Depth**: Average number of correct steps before first error\n")
        lines.append("- **Recovery Rate**: Fraction of errors that models self-correct from\n")
        lines.append("- **Consistency Score**: Agreement across multiple samples (0-1)\n")
        lines.append("- **Step Efficiency**: Ratio of optimal to actual reasoning steps\n")
        lines.append("- **Error Propagation Rate**: Fraction of errors that cascade to subsequent steps\n")
        
        return "".join(lines)

    
    def _generate_results_section(
        self, 
        results: pd.DataFrame, 
        metrics: Dict[str, ReasoningMetrics]
    ) -> str:
        """Generate results section of the report."""
        lines = []
        
        # Overall accuracy
        lines.append("### Overall Performance\n")
        
        if metrics:
            lines.append("| Model | Accuracy | Reasoning Depth | Recovery Rate | Consistency | Step Efficiency |\n")
            lines.append("|-------|----------|-----------------|---------------|-------------|----------------|\n")
            
            for model, m in metrics.items():
                lines.append(
                    f"| {model} | {m.accuracy:.1%} | {m.reasoning_depth:.2f} | "
                    f"{m.recovery_rate:.1%} | {m.consistency_score:.1%} | {m.step_efficiency:.1%} |\n"
                )
        
        # Accuracy by category
        if 'category' in results.columns:
            lines.append("\n### Accuracy by Category\n")
            acc_by_cat = results.groupby('category')['is_correct'].mean()
            lines.append("| Category | Accuracy |\n")
            lines.append("|----------|----------|\n")
            for cat, acc in acc_by_cat.items():
                lines.append(f"| {cat} | {acc:.1%} |\n")
        
        # Accuracy by difficulty
        if 'difficulty' in results.columns:
            lines.append("\n### Accuracy by Difficulty\n")
            acc_by_diff = results.groupby('difficulty')['is_correct'].mean()
            lines.append("| Difficulty | Accuracy |\n")
            lines.append("|------------|----------|\n")
            for diff, acc in acc_by_diff.items():
                lines.append(f"| {diff} | {acc:.1%} |\n")
        
        return "".join(lines)
    
    def _generate_findings_section(
        self, 
        results: pd.DataFrame, 
        metrics: Dict[str, ReasoningMetrics]
    ) -> str:
        """Generate key findings section of the report."""
        lines = []
        findings = []
        
        # Find best and worst performing categories
        if 'category' in results.columns:
            acc_by_cat = results.groupby('category')['is_correct'].mean()
            best_cat = acc_by_cat.idxmax()
            worst_cat = acc_by_cat.idxmin()
            findings.append(
                f"**Category Performance**: Models performed best on *{best_cat}* problems "
                f"({acc_by_cat[best_cat]:.1%}) and worst on *{worst_cat}* problems "
                f"({acc_by_cat[worst_cat]:.1%})."
            )
        
        # Difficulty trend
        if 'difficulty' in results.columns:
            acc_by_diff = results.groupby('difficulty')['is_correct'].mean()
            if len(acc_by_diff) > 1:
                # Check if accuracy decreases with difficulty
                correlation = acc_by_diff.index.to_series().corr(acc_by_diff)
                if correlation < -0.3:
                    findings.append(
                        f"**Difficulty Scaling**: Accuracy decreases with difficulty "
                        f"(correlation: {correlation:.2f}), indicating models struggle "
                        f"with more complex reasoning."
                    )
                elif correlation > 0.3:
                    findings.append(
                        f"**Unexpected Pattern**: Accuracy increases with difficulty "
                        f"(correlation: {correlation:.2f}), suggesting difficulty labels "
                        f"may not align with model capabilities."
                    )
        
        # Model comparison findings
        if len(metrics) > 1:
            best_model = max(metrics.items(), key=lambda x: x[1].accuracy)
            worst_model = min(metrics.items(), key=lambda x: x[1].accuracy)
            
            if best_model[0] != worst_model[0]:
                findings.append(
                    f"**Model Comparison**: *{best_model[0]}* achieved the highest accuracy "
                    f"({best_model[1].accuracy:.1%}), while *{worst_model[0]}* had the lowest "
                    f"({worst_model[1].accuracy:.1%})."
                )
            
            # Recovery rate comparison
            best_recovery = max(metrics.items(), key=lambda x: x[1].recovery_rate)
            if best_recovery[1].recovery_rate > 0:
                findings.append(
                    f"**Error Recovery**: *{best_recovery[0]}* showed the best error recovery "
                    f"capability ({best_recovery[1].recovery_rate:.1%})."
                )
        
        # Single model findings
        if len(metrics) == 1:
            model, m = list(metrics.items())[0]
            findings.append(
                f"**Overall Performance**: {model} achieved {m.accuracy:.1%} accuracy "
                f"with an average reasoning depth of {m.reasoning_depth:.2f} steps."
            )
            
            if m.error_propagation_rate > 0.5:
                findings.append(
                    f"**Error Propagation**: High error propagation rate "
                    f"({m.error_propagation_rate:.1%}) indicates errors tend to cascade "
                    f"through reasoning chains."
                )
        
        # Format findings
        for i, finding in enumerate(findings, 1):
            lines.append(f"{i}. {finding}\n\n")
        
        if not findings:
            lines.append("No significant findings to report.\n")
        
        return "".join(lines)
    
    def _generate_statistical_section(self, results: pd.DataFrame) -> str:
        """Generate statistical analysis section with effect sizes and CIs."""
        lines = []
        
        # Check if we have enough data for statistical tests
        if 'model' not in results.columns or results['model'].nunique() < 2:
            lines.append("Statistical comparison requires multiple models.\n")
            return "".join(lines)
        
        models = results['model'].unique()
        
        # Pairwise comparisons
        lines.append("### Pairwise Model Comparisons\n")
        lines.append("| Comparison | Accuracy Diff | Effect Size (Cohen's h) | 95% CI | p-value |\n")
        lines.append("|------------|---------------|-------------------------|--------|--------|\n")
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                comparison = self._compute_pairwise_comparison(
                    results, model1, model2
                )
                lines.append(
                    f"| {model1} vs {model2} | {comparison['diff']:.1%} | "
                    f"{comparison['effect_size']:.3f} | "
                    f"[{comparison['ci_low']:.1%}, {comparison['ci_high']:.1%}] | "
                    f"{comparison['p_value']:.4f} |\n"
                )
        
        lines.append("\n*Effect size interpretation: |h| < 0.2 = small, 0.2-0.8 = medium, > 0.8 = large*\n")
        
        return "".join(lines)

    
    def _compute_pairwise_comparison(
        self, 
        results: pd.DataFrame, 
        model1: str, 
        model2: str
    ) -> Dict[str, float]:
        """Compute pairwise statistical comparison between two models.
        
        Args:
            results: DataFrame with evaluation results
            model1: First model name
            model2: Second model name
            
        Returns:
            Dictionary with diff, effect_size, ci_low, ci_high, p_value
        """
        data1 = results[results['model'] == model1]['is_correct']
        data2 = results[results['model'] == model2]['is_correct']
        
        n1, n2 = len(data1), len(data2)
        p1, p2 = data1.mean(), data2.mean()
        
        # Accuracy difference
        diff = p1 - p2
        
        # Cohen's h effect size for proportions
        h1 = 2 * math.asin(math.sqrt(p1)) if p1 > 0 else 0
        h2 = 2 * math.asin(math.sqrt(p2)) if p2 > 0 else 0
        effect_size = h1 - h2
        
        # Confidence interval for difference in proportions
        se = math.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2)) if n1 > 0 and n2 > 0 else 0
        z = 1.96  # 95% CI
        ci_low = diff - z * se
        ci_high = diff + z * se
        
        # Chi-square test for independence
        try:
            contingency = pd.crosstab(
                results[results['model'].isin([model1, model2])]['model'],
                results[results['model'].isin([model1, model2])]['is_correct']
            )
            if contingency.shape == (2, 2):
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            else:
                p_value = 1.0
        except Exception:
            p_value = 1.0
        
        return {
            'diff': diff,
            'effect_size': effect_size,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p_value
        }
    
    def _generate_conclusion_section(
        self, 
        metrics: Dict[str, ReasoningMetrics]
    ) -> str:
        """Generate conclusion section of the report."""
        lines = []
        
        if not metrics:
            lines.append("No metrics available for conclusion.\n")
            return "".join(lines)
        
        # Overall summary
        avg_accuracy = sum(m.accuracy for m in metrics.values()) / len(metrics)
        avg_depth = sum(m.reasoning_depth for m in metrics.values()) / len(metrics)
        avg_recovery = sum(m.recovery_rate for m in metrics.values()) / len(metrics)
        
        lines.append(
            f"This evaluation assessed {len(metrics)} model(s) on multi-step reasoning tasks. "
            f"The average accuracy across models was **{avg_accuracy:.1%}**, with an average "
            f"reasoning depth of **{avg_depth:.2f}** steps before encountering errors.\n\n"
        )
        
        if avg_recovery > 0.1:
            lines.append(
                f"Models demonstrated some ability to recover from errors, with an average "
                f"recovery rate of **{avg_recovery:.1%}**. "
            )
        else:
            lines.append(
                "Error recovery was limited, suggesting that once models make mistakes, "
                "they rarely self-correct. "
            )
        
        # Recommendations
        lines.append("\n### Recommendations\n")
        lines.append("Based on these findings:\n")
        lines.append("1. Focus on improving early-step accuracy to prevent error cascades\n")
        lines.append("2. Investigate training approaches that enhance error recovery\n")
        lines.append("3. Consider difficulty-aware prompting strategies\n")
        
        return "".join(lines)
    
    def generate_abstract(
        self, 
        metrics: Dict[str, ReasoningMetrics]
    ) -> str:
        """Generate paper-ready abstract.
        
        Args:
            metrics: Dictionary mapping model names to ReasoningMetrics
            
        Returns:
            Abstract string suitable for research paper
        """
        if not metrics:
            return "No evaluation data available for abstract generation.\n"
        
        # Compute aggregate statistics
        num_models = len(metrics)
        accuracies = [m.accuracy for m in metrics.values()]
        avg_accuracy = sum(accuracies) / len(accuracies)
        max_accuracy = max(accuracies)
        min_accuracy = min(accuracies)
        
        depths = [m.reasoning_depth for m in metrics.values()]
        avg_depth = sum(depths) / len(depths)
        
        recovery_rates = [m.recovery_rate for m in metrics.values()]
        avg_recovery = sum(recovery_rates) / len(recovery_rates)
        
        propagation_rates = [m.error_propagation_rate for m in metrics.values()]
        avg_propagation = sum(propagation_rates) / len(propagation_rates)
        
        # Generate abstract
        abstract_parts = []
        
        abstract_parts.append(
            "We present a comprehensive evaluation of large language model (LLM) "
            "reasoning capabilities using a novel multi-step reasoning benchmark. "
        )
        
        if num_models > 1:
            abstract_parts.append(
                f"Our analysis covers {num_models} models, revealing significant "
                f"variation in reasoning performance (accuracy range: {min_accuracy:.1%} "
                f"to {max_accuracy:.1%}). "
            )
        else:
            model_name = list(metrics.keys())[0]
            abstract_parts.append(
                f"Our analysis of {model_name} reveals an overall accuracy of "
                f"{avg_accuracy:.1%} on multi-step reasoning tasks. "
            )
        
        abstract_parts.append(
            f"We introduce novel metrics including reasoning depth (average: "
            f"{avg_depth:.2f} steps), error recovery rate ({avg_recovery:.1%}), "
            f"and error propagation rate ({avg_propagation:.1%}). "
        )
        
        # Key finding
        if avg_propagation > 0.5:
            abstract_parts.append(
                "Our findings indicate that errors in early reasoning steps "
                "frequently cascade through subsequent steps, highlighting the "
                "importance of accurate initial reasoning. "
            )
        elif avg_recovery > 0.2:
            abstract_parts.append(
                "Notably, models demonstrate meaningful error recovery capabilities, "
                "suggesting potential for self-correction in reasoning chains. "
            )
        else:
            abstract_parts.append(
                "Results suggest that current models struggle with complex "
                "multi-step reasoning, with limited ability to recover from errors. "
            )
        
        abstract_parts.append(
            "These findings provide insights for improving LLM reasoning capabilities "
            "and inform future research directions in this area."
        )
        
        return "".join(abstract_parts) + "\n"
    
    def save_figure(
        self, 
        fig: plt.Figure, 
        path: str, 
        dpi: int = 300
    ) -> None:
        """Save a figure to file.
        
        Args:
            fig: Matplotlib figure to save
            path: Output file path
            dpi: Resolution in dots per inch
        """
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    def save_report(
        self, 
        report: str, 
        path: str
    ) -> None:
        """Save markdown report to file.
        
        Args:
            report: Markdown report string
            path: Output file path
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
