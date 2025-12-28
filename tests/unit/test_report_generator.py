"""Unit tests for ResearchReportGenerator.

Tests figure generation produces valid files and markdown contains required sections.
Requirements: 6.1, 6.2, 6.3, 6.4
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from src.report_generator import ResearchReportGenerator
from src.metrics import ReasoningMetrics


@pytest.fixture
def generator():
    """Create a ResearchReportGenerator instance."""
    return ResearchReportGenerator()


@pytest.fixture
def sample_results():
    """Create sample evaluation results DataFrame."""
    return pd.DataFrame({
        'problem_id': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
        'model': ['gpt-4', 'gpt-4', 'gpt-4', 'claude', 'claude', 'claude'],
        'category': ['math', 'logic', 'math', 'logic', 'causal', 'math'],
        'difficulty': [1, 2, 3, 2, 4, 5],
        'is_correct': [True, True, False, True, False, True],
        'error_categories': [[], [], ['arithmetic'], [], ['logical_fallacy'], []],
        'step_scores': [
            [{'is_correct': True}, {'is_correct': True}],
            [{'is_correct': True}, {'is_correct': True}, {'is_correct': True}],
            [{'is_correct': True}, {'is_correct': False}],
            [{'is_correct': True}, {'is_correct': True}],
            [{'is_correct': False}],
            [{'is_correct': True}, {'is_correct': True}, {'is_correct': True}],
        ]
    })


@pytest.fixture
def sample_metrics():
    """Create sample metrics dictionary."""
    return {
        'gpt-4': ReasoningMetrics(
            accuracy=0.67,
            reasoning_depth=2.5,
            recovery_rate=0.1,
            consistency_score=0.9,
            step_efficiency=0.85,
            error_propagation_rate=0.3
        ),
        'claude': ReasoningMetrics(
            accuracy=0.67,
            reasoning_depth=2.0,
            recovery_rate=0.15,
            consistency_score=0.85,
            step_efficiency=0.9,
            error_propagation_rate=0.25
        )
    }


@pytest.fixture
def single_model_results():
    """Create results for a single model."""
    return pd.DataFrame({
        'problem_id': ['P1', 'P2', 'P3'],
        'category': ['math', 'logic', 'causal'],
        'difficulty': [1, 3, 5],
        'is_correct': [True, True, False],
    })


@pytest.fixture
def single_model_metrics():
    """Create metrics for a single model."""
    return {
        'gpt-4': ReasoningMetrics(
            accuracy=0.67,
            reasoning_depth=2.0,
            recovery_rate=0.1,
            consistency_score=0.9,
            step_efficiency=0.8,
            error_propagation_rate=0.4
        )
    }


class TestAccuracyByCategory:
    """Tests for generate_accuracy_by_category."""
    
    def test_returns_figure(self, generator, sample_results):
        """Test that method returns a matplotlib Figure."""
        fig = generator.generate_accuracy_by_category(sample_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_figure_can_be_saved(self, generator, sample_results):
        """Test that generated figure can be saved to file."""
        fig = generator.generate_accuracy_by_category(sample_results)
        temp_path = tempfile.mktemp(suffix='.png')
        try:
            fig.savefig(temp_path)
            plt.close(fig)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_single_model_results(self, generator, single_model_results):
        """Test with single model data."""
        fig = generator.generate_accuracy_by_category(single_model_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)



class TestAccuracyByDifficulty:
    """Tests for generate_accuracy_by_difficulty."""
    
    def test_returns_figure(self, generator, sample_results):
        """Test that method returns a matplotlib Figure."""
        fig = generator.generate_accuracy_by_difficulty(sample_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_figure_can_be_saved(self, generator, sample_results):
        """Test that generated figure can be saved to file."""
        fig = generator.generate_accuracy_by_difficulty(sample_results)
        temp_path = tempfile.mktemp(suffix='.png')
        try:
            fig.savefig(temp_path)
            plt.close(fig)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestErrorHeatmap:
    """Tests for generate_error_heatmap."""
    
    def test_returns_figure(self, generator, sample_results):
        """Test that method returns a matplotlib Figure."""
        fig = generator.generate_error_heatmap(sample_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_handles_empty_errors(self, generator):
        """Test handling of results with no errors."""
        results = pd.DataFrame({
            'model': ['gpt-4', 'claude'],
            'error_categories': [[], []]
        })
        fig = generator.generate_error_heatmap(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestStepAccuracyCurves:
    """Tests for generate_step_accuracy_curves."""
    
    def test_returns_figure(self, generator, sample_results):
        """Test that method returns a matplotlib Figure."""
        fig = generator.generate_step_accuracy_curves(sample_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_handles_missing_step_scores(self, generator):
        """Test handling of results without step scores."""
        results = pd.DataFrame({
            'model': ['gpt-4'],
            'step_scores': [[]]
        })
        fig = generator.generate_step_accuracy_curves(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestModelComparisonRadar:
    """Tests for generate_model_comparison_radar."""
    
    def test_returns_figure(self, generator, sample_metrics):
        """Test that method returns a matplotlib Figure."""
        fig = generator.generate_model_comparison_radar(sample_metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_handles_empty_metrics(self, generator):
        """Test handling of empty metrics dictionary."""
        fig = generator.generate_model_comparison_radar({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_single_model(self, generator, single_model_metrics):
        """Test with single model metrics."""
        fig = generator.generate_model_comparison_radar(single_model_metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMarkdownReport:
    """Tests for generate_markdown_report."""
    
    def test_returns_string(self, generator, sample_results, sample_metrics):
        """Test that method returns a string."""
        report = generator.generate_markdown_report(sample_results, sample_metrics)
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_contains_title(self, generator, sample_results, sample_metrics):
        """Test that report contains title."""
        report = generator.generate_markdown_report(sample_results, sample_metrics)
        assert "# LLM Reasoning Evaluation Research Report" in report
    
    def test_contains_abstract_section(self, generator, sample_results, sample_metrics):
        """Test that report contains abstract section."""
        report = generator.generate_markdown_report(sample_results, sample_metrics)
        assert "## Abstract" in report
    
    def test_contains_methodology_section(self, generator, sample_results, sample_metrics):
        """Test that report contains methodology section."""
        report = generator.generate_markdown_report(sample_results, sample_metrics)
        assert "## Methodology" in report
    
    def test_contains_results_section(self, generator, sample_results, sample_metrics):
        """Test that report contains results section."""
        report = generator.generate_markdown_report(sample_results, sample_metrics)
        assert "## Results" in report
    
    def test_contains_findings_section(self, generator, sample_results, sample_metrics):
        """Test that report contains key findings section."""
        report = generator.generate_markdown_report(sample_results, sample_metrics)
        assert "## Key Findings" in report
    
    def test_contains_statistical_section(self, generator, sample_results, sample_metrics):
        """Test that report contains statistical analysis section."""
        report = generator.generate_markdown_report(sample_results, sample_metrics)
        assert "## Statistical Analysis" in report
    
    def test_contains_conclusion_section(self, generator, sample_results, sample_metrics):
        """Test that report contains conclusion section."""
        report = generator.generate_markdown_report(sample_results, sample_metrics)
        assert "## Conclusion" in report
    
    def test_contains_metrics_table(self, generator, sample_results, sample_metrics):
        """Test that report contains metrics table with model data."""
        report = generator.generate_markdown_report(sample_results, sample_metrics)
        assert "| Model |" in report
        assert "gpt-4" in report
        assert "claude" in report


class TestAbstract:
    """Tests for generate_abstract."""
    
    def test_returns_string(self, generator, sample_metrics):
        """Test that method returns a string."""
        abstract = generator.generate_abstract(sample_metrics)
        assert isinstance(abstract, str)
        assert len(abstract) > 0
    
    def test_handles_empty_metrics(self, generator):
        """Test handling of empty metrics."""
        abstract = generator.generate_abstract({})
        assert "No evaluation data" in abstract
    
    def test_mentions_accuracy(self, generator, sample_metrics):
        """Test that abstract mentions accuracy."""
        abstract = generator.generate_abstract(sample_metrics)
        assert "accuracy" in abstract.lower()
    
    def test_mentions_reasoning_depth(self, generator, sample_metrics):
        """Test that abstract mentions reasoning depth."""
        abstract = generator.generate_abstract(sample_metrics)
        assert "reasoning depth" in abstract.lower()
    
    def test_single_model_abstract(self, generator, single_model_metrics):
        """Test abstract generation for single model."""
        abstract = generator.generate_abstract(single_model_metrics)
        assert isinstance(abstract, str)
        assert "gpt-4" in abstract


class TestSaveOperations:
    """Tests for save_figure and save_report."""
    
    def test_save_figure(self, generator, sample_results):
        """Test saving figure to file."""
        fig = generator.generate_accuracy_by_category(sample_results)
        temp_path = tempfile.mktemp(suffix='.png')
        try:
            generator.save_figure(fig, temp_path)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_report(self, generator, sample_results, sample_metrics):
        """Test saving markdown report to file."""
        report = generator.generate_markdown_report(sample_results, sample_metrics)
        temp_path = tempfile.mktemp(suffix='.md')
        try:
            generator.save_report(report, temp_path)
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                saved_content = f.read()
            assert saved_content == report
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
