# src/api_logger.py
"""Comprehensive API logging for inference engine.

Logs all API calls with timestamps, parameters, and responses to support
reproducibility and debugging of LLM reasoning evaluation experiments.

Requirements: 8.1 - Log all API calls with timestamps, parameters, responses
"""
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field

from .config import REPO_ROOT


# ------------------- Configuration -------------------

# Default log directory
DEFAULT_LOG_DIR = REPO_ROOT / "results" / "logs"

# Log file naming pattern
LOG_FILE_PREFIX = "api_calls"


# ------------------- Data Classes -------------------

@dataclass
class APICallLog:
    """Represents a single API call log entry."""
    call_id: str
    timestamp: str
    provider: str
    model: str
    model_key: str
    strategy: str
    messages: List[Dict[str, str]]
    parameters: Dict[str, Any]
    response: Optional[str] = None
    error: Optional[str] = None
    latency_ms: int = 0
    retry_count: int = 0
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


# ------------------- API Logger Class -------------------

class APILogger:
    """Comprehensive logger for API calls.
    
    Logs all API calls with:
    - Timestamps (ISO format)
    - Request parameters (model, messages, config)
    - Response content
    - Latency measurements
    - Error information
    - Retry counts
    """
    
    def __init__(
        self, 
        log_dir: Optional[Path] = None,
        experiment_id: Optional[str] = None,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
    ):
        """Initialize the API logger.
        
        Args:
            log_dir: Directory for log files. Defaults to results/logs/
            experiment_id: Unique identifier for this experiment run
            enable_file_logging: Whether to write logs to files
            enable_console_logging: Whether to log to console
        """
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        
        # In-memory log storage for current session
        self._call_logs: List[APICallLog] = []
        self._call_count = 0
        
        # Set up logging
        self._setup_logging()
        
        # Ensure log directory exists
        if self.enable_file_logging:
            self._ensure_log_dir()
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"exp_{timestamp}_{short_uuid}"
    
    def _setup_logging(self) -> None:
        """Configure Python logging for API calls."""
        self.logger = logging.getLogger(f"api_logger.{self.experiment_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
    
    def _ensure_log_dir(self) -> None:
        """Create log directory if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Log directory ensured: {self.log_dir}")
    
    def _get_log_file_path(self) -> Path:
        """Get the path for the current log file."""
        filename = f"{LOG_FILE_PREFIX}_{self.experiment_id}.jsonl"
        return self.log_dir / filename
    
    def _write_log_entry(self, log_entry: APICallLog) -> None:
        """Write a log entry to the JSONL file."""
        if not self.enable_file_logging:
            return
        
        log_file = self._get_log_file_path()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry.to_dict(), default=str) + "\n")

    def log_call_start(
        self,
        provider: str,
        model: str,
        model_key: str,
        strategy: str,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
    ) -> str:
        """Log the start of an API call.
        
        Args:
            provider: API provider (openai, anthropic, google, hf)
            model: Model name
            model_key: Model configuration key
            strategy: Prompting strategy used
            messages: List of message dicts with role and content
            parameters: Model parameters (max_tokens, temperature, etc.)
            
        Returns:
            call_id: Unique identifier for this call
        """
        self._call_count += 1
        call_id = f"{self.experiment_id}_call_{self._call_count:06d}"
        timestamp = datetime.now().isoformat()
        
        log_entry = APICallLog(
            call_id=call_id,
            timestamp=timestamp,
            provider=provider,
            model=model,
            model_key=model_key,
            strategy=strategy,
            messages=messages,
            parameters=parameters,
        )
        
        self._call_logs.append(log_entry)
        
        self.logger.info(
            f"API call started: {call_id} | {provider}/{model} | strategy={strategy}"
        )
        
        return call_id
    
    def log_call_success(
        self,
        call_id: str,
        response: str,
        latency_ms: int,
        retry_count: int = 0,
    ) -> None:
        """Log a successful API call completion.
        
        Args:
            call_id: The call ID returned from log_call_start
            response: The model's response text
            latency_ms: Time taken for the call in milliseconds
            retry_count: Number of retries before success
        """
        # Find and update the log entry
        for log_entry in self._call_logs:
            if log_entry.call_id == call_id:
                log_entry.response = response
                log_entry.latency_ms = latency_ms
                log_entry.retry_count = retry_count
                log_entry.success = True
                
                # Write to file
                self._write_log_entry(log_entry)
                
                self.logger.info(
                    f"API call success: {call_id} | latency={latency_ms}ms | "
                    f"retries={retry_count} | response_len={len(response)}"
                )
                return
        
        self.logger.warning(f"Call ID not found for success logging: {call_id}")
    
    def log_call_failure(
        self,
        call_id: str,
        error: str,
        latency_ms: int,
        retry_count: int = 0,
    ) -> None:
        """Log a failed API call.
        
        Args:
            call_id: The call ID returned from log_call_start
            error: Error message or exception string
            latency_ms: Time taken before failure in milliseconds
            retry_count: Number of retries attempted
        """
        # Find and update the log entry
        for log_entry in self._call_logs:
            if log_entry.call_id == call_id:
                log_entry.error = error
                log_entry.latency_ms = latency_ms
                log_entry.retry_count = retry_count
                log_entry.success = False
                
                # Write to file
                self._write_log_entry(log_entry)
                
                self.logger.error(
                    f"API call failed: {call_id} | latency={latency_ms}ms | "
                    f"retries={retry_count} | error={error[:100]}"
                )
                return
        
        self.logger.warning(f"Call ID not found for failure logging: {call_id}")
    
    def get_call_logs(self) -> List[APICallLog]:
        """Get all logged API calls for this session."""
        return self._call_logs.copy()
    
    def get_call_count(self) -> int:
        """Get the total number of API calls made."""
        return self._call_count
    
    def get_success_rate(self) -> float:
        """Calculate the success rate of API calls."""
        if not self._call_logs:
            return 0.0
        successful = sum(1 for log in self._call_logs if log.success)
        return successful / len(self._call_logs)
    
    def get_average_latency(self) -> float:
        """Calculate the average latency of successful API calls."""
        successful_logs = [log for log in self._call_logs if log.success]
        if not successful_logs:
            return 0.0
        return sum(log.latency_ms for log in successful_logs) / len(successful_logs)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all API calls in this session."""
        return {
            "experiment_id": self.experiment_id,
            "total_calls": self._call_count,
            "successful_calls": sum(1 for log in self._call_logs if log.success),
            "failed_calls": sum(1 for log in self._call_logs if not log.success),
            "success_rate": self.get_success_rate(),
            "average_latency_ms": self.get_average_latency(),
            "total_retries": sum(log.retry_count for log in self._call_logs),
            "log_file": str(self._get_log_file_path()) if self.enable_file_logging else None,
        }
    
    def save_summary(self, path: Optional[Path] = None) -> Path:
        """Save the session summary to a JSON file.
        
        Args:
            path: Optional path for the summary file
            
        Returns:
            Path to the saved summary file
        """
        if path is None:
            path = self.log_dir / f"summary_{self.experiment_id}.json"
        
        summary = self.get_summary()
        summary["generated_at"] = datetime.now().isoformat()
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Summary saved to: {path}")
        return path


# ------------------- Global Logger Instance -------------------

# Global logger instance (can be replaced per experiment)
_global_logger: Optional[APILogger] = None


def get_api_logger() -> APILogger:
    """Get or create the global API logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = APILogger()
    return _global_logger


def set_api_logger(logger: APILogger) -> None:
    """Set the global API logger instance."""
    global _global_logger
    _global_logger = logger


def reset_api_logger() -> None:
    """Reset the global API logger (creates a new instance on next get)."""
    global _global_logger
    _global_logger = None
