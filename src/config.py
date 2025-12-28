# src/config.py
"""Global configuration for the LLM reasoning evaluation project.
All paths are absolute (via pathlib) so the code works regardless of cwd.
"""
import os
from pathlib import Path

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# ------------------- General -------------------
RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

# ------------------- Data -------------------
# Root of the repository (two levels up from this file)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "reasoning_dataset.jsonl"

# ------------------- Models -------------------
# API keys loaded from .env file or environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------- Retry Configuration -------------------
RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 1.0,  # seconds
    "max_delay": 60.0,  # seconds
    "exponential_base": 2,
}

MODELS = {
    # OpenAI GPT models
    "gpt4o": {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "gpt4": {
        "provider": "openai",
        "model_name": "gpt-4",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "gpt35": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    # Anthropic Claude models
    "claude3_opus": {
        "provider": "anthropic",
        "model_name": "claude-3-opus-20240229",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "claude3_sonnet": {
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "claude3_haiku": {
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    # Google Gemini models
    "gemini_pro": {
        "provider": "google",
        "model_name": "gemini-pro",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "gemini_15_pro": {
        "provider": "google",
        "model_name": "gemini-1.5-pro",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "gemini_15_flash": {
        "provider": "google",
        "model_name": "gemini-1.5-flash",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    # HuggingFace open-source models
    "llama2_70b": {
        "provider": "hf",
        "model_name": "meta-llama/Llama-2-70b-chat-hf",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "mistral_7b": {
        "provider": "hf",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    # Groq - Fast inference for open-source models
    "groq_llama3_70b": {
        "provider": "groq",
        "model_name": "llama-3.3-70b-versatile",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "groq_llama3_8b": {
        "provider": "groq",
        "model_name": "llama-3.1-8b-instant",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "groq_qwen3_32b": {
        "provider": "groq",
        "model_name": "qwen/qwen3-32b",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "groq_gpt_oss_120b": {
        "provider": "groq",
        "model_name": "openai/gpt-oss-120b",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "groq_gpt_oss_20b": {
        "provider": "groq",
        "model_name": "openai/gpt-oss-20b",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
}

# ------------------- Prompting Strategies -------------------
STRATEGIES = {
    "zero_shot": {
        "system_prompt": "You are a helpful assistant. Answer the question directly.",
        "user_template": "{prompt}",
    },
    "cot": {
        "system_prompt": "You are a helpful assistant. Solve the problem step‑by‑step and give the final answer.",
        "user_template": "{prompt}",
    },
    "self_consistency": {
        "system_prompt": "You are a helpful assistant. Solve the problem step‑by‑step. Provide **multiple** independent solutions and then give the most common final answer.",
        "user_template": "{prompt}",
        "num_samples": 5,
    },
}
