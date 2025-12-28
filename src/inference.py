# src/inference.py
"""Model inference wrappers supporting multiple providers and prompting strategies.
All calls are deterministic (temperature=0) unless the user changes config.

Supports:
- OpenAI GPT models (GPT-4, GPT-4o, GPT-3.5-turbo)
- Anthropic Claude models (Claude 3 Opus, Sonnet, Haiku)
- Google Gemini models (Gemini Pro, Gemini 1.5 Pro/Flash)
- HuggingFace open-source models (Llama 2, Mistral)

Requirements: 8.1 - Log all API calls with timestamps, parameters, responses
"""
import time
import logging
from typing import List, Dict, Any, Optional
from functools import wraps

import openai
import requests

from .config import (
    MODELS, 
    OPENAI_API_KEY, 
    HF_TOKEN, 
    ANTHROPIC_API_KEY, 
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    RETRY_CONFIG,
)
from .normalization import normalize_output
from .api_logger import get_api_logger, APILogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------- Custom Exceptions -------------------

class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class ModelUnavailableError(APIError):
    """Raised when a model endpoint is unavailable."""
    def __init__(self, message: str, model: str):
        super().__init__(message)
        self.model = model


# ------------------- Retry Logic -------------------

def retry_with_exponential_backoff(
    max_retries: int = None,
    initial_delay: float = None,
    max_delay: float = None,
    exponential_base: int = None,
):
    """Decorator for retrying API calls with exponential backoff."""
    max_retries = max_retries or RETRY_CONFIG["max_retries"]
    initial_delay = initial_delay or RETRY_CONFIG["initial_delay"]
    max_delay = max_delay or RETRY_CONFIG["max_delay"]
    exponential_base = exponential_base or RETRY_CONFIG["exponential_base"]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    last_exception = e
                    wait_time = min(e.retry_after, max_delay)
                    logger.warning(
                        f"Rate limit hit. Waiting {wait_time}s before retry "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    time.sleep(wait_time)
                except (APIError, requests.RequestException, Exception) as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    logger.warning(
                        f"API call failed: {e}. Retrying in {delay}s "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
            
            raise last_exception
        return wrapper
    return decorator


# ------------------- Provider-Specific Implementations -------------------

@retry_with_exponential_backoff()
def _openai_chat(messages: List[Dict], model_cfg: dict) -> str:
    """Call OpenAI ChatCompletion and return the assistant's content."""
    if not OPENAI_API_KEY:
        raise APIError("OPENAI_API_KEY environment variable not set")
    
    # Use the new OpenAI SDK (v1.0+)
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        resp = client.chat.completions.create(
            model=model_cfg["model_name"],
            messages=messages,
            max_tokens=model_cfg["max_tokens"],
            temperature=model_cfg["temperature"],
        )
        return resp.choices[0].message.content
    except openai.RateLimitError as e:
        raise RateLimitError(str(e), retry_after=60)
    except openai.APIError as e:
        raise APIError(f"OpenAI API error: {e}")
    except openai.BadRequestError as e:
        raise ModelUnavailableError(str(e), model_cfg["model_name"])


@retry_with_exponential_backoff()
def _anthropic_chat(messages: List[Dict], model_cfg: dict) -> str:
    """Call Anthropic Claude API and return the assistant's content."""
    if not ANTHROPIC_API_KEY:
        raise APIError("ANTHROPIC_API_KEY environment variable not set")
    
    try:
        import anthropic
    except ImportError:
        raise APIError("anthropic package not installed. Run: pip install anthropic")
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Convert messages format: extract system message and user messages
    system_content = ""
    anthropic_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            anthropic_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    try:
        response = client.messages.create(
            model=model_cfg["model_name"],
            max_tokens=model_cfg["max_tokens"],
            system=system_content,
            messages=anthropic_messages,
        )
        return response.content[0].text
    except anthropic.RateLimitError as e:
        raise RateLimitError(str(e), retry_after=60)
    except anthropic.APIError as e:
        raise APIError(f"Anthropic API error: {e}")
    except anthropic.NotFoundError as e:
        raise ModelUnavailableError(str(e), model_cfg["model_name"])


@retry_with_exponential_backoff()
def _google_chat(messages: List[Dict], model_cfg: dict) -> str:
    """Call Google Gemini API and return the model's content."""
    if not GOOGLE_API_KEY:
        raise APIError("GOOGLE_API_KEY environment variable not set")
    
    try:
        import google.generativeai as genai
    except ImportError:
        raise APIError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Build the prompt from messages
    system_content = ""
    user_content = ""
    
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            user_content = msg["content"]
    
    # Combine system and user content for Gemini
    full_prompt = f"{system_content}\n\n{user_content}" if system_content else user_content
    
    try:
        model = genai.GenerativeModel(model_cfg["model_name"])
        
        # Configure generation settings
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=model_cfg["max_tokens"],
            temperature=model_cfg["temperature"],
        )
        
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
        return response.text
    except Exception as e:
        error_str = str(e).lower()
        if "rate" in error_str or "quota" in error_str:
            raise RateLimitError(str(e), retry_after=60)
        elif "not found" in error_str or "unavailable" in error_str:
            raise ModelUnavailableError(str(e), model_cfg["model_name"])
        else:
            raise APIError(f"Google API error: {e}")


@retry_with_exponential_backoff()
def _groq_chat(messages: List[Dict], model_cfg: dict) -> str:
    """Call Groq API (FREE) and return the assistant's content."""
    if not GROQ_API_KEY:
        raise APIError("GROQ_API_KEY environment variable not set. Get free key at https://console.groq.com")
    
    try:
        from groq import Groq
    except ImportError:
        raise APIError("groq package not installed. Run: pip install groq")
    
    client = Groq(api_key=GROQ_API_KEY)
    
    try:
        response = client.chat.completions.create(
            model=model_cfg["model_name"],
            messages=messages,
            max_tokens=model_cfg["max_tokens"],
            temperature=model_cfg["temperature"],
        )
        return response.choices[0].message.content
    except Exception as e:
        error_str = str(e).lower()
        if "rate" in error_str or "limit" in error_str:
            raise RateLimitError(str(e), retry_after=60)
        elif "not found" in error_str:
            raise ModelUnavailableError(str(e), model_cfg["model_name"])
        else:
            raise APIError(f"Groq API error: {e}")


@retry_with_exponential_backoff()
def _hf_chat(prompt: str, model_cfg: dict) -> str:
    """Call HuggingFace Inference API (text-generation endpoint)."""
    if not HF_TOKEN:
        raise APIError("HF_TOKEN environment variable not set")
    
    endpoint = f"https://api-inference.huggingface.co/models/{model_cfg['model_name']}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": model_cfg["max_tokens"],
            "temperature": model_cfg["temperature"] if model_cfg["temperature"] > 0 else 0.01,
        },
    }
    
    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        raise RateLimitError("HuggingFace rate limit exceeded", retry_after=retry_after)
    elif response.status_code == 503:
        raise ModelUnavailableError(
            f"Model {model_cfg['model_name']} is loading or unavailable",
            model_cfg["model_name"]
        )
    
    response.raise_for_status()
    
    # HF returns a list of dicts; the generated text is under 'generated_text'
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "")
    return ""


# ------------------- Main Interface -------------------

def _call_model(
    model_cfg: dict, 
    messages: List[Dict], 
    prompt_text: str,
    model_key: str = "",
    strategy: str = "",
    api_logger: Optional[APILogger] = None,
) -> str:
    """Route the call to the appropriate provider with comprehensive logging.
    
    Args:
        model_cfg: Model configuration dictionary
        messages: List of message dicts with role and content
        prompt_text: Original prompt text
        model_key: Key identifying the model in MODELS config
        strategy: Prompting strategy being used
        api_logger: Optional API logger instance
        
    Returns:
        Model response string
    """
    provider = model_cfg["provider"]
    
    # Get or create logger
    if api_logger is None:
        api_logger = get_api_logger()
    
    # Log call start
    call_id = api_logger.log_call_start(
        provider=provider,
        model=model_cfg["model_name"],
        model_key=model_key,
        strategy=strategy,
        messages=messages,
        parameters={
            "max_tokens": model_cfg["max_tokens"],
            "temperature": model_cfg["temperature"],
        },
    )
    
    start_time = time.time()
    retry_count = 0
    
    try:
        if provider == "openai":
            result = _openai_chat(messages, model_cfg)
        elif provider == "anthropic":
            result = _anthropic_chat(messages, model_cfg)
        elif provider == "google":
            result = _google_chat(messages, model_cfg)
        elif provider == "groq":
            result = _groq_chat(messages, model_cfg)
        elif provider == "hf":
            # HF does not have a system/user separation – concatenate
            system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            user = messages[-1]["content"] if messages else prompt_text
            full_prompt = f"{system}\n\n{user}" if system else user
            result = _hf_chat(full_prompt, model_cfg)
        else:
            raise APIError(f"Unknown provider: {provider}")
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log success
        api_logger.log_call_success(
            call_id=call_id,
            response=result,
            latency_ms=latency_ms,
            retry_count=retry_count,
        )
        
        return result
        
    except Exception as e:
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log failure
        api_logger.log_call_failure(
            call_id=call_id,
            error=str(e),
            latency_ms=latency_ms,
            retry_count=retry_count,
        )
        
        raise


def get_responses(
    model_key: str, 
    strategy_key: str, 
    prompt: str, 
    strategy_cfg: dict,
    normalize: bool = True,
    api_logger: Optional[APILogger] = None,
) -> List[str]:
    """Return a list of model outputs.
    
    * For `self_consistency` we return *num_samples* independent CoT generations.
    * For other strategies we return a single-element list.
    
    Args:
        model_key: Key identifying the model in MODELS config
        strategy_key: Key identifying the prompting strategy
        prompt: The user's prompt/question
        strategy_cfg: Strategy configuration dict with system_prompt and user_template
        normalize: Whether to normalize outputs (default True)
        api_logger: Optional API logger instance for comprehensive logging
        
    Returns:
        List of model output strings
        
    Raises:
        APIError: If API call fails after retries
        RateLimitError: If rate limit is exceeded
        ModelUnavailableError: If model is not available
    """
    if model_key not in MODELS:
        raise APIError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    
    model_cfg = MODELS[model_key]
    system = strategy_cfg["system_prompt"]
    user = strategy_cfg["user_template"].format(prompt=prompt)
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    if strategy_key == "self_consistency":
        n = strategy_cfg.get("num_samples", 5)
        outputs = []
        for i in range(n):
            logger.info(f"Self-consistency sample {i + 1}/{n} for model {model_key}")
            out = _call_model(
                model_cfg, 
                messages, 
                user,
                model_key=model_key,
                strategy=strategy_key,
                api_logger=api_logger,
            )
            if normalize:
                out = normalize_output(out)
            outputs.append(out)
            time.sleep(0.2)  # be nice to the API
        return outputs
    else:
        out = _call_model(
            model_cfg, 
            messages, 
            user,
            model_key=model_key,
            strategy=strategy_key,
            api_logger=api_logger,
        )
        if normalize:
            out = normalize_output(out)
        return [out]


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of available models and their configurations."""
    return MODELS.copy()


def get_models_by_provider(provider: str) -> Dict[str, Dict[str, Any]]:
    """Return models filtered by provider.
    
    Args:
        provider: One of 'openai', 'anthropic', 'google', 'hf'
        
    Returns:
        Dictionary of models for the specified provider
    """
    return {k: v for k, v in MODELS.items() if v["provider"] == provider}
