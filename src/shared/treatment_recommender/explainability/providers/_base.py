"""
Base LLM provider interface for explainability system.

This module defines the abstract interface that all LLM providers
(Gemini, GPT, Claude) must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import re


# =============================================================================
# BASE LLM PROVIDER
# =============================================================================

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (Gemini, OpenAI, Anthropic) must implement this interface.

    Key Responsibilities:
    1. API communication (handle authentication, rate limits, errors)
    2. Prompt formatting (provider-specific formatting)
    3. Response parsing (extract JSON from LLM response)
    4. Error handling (retries, fallbacks)

    Implementation Example:
        class MyLLMProvider(BaseLLMProvider):
            def __init__(self, api_key: str):
                super().__init__(
                    provider_name='my_provider',
                    model_version='my-model-v1'
                )
                self.api_key = api_key

            def generate_explanation(self, prompt, model_name, temperature):
                response = requests.post(...)
                return response.json()['text']
    """

    def __init__(self, provider_name: str, model_version: str):
        """
        Initialize base LLM provider.

        Args:
            provider_name: Provider identifier (e.g., 'gemini', 'openai', 'claude')
            model_version: Model version string (e.g., 'gemini-pro', 'gpt-4')
        """
        self._provider_name = provider_name
        self._model_version = model_version

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._provider_name

    @property
    def model_version(self) -> str:
        """Get model version."""
        return self._model_version

    @abstractmethod
    def generate_explanation(self,
                             prompt: str,
                             model_name: Optional[str] = None,
                             temperature: float = 0.3) -> str:
        """
        Generate explanation text from prompt.

        This method must:
        1. Call the LLM API with the prompt
        2. Handle API errors gracefully
        3. Return the generated text

        Args:
            prompt: Formatted prompt with context
            model_name: Specific model version (overrides default)
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated explanation text (may contain JSON)

        Raises:
            LLMProviderError: If API call fails
        """
        pass

    def _extract_json(self, text: str) -> dict:
        """
        Extract JSON from LLM response text.

        Handles:
        - Plain JSON
        - JSON wrapped in markdown code blocks (```json ... ```)
        - JSON with extra whitespace/newlines

        Args:
            text: Raw response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If no valid JSON found
        """
        import re

        # Remove markdown code fences if present
        # Match ```json ... ``` or ``` ... ```
        code_fence_pattern = r'```(?:json)?\s*(.*?)\s*```'
        matches = re.findall(code_fence_pattern, text, re.DOTALL)

        if matches:
            # Use the content inside code fences
            text = matches[0]

        # Try to find JSON object boundaries
        text = text.strip()

        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')

        if start == -1 or end == -1:
            raise ValueError(f"No valid JSON dict found in response. First 200 chars: {text[:200]}...")

        json_str = text[start:end + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}. First 200 chars: {json_str[:200]}...")

    def parse_structured_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured dictionary.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed dictionary

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            result = self._extract_json(response_text)
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict, got {type(result).__name__}")
            return result
        except ValueError:
            # Try to fix common JSON issues and retry
            fixed = self._fix_json(response_text)
            result = self._extract_json(fixed)
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict after fix, got {type(result).__name__}")
            return result

    def _fix_json(self, json_text: str) -> str:
        """
        Attempt to fix common JSON formatting issues.

        Fixes:
        - Trailing commas before } or ]
        - Unescaped quotes in strings
        - Missing commas between elements
        - Common formatting issues

        Args:
            json_text: Potentially malformed JSON

        Returns:
            Fixed JSON string
        """
        # Remove markdown fences first
        code_fence_pattern = r'```(?:json)?\s*(.*?)\s*```'
        matches = re.findall(code_fence_pattern, json_text, re.DOTALL)
        if matches:
            json_text = matches[0]

        # Remove trailing commas before closing braces/brackets
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)

        # Fix missing commas between array elements (common LLM error)
        # This regex looks for }" followed by { without a comma
        json_text = re.sub(r'\}\s*\n\s*\{', '},\n{', json_text)

        # Fix missing commas between object properties
        # Looks for }" or ]" followed by " without a comma
        json_text = re.sub(r'([}\]])\s*\n\s*"', r'\1,\n"', json_text)

        return json_text.strip()

    def validate_response_schema(self, parsed_response: Dict[str, Any]) -> bool:
        """
        Validate that parsed response has required fields.

        Expected schema:
        {
            "summary": {...},
            "model_reasoning": {...},
            "clinical_context": {...},
            "safety_considerations": {...},
            "alternatives_explanation": {...}
        }

        Args:
            parsed_response: Parsed LLM response

        Returns:
            True if schema is valid, False otherwise
        """
        required_keys = [
            'summary',
            'model_reasoning',
            'clinical_context',
            'safety_considerations',
            'alternatives_explanation'
        ]

        for key in required_keys:
            if key not in parsed_response:
                return False

        return True

    def test_connection(self) -> bool:
        """
        Test that API connection is working.

        Sends a simple test prompt to verify API key and connectivity.

        Returns:
            True if connection works, False otherwise
        """
        try:
            test_prompt = "Respond with: 'OK'"
            response = self.generate_explanation(
                prompt=test_prompt,
                temperature=0.0
            )
            return len(response) > 0
        except Exception:
            return False


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class APIKeyError(LLMProviderError):
    """API key is invalid or missing."""
    pass


class RateLimitError(LLMProviderError):
    """API rate limit exceeded."""
    pass


class InvalidResponseError(LLMProviderError):
    """LLM response is invalid or cannot be parsed."""
    pass


class ModelNotFoundError(LLMProviderError):
    """Requested model is not available."""
    pass


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry_on_rate_limit(max_retries: int = 3, backoff_seconds: int = 2):
    """
    Decorator to retry API calls on rate limit errors.

    Uses exponential backoff: 2s, 4s, 8s, etc.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_seconds: Initial backoff time (doubles each retry)

    Example:
        @retry_on_rate_limit(max_retries=3)
        def call_api(self, prompt):
            response = requests.post(...)
            return response
    """
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = backoff_seconds

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except RateLimitError:
                    retries += 1
                    if retries >= max_retries:
                        raise

                    print(f"Rate limit hit. Retrying in {backoff}s... ({retries}/{max_retries})")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff

            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_api_key(api_key: Optional[str], provider_name: str) -> str:
    """
    Validate API key is provided and not empty.

    Args:
        api_key: API key to validate
        provider_name: Provider name for error message

    Returns:
        Validated API key

    Raises:
        APIKeyError: If API key is invalid
    """
    if not api_key:
        raise APIKeyError(
            f"{provider_name} API key is required. "
            f"Provide it via the 'api_key' parameter or set the appropriate environment variable."
        )

    if not isinstance(api_key, str):
        raise APIKeyError(f"API key must be a string, got {type(api_key).__name__}")

    if len(api_key.strip()) == 0:
        raise APIKeyError(f"{provider_name} API key cannot be empty")

    return api_key.strip()


def format_error_message(error: Exception, context: str) -> str:
    """
    Format error message with context.

    Args:
        error: Exception that occurred
        context: Context description

    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)

    return f"{context}: {error_type} - {error_msg}"


# =============================================================================
# RESPONSE VALIDATION
# =============================================================================

def validate_explanation_response(response: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate that LLM response contains required fields.

    Args:
        response: Parsed LLM response

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        is_valid, error = validate_explanation_response(parsed)
        if not is_valid:
            raise InvalidResponseError(error)
    """
    # Check required top-level keys
    required_keys = {
        'summary': ['one_sentence', 'clinical_priority'],
        'model_reasoning': ['why_this_treatment', 'key_factors'],
        'clinical_context': ['guideline_alignment'],
        'safety_considerations': ['warnings', 'monitoring_requirements'],
        'alternatives_explanation': ['why_not_alternatives', 'alternatives']
    }

    for key, subkeys in required_keys.items():
        if key not in response:
            return False, f"Missing required key: '{key}'"

        for subkey in subkeys:
            if subkey not in response[key]:
                return False, f"Missing required subkey: '{key}.{subkey}'"

    # Validate data types
    if not isinstance(response['model_reasoning']['key_factors'], list):
        return False, "model_reasoning.key_factors must be a list"

    if not isinstance(response['safety_considerations']['warnings'], list):
        return False, "safety_considerations.warnings must be a list"

    if not isinstance(response['alternatives_explanation']['alternatives'], list):
        return False, "alternatives_explanation.alternatives must be a list"

    return True, None