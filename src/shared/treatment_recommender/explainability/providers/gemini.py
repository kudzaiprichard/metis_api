"""
Google Gemini provider for explainability system.

This module provides integration with Google's Gemini API for
generating clinical explanations using the official google-genai SDK.
"""
from typing import Optional
from google import genai
from google.genai import types

from ._base import (
    BaseLLMProvider,
    LLMProviderError,
    APIKeyError,
    RateLimitError,
    InvalidResponseError,
    validate_api_key,
    retry_on_rate_limit,
    format_error_message
)


# =============================================================================
# GEMINI PROVIDER
# =============================================================================

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider for clinical explanation generation.

    Uses the official google-genai Python SDK.

    Usage:
        provider = GeminiProvider(
            api_key='YOUR_GEMINI_API_KEY',
            model_name='gemini-2.0-flash-exp'
        )

        response = provider.generate_explanation(
            prompt="Explain why...",
            temperature=0.3
        )

    API Key:
        Get your API key from: https://aistudio.google.com/app/apikey

    Available Models:
        - gemini-2.0-flash-exp (latest)
        - gemini-1.5-pro
        - gemini-1.5-flash
    """

    def __init__(self,
                 api_key: str,
                 model_name: str,
                 timeout: int = 60,
                 verbose: bool = False):
        """
        Initialize Gemini provider.

        Args:
            api_key: Gemini API key (required)
            model_name: Model to use (e.g., 'gemini-2.0-flash-exp') (required)
            timeout: Request timeout in seconds
            verbose: If True, print detailed logs

        Raises:
            APIKeyError: If API key is invalid
        """
        # Validate API key
        self.api_key = validate_api_key(api_key, 'Gemini')
        self.model_name = model_name
        self.timeout = timeout
        self.verbose = verbose

        # Initialize client
        self.client = genai.Client(api_key=self.api_key)

        # Initialize base class
        super().__init__(
            provider_name='gemini',
            model_version=model_name
        )

        if self.verbose:
            print(f"[GeminiProvider] Initialized")
            print(f"[GeminiProvider] Model: {model_name}")
            print(f"[GeminiProvider] Timeout: {timeout}s")

    @retry_on_rate_limit(max_retries=3, backoff_seconds=2)
    def generate_explanation(self,
                             prompt: str,
                             model_name: Optional[str] = None,
                             temperature: float = 0.3) -> str:
        """
        Generate explanation using Gemini API.

        Args:
            prompt: Formatted prompt with context
            model_name: Specific model (overrides default)
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated explanation text

        Raises:
            LLMProviderError: If API call fails
            RateLimitError: If rate limit exceeded
            APIKeyError: If API key is invalid
        """
        # Use provided model or default
        model = model_name or self.model_name

        if self.verbose:
            print(f"[GeminiProvider] Calling API")
            print(f"[GeminiProvider] Model: {model}")
            print(f"[GeminiProvider] Temperature: {temperature}")
            print(f"[GeminiProvider] Prompt length: {len(prompt)} chars")

        try:
            # Configure generation
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
            )

            # Safety settings - allow all for medical content
            safety_settings = [
                types.SafetySetting(
                    category='HARM_CATEGORY_HARASSMENT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_HATE_SPEECH',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_DANGEROUS_CONTENT',
                    threshold='BLOCK_NONE'
                ),
            ]

            # Generate content
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=generation_config,
                safety_settings=safety_settings
            )

            # Extract text
            if not response.text:
                # Check if blocked
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    block_reason = response.prompt_feedback.block_reason
                    raise InvalidResponseError(f"Content blocked: {block_reason}")

                raise InvalidResponseError("No text in response")

            if self.verbose:
                print(f"[GeminiProvider] Response received")
                print(f"[GeminiProvider] Response length: {len(response.text)} chars")

            return response.text

        except Exception as e:
            # Handle SDK-specific errors
            error_str = str(e).lower()

            if 'api key' in error_str or 'authentication' in error_str or '401' in error_str:
                raise APIKeyError(f"Invalid API key: {str(e)}")

            elif 'quota' in error_str or 'rate limit' in error_str or '429' in error_str:
                raise RateLimitError(f"Rate limit exceeded: {str(e)}")

            elif 'not found' in error_str or '404' in error_str:
                raise LLMProviderError(f"Model not found: {model}")

            elif 'timeout' in error_str:
                raise LLMProviderError(f"Request timed out after {self.timeout}s")

            else:
                raise LLMProviderError(format_error_message(e, "API call failed"))

    def test_connection(self) -> bool:
        """
        Test Gemini API connection.

        Returns:
            True if connection works, False otherwise
        """
        try:
            if self.verbose:
                print("[GeminiProvider] Testing API connection...")

            response = self.generate_explanation(
                prompt="Respond with: OK",
                temperature=0.0
            )

            success = len(response) > 0

            if self.verbose:
                if success:
                    print("[GeminiProvider] ✓ Connection test passed")
                else:
                    print("[GeminiProvider] ✗ Connection test failed")

            return success

        except Exception as e:
            if self.verbose:
                print(f"[GeminiProvider] ✗ Connection test failed: {str(e)}")
            return False


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_gemini_provider(api_key: str,
                           model_name: str,
                           timeout: int = 60,
                           verbose: bool = False) -> GeminiProvider:
    """
    Factory function to create Gemini provider.

    Args:
        api_key: Gemini API key (required)
        model_name: Model to use (e.g., 'gemini-2.0-flash-exp') (required)
        timeout: Request timeout in seconds
        verbose: Enable detailed logging

    Returns:
        Configured GeminiProvider instance

    Example:
        provider = create_gemini_provider(
            api_key='YOUR_KEY',
            model_name='gemini-2.0-flash-exp'
        )
    """
    return GeminiProvider(
        api_key=api_key,
        model_name=model_name,
        timeout=timeout,
        verbose=verbose
    )