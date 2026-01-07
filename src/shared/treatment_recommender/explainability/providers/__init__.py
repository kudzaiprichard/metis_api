"""
LLM providers for explainability system.

This package provides integrations with various LLM providers
(Gemini, OpenAI GPT, Anthropic Claude) for generating clinical explanations.

Usage:
    from treatment_recommender.explainability.providers import create_gemini_provider

    provider = create_gemini_provider(
        api_key='YOUR_KEY',
        model_name='gemini-1.5-pro-latest'
    )
"""

from ._base import (
    BaseLLMProvider,
    LLMProviderError,
    APIKeyError,
    RateLimitError,
    InvalidResponseError,
    ModelNotFoundError,
)

from .gemini import (
    GeminiProvider,
    create_gemini_provider,
)

__all__ = [
    # Base
    'BaseLLMProvider',
    'LLMProviderError',
    'APIKeyError',
    'RateLimitError',
    'InvalidResponseError',
    'ModelNotFoundError',

    # Gemini
    'GeminiProvider',
    'create_gemini_provider',
]

__version__ = '1.0.0'