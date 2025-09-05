"""
LLM Provider Management System
Handles different providers: OpenRouter, Anthropic, Cohere
"""

import os
import time
from typing import Dict, List, Optional, Any
import openai
import anthropic
import cohere
from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._setup_client()
    
    @abstractmethod
    def _setup_client(self):
        """Setup the provider client"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider"""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate response from the model"""
        pass


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider implementation"""
    
    def _setup_client(self):
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    def get_available_models(self) -> List[str]:
        """Get available models from OpenRouter"""
        return [
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-405b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "mistralai/mistral-7b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "mistralai/mixtral-8x22b-instruct"
        ]
    
    def generate_response(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate response using OpenRouter"""
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000)
            )
            
            execution_time = time.time() - start_time
            
            return {
                "content": response.choices[0].message.content,
                "execution_time": execution_time,
                "model": model,
                "provider": "openrouter"
            }
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "execution_time": 0,
                "model": model,
                "provider": "openrouter",
                "error": str(e)
            }


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation"""
    
    def _setup_client(self):
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def get_available_models(self) -> List[str]:
        """Get available Anthropic models"""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    
    def generate_response(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Anthropic"""
        try:
            start_time = time.time()
            
            response = self.client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            execution_time = time.time() - start_time
            
            return {
                "content": response.content[0].text,
                "execution_time": execution_time,
                "model": model,
                "provider": "anthropic"
            }
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "execution_time": 0,
                "model": model,
                "provider": "anthropic",
                "error": str(e)
            }


class CohereProvider(BaseProvider):
    """Cohere provider implementation"""
    
    def _setup_client(self):
        self.client = cohere.Client(api_key=self.api_key)
    
    def get_available_models(self) -> List[str]:
        """Get available Cohere models"""
        return [
            "command-r-plus",
            "command-r",
            "command",
            "command-light",
            "command-nightly"
        ]
    
    def generate_response(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Cohere"""
        try:
            start_time = time.time()
            
            response = self.client.chat(
                model=model,
                message=prompt,
            )
            
            execution_time = time.time() - start_time
            
            return {
                "content": response.text,
                "execution_time": execution_time,
                "model": model,
                "provider": "cohere"
            }
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "execution_time": 0,
                "model": model,
                "provider": "cohere",
                "error": str(e)
            }


class ProviderManager:
    """Manages different LLM providers"""
    
    def __init__(self):
        self.providers = {}
        self.current_provider = None
    
    def add_provider(self, provider_name: str, api_key: str):
        """Add a provider with API key"""
        provider_name = provider_name.lower()
        
        if provider_name == "openrouter":
            self.providers[provider_name] = OpenRouterProvider(api_key)
        elif provider_name == "anthropic":
            self.providers[provider_name] = AnthropicProvider(api_key)
        elif provider_name == "cohere":
            self.providers[provider_name] = CohereProvider(api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
    
    def set_current_provider(self, provider_name: str):
        """Set the current active provider"""
        provider_name = provider_name.lower()
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found. Please add it first.")
        self.current_provider = provider_name
    
    def get_available_models(self, provider_name: Optional[str] = None) -> List[str]:
        """Get available models for a provider"""
        if provider_name:
            provider_name = provider_name.lower()
            if provider_name not in self.providers:
                raise ValueError(f"Provider {provider_name} not found.")
            return self.providers[provider_name].get_available_models()
        elif self.current_provider:
            return self.providers[self.current_provider].get_available_models()
        else:
            raise ValueError("No provider selected")
    
    def generate_response(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate response using current provider"""
        if not self.current_provider:
            raise ValueError("No provider selected")
        
        return self.providers[self.current_provider].generate_response(prompt, model, **kwargs)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers"""
        return {
            "available_providers": list(self.providers.keys()),
            "current_provider": self.current_provider,
            "provider_models": {
                provider: provider_obj.get_available_models() 
                for provider, provider_obj in self.providers.items()
            }
        }
