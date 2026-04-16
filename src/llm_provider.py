"""
LLM Provider Module
===================
Supports multiple LLM providers: OpenAI and Ollama

This module provides a unified interface for different LLM backends,
allowing easy switching between cloud-based (OpenAI) and local (Ollama) models.
"""

from typing import List, Optional
from abc import ABC, abstractmethod
import requests
from langchain_core.messages import BaseMessage


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def invoke(self, messages: List[BaseMessage]) -> str:
        """Generate response from messages."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local model inference."""
    
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model name (e.g., "mistral", "llama2", "neural-chat")
            base_url: Ollama server URL (default: localhost:11434)
            
        Popular Ollama models:
        - mistral: Fast, good quality (recommended)
        - llama2: Larger, better quality
        - neural-chat: Optimized for chat
        - dolphin-mixtral: Very capable
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                raise ConnectionError(
                    f"Ollama server not responding. "
                    f"Make sure Ollama is running at {self.base_url}"
                )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Please start Ollama with: ollama serve"
            )
    
    def invoke(self, messages: List[BaseMessage]) -> str:
        """
        Generate response using Ollama.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Generated response text
        """
        # Convert messages to prompt format
        prompt = self._format_messages(messages)
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                },
                timeout=300  # 5 minute timeout for long responses
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.text}")
            
            result = response.json()
            return result.get("response", "").strip()
        
        except requests.exceptions.Timeout:
            raise TimeoutError(
                "Ollama request timed out. The model might be too large "
                "or your system is slow. Try a smaller model."
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Please start Ollama with: ollama serve"
            )
    
    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Format messages into a prompt string."""
        prompt_parts = []
        
        for msg in messages:
            role = msg.type  # "human", "ai", "system"
            content = msg.content
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "human":
                prompt_parts.append(f"User: {content}")
            elif role == "ai":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:"


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider (wrapper for existing ChatOpenAI)."""
    
    def __init__(self, llm):
        """
        Initialize OpenAI provider.
        
        Args:
            llm: ChatOpenAI instance from langchain_openai
        """
        self.llm = llm
    
    def invoke(self, messages: List[BaseMessage]) -> str:
        """Generate response using OpenAI."""
        response = self.llm.invoke(messages)
        return response.content


def get_llm_provider(provider: str = "ollama", **kwargs) -> LLMProvider:
    """
    Factory function to get LLM provider.
    
    Args:
        provider: "ollama" or "openai"
        **kwargs: Provider-specific arguments
        
    Returns:
        LLMProvider instance
    """
    if provider.lower() == "ollama":
        return OllamaProvider(
            model=kwargs.get("model", "mistral"),
            base_url=kwargs.get("base_url", "http://localhost:11434")
        )
    elif provider.lower() == "openai":
        from langchain_openai import ChatOpenAI
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("OpenAI provider requires api_key")
        
        llm = ChatOpenAI(
            api_key=api_key,
            model=kwargs.get("model", "gpt-3.5-turbo"),
            temperature=kwargs.get("temperature", 0.7)
        )
        return OpenAIProvider(llm)
    else:
        raise ValueError(f"Unknown provider: {provider}")
