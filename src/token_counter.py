"""
Token Counter Module
====================
Tracks OpenAI token usage for embeddings and LLM calls.

This helps you understand exactly how many tokens are consumed
and estimate costs for your RAG system.
"""

import tiktoken
from typing import List, Tuple


class TokenCounter:
    """Counts tokens for OpenAI API calls."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize token counter.
        
        Args:
            model: OpenAI model name (gpt-3.5-turbo, gpt-4, etc.)
        """
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Pricing per 1K tokens (as of 2024)
        self.pricing = {
            "gpt-3.5-turbo": {
                "input": 0.0005,   # $0.0005 per 1K input tokens
                "output": 0.0015   # $0.0015 per 1K output tokens
            },
            "gpt-4": {
                "input": 0.03,     # $0.03 per 1K input tokens
                "output": 0.06     # $0.06 per 1K output tokens
            }
        }
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Number of tokens
        """
        tokens = self.encoding.encode(text)
        return len(tokens)
    
    def count_messages_tokens(self, messages: List[dict]) -> int:
        """
        Count tokens in a list of messages (for chat API).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Number of tokens
        """
        total = 0
        for message in messages:
            # Each message has overhead
            total += 4  # Message overhead
            total += self.count_tokens(message.get("content", ""))
        
        # Add buffer for response format
        total += 2
        return total
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Tuple[float, dict]:
        """
        Estimate cost for input and output tokens.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Tuple of (total_cost, breakdown_dict)
        """
        pricing = self.pricing.get(self.model, self.pricing["gpt-3.5-turbo"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return total_cost, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def format_cost_breakdown(self, breakdown: dict) -> str:
        """
        Format cost breakdown for display.
        
        Args:
            breakdown: Cost breakdown dict from estimate_cost()
        
        Returns:
            Formatted string
        """
        return f"""
📊 Token Usage:
  Input Tokens:  {breakdown['input_tokens']:,}
  Output Tokens: {breakdown['output_tokens']:,}
  Total Tokens:  {breakdown['input_tokens'] + breakdown['output_tokens']:,}

💰 Cost Breakdown:
  Input Cost:    ${breakdown['input_cost']:.6f}
  Output Cost:   ${breakdown['output_cost']:.6f}
  Total Cost:    ${breakdown['total_cost']:.6f}
"""


class RAGTokenTracker:
    """Tracks tokens for entire RAG pipeline."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize RAG token tracker."""
        self.counter = TokenCounter(model)
        self.embeddings_tokens = 0
        self.retrieval_tokens = 0
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.total_cost = 0.0
    
    def track_embeddings(self, chunks: List[str]) -> int:
        """
        Track tokens used for generating embeddings.
        
        Note: Embeddings use a different model (text-embedding-3-small)
        and have different pricing: $0.02 per 1M tokens
        
        Args:
            chunks: List of text chunks
        
        Returns:
            Total tokens used
        """
        total_tokens = sum(self.counter.count_tokens(chunk) for chunk in chunks)
        
        # Embeddings pricing: $0.02 per 1M tokens
        embedding_cost = (total_tokens / 1_000_000) * 0.02
        
        self.embeddings_tokens = total_tokens
        self.total_cost += embedding_cost
        
        return total_tokens
    
    def track_retrieval(self, query: str, retrieved_chunks: List[str]) -> int:
        """
        Track tokens used for retrieval (context formatting).
        
        Args:
            query: User query
            retrieved_chunks: Retrieved chunks
        
        Returns:
            Total tokens used
        """
        query_tokens = self.counter.count_tokens(query)
        context_tokens = sum(self.counter.count_tokens(chunk) for chunk in retrieved_chunks)
        
        total_tokens = query_tokens + context_tokens
        self.retrieval_tokens = total_tokens
        
        return total_tokens
    
    def track_llm_call(self, input_text: str, output_text: str) -> Tuple[int, int, float]:
        """
        Track tokens used for LLM call.
        
        Args:
            input_text: Input to LLM (system + user prompt + context)
            output_text: Output from LLM (answer)
        
        Returns:
            Tuple of (input_tokens, output_tokens, cost)
        """
        input_tokens = self.counter.count_tokens(input_text)
        output_tokens = self.counter.count_tokens(output_text)
        
        cost, _ = self.counter.estimate_cost(input_tokens, output_tokens)
        
        self.llm_input_tokens = input_tokens
        self.llm_output_tokens = output_tokens
        self.total_cost += cost
        
        return input_tokens, output_tokens, cost
    
    def get_summary(self) -> dict:
        """
        Get summary of all tokens and costs.
        
        Returns:
            Dictionary with token and cost breakdown
        """
        return {
            "embeddings_tokens": self.embeddings_tokens,
            "retrieval_tokens": self.retrieval_tokens,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "total_tokens": (
                self.embeddings_tokens +
                self.retrieval_tokens +
                self.llm_input_tokens +
                self.llm_output_tokens
            ),
            "total_cost": self.total_cost
        }
    
    def format_summary(self) -> str:
        """Format summary for display."""
        summary = self.get_summary()
        
        return f"""
═══════════════════════════════════════════════════════════
                    📊 COMPLETE TOKEN USAGE REPORT
═══════════════════════════════════════════════════════════

🔤 Token Breakdown:
  Embeddings (PDF chunks):     {summary['embeddings_tokens']:>8,} tokens
  Retrieval (context):         {summary['retrieval_tokens']:>8,} tokens
  LLM Input (prompt):          {summary['llm_input_tokens']:>8,} tokens
  LLM Output (answer):         {summary['llm_output_tokens']:>8,} tokens
  ─────────────────────────────────────────
  TOTAL:                       {summary['total_tokens']:>8,} tokens

💰 Cost Breakdown:
  Embeddings Cost:             ${(summary['embeddings_tokens'] / 1_000_000 * 0.02):>10.6f}
  LLM Cost:                    ${(summary['total_cost'] - summary['embeddings_tokens'] / 1_000_000 * 0.02):>10.6f}
  ─────────────────────────────────────────
  TOTAL COST:                  ${summary['total_cost']:>10.6f}

📈 Estimates:
  Cost per question:           ${summary['total_cost']:>10.6f}
  Questions per $1:            {int(1 / max(summary['total_cost'], 0.0001)):>10,}
  Questions per $10:           {int(10 / max(summary['total_cost'], 0.0001)):>10,}

═══════════════════════════════════════════════════════════
"""
