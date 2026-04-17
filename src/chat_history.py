"""
Chat History Module
===================
Manages conversation history and context for multi-turn conversations.

Features:
- Stores all questions and answers
- Maintains conversation context
- Learns from previous exchanges
- Provides context to LLM for better answers
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    """Single message in conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "sources": self.sources or []
        }


class ChatHistory:
    """Manages conversation history."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize chat history.
        
        Args:
            max_history: Maximum number of messages to keep
        """
        self.messages: List[Message] = []
        self.max_history = max_history
    
    def add_message(
        self,
        role: str,
        content: str,
        sources: Optional[List[str]] = None
    ) -> None:
        """
        Add message to history.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            sources: Optional list of sources (for assistant messages)
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            sources=sources
        )
        self.messages.append(message)
        
        # Keep only recent messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_history(self) -> List[Message]:
        """Get all messages in history."""
        return self.messages
    
    def get_context(self, num_turns: int = 3) -> str:
        """
        Get recent conversation context for LLM.
        
        Args:
            num_turns: Number of recent exchanges to include
        
        Returns:
            Formatted conversation context
        """
        if not self.messages:
            return ""
        
        # Get last num_turns exchanges (user + assistant pairs)
        recent_messages = self.messages[-(num_turns * 2):]
        
        context_parts = []
        for msg in recent_messages:
            if msg.role == "user":
                context_parts.append(f"User: {msg.content}")
            else:
                context_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_system_prompt_with_context(self) -> str:
        """
        Get enhanced system prompt with conversation context.
        
        Returns:
            System prompt with history context
        """
        base_prompt = """You are a helpful assistant that answers questions based on provided context.
Always answer based on the context given. If the context doesn't contain the answer, say so clearly.
Be concise and accurate.
Remember the conversation history and use it to provide better answers."""
        
        if not self.messages:
            return base_prompt
        
        # Add recent conversation context
        context = self.get_context(num_turns=2)
        if context:
            return f"""{base_prompt}

Recent conversation history:
{context}

Use this history to provide consistent and contextual answers."""
        
        return base_prompt
    
    def clear(self) -> None:
        """Clear all history."""
        self.messages = []
    
    def get_summary(self) -> dict:
        """Get conversation summary."""
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m.role == "user"),
            "assistant_messages": sum(1 for m in self.messages if m.role == "assistant"),
            "messages": [m.to_dict() for m in self.messages]
        }
    
    def export_conversation(self) -> str:
        """Export conversation as formatted text."""
        if not self.messages:
            return "No conversation history."
        
        lines = ["=== Conversation History ===\n"]
        for msg in self.messages:
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            role = msg.role.upper()
            lines.append(f"[{timestamp}] {role}:")
            lines.append(msg.content)
            if msg.sources:
                lines.append(f"Sources: {len(msg.sources)} document(s)")
            lines.append("")
        
        return "\n".join(lines)


class ContextualRAGRetriever:
    """
    RAG Retriever that uses conversation history for better context.
    
    Enhances retrieval by considering previous questions and answers.
    """
    
    def __init__(self, base_retriever, chat_history: ChatHistory):
        """
        Initialize contextual retriever.
        
        Args:
            base_retriever: Base RAG retriever
            chat_history: Chat history manager
        """
        self.base_retriever = base_retriever
        self.chat_history = chat_history
    
    def generate_answer_with_history(
        self,
        question: str,
        k: int = 5,
        include_sources: bool = True,
        use_history: bool = True
    ) -> dict:
        """
        Generate answer considering conversation history.
        
        Args:
            question: User question
            k: Number of context chunks
            include_sources: Whether to include sources
            use_history: Whether to use conversation history
        
        Returns:
            Answer with sources
        """
        # Get base answer
        result = self.base_retriever.generate_answer(
            question,
            k=k,
            include_sources=include_sources
        )
        
        # Enhance with history if enabled
        if use_history and self.chat_history.get_history():
            history_context = self.chat_history.get_context(num_turns=2)
            result['history_context'] = history_context
        
        return result
    
    def add_to_history(
        self,
        question: str,
        answer: str,
        sources: Optional[List[str]] = None
    ) -> None:
        """Add Q&A to history."""
        self.chat_history.add_message("user", question)
        self.chat_history.add_message("assistant", answer, sources=sources)
