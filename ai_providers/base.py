"""Abstract base class every AI provider implements.

Both ``generate`` and ``generate_stream`` accept either a single user prompt
string OR an OpenAI-style messages array (list of {role, content}). The
simple case (analysis, selects, profile_creation) passes a string; chat
passes the full message history. Each provider serializes whichever it
received using the appropriate native format.

``task_type`` selects the model when the provider has tiered options:

  - "analysis"          — story/social/soundbite analysis chunks (flagship)
  - "chat"              — interactive AI chat
  - "story_builder"     — Story Builder timeline assembly
  - "selects"           — clip selection / labeling
  - "profile_creation"  — My Style synthesis (flagship)
  - "general"           — anything else / test connection

Ollama ignores ``task_type`` entirely — the user picks one local model.
Anthropic and OpenAI route the flagship model (Opus 4.8 / gpt-5.4) to
``analysis`` and ``profile_creation``, and a faster, cheaper model
(Sonnet 4.6 / gpt-5.4-mini) to everything else.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Union


class BaseProvider(ABC):
    """Common interface every provider must implement."""

    name: str = ""  # 'ollama' | 'anthropic' | 'openai'

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_or_messages: Union[str, list],
        task_type: str = "general",
        **kwargs,
    ) -> str:
        """Send a prompt; return the full response text."""

    @abstractmethod
    def generate_stream(
        self,
        system_prompt: str,
        user_or_messages: Union[str, list],
        task_type: str = "general",
        **kwargs,
    ) -> Iterator[str]:
        """Send a prompt; yield response chunks for streaming display."""

    @abstractmethod
    def test_connection(self) -> dict:
        """Verify reachability + credentials.

        Returns ``{"success": True, ...}`` on success, or
        ``{"success": False, "error": "<message>"}`` on failure.
        """
