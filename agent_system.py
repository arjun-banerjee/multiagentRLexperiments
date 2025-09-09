from abc import ABC, abstractmethod
from typing import Any

class MultiAgentSystem(ABC):
    """
    Base class for all multi-agent systems.
    Subclasses must implement `answer`, which maps a system prompt, question -> answer.
    """

    @abstractmethod
    def answer(self, system_prompt: str, question: str, **kwargs: Any) -> str:
        """Return an answer to `question`."""
        raise NotImplementedError

