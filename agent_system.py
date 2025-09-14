from abc import ABC, abstractmethod
from typing import Any

class MultiAgentSystem(ABC):

    # main api for multi agent system 
    # TODO: batch it later 
    @abstractmethod
    def answer(self, system_prompt: str, question: str, **kwargs: Any) -> str:
        return "bello"

