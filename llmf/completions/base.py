"""
Text completions base module
"""

from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

T = TypeVar('T')

class TextCompletionsClient(ABC, Generic[T]):
    """ Client for generating text completions from prompts """

    def run_batch(self, prompts: List[T]) -> List[str]:
        """
        Generate a completion for the given prompts
        """
        return [self.run(prompt) for prompt in prompts]

    @abstractmethod
    def run(self, prompt: T) -> str:
        """
        Generate a completion for the given prompt
        """
