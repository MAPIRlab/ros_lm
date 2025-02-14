from abc import ABC, abstractmethod
from typing import Tuple

class LanguageModel(ABC):

    @abstractmethod
    def get_model_id(self) -> str:
        # TODO: create documentation
        pass

    @abstractmethod
    def generate_text(self, prompt: str, params: dict) -> Tuple[str, float]:
        # TODO: create documentation
        raise NotImplementedError("Method generate_text is not implemented! Is your model a LLM?")

    @abstractmethod
    # TODO: type of images
    def generate_text_with_images(self, prompt: str, images: list, params: dict) -> Tuple[str, float]:
        # TODO: create documentation
        raise NotImplementedError("Method generate_text_with_images is not implemented! Is your model a LVLM?")

    