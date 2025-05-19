from abc import ABC, abstractmethod
from typing import Tuple
from PIL import Image
import base64
from io import BytesIO

class LanguageModel(ABC):
    """Abstract base class for language models (LLMs and LVLMs)."""

    @abstractmethod
    def get_model_id(self) -> str:
        """Returns the unique identifier of the language model."""
        pass

    @abstractmethod
    def generate_text(self, prompt: str, params: dict) -> Tuple[str, float]:
        """
        Generates text based on the given prompt and parameters.

        Args:
            prompt (str): The input text prompt.
            params (dict): Model-specific generation parameters.

        Returns:
            Tuple[str, float]: The generated text and processing time.
        """
        raise NotImplementedError("Method generate_text is not implemented! Is your model a LLM?")

    @abstractmethod
    def generate_text_with_images(self, prompt: str, images: list[str], params: dict) -> Tuple[str, float]:
        """
        Generates text based on a prompt and accompanying images.

        Args:
            prompt (str): The input text prompt.
            images (list[str]): A list of base64-encoded images or image paths.
            params (dict): Model-specific generation parameters.

        Returns:
            Tuple[str, float]: The generated text and processing time.
        """
        raise NotImplementedError("Method generate_text_with_images is not implemented! Is your model a LVLM?")
    
    def base64_to_PIL(self, image_b64: str) -> list[Image.Image]:
        """Converts base64-encoded images to PIL images."""
        try:
            img_data = base64.b64decode(image_b64)
            img = Image.open(BytesIO(img_data))
            return img
        except Exception as e:
            print(f"Error decoding image: {e}")