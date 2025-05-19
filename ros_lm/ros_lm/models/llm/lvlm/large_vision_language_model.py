import base64
from io import BytesIO
import torch
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from ros_lm.models.language_model import LanguageModel


class LargeVisionLanguageModel(LanguageModel):
    """A vision-language model for generating text from images and text prompts."""

    def __init__(self, model_id: str, processor, model):
        """Initializes the model with a processor and a pre-trained vision-language model."""
        self._model_id = model_id
        self.processor = processor
        self._model = model

    def get_model_id(self):
        """Returns the model identifier."""
        return self._model_id

    def generate_text(self, prompt, params):
        return super().generate_text(prompt, params)

    def generate_text_with_images(self, prompt: str, images: list[str], params: dict):
        """Generates text based on a text prompt and input images."""
        inputs = self.processor(
            text=prompt, 
            images=list(map(self.base64_to_PIL, images)), 
            padding=True, 
            return_tensors="pt"
        )
        output = self._model.generate(**inputs, max_new_tokens=200)
        return self.processor.batch_decode(output, skip_special_tokens=True)[-1]

    @staticmethod
    def create(model_id: str):
        """Creates a LargeVisionLanguageModel instance with the specified model ID."""
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, 
                quantization_config=quantization_config, 
                device_map="auto"
            )
            return True, LargeVisionLanguageModel(model_id, processor, model)
        
        except OSError:
            return False, None
