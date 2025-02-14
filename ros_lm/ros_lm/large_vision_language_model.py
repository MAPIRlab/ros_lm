
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
import torch

class LargeVisionLanguageModel:

    def __init__(self, model_id : str, processor, model):
        # TODO: documentation
        self._model_id = model_id
        self.processor = processor
        self._model = model

    def get_model_id(self,):
        # TODO: documentation
        return self._model_id()

    def generate_text_with_images(self, prompt: str, images: list):
        # TODO: documentation
        
        # Process input
        inputs = self.processor(text=prompt, images=images, padding=True, return_tensors="pt").to("cuda")

        # Generate output
        output = self.model.generate(**inputs, max_new_tokens=200)
        
        # Decode generated text
        return self.processor.batch_decode(output, skip_special_tokens=True).join(" ")
    
    @staticmethod
    def create(model_id: str):
        # TODO: documentation
        try:
            # TODO: review
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            # Load processor and model
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

            return True, LargeVisionLanguageModel(model_id, processor, model)
        
        except OSError as e:
            return False, None