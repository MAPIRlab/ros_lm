import torch
from typing import Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


from ros_lm.models.language_model import LanguageModel

class Qwen25VLModel(LanguageModel):
    
    def __init__(self):
        self._model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
        self._model = self.__create_model()
        self._processor = self.__create_processor()

    def get_model_id(self) -> str:
        return self._model_id

    def generate_text(self, prompt: str, params: dict) -> Tuple[str, float]:
        prompt = {"type": "text", "text": prompt}
        msgs = [{'role': 'user', 'content': [prompt]}]

        text = self._processor.apply_chat_template(
            conversation=msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=text,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self._model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if isinstance(output_text, list):
            return output_text[0]
        return str(output_text)
    

    def generate_text_with_images(self, prompt: str, images: list[str], params: dict) -> Tuple[str, float]:

        images_dict = [{"type": "image", "image": f"data:image;base64,{image}"} for image in images]
        prompt = {"type": "text", "text": prompt}
        msgs = [{'role': 'user', 'content': [*images_dict, prompt]}]

        text = self._processor.apply_chat_template(
            conversation=msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs,_ = process_vision_info(msgs)
        inputs = self._processor(
            text=text,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self._model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if isinstance(output_text, list):
            return output_text[0]
        return str(output_text)
    
    @staticmethod
    def create() -> Tuple[bool, LanguageModel]:
        """Creates a MiniCPMModel instance with the specified model ID."""
        try:
            model = Qwen25VLModel()
            return True, model
        except OSError:
            return False, None

    def __create_model(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct",
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        return model

    def __create_processor(self):
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
        return processor