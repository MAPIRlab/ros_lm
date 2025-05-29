
from typing import Tuple
import torch
from transformers import AutoModel, AutoTokenizer

from ros_lm.models.llm.lvlm.large_vision_language_model import LargeVisionLanguageModel
from ros_lm.models.language_model import LanguageModel

class MiniCPMModel(LanguageModel):
    
    def __init__(self):
        self._model_id = "openbmb/MiniCPM-o-2_6"
        self._model = self.__create_model()
        self._tokenizer = self.__create_tokenizer()

    def get_model_id(self) -> str:
        return self._model_id

    def generate_text(self, prompt: str, params: dict) -> Tuple[str, float]:
        msgs = [{'role': 'user', 'content': prompt}]
        res = self._model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self._tokenizer,
        )

        return str(res)
    

    def generate_text_with_images(self, prompt: str, images: list[str], params: dict) -> Tuple[str, float]:
        pil_images = [self.base64_to_PIL(image) for image in images]
        msgs = [{'role': 'user', 'content': [*pil_images,prompt]}]

        res = self._model.chat(
            image=pil_images,
            temperature=params['temperature'],
            msgs=msgs,
            tokenizer=self._tokenizer,
        )
        return str(res)
    
    @staticmethod
    def create() -> Tuple[bool, LanguageModel]:
        """Creates a MiniCPMModel instance with the specified model ID."""
        try:
            model = MiniCPMModel()
            return True, model
        except OSError:
            return False, None

    def __create_model(self):
        model = AutoModel.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False,
        )
        model = model.eval().cuda()
        return model

    def __create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
        return tokenizer