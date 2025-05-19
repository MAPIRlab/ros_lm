
from ros_lm.models.llm.lvlm.large_vision_language_model import LargeVisionLanguageModel


class MiniCPMModel(LargeVisionLanguageModel):
    
    def __init__(self, model_id, processor, model):
        self._model_id = model_id
        self.processor = processor
        self._model = model