from typing import Tuple
from ros_lm.models.language_model import LanguageModel
from ros_lm.models.model_database import ModelDatabase

from ros_lm.models.llm.lvlm.minicpm_model import MiniCPMModel
from ros_lm.models.llm.lvlm.qwen2_5vl_model import Qwen25VLModel
from ros_lm.models.llm.lvlm.large_vision_language_model import LargeVisionLanguageModel

class LVLMFactory:

    @staticmethod
    def create_lvlm(model_name: str) -> Tuple[bool, LanguageModel]:
        if not ModelDatabase.is_lvlm(model_name):
            raise ValueError(f"Model {model_name} not found in the LVLM database.")
        
        match model_name:
            case "openbmb/MiniCPM-o-2_6":
                return MiniCPMModel.create()
            case "Qwen/Qwen2.5-VL-32B-Instruct":
                return Qwen25VLModel.create()
            case _:
                return LargeVisionLanguageModel.create(model_name)


