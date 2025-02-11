import rclpy
from rclpy.node import Node
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
import torch

from .large_vision_language_model import LargeVisionLanguageModel
from .large_language_model import LargeLanguageModel
from .constants import ACTION_LOAD_LLM, ACTION_GENERATE_TEXT, ACTION_UNLOAD_LLM, is_llm, is_lvlm
from .request_validator import RequestValidator
from ros_lm_interfaces.srv import OpenLLMRequest


class RosLMServiceServer(Node):

    def __init__(self):
        super().__init__('ros_lm_service_server')
        self._service = self.create_service(
            OpenLLMRequest,
            'llm_generate_text',
            self.service_callback
        )
        
        self.loaded_models = dict()

    def service_callback(self, request, response):
        
        # Validate request
        request_validator = RequestValidator(self.get_logger(), request, response, self.loaded_models, self.loaded_tokenizers)
        if not request_validator.validate():
            return request_validator.get_error_response()

        # Execute the requested action
        if request.action == ACTION_LOAD_LLM:
            
            self.get_logger().info(f"Loading model {request.model_id}...")
            if self.load_model(request.model_id):
                response.status_code = 1
                response.status_message = f"Model {request.model_id} successfully loaded."
                response.generated_text = ""
            else:
                response.status_code = 0
                response.status_message = f"Failed to load model {request.model_id}."
                response.generated_text = ""
        
        elif request.action == ACTION_GENERATE_TEXT:
            
            self.get_logger().info(f"Generating text using model {request.model_id}...")
            generated_text = self.generate_text(request.model_id, request.prompt, request.images)
            response.status_code = 1
            response.status_message = "Text generated successfully."
            response.generated_text = generated_text

        elif request.action == ACTION_UNLOAD_LLM:

            self.get_logger().info(f"Unloading model {request.model_id}...")
            self.unload_model(request.model_id)
            response.status_code = 1
            response.status_message = f"Model {request.model_id} successfully unloaded."
            response.generated_text = ""

        return response

    def load_model(self, model_id: str):
        if is_llm(model_id):
            return self._load_llm(model_id)
        elif is_lvlm(model_id):
            return self._load_lvlm(model_id)
        else:
            raise ValueError(f"Model {model_id} is not LLM nor LVLM")

    def _load_llm(self, model_id: str):
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

            # Create and store LargeLanguageModel
            self.loaded_models[model_id] = LargeLanguageModel(model_id, tokenizer, model)

            self.get_logger().info(f"Large Language Model {model_id} and tokenizer successfully loaded.")
            return True
        
        except OSError as e:
            self.get_logger().error(f"Error loading model {model_id}: {e}")
            return False
    
    def _load_lvlm(self, model_id: str):
        try:
            # Load processor and model
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            # Load processor and model
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

            self.loaded_models[model_id] = LargeVisionLanguageModel(model_id, processor, model)

            self.get_logger().info(f"Large Vision-Language Model {model_id} and processor successfully loaded.")
            return True
        
        except OSError as e:
            self.get_logger().error(f"Error loading model {model_id}: {e}")
            return False
        
    def generate_text(self, model_id, prompt: str, images: list):
        
        model = self.loaded_models[model_id]
        if is_llm(model_id):
            return model.generate_text(prompt)
        elif is_lvlm(model_id):
            return model.generate_text(prompt, images)
        else:
            raise ValueError(f"Model {model_id} is not LLM nor LVLM")

    def unload_model(self, model_id: str):
        # Remove from loaded_models and loaded_tokenizers
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
        # Collect garbage
        gc.collect()
        # Empty CUDA cache
        torch.cuda.empty_cache()
        self.get_logger().info(f"Model {model_id} successfully unloaded.")

def main(args=None):
    rclpy.init(args=args)
    ros_lm_service_server = RosLMServiceServer()
    rclpy.spin(ros_lm_service_server)

if __name__ == '__main__':
    main()
