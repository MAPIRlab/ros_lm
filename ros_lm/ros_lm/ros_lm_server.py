import rclpy
from rclpy.node import Node
import gc
import torch

from ros_lm.ros_lm.ros_lm.language_model import LanguageModel
from ros_lm.ros_lm.ros_lm.response_utils import ResponseUtils

from .model_database import ModelDatabase
from .large_vision_language_model import LargeVisionLanguageModel
from .large_language_model import LargeLanguageModel
from .constants import ACTION_LOAD_LLM, ACTION_GENERATE_TEXT, ACTION_UNLOAD_LLM
from .request_validator import RequestValidator
from ros_lm_interfaces.srv import OpenLLMRequest

from ros_lm.ros_lm.ros_lm import constants


class RosLMServiceServer(Node):

    def __init__(self):
        # TODO: documentation
        super().__init__('ros_lm_service_server')
        self._service = self.create_service(
            OpenLLMRequest,
            'llm_generate_text',
            self.service_callback
        )
        
        self.loaded_models = dict()

    def service_callback(self, request, response):
        # TODO: documentation
        # Validate request
        request_validator = RequestValidator(self.get_logger(), request, response, self.loaded_models, self.loaded_tokenizers)
        if not request_validator.validate():
            return request_validator.get_error_response()

        # Execute the requested action
        if request.action == ACTION_LOAD_LLM:
            
            self.get_logger().info(f"Loading model {request.model_id}...")
            if self.load_model(request.model_id):
                response = ResponseUtils.create_response(response,
                                                         status_code=constants.STATUS_CODE_SUCCESS,
                                                         status_message = f"Model {request.model_id} successfully loaded into memory.")
            else:
                response = ResponseUtils.create_response(response,
                                                         status_code=constants.STATUS_CODE_ERROR,
                                                         status_message = f"Failed to load model {request.model_id} into memory.")
        
        elif request.action == ACTION_GENERATE_TEXT:
            
            self.get_logger().info(f"Generating text using model {request.model_id}...")
            params = {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p
            }

            generated_text = self.generate_text(request.model_id, request.prompt, params, request.images)

            response = ResponseUtils.create_response(response,
                                                     status_code=constants.STATUS_CODE_SUCCESS,
                                                     status_message = "Text generated successfully.",
                                                     generated_text = generated_text)

        elif request.action == ACTION_UNLOAD_LLM:

            self.get_logger().info(f"Unloading model {request.model_id}...")
            self.unload_model(request.model_id)
            response = ResponseUtils.create_response(response,
                                            status_code=constants.STATUS_CODE_SUCCESS,
                                            status_message = f"Model {request.model_id} successfully unloaded.")

        return response

    def load_model(self, model_id: str):
        # TODO: documentation

        if ModelDatabase.is_llm(model_id):
            result, model = LargeLanguageModel.create(model_id)
        elif ModelDatabase.is_llm(model_id):
            result, model = LargeVisionLanguageModel.create(model_id)

        if result:
            self.loaded_models[model_id] = model
        return result
        
    def generate_text(self, model_id, prompt: str, images: list, params: dict):
        # TODO: documentation
        model:LanguageModel = self.loaded_models[model_id]
        
        if len(images) == 0:
            return model.generate_text(prompt, params)
        else:
            return model.generate_text_with_images(prompt, images, params)


    def unload_model(self, model_id: str):
        # TODO: documentation
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
