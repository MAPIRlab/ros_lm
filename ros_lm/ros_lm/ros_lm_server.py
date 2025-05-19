import rclpy
from rclpy.node import Node
import gc
import torch

from ros_lm.models.language_model import LanguageModel

from ros_lm.response_utils import ResponseUtils

from ros_lm.models.model_database import ModelDatabase
from ros_lm.models.llm.lvlm.large_vision_language_model import LargeVisionLanguageModel
from ros_lm.models.llm.large_language_model import LargeLanguageModel
from ros_lm.request_validator import RequestValidator
import ros_lm.constants as constants

from ros_lm_interfaces.srv import OpenLLMRequest



class RosLMServiceServer(Node):
    """ROS 2 service server for handling LLM requests."""

    def __init__(self):
        """Initializes the ROS 2 service server for LLM processing."""
        super().__init__('ros_lm_service_server')
        self._service = self.create_service(
            OpenLLMRequest,
            'llm_generate_text',
            self.service_callback
        )
        self.loaded_models = dict()

    def service_callback(self, request, response):
        """Handles incoming service requests and executes the corresponding actions."""
        request_validator = RequestValidator(
            self.get_logger(), request, response, self.loaded_models
        )
        if not request_validator.validate():
            return request_validator.get_error_response()

        if request.action == constants.ACTION_LOAD_LLM:
            self.get_logger().info(f"Loading model {request.model_id} into memory...")
            if self.load_model(request.model_id):
                response = ResponseUtils.create_response(
                    response,
                    status_code=constants.STATUS_CODE_SUCCESS,
                    status_message=f"Model {request.model_id} successfully loaded into memory."
                )
                self.get_logger().info(f"Model {request.model_id} successfully loaded into memory.")
            else:
                response = ResponseUtils.create_response(
                    response,
                    status_code=constants.STATUS_CODE_ERROR,
                    status_message=f"Failed to load model {request.model_id} into memory."
                )
                self.get_logger().error(f"Failed to load model {request.model_id} into memory.")

        elif request.action == constants.ACTION_GENERATE_TEXT:
            self.get_logger().info(f"Generating text using model {request.model_id}...")
            params = {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p
            }

            generated_text = self.generate_text(
                request.model_id, request.prompt, request.images, params
            )

            response = ResponseUtils.create_response(
                response,
                status_code=constants.STATUS_CODE_SUCCESS,
                status_message="Text generated successfully.",
                generated_text=generated_text
            )

        elif request.action == constants.ACTION_UNLOAD_LLM:
            self.get_logger().info(f"Unloading model {request.model_id}...")
            self.unload_model(request.model_id)
            response = ResponseUtils.create_response(
                response,
                status_code=constants.STATUS_CODE_SUCCESS,
                status_message=f"Model {request.model_id} successfully unloaded."
            )

        return response

    def load_model(self, model_id: str):
        """Loads the specified model into memory."""
        if ModelDatabase.is_llm(model_id):
            result, model = LargeLanguageModel.create(model_id)
        elif ModelDatabase.is_lvlm(model_id):
            result, model = LargeVisionLanguageModel.create(model_id)

        if result:
            self.loaded_models[model_id] = model
        return result

    def generate_text(self, model_id: str, prompt: str, images: list, params: dict):
        """Generates text using the specified model, with optional image inputs."""
        model: LanguageModel = self.loaded_models[model_id]
        if len(images) == 0:
            return model.generate_text(prompt, params)
        else:
            return model.generate_text_with_images(prompt, images, params)

    def unload_model(self, model_id: str):
        """Unloads the specified model from memory and clears cache."""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
        gc.collect()
        torch.cuda.empty_cache()
        self.get_logger().info(f"Model {model_id} successfully unloaded.")


def main(args=None):
    """Entry point for running the ROS 2 service server."""
    rclpy.init(args=args)
    ros_lm_service_server = RosLMServiceServer()
    rclpy.spin(ros_lm_service_server)


if __name__ == '__main__':
    main()
