

from . import constants
from .model_database import ModelDatabase

class RequestValidator:

    def __init__(self, logger, request, response, loaded_models):
        self.logger = logger
        self.request = request
        self.error_response = response
        self.loaded_models = loaded_models

    def is_model_loaded(self, model_id: str):
        return model_id in self.loaded_models

    def validate(self):
        
        # Validate requested action
        if self.request.action not in [constants.ACTION_LOAD_LLM, constants.ACTION_GENERATE_TEXT, constants.ACTION_UNLOAD_LLM]:
            self.logger.error(f"Service request rejected: action {self.request.action} is not supported")
            self.error_response.status_code = constants.STATUS_CODE_ERROR
            self.error_response.status_message = "Error: Unsupported action."
            self.error_response.generated_text = ""
            return False

        # Validate model ID
        if not ModelDatabase.exists(self.request.model_id):
            self.logger.error(f"Service request rejected: model_id {self.request.model_id} is not supported")
            self.error_response.status_code = constants.STATUS_CODE_ERROR
            self.error_response.status_message = "Error: Unsupported model ID."
            self.error_response.generated_text = ""
            return False
        
        # Check if model is already loaded for ACTION_LOAD_LLM
        if self.request.action == constants.ACTION_LOAD_LLM and self.is_model_loaded(self.request.model_id):
            self.logger.info(f"Model {self.request.model_id} is already loaded")
            self.error_response.status_code = constants.STATUS_CODE_ERROR
            self.error_response.status_message = f"Model {self.request.model_id} is already loaded."
            self.error_response.generated_text = ""
            return False
        
        # Check if model is not loaded for ACTION_GENERATE_TEXT and ACTION_UNLOAD_LLM
        if self.request.action in [constants.ACTION_GENERATE_TEXT, constants.ACTION_UNLOAD_LLM] and not self.is_model_loaded(self.request.model_id):
            self.logger.error(f"Service request rejected: model {self.request.model_id} is not loaded")
            self.error_response.status_code = constants.STATUS_CODE_ERROR
            self.error_response.status_message = "Error: Model is not loaded."
            self.error_response.generated_text = ""
            return False
        
        return True

    def get_error_response(self):
        return self.error_response
        