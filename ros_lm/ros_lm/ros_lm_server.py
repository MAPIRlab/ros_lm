import rclpy
from rclpy.node import Node
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import torch

from .constants import ACTION_LOAD_LLM, ACTION_GENERATE_TEXT, ACTION_UNLOAD_LLM
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
        
        self.loaded_tokenizers = dict()
        self.loaded_models = dict()

    def service_callback(self, request, response):
        
        # Validate request
        request_validator = RequestValidator(self.get_logger(), request, response, self.loaded_models, self.loaded_tokenizers)
        if not request_validator.validate():
            return request_validator.get_error_response()

        # Execute the requested action
        if request.action == ACTION_LOAD_LLM:
            self.get_logger().info(f"Loading model {request.model_id}...")
            if self.load_tokenizer_and_model(request.model_id):
                response.status_code = 1
                response.status_message = f"Model {request.model_id} successfully loaded."
                response.generated_text = ""
            else:
                response.status_code = 0
                response.status_message = f"Failed to load model {request.model_id}."
                response.generated_text = ""
        
        elif request.action == ACTION_GENERATE_TEXT:
            self.get_logger().info(f"Generating text using model {request.model_id}...")
            generated_text = self.generate_text(request.model_id, request.prompt, request)
            response.status_code = 1
            response.status_message = "Text generated successfully."
            response.generated_text = generated_text

        elif request.action == ACTION_UNLOAD_LLM:

            self.get_logger().info(f"Unloading model {request.model_id}...")
            self.unload_tokenizer_and_model(request.model_id)
            response.status_code = 1
            response.status_message = f"Model {request.model_id} successfully unloaded."
            response.generated_text = ""

        return response

    def load_tokenizer_and_model(self, model_id: str):
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

            # Store tokenizer and model
            self.loaded_tokenizers[model_id] = tokenizer
            self.loaded_models[model_id] = model

            self.get_logger().info(f"Model {model_id} and tokenizer successfully loaded.")
            return True
        except OSError as e:
            self.get_logger().error(f"Error loading model {model_id}: {e}")
            return False
        
    def unload_tokenizer_and_model(self, model_id: str):
        # Remove from loaded_models and loaded_tokenizers
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
        if model_id in self.loaded_tokenizers:
            del self.loaded_tokenizers[model_id]
        # Collect garbage
        gc.collect()
        # Empty CUDA cache
        torch.cuda.empty_cache()
        self.get_logger().info(f"Model {model_id} and tokenizer successfully unloaded.")
        
    def get_tokenizer_and_model(self, model_id: str):
        return self.loaded_tokenizers[model_id], self.loaded_models[model_id]

    def generate_text(self, model_id: str, prompt: str, parameters: dict):
        
        # Retrieve tokenizer and model
        tokenizer, model = self.get_tokenizer_and_model(model_id)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate text output
        output = model.generate(
            inputs.input_ids, 
            max_length=parameters['max_length'],
            num_return_sequences=1,
            temperature=parameters['temperature'],
            top_k=parameters['top_k'],
            top_p=parameters['top_p'],
            do_sample=True
        )

        # Decode generated text
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        self.get_logger().info(f"Generated text: {response_text}")
        return response_text


def main(args=None):
    rclpy.init(args=args)
    ros_lm_service_server = RosLMServiceServer()
    rclpy.spin(ros_lm_service_server)


if __name__ == '__main__':
    main()
