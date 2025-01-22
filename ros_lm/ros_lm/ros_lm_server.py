import rclpy
from rclpy.node import Node
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import torch

from ros_lm_interfaces.srv import OpenLLMRequest


class RosLMServiceServer(Node):

    # Available service action types
    ACTION_LOAD_LLM = 1
    ACTION_GENERATE_TEXT = 2
    ACTION_UNLOAD_LLM = 3

    # Model Ids List
    MODEL_LLAMA_3DOT1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_LLAMA_3DOT2_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"
    MODEL_IDS = (MODEL_LLAMA_3DOT1_8B_INSTRUCT, MODEL_LLAMA_3DOT2_3B_INSTRUCT)

    def __init__(self):
        super().__init__('ros_lm_service_server')
        self._service = self.create_service(
            OpenLLMRequest,
            'llm_generate_text',
            self.service_callback
        )
        
        self.loaded_tokenizers = dict()
        self.loaded_models = dict()

    def is_model_loaded(self, model_id: str):
        return model_id in self.loaded_models and model_id in self.loaded_tokenizers
    
    def get_tokenizer_and_model(self, model_id: str):
        return self.loaded_tokenizers[model_id], self.loaded_models[model_id]

    def service_callback(self, request, response):
        
        # Validate requested action
        if request.action not in (self.ACTION_LOAD_LLM, self.ACTION_GENERATE_TEXT, self.ACTION_UNLOAD_LLM):
            self.get_logger().error(f"Service request rejected: action {request.action} is not supported")
            response.status_code = 0
            response.status_message = "Error: Unsupported action."
            response.generated_text = ""
            return response

        # Validate model ID
        if request.model_id not in self.MODEL_IDS:
            self.get_logger().error(f"Service request rejected: model_id {request.model_id} is not supported")
            response.status_code = 0
            response.status_message = "Error: Unsupported model ID."
            response.generated_text = ""
            return response
        
        # Check if model is already loaded for ACTION_LOAD_LLM
        if request.action == self.ACTION_LOAD_LLM and self.is_model_loaded(request.model_id):
            self.get_logger().info(f"Model {request.model_id} is already loaded")
            response.status_code = 1
            response.status_message = f"Model {request.model_id} is already loaded."
            response.generated_text = ""
            return response
        
        # Check if model is not loaded for ACTION_GENERATE_TEXT and ACTION_UNLOAD_LLM
        if request.action in (self.ACTION_GENERATE_TEXT, self.ACTION_UNLOAD_LLM) and not self.is_model_loaded(request.model_id):
            self.get_logger().error(f"Service request rejected: model {request.model_id} is not loaded")
            response.status_code = 0
            response.status_message = "Error: Model is not loaded."
            response.generated_text = ""
            return response
        
        # Execute the requested action
        if request.action == self.ACTION_LOAD_LLM:
            self.get_logger().info(f"Loading model {request.model_id}...")
            if self.load_model_and_tokenizer(request.model_id):
                response.status_code = 1
                response.status_message = f"Model {request.model_id} successfully loaded."
                response.generated_text = ""
            else:
                response.status_code = 0
                response.status_message = f"Failed to load model {request.model_id}."
                response.generated_text = ""
        
        elif request.action == self.ACTION_GENERATE_TEXT:
            
            self.get_logger().info(f"Generating text using model {request.model_id}...")
            generated_text = self.generate_text(request.model_id, request.prompt)
            response.status_code = 1
            response.status_message = "Text generated successfully."
            response.generated_text = generated_text

        elif request.action == self.ACTION_UNLOAD_LLM:

            self.get_logger().info(f"Unloading model {request.model_id}...")
            self.unload_model_and_tokenizer(request.model_id)
            response.status_code = 1
            response.status_message = f"Model {request.model_id} successfully unloaded."
            response.generated_text = ""

        return response

    def load_model_and_tokenizer(self, model_id: str):
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
        
    def unload_model_and_tokenizer(self, model_id: str):
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
        if model_id in self.loaded_tokenizers:
            del self.loaded_tokenizers[model_id]
        gc.collect()
        torch.cuda.empty_cache()
        self.get_logger().info(f"Model {model_id} and tokenizer successfully unloaded.")
        
    def generate_text(self, model_id: str, prompt: str):
        
        # Retrieve tokenizer and model
        tokenizer, model = self.get_tokenizer_and_model(model_id)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate text output
        output = model.generate(
            inputs.input_ids, 
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
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
