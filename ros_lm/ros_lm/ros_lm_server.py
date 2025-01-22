import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.node import Node
from transformers import AutoTokenizer, AutoModelForCausalLM


from ros_lm_interfaces.action import LLMGenerateText

class RosLMActionServer(Node):

    # Available action types
    ACTION_LOAD_LLM = 1
    ACTION_GENERATE_TEXT = 2

    # Model Ids List
    MODEL_LLAMA_2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
    MODEL_IDS = (MODEL_LLAMA_2_7B_CHAT)

    def __init__(self):
        super().__init__('ros_lm_action_server')
        self._action_server = ActionServer(
            self,
            LLMGenerateText,
            'llm_generate_text',
            self.execute_callback,
            goal_callback=self.goal_callback)
        
        self.loaded_tokenizers = dict()
        self.loaded_models = dict()

    def is_model_loaded(self, model_id: str):
        return model_id in self.loaded_models and model_id in self.loaded_tokenizers
    
    def get_tokenizer_and_model(self, model_id: str):
        return self.loaded_tokenizers[model_id], self.loaded_models[model_id]

    def goal_callback(self, goal_request):
        
        # action does not exist
        if goal_request.action not in (self.ACTION_LOAD_LLM, self.ACTION_GENERATE_TEXT):
            self.get_logger().debug(f"Goal rejected because action {goal_request.action} is not supported")
            return GoalResponse.REJECT

        # model_id does not exist
        if goal_request.model_id not in self.MODEL_IDS:
            self.get_logger().debug(f"Goal rejected because model_id {goal_request.model_id} is not supported")
            return GoalResponse.REJECT
        
        # load model already loaded
        if goal_request.action == self.ACTION_LOAD_LLM and self.is_model_loaded(goal_request.model_id):
            self.get_logger().debug(f"Goal rejected because model {goal_request.model_id} is already loaded")
            return GoalResponse.REJECT
        
        # generate text for model not loaded
        if goal_request.action == self.ACTION_GENERATE_TEXT and not self.is_model_loaded(goal_request.model_id):
            self.get_logger().debug(f"Goal rejected because model {goal_request.model_id} is not loaded")
            return GoalResponse.REJECT
        
        return GoalResponse.ACCEPT

    def execute_callback(self, goal_handle):

        # Get request fields
        request_action = goal_handle.request.action
        request_model_id = goal_handle.request.model_id
        request_prompt = goal_handle.request.prompt

        # Build result
        result = LLMGenerateText.Result() 

        if request_action == self.ACTION_LOAD_LLM:
            self.get_logger().info(f"Executing action = {self.ACTION_LOAD_LLM} (load LLM) model_id = {goal_handle.request.model_id}...")

            # Load LLM model
            ok = self.load_model_and_tokenizer(request_model_id)
            if not ok:
                return
        
        elif request_action == self.ACTION_GENERATE_TEXT:
            self.get_logger().info(f"Executing action = {self.ACTION_GENERATE_TEXT} (generate text) model_id = {goal_handle.request.model_id}...")
            
            # Generate text
            messages = [{"role": "user", "content": request_prompt}]
            result.response = self.loaded_models[request_model_id](messages)

        else:
            self.get_logger().error(f"Invalid action = {goal_handle.request.action} requested.")

        goal_handle.succeed()

        return result

    def load_model_and_tokenizer(self, model_id: str):
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")

            # Save tokenizer and model
            self.loaded_tokenizers[model_id] = tokenizer
            self.loaded_models[model_id] = model

            self.get_logger().info(f"Model {model_id} and tokenizer successfully loaded!")
            return True
        except OSError:
            return False
        
    def generate_text(self, model_id: str, prompt: str):
        
        # Get tokenizer and model
        tokenizer, model = self.get_tokenizer_and_model(model_id)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate text
        output = model.generate(
            inputs.input_ids, 
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )

        # De-tokenize
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        self.get_logger().debug(f"Model {model_id} generated text: {response}")
        return response

def main(args=None):
    
    rclpy.init(args=args)

    ros_lm_action_server = RosLMActionServer()

    rclpy.spin(ros_lm_action_server)

if __name__ == '__main__':
    main()