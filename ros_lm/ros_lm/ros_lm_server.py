import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.node import Node
from transformers import pipeline


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
        
        self.loaded_models = dict()

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
        if goal_request.action == self.ACTION_LOAD_LLM and goal_request.model_id in self.loaded_models:
            self.get_logger().debug(f"Goal rejected because model {goal_request.model_id} is already loaded")
            return GoalResponse.REJECT
        
        # generate text for model not loaded
        if goal_request.action == self.ACTION_GENERATE_TEXT and goal_request.model_id not in self.loaded_models:
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
            ok = self.load_model(request_model_id)
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

    def load_model(self, model_id: str):
        try:
            pipe = pipeline("text-generation", model=model_id)
            self.loaded_models[model_id] = pipe
            self.get_logger().info(f"Model {model_id} successfully loaded!")
            return True
        except OSError:
            return False

def main(args=None):
    rclpy.init(args=args)

    ros_lm_action_server = RosLMActionServer()

    rclpy.spin(ros_lm_action_server)

if __name__ == '__main__':
    main()