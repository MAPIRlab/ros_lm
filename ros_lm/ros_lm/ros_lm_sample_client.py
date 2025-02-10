from rclpy.node import Node
import rclpy
from threading import Thread
from ros_lm_interfaces.srv import OpenLLMRequest
from .constants import ACTION_GENERATE_TEXT, ACTION_LOAD_LLM, ACTION_UNLOAD_LLM

class SampleROSLMServiceClient(Node):

    def __init__(self):
        super().__init__("sample_ros_llm_service_client")

        self.client = self.create_client("ros_lm_service_server")
        # Waiting for service
        while not self.client.wait_for_service(timeout_sec=10):
            self.get_logger().warning("Service is not available, waiting to check again...")

    def load_model_request(self, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        req = OpenLLMRequest.Request()
        req.action = ACTION_LOAD_LLM
        req.model_id = model
        self.get_logger().info("Loading model...")
        return self.client.call(req)
    
    def generate_text_request(self, prompt: str, model : str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        req = OpenLLMRequest.Request()
        req.action = ACTION_GENERATE_TEXT
        req.model_id = model
        req.prompt = prompt
        self.get_logger().info("Waiting for response...")
        return self.client.call(req)
    
    def unload_model_request(self, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        req = OpenLLMRequest.Request()
        req.action = ACTION_UNLOAD_LLM
        req.model_id = model
        self.get_logger().info("Unloading model...")
        return self.client.call(req)



def main(args=None):
    rclpy.init(args=args)

    client = SampleROSLMServiceClient()

    spin_thread = Thread(target=rclpy.spin, args=(client,))
    spin_thread.start()

    logger = client.get_logger()
    lm_res = client.load_model_request()
    
    if lm_res.status_code == 1:
        logger.info(lm_res.status_message)
        prompt = input("Introduce a prompt: ")
        gt_res = client.generate_text_request(prompt=prompt)

        if gt_res.status_code == 1:
            logger.info(um_res.status_message)
            logger.info(f"Model response: {gt_res.generated_text}")

            um_res = client.unload_model_request()
            if um_res.status_code == 1:
                logger.info(um_res.status_message)
            else:
                logger.error(um_res.status_message)
        else:
            logger.error(gt_res.status_message)
    else:
        logger.error(lm_res.status_message)

    client.destroy_client()
    rclpy.shutdown()

    spin_thread.join()

if __name__ == "__main__":
    main()