import random
from rclpy.node import Node
import rclpy
from threading import Thread
from ros_lm.models.model_database import ModelDatabase
from ros_lm_interfaces.srv import OpenLLMRequest
from ros_lm.constants import ACTION_GENERATE_TEXT, ACTION_LOAD_LLM, ACTION_UNLOAD_LLM, STATUS_CODE_SUCCESS, BASE64_EXAMPLE_IMAGE

class SampleROSLMServiceClient(Node):

    def __init__(self):
        super().__init__("sample_ros_llm_service_client")

        self.client = self.create_client(OpenLLMRequest, "llm_generate_text")
        
        # Waiting for service
        while not self.client.wait_for_service(timeout_sec=10):
            self.get_logger().warning("Service is not available, waiting to check again...")

    def load_model_request(self, model_id: str):
        req = OpenLLMRequest.Request()
        req.action = ACTION_LOAD_LLM
        req.model_id = model_id
        self.get_logger().info("Load model request sent...")
        return self.client.call(req)
    
    def generate_text_request(self, model_id : str, prompt: str, images: list[str] = []):
        req = OpenLLMRequest.Request()
        req.action = ACTION_GENERATE_TEXT
        req.model_id = model_id
        req.prompt = prompt
        req.images = images
        self.get_logger().info(f"Generate text request sent (prompt = {prompt}, nº images = {len(images)})...")
        return self.client.call(req)
    
    def unload_model_request(self, model_id: str):
        req = OpenLLMRequest.Request()
        req.action = ACTION_UNLOAD_LLM
        req.model_id = model_id
        self.get_logger().info("Unload model request sent...")
        return self.client.call(req)

def main(args=None):
    rclpy.init(args=args)

    client_node = SampleROSLMServiceClient()

    spin_thread = Thread(target=rclpy.spin, args=(client_node,))
    spin_thread.start()

    logger = client_node.get_logger()
    
    llm_model_id = random.choice(ModelDatabase().get_llms())
    logger.info(f"Chosen LLM: {llm_model_id}")
    lvlm_model_id = random.choice(ModelDatabase().get_lvlms())
    logger.info(f"Chosen LVLM: {lvlm_model_id}")

    ############
    ## LLM tests
    ############
    
    # Load
    lm_res = client_node.load_model_request(llm_model_id)
    if lm_res.status_code == STATUS_CODE_SUCCESS:
        logger.info(f"Response: {lm_res.status_message}")
        
        prompt = ("Hello! How are you?")

        # Generate text
        gt_res = client_node.generate_text_request(model_id=llm_model_id, prompt=prompt)
        if gt_res.status_code == STATUS_CODE_SUCCESS:
            logger.info(f"Response: {gt_res.status_message} | {gt_res.generated_text}")

            # Unload model
            um_res = client_node.unload_model_request(llm_model_id)
            if um_res.status_code == STATUS_CODE_SUCCESS:
                logger.info(f"Response: {um_res.status_message}")
            else:
                logger.error(f"Response: {um_res.status_message}")
        else:
            logger.error(f"Response: {gt_res.status_message}")
    else:
        logger.error(f"Response: {lm_res.status_message}")

    #############
    ## LVLM tests
    #############

    # Load
    lm_res = client_node.load_model_request(lvlm_model_id)
    if lm_res.status_code == STATUS_CODE_SUCCESS:
        logger.info(f"Response: {lm_res.status_message}")
        
        prompt = "What is this animal? <image>"

        # Generate text
        gt_res = client_node.generate_text_request(model_id=lvlm_model_id, prompt=prompt, images=[BASE64_EXAMPLE_IMAGE])
        if gt_res.status_code == STATUS_CODE_SUCCESS:
            logger.info(f"Response: {gt_res.status_message} | {gt_res.generated_text}")

            # Unload model
            um_res = client_node.unload_model_request(lvlm_model_id)
            if um_res.status_code == STATUS_CODE_SUCCESS:
                logger.info(f"Response: {um_res.status_message}")
            else:
                logger.error(f"Response: {um_res.status_message}")
        else:
            logger.error(f"Response: {gt_res.status_message}")
    else:
        logger.error(f"Response: {lm_res.status_message}")

    client_node.destroy_node()
    rclpy.shutdown()

    spin_thread.join()

if __name__ == "__main__":
    main()