import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from ros_lm.action import Count
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class CountActionServer(Node):
    def __init__(self):
        super().__init__('count_action_server')
        self._action_server = ActionServer(
            self,
            Count,
            'count',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup()  # Allow multi-threading
        )
        self.get_logger().info('Action server started.')

    def execute_callback(self, goal_handle):
        self.get_logger().info(f'Receiving goal: {goal_handle.request.target_count}')
        
        feedback_msg = Count.Feedback()
        for i in range(1, goal_handle.request.target_count + 1):
            feedback_msg.current_count = i
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {i}')
            
        goal_handle.succeed()
        result = Count.Result()
        result.success = True
        self.get_logger().info(f'Action completed successfully.')
        return result

def main(args=None):
    rclpy.init(args=args)
    action_server = CountActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(action_server)
    try:
        executor.spin()
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
