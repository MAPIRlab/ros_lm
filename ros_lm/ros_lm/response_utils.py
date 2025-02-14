
from ros_lm.ros_lm.ros_lm import constants


class ResponseUtils:

    @staticmethod
    def create_response(response, status_code: int = constants.STATUS_CODE_ERROR, status_message: str = "", generated_text: str = ""):
        # TODO: documentation
        response.status_code = status_code
        response.status_message = status_message
        response.generated_text = generated_text
        return response