class ModelDatabase:
    
    # Types of models
    TYPE_LLM = "llm"
    TYPE_LVLM = "lvlm"

    # Models considered
    MODEL_LIST = {
        "meta-llama/Llama-3.1-8B-Instruct" : TYPE_LLM,
        "meta-llama/Llama-3.2-3B-Instruct" : TYPE_LLM,
        "deepseek-ai/deepseek-llm-7b-chat" : TYPE_LLM,
        "llava-hf/llava-1.5-7b-hf": TYPE_LVLM
    }

    @staticmethod
    def is_llm(model_id):
        return model_id in ModelDatabase.MODEL_LIST and ModelDatabase.MODEL_LIST[model_id] == ModelDatabase.TYPE_LLM
    
    @staticmethod
    def get_llms():
        return list(filter(lambda model_id : ModelDatabase.is_llm(model_id), ModelDatabase.MODEL_LIST.keys()))

    @staticmethod
    def is_lvlm(model_id):
        return model_id in ModelDatabase.MODEL_LIST and ModelDatabase.MODEL_LIST[model_id] == ModelDatabase.TYPE_LVLM
    
    @staticmethod
    def get_lvlms():
        return list(filter(lambda model_id : ModelDatabase.is_lvlm(model_id), ModelDatabase.MODEL_LIST.keys()))
    
    @staticmethod
    def exists(model_id):
        return model_id in ModelDatabase.MODEL_LIST