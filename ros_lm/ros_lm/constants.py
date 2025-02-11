
# Constants
ACTION_LOAD_LLM = 1
ACTION_GENERATE_TEXT = 2
ACTION_UNLOAD_LLM = 3

STATUS_CODE_ERROR = 0
STATUS_CODE_SUCCESS = 1

# Types of models
TYPE_LLM = "llm"
TYPE_LVLM = "lvlm"

# Models
MODEL_LLAMA_3DOT1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_LLAMA_3DOT2_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"

MODEL_LIST = {
    MODEL_LLAMA_3DOT1_8B_INSTRUCT : TYPE_LLM,
    MODEL_LLAMA_3DOT2_3B_INSTRUCT : TYPE_LLM
}

def is_llm(model_id):
    return model_id in MODEL_LIST and MODEL_LIST[model_id] == TYPE_LLM

def is_lvlm(model_id):
    return model_id in MODEL_LIST and MODEL_LIST[model_id] == TYPE_LVLM

# Models