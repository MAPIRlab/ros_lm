from .language_model import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class LargeLanguageModel(LanguageModel):

    # TODO: type of tokenizer and model
    def __init__(self, model_id : str, tokenizer, model):
        # TODO: documentation

        self._model_id = model_id
        self._tokenizer = tokenizer
        self._model = model

    def get_model_id(self,):
        # TODO: documentation

        return self._model_id()

    def generate_text(self, prompt: str, params: dict):
        # TODO: documentation   
        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")

        # Generate output
        output = self._model.generate(
            inputs.input_ids, 
            max_length=params["max_length"],
            num_return_sequences=1,
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            do_sample=True
        )

        # Decode generated text
        response_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return response_text
    
    @staticmethod
    def create(self, model_id: str):
        # TODO: documentation
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

            return True, LargeLanguageModel(model_id, tokenizer, model)
        
        except OSError as e:
            return False, None