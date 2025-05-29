from ros_lm.models.language_model import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class LargeLanguageModel(LanguageModel):
    """A wrapper for a large language model with text generation capabilities."""

    def __init__(self, model_id: str, tokenizer, model):
        """Initializes the model with a tokenizer and a pre-trained model."""
        self._model_id = model_id
        self._tokenizer = tokenizer
        self._model = model

    def get_model_id(self):
        """Returns the model identifier."""
        return self._model_id

    def generate_text(self, prompt: str, params: dict):
        """Generates text based on the given prompt and parameters."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        output = self._model.generate(
            inputs.input_ids, 
            max_length=params["max_length"],
            num_return_sequences=1,
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id # To avoid warning: The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
        )
        return self._tokenizer.decode(output[0], skip_special_tokens=True)
    
    def generate_text_with_images(self, prompt, images, params):
        return super().generate_text_with_images(prompt, images, params)

    @staticmethod
    def create(model_id: str):
        """Creates a LargeLanguageModel instance with the specified model ID."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
            return True, LargeLanguageModel(model_id, tokenizer, model)
        except OSError:
            return False, None
