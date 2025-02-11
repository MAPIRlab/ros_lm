class LargeLanguageModel:

    def __init__(self, model_id : str, tokenizer, model):
        
        self._model_id = model_id
        self._tokenizer = tokenizer
        self._model = model

    def get_model_id(self,):
        return self._model_id()

    def generate_text(self, prompt: str):
        
        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")

        # Generate output
        output = self._model.generate(
            inputs.input_ids, 
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )

        # Decode generated text
        response_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return response_text