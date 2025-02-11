class LargeVisionLanguageModel:

    def __init__(self, model_id : str, processor, model):
        
        self._model_id = model_id
        self.processor = processor
        self._model = model

    def get_model_id(self,):
        return self._model_id()

    def generate_text(self, prompt: str, images: list):
        
        # Process input
        inputs = self.processor(text=prompt, images=images, padding=True, return_tensors="pt").to("cuda")

        # Generate output
        output = self.model.generate(**inputs, max_new_tokens=200)
        
        # Decode generated text
        return self.processor.batch_decode(output, skip_special_tokens=True).join(" ")