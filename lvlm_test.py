from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

import requests
from PIL import Image

image1 = Image.open("./image.png")
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

prompt1 = "USER: <image>\nWhat is this image?\nASSISTANT:"
prompt2 = "USER: <image>\nPlease describe this image\nASSISTANT:"
prompt3 = "How are you?"

inputs = processor(text=prompt1, images=image1, padding=True, return_tensors="pt").to("cuda")

for k,v in inputs.items():
  print(k,v.shape)

output = model.generate(**inputs, max_new_tokens=200)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print('generated_text', generated_text)
for text in generated_text:
    print(text.split("ASSISTANT:")[-1])