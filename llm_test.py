from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")

# Define the prompt for text generation
prompt = "Once upon a time in a distant kingdom, there was a"

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    inputs.input_ids, 
    max_length=200,   # Adjust the max output length
    num_return_sequences=1,  # Number of generated sequences
    temperature=0.7,  # Sampling temperature (lower = more deterministic)
    top_k=50,         # Top-k sampling (set to 0 for no filtering)
    top_p=0.9,        # Nucleus sampling (set to 1.0 for no filtering)
    do_sample=True    # Enable sampling
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:\n", generated_text)