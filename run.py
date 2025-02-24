import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the dataset
#dataset = load_dataset("deepmind/code_contests")

# Example usage
def generate_code(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    print("Inputs", inputs)
    attention_mask =  inputs.attention_mask
    print("Attention Mask", attention_mask)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=2,
        do_sample=True,
        top_k=150,
        top_p=.95,
        temperature=10.7
    )
    print("Outputs", outputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model
prompt = "How are you feeling?"
generated_code = generate_code(prompt)
print("Response: ", generated_code)