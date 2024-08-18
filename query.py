from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the fine-tuned model and tokenizer
model_name = './fine-tuned-model'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if a GPU is available and move model to appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def generate_response(prompt):
    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device as the model
    
    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,  # Use beam search for better quality
            no_repeat_ngram_size=2,  # Avoid repetition of n-grams
            early_stopping=True
        )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    while True:
        prompt = input("Enter your question (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        response = generate_response(prompt)
        print("Response:", response)
