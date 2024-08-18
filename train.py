import os
import json
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
import logging

# Configure Python logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define your model and tokenizer
model_name = 'gpt2'  # You can choose a different pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if tokenizer has a pad token, if not, add one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Prepare dataset from JSON files in 'data/' directory
def load_json_files_from_directory(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list):
                    texts.extend(data)
                elif isinstance(data, dict):
                    # Adjust the key if your JSON structure is different
                    texts.extend(data.get('text', []))
    return texts

# Load and prepare dataset
texts = load_json_files_from_directory('data/')
data = {'text': texts}
df = pd.DataFrame(data)

# Ensure all entries in the 'text' column are strings
df['text'] = df['text'].astype(str)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

def preprocess_function(examples):
    # Tokenize the input and include labels
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    # Copy input_ids to labels
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

dataset = dataset.map(preprocess_function, batched=True)

# Set up training arguments with detailed logging
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=10,  # Log every 10 steps
    report_to='none',  # Disable reporting to avoid issues with some logging systems
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save model checkpoints at the end of each epoch
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model='loss',  # The metric to use to evaluate the best model
)

# Detect if GPU is available and use it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
try:
    trainer.train()
except Exception as e:
    logger.error(f"An error occurred during training: {e}")

# Save the model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')
