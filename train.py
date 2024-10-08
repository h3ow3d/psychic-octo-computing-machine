import os
import json
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG
logger = logging.getLogger(__name__)

# Define your model and tokenizer
model_name = 'gpt2'
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
                    texts.extend(data.get('text', []))
    return texts

# Load and prepare dataset
texts = load_json_files_from_directory('data/')
data = {'text': texts}
df = pd.DataFrame(data)
df['text'] = df['text'].astype(str)

# Split the data into training and evaluation datasets
train_df = df.sample(frac=0.9, random_state=42)  # 90% for training
eval_df = df.drop(train_df.index)  # Remaining 10% for evaluation

# Convert to datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

def preprocess_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

# Apply preprocessing to both training and evaluation datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Adjust as needed
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=50,  # Log every 50 steps
    evaluation_strategy="steps",  # Evaluate every 500 steps
    eval_steps=500,  # Evaluate every 500 steps
    save_strategy="steps",  # Save every 500 steps
    save_steps=500,
    report_to='tensorboard',  # Use TensorBoard for detailed logs
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    disable_tqdm=False,  # Ensure tqdm is enabled for progress bar
)

# Detect if GPU is available and use it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset  # Provide the evaluation dataset
)

# Train the model
logger.info("Starting training...")
trainer.train()
logger.info("Training completed.")

# Save the model
logger.info("Saving the model and tokenizer...")
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')
logger.info("Model and tokenizer saved.")
