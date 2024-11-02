import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the path to the trained model
model_dir = "./results"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Function to make predictions on new text
def predict_procrastination(task_texts):
    inputs = tokenizer(task_texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.numpy()  # Convert predictions to NumPy for easy handling

# Example input data: Replace with actual new data or load from a file
new_tasks = [
    "get medication  [Project: Health Management]",
    "university project  [Project: university]",
    
]

# Make predictions
predictions = predict_procrastination(new_tasks)

# Display predictions
for task, prediction in zip(new_tasks, predictions):
    print(f"Task: {task} - Procrastination Likelihood: {prediction}")
