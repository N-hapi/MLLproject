from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load the cleaned data
data = pd.read_csv("cleaned_procrastination_tasks.csv")
print(data.columns)  # Ensure 'text' and 'label' columns are present

# Convert the 'label' to float for regression
data['label'] = data['label'].astype(float)

# Split data into training and evaluation sets
train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # Single output for regression

# Set the model to treat this as a regression task
model.config.problem_type = "regression"

# Tokenize the data
def preprocess_function(examples):
    texts = examples['text']
    return tokenizer(texts, padding="max_length", truncation=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=5,
    save_strategy="epoch",              # Save at the end of each epoch
    evaluation_strategy="epoch",        # Evaluate after each epoch
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True
)

# Define custom metrics for regression
def compute_metrics(pred):
    predictions = pred.predictions.squeeze()  # Remove extra dimensions
    labels = pred.label_ids
    mse = ((predictions - labels) ** 2).mean()
    return {"mse": mse}

# Initialize the Trainer with both training and evaluation datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save both model and tokenizer
trainer.save_model("./results")
tokenizer.save_pretrained("./results")  

print("Training complete. Model saved in './results'")