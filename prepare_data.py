import pandas as pd

# Load the CSV data
data = pd.read_csv("frogs.csv")  # Replace with the path to your raw CSV file

# Fill missing values in "Task name" and "Project Formula" with placeholders
data['Task name'] = data['Task name'].fillna("Unknown Task")
data['Project Formula'] = data['Project Formula'].fillna("Unknown Project")

# Combine "Task name" and "Project Formula" with an explicit task instruction
data['text'] = "Predict procrastination likelihood: " + data['Task name'] + " [Project: " + data['Project Formula'] + "]"

# Use the "Count" column as the "label" (likelihood of procrastination) and convert it to integer
data['label'] = "the higher the number the higher the likleyhood of rocrastination the number is:" +data['Count'].fillna(0).astype(int)

# Select only the required columns
cleaned_data = data[['text', 'label']]

# Save to a new CSV file
cleaned_data.to_csv("cleaned_procrastination_tasks.csv", index=False)

print("Data preparation complete. Saved as 'cleaned_procrastination_tasks.csv'")
