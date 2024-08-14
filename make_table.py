import json
import numpy as np
import pandas as pd

# Load the JSON file
file_path = './results/evaluation_results.json'  # Replace with your file path if different
with open(file_path, 'r') as f:
    data = json.load(f)

# Initialize a list to store the results
results = []

# Iterate through each model in the data
for model_name, severity_data in data.items():
    acc_list = []
    ece_list = []
    nll_list = []
    
    # Iterate through each severity level
    for severity, metrics in severity_data.items():
        acc_list.append(metrics[0]['acc'])
        ece_list.append(metrics[0]['ece'])
        nll_list.append(metrics[0]['nll'])
    
    # Calculate mean across all severities
    mean_acc = sum(acc_list) / len(acc_list)
    mean_ece = sum(ece_list) / len(ece_list)
    mean_nll = sum(nll_list) / len(nll_list)
    
    # Store the results in a dictionary
    results.append({
        'Model': model_name,
        'Mean Accuracy': mean_acc,
        'Mean ECE': mean_ece,
        'Mean NLL': mean_nll
    })

# Convert the results to a DataFrame
df = pd.DataFrame(results)

# Generate the table in markdown format
markdown_table = df.to_markdown(index=False)

# Print the markdown table
print(markdown_table)
