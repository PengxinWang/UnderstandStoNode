import os
import json
import matplotlib.pyplot as plt

def load_results_from_json(result_dir):
    results = {}
    for root, _, files in os.walk(result_dir):
        for file in files:
            if file.endswith(".json"):
                model_name = os.path.basename(root)
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if model_name not in results:
                        results[model_name] = data
    return results

def process_data(results):
    processed_data = {}
    for model_name, data in results.items():
        model_data = data[model_name]
        intensities = sorted(set(d['intensity'] for d in model_data))
        acc_data = {intensity: [] for intensity in intensities}
        ece_data = {intensity: [] for intensity in intensities}

        for entry in model_data:
            intensity = entry['intensity']
            acc_data[intensity].append(entry['acc'])
            ece_data[intensity].append(entry['ece'])

        processed_data[model_name] = {
            'intensities': intensities,
            'accuracies': [sum(acc_data[i])/len(acc_data[i]) for i in intensities],
            'eces': [sum(ece_data[i])/len(ece_data[i]) for i in intensities],
        }
    return processed_data

def plot_results(processed_data):
    markers = ['o', 's', '^', '*']
    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    for idx, (model_name, data) in enumerate(processed_data.items()):
        marker = markers[idx]
        plt.plot(data['intensities'], data['accuracies'], marker=marker, label=model_name)
    plt.xlabel('Corruption Intensity')
    plt.ylabel('Accuracy')
    plt.title('Accuracy (up)')
    plt.legend()
    plt.grid(True)

    # Plot ECE
    plt.subplot(1, 2, 2)
    for idx, (model_name, data) in enumerate(processed_data.items()):
        marker = markers[idx]
        plt.plot(data['intensities'], data['eces'], marker=marker, label=model_name)
    plt.xlabel('Corruption Intensity')
    plt.ylabel('ECE')
    plt.title('ECE (down)')
    plt.legend()
    plt.grid(True)

    plt.suptitle('Evaluation Results Across Models')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'./plot_results/quantitative_result.png')
    plt.show()

result_dir = './results'

# Load and process results
results = load_results_from_json(result_dir)
processed_data = process_data(results)

# Plot the results
plot_results(processed_data)
