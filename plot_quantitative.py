import json
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# model_name_list = ['resnet18_v0', 'resnet18_v1', 'resnet18_v2', 'resnet18_v3']
model_name_list = ['resnet18_v0', 'storesnet18_v1', 'resnet18_v1', 'resnet18_v2', 'resnet18_v3']
dataset_name = 'CIFAR10'
result_file = './results/evaluation_results.json'
# result_file = './results/evaluation_results.json'

def load_results_from_json(result_file):
    with open(result_file, 'r') as f:
        results = json.load(f)
    return results

def process_data(results):
    processed_data = {}
    for model_name, intensities_data in results.items():
        if model_name not in model_name_list:
            continue
        processed_data[model_name] = {
            'intensities': [],
            'accuracies': [],
            'eces': [],
            'nlls': [],
        }
        for intensity, metrics_list in intensities_data.items():
            # Since there's only one entry per intensity, we don't need to average
            metrics = metrics_list[0]
            processed_data[model_name]['intensities'].append(int(intensity))
            processed_data[model_name]['accuracies'].append(metrics['acc'])
            processed_data[model_name]['eces'].append(metrics['ece'])
            processed_data[model_name]['nlls'].append(metrics['nll'])

    return processed_data

def plot_results(processed_data):
    plt.figure(figsize=(18, 6))  # Adjusted to fit three subplots

    # Assign markers based on model name
    markers = {
        'resnet': 'o',
        'storesnet': 's',
        'v1': 's',
    }

    # Plot Accuracy
    plt.subplot(1, 3, 1)
    for model_name, data in processed_data.items():
        marker = markers['v1'] if 'v1' in model_name else markers['storesnet'] if 'storesnet' in model_name else markers['resnet']
        plt.plot(data['intensities'], data['accuracies'], marker=marker, label=model_name)
    plt.xlabel('Corruption Intensity')
    plt.ylabel('acc')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot ECE
    plt.subplot(1, 3, 2)
    for model_name, data in processed_data.items():
        marker = markers['v1'] if 'v1' in model_name else markers['storesnet'] if 'storesnet' in model_name else markers['resnet']
        plt.plot(data['intensities'], data['eces'], marker=marker, label=model_name)
    plt.xlabel('Corruption Intensity')
    plt.ylabel('ECE')
    plt.title('Expected Calibration Error')
    plt.legend()
    plt.grid(True)

    # Plot NLL
    plt.subplot(1, 3, 3)
    for model_name, data in processed_data.items():
        marker = markers['v1'] if 'v1' in model_name else markers['storesnet'] if 'storesnet' in model_name else markers['resnet']
        plt.plot(data['intensities'], data['nlls'], marker=marker, label=model_name)
    plt.xlabel('Corruption Intensity')
    plt.ylabel('NLL')
    plt.title('Negative Log-Likelihood')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'Evaluation Results on {dataset_name}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'./plot_results/quantitative_result.png')
    plt.show()

def generate_results_table(processed_data, output_file='./plot_results/evaluation_results.csv'):
    tables = []
    for model_name, data in processed_data.items():
        table = pd.DataFrame({
            'Model': model_name,
            'Intensity': data['intensities'],
            'Accuracy': data['accuracies'],
            'ECE': data['eces'],
            'NLL': data['nlls']
        })
        tables.append(table)

    full_table = pd.concat(tables)
    full_table.to_csv(output_file, index=False)
    print(f"Results table saved to {output_file}")

# Load and process results
results = load_results_from_json(result_file)
processed_data = process_data(results)

plot_results(processed_data)
# generate_results_table(processed_data)