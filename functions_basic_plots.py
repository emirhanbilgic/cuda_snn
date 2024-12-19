# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:10:06 2024

@author: ege-demir
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List
import numpy as np
import os
from collections import Counter
from functions_basic import load_synapse_attributes, normalize_to_grayscale


#PLOTS PLOTS PLOTS PLOTS PLOTS PLOTS PLOTS PLOTS PLOTS PLOTS PLOTS PLOTS PLOTS PLOTS
def plot_accuracies_by_batch(batch_accuracies: List[float], output_dir: str) -> None:
    batch_numbers = list(range(1, len(batch_accuracies) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(batch_numbers, batch_accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel("Batch Number")
    plt.ylabel("Batch (Training) Accuracy (%)")
    plt.title("Training Accuracy for Each Prediction Batch")
    
    # Use MaxNLocator for x-axis ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins='auto'))
    
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(f"{output_dir}/batch_training_accuracy_by_batch_number.png")
    plt.show()

def plot_accuracies_by_seen_images(batch_accuracies: List[float], label_predict_range: int, output_dir: str) -> None:
    
    cumulative_images_seen = [((idx + 1) * label_predict_range) + label_predict_range for idx in range(len(batch_accuracies))]

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_images_seen, batch_accuracies, marker='o', linestyle='-', color='g')
    plt.xlabel("Cumulative Images Seen")
    plt.ylabel("Training Accuracy (%)")
    plt.title("Training Accuracy for Each Prediction Batch per Seen Images")
    
    # Use MaxNLocator for x-axis ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins='auto'))
    
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(f"{output_dir}/batch_training_accuracy_by_seen_images.png")
    plt.show()

def plot_cumulative_accuracy(cumulative_accuracies: List[float], image_indexes_in_loop: List[int], output_dir: str) -> None:
    """
    Plots cumulative accuracies with respect to seen examples, adjusting x-axis ticks dynamically.

    :param cumulative_accuracies: List of cumulative accuracy values.
    :param image_indexes_in_loop: List of image indexes corresponding to cumulative accuracies.
    :param output_dir: Directory where the plot will be saved.
    """
    
    adjusted_image_indexes = [index + 1 for index in image_indexes_in_loop]
    
    plt.figure(figsize=(10, 6))
    plt.plot(adjusted_image_indexes, cumulative_accuracies, marker='o', linestyle='-', color='b', label='Cumulative Accuracy')
    plt.xlabel('Seen Images')
    plt.ylabel('Cumulative Accuracy (%)')
    plt.title('Cumulative Accuracy vs. Seen Examples')
    plt.ylim(0, 100)

    # Use MaxNLocator for x-axis to avoid overcrowding ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins='auto'))

    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/cumulative_accuracy_vs_seen_examples.png")
    plt.show()

    
def plot_training_results(output_dir: str, label_predict_range: int) -> None:
    """
    Plots cumulative accuracy, batch accuracies, and accuracies by seen images.

    Parameters:
    - output_dir: Directory containing data for plotting and saving plot images.
    - label_predict_range: Number of images per label prediction batch.
    """
    # Load required data for plotting
    cumulative_accuracies = np.load(f'{output_dir}/cumulative_accuracies.npy')
    image_indexes_in_loop = np.load(f'{output_dir}/image_indexes_in_loop.npy')
    batch_accuracies = np.load(f"{output_dir}/batch_accuracies.npy")

    # Plotting functions
    plot_cumulative_accuracy(cumulative_accuracies, image_indexes_in_loop, output_dir)
    plot_accuracies_by_batch(batch_accuracies, output_dir)
    plot_accuracies_by_seen_images(batch_accuracies, label_predict_range, output_dir)



def plot_final_synapse_weights(run_dir: str, population_exc: int, size_selected: int, final_weights_path: str = None) -> None:
    grid_size = int(np.sqrt(population_exc))
    # Define default path if none provided
    if final_weights_path is None:
        final_weights_path = f'{run_dir}/final_synaptic_weights.npy'
    
    # Load the final synaptic weights
    final_weights = np.load(final_weights_path)
    print(f'[INFO] Loaded from {final_weights_path}')
    # Initialize the figure for plotting

    syn_input_exc_j, syn_input_exc_w_ee = load_synapse_attributes(run_dir)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(size_selected, size_selected))
    
    # Ensure that weights are extracted and mapped correctly
    with tqdm(total=population_exc, desc="Processing Excitatory Neurons", dynamic_ncols=True) as pbar:
        for exc_neuron_idx in range(population_exc):
            # Get neuron-specific synapse indices to maintain the correct mapping
            synapse_indices = np.where(syn_input_exc_j == exc_neuron_idx)[0]
            neuron_weights = final_weights[synapse_indices]
            
            # Reshape weights into the original size
            neuron_weights_reshaped = neuron_weights.reshape(size_selected, size_selected)
            
            # Normalize weights to grayscale (0-255)
            normalized_weights = normalize_to_grayscale([neuron_weights_reshaped], vmin=0, vmax=1)[0] 
            
            # Plot the weights in the grid layout
            row, col = divmod(exc_neuron_idx, grid_size)
            axes[row, col].imshow(normalized_weights, cmap='gray', vmin=0, vmax=255)
            axes[row, col].axis('off')
            pbar.update(1)

    plt.tight_layout()
    plt.savefig(f'{run_dir}/final_synapse_weights.png')
    plt.show()
    plt.close()


def plot_spike_counts_with_cbar(run_dir: str, population_exc: int) -> None:
    """
    Plot spike counts with a color bar for the current run.

    :param run_dir: The current run directory for saving files.
    :param population_exc: The total number of excitatory neurons.
    """
    grid_size = int(np.sqrt(population_exc))
    
    # Load current spike data
    current_data_path = f'{run_dir}/spike_data_with_labels.npz'
    current_data = np.load(current_data_path)
    print(f'[INFO] Loaded from {current_data_path}')
    current_spike_counts = current_data['spike_counts']
    current_spike_counts_per_neuron = np.sum(current_spike_counts, axis=0)

    # Plot spike counts
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
    max_count = np.max(current_spike_counts_per_neuron)
    
    with tqdm(total=population_exc, desc="Processing Spike Counts", dynamic_ncols=True) as pbar:
        for exc_neuron_idx in range(population_exc):
            neuron_spike_count = current_spike_counts_per_neuron[exc_neuron_idx]
            spike_count_matrix = np.array([[neuron_spike_count]])
            row, col = divmod(exc_neuron_idx, grid_size)
            im = axes[row, col].imshow(spike_count_matrix, cmap='hot', vmin=0, vmax=max_count)
            axes[row, col].axis('off')
            axes[row, col].text(0.5, 0.5, str(neuron_spike_count), color='black', fontsize=12, ha='center', va='center')
            pbar.update(1)

    # Add vertical color bar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Spike Count', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Save and display plot
    output_path = f'{run_dir}/spike_counts_with_cbar.png'
    plt.savefig(output_path)
    plt.show()
    plt.close()
    print(f'[INFO] Plot saved to {output_path}')
    
    
    
    
def plot_neurons_with_label_spike_counts(
    selected_spike_data_path: str,
    spike_data_with_labels_folder_path: str,
    population_exc: int
) -> None:
    """
    Plots neurons and their average spike counts for each label (0-9).
    For each neuron, it will display the average spike count for each label,
    with the label that has the highest spike count highlighted in red.

    :param selected_spike_data_path: Path to the file containing spike data with labels.
    :param population_exc: Number of excitatory neurons.
    """
    # Calculate grid size
    grid_size = int(np.sqrt(population_exc))
    
    # Load spike counts and labels
    loaded_data = np.load(selected_spike_data_path)
    loaded_spike_counts = loaded_data['spike_counts']
    loaded_labels = loaded_data['labels']
    print(f'[INFO] Loaded spike data from {selected_spike_data_path}')
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(40, 40))

    # Calculate average spike counts per label for each neuron
    average_spike_counts_per_label = np.zeros((population_exc, 10))  # shape: (num_neurons, 10 labels)

    for label in range(10):
        # Get indices of images with the current label
        label_indices = np.where(loaded_labels == label)[0]
        if len(label_indices) > 0:
            # Sum spike counts for images of this label
            label_spike_counts = np.sum(loaded_spike_counts[label_indices], axis=0)
            # Calculate average spike count per neuron for the current label
            average_spike_counts_per_label[:, label] = label_spike_counts / len(label_indices)

    # Create the plot for each neuron
    for exc_neuron_idx in range(population_exc):
        row = exc_neuron_idx // grid_size  # Row index in grid
        col = exc_neuron_idx % grid_size   # Column index in grid    

        # Find the maximum spike count for this neuron
        max_spike_count = np.max(average_spike_counts_per_label[exc_neuron_idx])
        # Identify labels that have the max spike count
        max_labels = np.where(average_spike_counts_per_label[exc_neuron_idx] == max_spike_count)[0]

        # Prepare the text with average spike counts for each label
        spike_text_lines = []
        for label in range(10):
            spike_count = average_spike_counts_per_label[exc_neuron_idx, label]
            if label in max_labels:
                spike_text_lines.append(f"lb {label}: {spike_count:.2f} sp/im (max)")  # Mark the max labels
            else:
                spike_text_lines.append(f"lb {label}: {spike_count:.2f} sp/im")

        # Display the neuron and label text
        axes[row, col].imshow(np.ones((1, 1)), cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')  # Hide axis for a clean visualization
        
        # Add the text, highlighting the max label(s) in red
        for idx, line in enumerate(spike_text_lines):
            color = 'red' if "max" in line else 'black'
            axes[row, col].text(0.5, 1 - 0.1 * idx, line.replace(' (max)', ''), color=color, fontsize=12, ha='center', va='center', transform=axes[row, col].transAxes)
    
    # Save the plot in the same folder as the spike data file
    plot_path = f'{spike_data_with_labels_folder_path}/neuron_spike_counts.png'
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    print(f'[INFO] Neuron spike counts plot saved at {plot_path}')
    

def plot_neurons_with_labels(
    assigned_labels_path: str,
    population_exc: int
) -> None:
    """
    Plots neurons with assigned labels in a grid and saves the plot in the specified directory.

    :param assigned_labels_path: Path to the file containing assigned neuron labels.
    :param population_exc: Number of excitatory neurons.
    """
    # Calculate grid size
    grid_size = int(np.sqrt(population_exc))
    
    # Load assigned labels
    assigned_labels = np.load(assigned_labels_path)
    print(f'[INFO] Loaded assigned labels from {assigned_labels_path}')

    # Create plot figure with neurons in a grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    for exc_neuron_idx in range(population_exc):
        row = exc_neuron_idx // grid_size  # Row index in grid
        col = exc_neuron_idx % grid_size   # Column index in grid    
        
        # Display neuron label on the grid
        axes[row, col].imshow(np.ones((1, 1)), cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')  # Turn off axis for a clean visualization
        axes[row, col].text(0.5, 0.5, str(assigned_labels[exc_neuron_idx]), color='black', fontsize=12, ha='center', va='center', transform=axes[row, col].transAxes)
    
    plt.tight_layout()
    
    # Define save path
    save_folder = os.path.dirname(assigned_labels_path)
    plot_path = f'{save_folder}/neuron_labels_plot.png'
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    print(f'[INFO] Neuron labels plot saved at {plot_path}')


def plot_neurons_with_labels_detailed(
    assigned_labels_path: str,
    population_exc: int
) -> None:
    """
    Plots neurons with assigned labels in a grid, saves the plot, and generates a histogram of label counts.

    :param assigned_labels_path: Path to the file containing assigned neuron labels.
    :param population_exc: Number of excitatory neurons.
    """
    # Calculate grid size
    grid_size = int(np.sqrt(population_exc))
    
    # Load assigned labels
    assigned_labels = np.load(assigned_labels_path)
    print(f'[INFO] Loaded assigned labels from {assigned_labels_path}')
    
    # Count label occurrences
    label_counts = Counter(assigned_labels)
    unassigned_count = label_counts.get(-1, 0)
    
    # Create plot figure with neurons in a grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
    
    
    for exc_neuron_idx in range(population_exc):
        row = exc_neuron_idx // grid_size  # Row index in grid
        col = exc_neuron_idx % grid_size   # Column index in grid    
        
        neuron_label = assigned_labels[exc_neuron_idx]
        
        # Set cell color based on label
        if neuron_label == -1:
            # If unassigned, make cell black
            axes[row, col].imshow(np.zeros((1, 1)), cmap='gray', vmin=0, vmax=1)
            color = 'white'  # Text color on black background
        else:
            # Otherwise, display as normal
            axes[row, col].imshow(np.ones((1, 1)), cmap='gray', vmin=0, vmax=1)
            color = 'black'  # Text color on white background
            
            
        # Display neuron label text
        axes[row, col].axis('off')  # Turn off axis for a clean visualization
        axes[row, col].text(
            0.5, 0.5, str(neuron_label),
            color=color, fontsize=16, ha='center', va='center',
            transform=axes[row, col].transAxes
        )
    
    # Set the title based on unassigned count
    if unassigned_count > 0:
        title = f"Neuron Labels, Unassigned Count: {unassigned_count}"
    else:
        title = "Neuron Labels"
    fig.suptitle(title, fontsize=24, y=0.93)
    
    # Save the neuron grid plot
    save_folder = os.path.dirname(assigned_labels_path)
    plot_path = f'{save_folder}/neuron_labels_plot_2.png'
    plt.savefig(plot_path)
    #plt.show()
    plt.close()
    
    print(f'[INFO] Neuron labels plot saved at {plot_path}')
    # Generate and save histogram of label counts (including unassigned)
    labels, counts = zip(*[(label, label_counts.get(label, 0)) for label in range(-1, 10)])
    plt.figure(figsize=(30, 15))
    plt.bar(labels, counts, color='black')
    
    # Set labels with specific font sizes
    plt.xlabel('Neuron Labels', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    
    # Set tick labels with specific font sizes
    plt.xticks(range(10), fontsize=16)  # X-axis ticks from 0 to 9
    plt.yticks(fontsize=16)
    
    # Set the title with specific font size and position
    plt.title('Distribution of Neuron Labels', fontsize=24, y=0.93)
    
    # Save and close the plot
    hist_path = f'{save_folder}/neuron_labels_histogram.png'
    plt.savefig(hist_path)
    #plt.show()
    plt.close()
    print(f'[INFO] Neuron label histogram saved at {hist_path}')
    
    
# Function for plotting the test image and its prediction
def plot_test_image_with_prediction(test_image: np.ndarray, true_label: int, predicted_label: int):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot the test image on the left
    axes[0].imshow(test_image, cmap='gray')
    axes[0].axis('off')  # Turn off axis for a clean image
    axes[0].set_title(f"Test Image (Label: {true_label})")

    # Plot the predicted label on the right
    axes[1].text(0.5, 0.5, str(predicted_label), fontsize=30, ha='center', va='center', color='black')
    axes[1].axis('off')  # No axis needed for this label
    axes[1].set_title("Predicted Label")

    plt.tight_layout()
    plt.show()
    plt.close()
    
    
# Function for complex 4-layer plot with improvements
def plot_prediction_process(spike_counts_current_image: np.ndarray, assigned_labels_path: str) -> None:
    """
    Plots the process of predicting a label for the current image based on neuron spike counts.
    
    :param spike_counts_current_image: Array of spike counts for the current image.
    :param assigned_labels_path: Path to the assigned labels file.
    """
    # Load the assigned labels
    assigned_labels = np.load(assigned_labels_path)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Number of labels (0-9)
    num_labels = 10

    # Calculate total spikes for neurons assigned to each label
    total_spike_counts = np.zeros(num_labels)
    for label in range(num_labels):
        assignment_indices = np.where(assigned_labels == label)[0]
        total_spike_counts[label] = np.sum(spike_counts_current_image[assignment_indices])

    # Layer 1: Total spikes for each label
    sns.heatmap(total_spike_counts.reshape(-1, 1), annot=True, fmt=".6f", cmap="Blues", cbar=True, ax=axes[0])
    axes[0].set_title("Total Spikes by Label")
    axes[0].set_yticks(np.arange(num_labels) + 0.5)
    axes[0].set_yticklabels(np.arange(num_labels))
    axes[0].set_xticks([])

    # Layer 2: Average spikes for each label
    avg_spike_counts = np.zeros(num_labels)
    for label in range(num_labels):
        assignment_indices = np.where(assigned_labels == label)[0]
        if len(assignment_indices) > 0:
            avg_spike_counts[label] = total_spike_counts[label] / len(assignment_indices)

    sns.heatmap(avg_spike_counts.reshape(-1, 1), annot=True, fmt=".6f", cmap="Blues", cbar=True, ax=axes[1])
    axes[1].set_title("Average Spikes by Label")
    axes[1].set_yticks(np.arange(num_labels) + 0.5)
    axes[1].set_yticklabels(np.arange(num_labels))
    axes[1].set_xticks([])

    # Layer 3: Maximum value from average spikes
    max_avg_spike = np.max(avg_spike_counts)
    sns.heatmap(np.array([[max_avg_spike]]), annot=True, fmt=".6f", cmap="Blues", cbar=False, ax=axes[2])
    axes[2].set_title("Max Avg Spike")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Layer 4: Label with the maximum average spike count
    label_with_max_avg_spike = np.argmax(avg_spike_counts)
    sns.heatmap(np.array([[label_with_max_avg_spike]]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[3])
    axes[3].set_title("Predicted Label")
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    plt.tight_layout()
    plt.show()
    plt.close()
    
    
    
def plot_spike_counts_with_cbar_test(prediction_folder_path: str, population_exc: int, 
                                    current_data_path: str = None) -> None:
    """
    Plot spike counts with a color bar for the current run, and optionally combine and plot with previous run data.
    :param prediction_folder_path: The current run name for saving files.
    :param population_exc: The total number of excitatory neurons.
    """
    grid_size = int(np.sqrt(population_exc))
    
    # Load current spike data
    if current_data_path is None:
        current_data_path = f'{prediction_folder_path}/spike_data_with_labels.npz'
    current_data = np.load(current_data_path)
    print(f'[INFO] Loaded from {current_data_path}')
    current_spike_counts = current_data['spike_counts']
    current_spike_counts_per_neuron_without_retries = np.sum(current_spike_counts, axis=0)

    # Plot spike counts for the current run
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
    with tqdm(total=population_exc, desc="Processing Current Spike Counts", dynamic_ncols=True) as pbar:
        for exc_neuron_idx in range(population_exc):
            neuron_spike_count = current_spike_counts_per_neuron_without_retries[exc_neuron_idx]
            spike_count_matrix = np.array([[neuron_spike_count]])
            row, col = divmod(exc_neuron_idx, grid_size)
            im = axes[row, col].imshow(spike_count_matrix, cmap='hot', vmin=0, vmax=np.max(current_spike_counts_per_neuron_without_retries))
            axes[row, col].axis('off')
            axes[row, col].text(0.5, 0.5, str(neuron_spike_count), color='black', fontsize=12, ha='center', va='center', transform=axes[row, col].transAxes)
            pbar.update(1)

    # Add vertical color bar on the right of the grid
    cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])  # Adjust position for vertical orientation
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Spike Count', fontsize=12)
    cbar.ax.tick_params(labelsize=10)  # Increase color bar tick font size
    plt.savefig(f'{prediction_folder_path}/spike_counts_with_numbers_and_cbar.png')
    plt.show()
    plt.close()
    
def plot_accuracy_per_label(
    prediction_folder_path: str,
    num_labels: int = 10
):
    """
    Plot accuracy per label as a bar chart.

    :param prediction_folder_path: Path where the results will be saved.
    :param accuracy_per_label: numpy array of accuracy per label.
    :param num_labels: Total number of labels (default is 10 for digits).
    """
    os.makedirs(prediction_folder_path, exist_ok=True)
    accuracy_per_label_path = os.path.join(prediction_folder_path, 'accuracy_per_label.npy')
    accuracy_per_label = np.load(accuracy_per_label_path)
    print(f"[INFO] Loaded accuracy per label from {accuracy_per_label_path}")
    labels = np.arange(num_labels)  # Labels for x-axis (0-9)
    plt.figure(figsize=(8, 6))
    plt.bar(labels, accuracy_per_label)
    plt.xlabel('Label')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Label')
    plt.xticks(labels)  # Ensure x-axis ticks match the label numbers
    plt.ylim(0, 100)  # Set y-axis limit to 100%

    plt.tight_layout()
    plot_path = os.path.join(prediction_folder_path, 'accuracy_per_label.png')
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    print(f"[INFO] Accuracy per label plot saved to {plot_path}")

def plot_confusion_matrix(
    prediction_folder_path: str,
    class_labels=None
):
    """
    Creates and saves a confusion matrix plot.

    Parameters:
    - prediction_folder_path (str): Path to save the confusion matrix plot.
    - true_labels_path (str): Path to the true labels file.
    - predicted_labels_path (str): Path to the predicted labels file.
    - class_labels (list, optional): List of labels for axes (default is 0-9).
    """
    os.makedirs(prediction_folder_path, exist_ok=True)

    true_labels_path = f'{prediction_folder_path}/image_labels_in_loop.npy'
    true_labels = np.load(true_labels_path)

    predicted_labels_path = f'{prediction_folder_path}/predicted_labels.npy'
    predicted_labels = np.load(predicted_labels_path)

    # Default class labels if none provided
    if class_labels is None:
        class_labels = np.arange(10)  # Default to 0-9

    # Create confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Save and display plot
    plot_path = os.path.join(prediction_folder_path, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    print(f"[INFO] Confusion matrix saved to {plot_path}")

def plot_cumulative_test_accuracy(image_indexes_in_loop_path: str, prediction_folder_path: str) -> None:
    """
    Plots cumulative accuracies with respect to seen examples, adjusting x-axis ticks dynamically.

    :param cumulative_accuracies: List of cumulative accuracy values.
    :param image_indexes_in_loop: List of image indexes corresponding to cumulative accuracies.
    :param prediction_folder_path: Directory where the plot will be saved.
    """
    cumulative_accuracies_path = f'{prediction_folder_path}/cumulative_accuracies.npy'
    cumulative_accuracies = np.load(cumulative_accuracies_path)
    image_indexes_in_loop_path = f'{prediction_folder_path}/image_indexes_in_loop.npy'
    image_indexes_in_loop = np.load(image_indexes_in_loop_path)
    
    adjusted_image_indexes = [index + 1 for index in image_indexes_in_loop]
    
    # Plot with dynamically adjusted x-axis ticks
    plt.figure(figsize=(10, 6))
    plt.plot(adjusted_image_indexes, cumulative_accuracies, marker='o', linestyle='-', color='b', label='Cumulative Accuracy')
    plt.xlabel('Seen Images')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs. Seen Examples')
    plt.ylim(0, 100)

    # Use MaxNLocator for x-axis to avoid overcrowding ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins='auto'))

    plt.grid(True)
    plt.legend()
    plt.savefig(f"{prediction_folder_path}/cumulative_test_accuracy_vs_seen_examples.png")
    plt.show()
    
    print("[INFO] Plot generated using cumulative accuracy, which is standard practice for evaluating test performance.")
