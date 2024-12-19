# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 03:16:15 2024

@author: ege-demir
"""

import os
import numpy as np               # For numerical operations and loading .npz files
import matplotlib.pyplot as plt  # For plotting
from tqdm import tqdm            # For progress bar visualization

first_x_elements_array = np.arange(1000, 60001, 1000)
first_x_elements_array = [60000]
selected_run = 'run_20241108131041'

run_array = ['run_20241119015918', 'run_20241112104344', 'run_20241119110308']
size_array = [100,400,1600]

# Ensure the directory exists
output_dir = f'{selected_run}/exc_theta_plots/'
os.makedirs(output_dir, exist_ok=True)

for first_x_elements in first_x_elements_array:
    population_exc = 400
    grid_size = int(np.sqrt(population_exc))
    current_data_path = f'{selected_run}/neuron_group_exc_theta.npy'
    current_data = np.load(current_data_path)
    current_theta_counts = current_data * 1000
 
    # Plot theta values for the current run
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
    #fig.suptitle(f"Theta Values in mV After {first_x_elements} Images", fontsize=24, y=0.93)
    
    with tqdm(total=population_exc, desc="Processing Theta Values", dynamic_ncols=True) as pbar:
        for exc_neuron_idx in range(population_exc):
            neuron_theta_count = current_theta_counts[exc_neuron_idx]
            theta_count_matrix = np.array([[neuron_theta_count]])
            
            row, col = divmod(exc_neuron_idx, grid_size)
            im = axes[row, col].imshow(theta_count_matrix, cmap='hot', vmin=0, vmax=np.max(current_theta_counts))
            axes[row, col].axis('off')
            # Format the text to show two decimal places
            axes[row, col].text(0.5, 0.5, f"{neuron_theta_count:.2f}", color='black', fontsize=24, ha='center', va='center', transform=axes[row, col].transAxes)
            pbar.update(1)
    
    # Add vertical color bar on the right of the grid
    cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])  # Adjust position for vertical orientation
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    #cbar.set_label('Theta Values in mV', fontsize=16)
    cbar.ax.tick_params(labelsize=24)  # Increase color bar tick font size
    plt.savefig(f'{output_dir}/neuron_group_exc_theta_{first_x_elements}.png')
    plt.show()
    plt.close()

# %%

import os
import numpy as np               # For numerical operations and loading .npz files
import matplotlib.pyplot as plt  # For plotting
from tqdm import tqdm            # For progress bar visualization

#first_x_elements = 100
#first_x_elements_array = np.arange(1000, 60001, 1000)
#first_x_elements_array = [1000,3000,5000,7000,8000,60000]
first_x_elements_array = [60000]
selected_run = 'run_20241113001457'
# Ensure the directory exists
output_dir = f'{selected_run}/spike_count_plots_bigger/'
os.makedirs(output_dir, exist_ok=True)

for first_x_elements in first_x_elements_array:
    population_exc = 400
    grid_size = int(np.sqrt(population_exc))
    current_data_path = f'{selected_run}/spike_data_with_labels.npz'
    current_data = np.load(current_data_path)
    current_spike_counts = current_data['spike_counts']
    first_x_spike_counts = current_spike_counts[:first_x_elements]
    sum_first_x_spike_counts = np.sum(first_x_spike_counts, axis=0)
    
    # Plot spike counts for the current run
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
    #fig.suptitle(f"Spike Counts per Neuron at Image {first_x_elements}", fontsize=24, y=0.93)
    with tqdm(total=population_exc, desc="Processing Current Spike Counts", dynamic_ncols=True) as pbar:
        for exc_neuron_idx in range(population_exc):
            neuron_spike_count = sum_first_x_spike_counts[exc_neuron_idx]
            spike_count_matrix = np.array([[neuron_spike_count]])
            row, col = divmod(exc_neuron_idx, grid_size)
            im = axes[row, col].imshow(spike_count_matrix, cmap='hot', vmin=0, vmax=np.max(sum_first_x_spike_counts))
            axes[row, col].axis('off')
            axes[row, col].text(0.5, 0.5, str(neuron_spike_count), color='black', fontsize=24, ha='center', va='center', transform=axes[row, col].transAxes)
            pbar.update(1)
    
    # Add vertical color bar on the right of the grid
    cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])  # Adjust position for vertical orientation
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    # cbar.set_label('Spike Count', fontsize=16)
    cbar.ax.tick_params(labelsize=24)  # Increase color bar tick font size
    plt.savefig(f'{output_dir}/spike_counts_with_numbers_and_cbar_{first_x_elements}.png')
    plt.show()
    plt.close()
    
# %%
import os
import numpy as np               # For numerical operations and loading .npz files
import matplotlib.pyplot as plt  # For plotting
from tqdm import tqdm            # For progress bar visualization

from collections import Counter

def plot_neurons_with_labels_detailed_2(
    assigned_labels_path: str,
    population_exc: int, save_folder: str) -> None:
    """
    Plots neurons with assigned labels in a grid, saves the plot, and generates a histogram of label counts.

    :param assigned_labels_path: Path to the file containing assigned neuron labels.
    :param population_exc: Number of excitatory neurons.
    """
    os.makedirs(save_folder, exist_ok=True)
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
            color=color, fontsize=40, ha='center', va='center',
            transform=axes[row, col].transAxes
        )
    
    # Set the title based on unassigned count
    if unassigned_count > 0:
        title = f"Neuron Labels, Unassigned Count: {unassigned_count}"
    else:
        title = "Neuron Labels"
    #fig.suptitle(title, fontsize=40, y=0.93)
    
    # Save the neuron grid plot
    plot_path = f'{save_folder}/neuron_labels_plot_2_bigger.png'
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    print(f'[INFO] Neuron labels plot saved at {plot_path}')
    # Generate and save histogram of label counts (including unassigned)
    labels, counts = zip(*[(label, label_counts.get(label, 0)) for label in range(0, 10)])
    plt.figure(figsize=(30, 15))
    plt.bar(labels, counts, color='black')
    
    # Set labels with specific font sizes
    plt.xlabel('Neuron Labels', fontsize=40)
    plt.ylabel('Count', fontsize=40)
    
    # Set tick labels with specific font sizes
    plt.xticks(range(10), fontsize=40)  # X-axis ticks from 0 to 9
    plt.yticks(fontsize=40)
    
    # Set the title with specific font size and position
    #plt.title('Distribution of Neuron Labels', fontsize=24, y=1.1)
    
    # Save and close the plot
    hist_path = f'{save_folder}/neuron_labels_histogram_bigger.png'
    plt.savefig(hist_path)
    plt.show()
    plt.close()
    print(f'[INFO] Neuron label histogram saved at {hist_path}')
    
run_array = ['run_20241119015918', 'run_20241112104344', 'run_20241119110308']
size_array = [100,400,1600]

run_array = ['run_20241111111504']
size_array = [400]
for idx, selected_run in enumerate(run_array):   
    print(idx)
    pop_size = size_array[idx]
    save_folder = f"{selected_run}/histogram_bigger"
    assigned_labels_path = f"{selected_run}/regular_full/assignments_from_training.npy"
        
    plot_neurons_with_labels_detailed_2(assigned_labels_path, pop_size, save_folder)