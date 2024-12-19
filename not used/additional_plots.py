# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:40:56 2024

@author: ege-demir
"""
import numpy as np

def normalize_to_grayscale(kernels, vmin, vmax):
    grayscaled_kernels = []
    for kernel in kernels:
        grayscaled_kernels.append(np.floor(np.clip(255 * (kernel - vmin) / (vmax - vmin), 0, 255)))
    return grayscaled_kernels
    
def load_synapse_attributes(directory, prefix="syn_input_exc"):
    """
    Loads the important attributes of a synapse object (e.g., syn_input_exc)
    from specified files in the directory.

    Parameters:
    - directory: Directory to load the attributes from.
    - prefix: Prefix used for filenames (default is "syn_input_exc").

    Returns:
    - syn_input_exc_j: The target neuron indices (e.g., syn_input_exc.j).
    - syn_input_exc_w_ee: The synaptic weights (e.g., syn_input_exc.w_ee).
    """
    # Paths to the attribute files
    weights_path = os.path.join(directory, f"{prefix}_weights.npy")
    indices_path = os.path.join(directory, f"{prefix}_indices.npy")
    
    # Initialize variables to store loaded attributes
    syn_input_exc_j = None
    syn_input_exc_w_ee = None

    # Load the synaptic weights if the file exists
    if os.path.exists(weights_path):
        syn_input_exc_w_ee = np.load(weights_path)
        print(f"[INFO] Loaded synaptic weights from {weights_path}")
    else:
        print(f"[WARNING] Synaptic weights file not found at {weights_path}")

    # Load the target neuron indices if the file exists
    if os.path.exists(indices_path):
        syn_input_exc_j = np.load(indices_path)
        print(f"[INFO] Loaded target neuron indices from {indices_path}")
    else:
        print(f"[WARNING] Target neuron indices file not found at {indices_path}")
    
    return syn_input_exc_j, syn_input_exc_w_ee


rf_dimensions_file_path = 'input_to_output_mapping/refined_rf_trials/28_9_1_0_20_rf_dimensions.npz'

rf_data = np.load(rf_dimensions_file_path)
print(f'[INFO] rf dimensions loaded from {rf_dimensions_file_path}')

# Extract the rf_dimensions array
rf_dimensions = rf_data['rf_dimensions']

# Access each column separately
neuron_indices = rf_dimensions[:, 0]  # Neuron Index column
rf_heights = rf_dimensions[:, 1]      # RF Height column
rf_widths = rf_dimensions[:, 2]       # RF Width column

import os
import numpy as np               # For numerical operations and loading .npz files
import matplotlib.pyplot as plt  # For plotting
from tqdm import tqdm            # For progress bar visualization

population_exc = 20*20
grid_size = int(np.sqrt(population_exc))

first_x_elements_array = np.arange(1000, 60001, 1000)
#first_x_elements_array = [37000]
first_x_elements_array = [60000]
selected_run = 'run_20241113001457'

# Ensure the directory exists
output_dir = f'{selected_run}/weights_plots/'
os.makedirs(output_dir, exist_ok=True)

for first_x_elements in first_x_elements_array:
    population_exc = 400
    grid_size = int(np.sqrt(population_exc))
    current_data_path = f'{selected_run}/save_state_totim_{first_x_elements}/final_synaptic_weights.npy'
    current_data = np.load(current_data_path)
    current_weights = current_data
    # Load the final synaptic weights
    final_weights = np.load(current_data_path)
    print(f'[INFO] Loaded from {current_data_path}')
    
    syn_input_exc_j, _ = load_synapse_attributes(selected_run)
    
    # Set up list to store each neuron's dimensions for combined plot
    neuron_dimensions = []
    
    # Process each excitatory neuron
    with tqdm(total=population_exc, desc="Processing Excitatory Neurons", dynamic_ncols=True) as pbar:
        for exc_neuron_idx in range(population_exc):
            
            neuron_index = neuron_indices[exc_neuron_idx]
            height_calculated = rf_heights[exc_neuron_idx]  # RF Height
            width_calculated = rf_widths[exc_neuron_idx]  # RF Width
            
            rf_det=f'Neuron {exc_neuron_idx}: Height = {height_calculated}, Width = {width_calculated}'
            
            if neuron_index  == exc_neuron_idx:
                ref_det2= f'SAME. exc_neuron_idx_arr: {neuron_index}, exc_neuron_idx: {exc_neuron_idx}'
            else:
                ref_det2=f'NOT SAME. exc_neuron_idx_arr: {neuron_index}, exc_neuron_idx: {exc_neuron_idx}'

            pbar.set_description(f'{rf_det}, {ref_det2}')
            synapse_indices = np.where(syn_input_exc_j == exc_neuron_idx)[0]
            neuron_weights = final_weights[synapse_indices]
            neuron_dimensions.append((width_calculated, height_calculated))
            pbar.update(1)
            
    
    # Calculate figsize for the combined plot based on neuron dimensions
    max_width = max(width for width, _ in neuron_dimensions)
    max_height = max(height for _, height in neuron_dimensions)
    
    combined_fig, combined_axes = plt.subplots(grid_size, grid_size, figsize=(max_width*3, max_height*3))
    combined_fig.suptitle(f"Synaptic Weights After {first_x_elements} Images", fontsize=24, y=0.91)
    
    # Plot each neuron's weights on the combined plot grid
    with tqdm(total=population_exc, desc="Creating Combined Plot", dynamic_ncols=True) as pbar:
        for exc_neuron_idx in range(population_exc):
            row, col = divmod(exc_neuron_idx, grid_size)
            width_calculated, height_calculated = neuron_dimensions[exc_neuron_idx]
            synapse_indices = np.where(syn_input_exc_j == exc_neuron_idx)[0]
            neuron_weights = final_weights[synapse_indices]
            neuron_weights_reshaped = neuron_weights.reshape(height_calculated, width_calculated)
            
            normalized_weights = normalize_to_grayscale([neuron_weights_reshaped], vmin=0, vmax=1)[0]

            
            padded_image = np.full((max_width, max_height), 255)  # Create a blank white 28x28 base
            padded_image[:height_calculated, :width_calculated] = normalized_weights  # Place the normalized weights in the top-left corner
            
            # Display the padded image in the subplot
            combined_axes[row, col].imshow(padded_image, cmap='gray', vmin=0, vmax=255)
            

            #combined_axes[row, col].axis('off')
            combined_axes[row, col].set_xticks([])
            combined_axes[row, col].set_yticks([])

            pbar.update(1)
    
    
    # Save the combined final synapse weights plot
    #plt.tight_layout()
    plt.savefig(f'{output_dir}/final_synapse_weights_{first_x_elements}.png')
    #plt.show()
    plt.close()
