# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:21:10 2024

@author: ege-demir
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import struct
from tqdm import tqdm
from PIL import Image
# import matplotlib.colorbar as cbar
import typing as ty
import seaborn as sns
from sklearn.metrics import confusion_matrix
from glob import glob
from matplotlib.ticker import MaxNLocator
from typing import List

#TRAINING


def divisive_weight_normalization(synapse, population_exc: int, normalization_value: float) -> None:
    """
    Normalizes the synaptic weights of the connections to each post-synaptic neuron, ensuring that the total
    sum of weights connecting to each neuron equals the specified `normalization_value`.

    Parameters:
    - synapse: The Synapses object that contains information about the synaptic connections and their weights.
    - population_exc: The number of excitatory neurons (i.e., the size of the post-synaptic population).
    - normalization_value: The target sum that the synaptic weights connecting to each post-synaptic neuron
      should add up to after normalization (e.g., 78).
    
    Process:
    - For each post-synaptic neuron, the function identifies all the synapses that connect to it.
    - It calculates the sum of these synaptic weights.
    - A normalization factor is computed by dividing the `normalization_value` by this sum.
    - Each weight is then scaled by this factor to ensure that the sum of the weights equals `normalization_value`
      while maintaining the same relative proportions between the original weights.

    Output:
    - The function does not return a value. It modifies the weights in `synapse.w_ee` in place, normalizing them
      according to the specified `normalization_value`.
    """
    for post_idx in range(population_exc):
        # Extract indices of synapses that connect to the current post-synaptic neuron
        target_indices = np.where(synapse.j == post_idx)[0]

        # Extract weights of these synapses
        weights_to_same_post = synapse.w_ee[target_indices]

        # Calculate sum of weights connected to the current post-synaptic neuron
        sum_of_weights = np.sum(weights_to_same_post)

        # Calculate normalization factor based on the provided normalization value
        normalization_factor = normalization_value / sum_of_weights
        
        # Update the weights in the Synapses object
        synapse.w_ee[target_indices] *= normalization_factor


def get_spiking_rates_and_labels(use_test_data_mnist, image_count, seed_data, size_selected, start_index, dataset_path: str = None):
    name = 't10k' if use_test_data_mnist else 'train'
    if dataset_path is None:
        dataset_path = "mnist/"
    # new_size = (size_selected, size_selected)
    
    # Load the images and labels
    image_intensities = _load_images(dataset_path + f'{name}-images.idx3-ubyte')
    
    # Resize the images
    resized_images = _resize_images(image_intensities, size_selected)
    
    # Convert the resized images to 1D arrays for PoissonGroup usage (if needed)
    resized_images_1d = _convert_indices_to_1d(resized_images)

    # Normalize pixel intensities to spiking rates
    image_rates = _convert_to_spiking_rates(resized_images_1d)

    # Load the labels
    image_labels = _load_labels(dataset_path + f'{name}-labels.idx1-ubyte')

    # Check if image_count + start_index does not exceed the number of available images
    num_images = image_rates.shape[0]
    if start_index + image_count > num_images:
        raise ValueError(f"Requested image_count starting from index {start_index} exceeds the number of available images {num_images}.")

    # Select random indices from start_index
    if seed_data:
        np.random.seed(42)

    random_indices = np.random.choice(np.arange(start_index, num_images), size=image_count, replace=False)

    # Select the subset of images and labels
    image_rates_subset = image_rates[random_indices]
    image_labels_subset = image_labels[random_indices]
    resized_images_subset = resized_images[random_indices]
    
    # Return the resized images along with the rates and labels
    return image_rates_subset, image_labels_subset, resized_images_subset

def _resize_images(images, size_selected):
    new_size = (size_selected, size_selected)  # Create new size tuple from size_selected
    resized_images = []
    for image in images:
        img = Image.fromarray(image)  # Convert numpy array to PIL image
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)  # Resize using high-quality resampling
        resized_images.append(np.array(img_resized))  # Convert back to numpy array and append
    return np.array(resized_images)

def _load_images(filename: str):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def _load_labels(filename: str):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Normalizes pixel intensities between 0 and 63.75. Normalized value will
# be spiking rate (Hz) of the cell.
def _convert_to_spiking_rates(images, max_rate = 63.75 ):
    return (images * max_rate) / 255

# Converts indices spiking rates from 2d to 1d, so that it can be used in
# PoissonGroup object.
def _convert_indices_to_1d(images):
    return images.reshape(images.shape[0], -1)

def save_synapse_attributes(syn_input_exc, directory, prefix="syn_input_exc"):
    """
    Saves important attributes of a synapse object (e.g., syn_input_exc) to the specified directory.

    Parameters:
    - syn_input_exc: The synapse object to save attributes from.
    - directory: Directory to save the attributes in.
    - prefix: Prefix for filenames (default is "syn_input_exc").
    """
    os.makedirs(directory, exist_ok=True)
    
    # Save synapse weights (e.g., `w_ee`) and target indices (`j`) if they exist
    if hasattr(syn_input_exc, 'w_ee'):
        np.save(os.path.join(directory, f"{prefix}_weights.npy"), syn_input_exc.w_ee[:])
    if hasattr(syn_input_exc, 'j'):
        np.save(os.path.join(directory, f"{prefix}_indices.npy"), syn_input_exc.j[:])

    print(f"[INFO] Synapse attributes saved in {directory}.")

def save_neuron_group_exc_attributes(neuron_group_exc, directory, prefix="neuron_group_exc"):
    """
    Saves important attributes of a neuron group object (e.g., neuron_group_exc) to the specified directory.

    Parameters:
    - neuron_group_exc: The neuron group object to save attributes from.
    - directory: Directory to save the attributes in.
    - prefix: Prefix for filenames (default is "neuron_group_exc").
    """
    os.makedirs(directory, exist_ok=True)
    
    # Save attributes like `v`, `theta`, etc., if they exist
    if hasattr(neuron_group_exc, 'v'):
        np.save(os.path.join(directory, f"{prefix}_voltages.npy"), neuron_group_exc.v[:])
    if hasattr(neuron_group_exc, 'theta'):
        np.save(os.path.join(directory, f"{prefix}_theta.npy"), neuron_group_exc.theta[:])

    print(f"[INFO] Neuron group (excitatory) attributes saved in {directory}.")

def save_neuron_group_inh_attributes(neuron_group_inh, directory, prefix="neuron_group_inh"):
    """
    Saves important attributes of a neuron group object (e.g., neuron_group_inh) to the specified directory.

    Parameters:
    - neuron_group_inh: The neuron group object to save attributes from.
    - directory: Directory to save the attributes in.
    - prefix: Prefix for filenames (default is "neuron_group_inh").
    """
    os.makedirs(directory, exist_ok=True)
    
    # Save attributes like `v`, `g_e`, etc., if they exist
    if hasattr(neuron_group_inh, 'v'):
        np.save(os.path.join(directory, f"{prefix}_voltages.npy"), neuron_group_inh.v[:])
    if hasattr(neuron_group_inh, 'g_e'):
        np.save(os.path.join(directory, f"{prefix}_g_e.npy"), neuron_group_inh.g_e[:])
    if hasattr(neuron_group_inh, 'g_i'):
        np.save(os.path.join(directory, f"{prefix}_g_i.npy"), neuron_group_inh.g_i[:])

    print(f"[INFO] Neuron group (inhibitory) attributes saved in {directory}.")

# Save the state of the network (including neuron states)
def save_simulation_state(run_dir, last_image_index, syn_input_exc, neuron_group_exc, neuron_group_inh, save_simulation_state_folder: str = None):

    if save_simulation_state_folder is None:
        save_simulation_state_folder = run_dir
    os.makedirs(save_simulation_state_folder, exist_ok=True)
    
    save_synapse_attributes(syn_input_exc, save_simulation_state_folder)
    save_neuron_group_exc_attributes(neuron_group_exc, save_simulation_state_folder)
    save_neuron_group_inh_attributes(neuron_group_inh, save_simulation_state_folder)

    final_weights = syn_input_exc.w_ee[:]  # Get the synaptic weights
    np.save(f'{save_simulation_state_folder}/final_synaptic_weights.npy', final_weights)  # Save the weights
    np.save(f'{save_simulation_state_folder}/neuron_group_exc_theta.npy', neuron_group_exc.theta[:])  # Threshold values for excitatory neurons
    print(f"[INFO] Final synaptic weights saved for run: {run_dir}")
    print(f"[INFO] Theta values saved for run: {run_dir}")

    np.save(f'{save_simulation_state_folder}/neuron_group_exc_v.npy', neuron_group_exc.v[:])  # Membrane potentials for excitatory neurons
    np.save(f'{save_simulation_state_folder}/neuron_group_inh_v.npy', neuron_group_inh.v[:])  # Membrane potentials for inhibitory neurons
    
    # Save the current image index
    np.save(f'{save_simulation_state_folder}/last_image_index.npy', np.array([last_image_index]))
    print(f"[INFO] Simulation state saved after processing {last_image_index+1} images.")

def plot_images(image_index, image_intensities, image_labels, size_selected):
    # Plot the original MNIST input image for comparison
    show_image_index = image_index
    mnist_image = image_intensities[show_image_index].reshape(size_selected, size_selected)  # Reshape the MNIST input to new size

    fig, (ax_mnist) = plt.subplots(1, 1, figsize=(5, 5))

    # Plot the resized image
    im_resized = ax_mnist.imshow(mnist_image, cmap='gray', vmin=0, vmax=255)
    ax_mnist.set_title(f'Resized Image ({size_selected}x{size_selected})\n(image no:{show_image_index} with label {image_labels[image_index]})')
    plt.colorbar(im_resized, ax=ax_mnist)

    plt.show()

def increase_spiking_rates(image, current_max_rate):
    new_maximum_rate = current_max_rate + 32
    return (image * new_maximum_rate) / current_max_rate

def normalize_to_grayscale(kernels, vmin=0, vmax=1):
    """
    Normalize weights to a grayscale range from 0 to 255.
    """
    grayscaled_kernels = []
    for kernel in kernels:
        grayscaled_kernels.append(np.floor(np.clip(255 * (kernel - vmin) / (vmax - vmin), 0, 255)))
    return grayscaled_kernels

def save_and_combine_spike_data(run_dir: str, image_labels: np.ndarray, all_spike_counts_per_image: np.ndarray,
                                spike_mon_exc_count: np.ndarray, load_spike_counts: bool = False, 
                                load_run_dir: str = None, previous_data_path: str = None,
                                previous_spike_counts_with_retries_path: str = None ) -> None:
    """
    Save and combine spike data and labels from the current run, and optionally load and combine with a previous run.
    
    :param run_dir: The current run name for saving files.
    :param image_labels: Array of labels for the current run images.
    :param all_spike_counts_per_image: Array of spike counts per image for the current run.
    :param spike_mon_exc_count: Array of spike counts per neuron with retries for the current run.
    :param load_spike_counts: Boolean indicating whether to load previous run data.
    :param load_run_dir: The previous run name to load data from if load_spike_counts is True.
    """
    
    # Save spike counts without and with retries for current run
    spike_counts_per_neuron_without_retries = np.sum(all_spike_counts_per_image, axis=0)
    np.save(f'{run_dir}/spike_counts_per_neuron_without_retries.npy', spike_counts_per_neuron_without_retries)
    np.save(f'{run_dir}/spike_counts_per_neuron_with_retries.npy', spike_mon_exc_count)
    
    # Save the spike data and labels for the current run
    np.savez(f'{run_dir}/spike_data_with_labels.npz', labels=image_labels, spike_counts=all_spike_counts_per_image)

    # If loading from previous data
    if load_spike_counts and load_run_dir:
        if previous_data_path is None:
            previous_data_path = f'{load_run_dir}/total_spike_data_with_labels.npz'
        previous_data = np.load(previous_data_path)
        print(f'[INFO] Loaded from {previous_data_path}')
        previous_spike_counts = previous_data['spike_counts']
        previous_labels = previous_data['labels']

        # Combine previous and current spike data and labels
        total_spike_counts = np.concatenate((previous_spike_counts, all_spike_counts_per_image), axis=0)
        total_labels = np.concatenate((previous_labels, image_labels), axis=0)

        # Load and combine spike counts with retries
        if previous_spike_counts_with_retries_path is None:
            previous_spike_counts_with_retries_path = f'{load_run_dir}/total_spike_counts_per_neuron_with_retries.npy'
        previous_spike_counts_with_retries = np.load(previous_spike_counts_with_retries_path)
        total_spike_counts_with_retries = previous_spike_counts_with_retries + spike_mon_exc_count

        # Calculate and combine spike counts without retries
        previous_spike_counts_without_retries = np.sum(previous_spike_counts, axis=0)
        total_spike_counts_without_retries = previous_spike_counts_without_retries + spike_counts_per_neuron_without_retries

    else:
        # If no previous data, use only current run data
        total_spike_counts = all_spike_counts_per_image
        total_labels = image_labels
        total_spike_counts_with_retries = spike_mon_exc_count
        total_spike_counts_without_retries = spike_counts_per_neuron_without_retries

    # Save the combined spike data, labels, and spike counts with and without retries
    np.savez(f'{run_dir}/total_spike_data_with_labels.npz', labels=total_labels, spike_counts=total_spike_counts)
    np.save(f'{run_dir}/total_spike_counts_per_neuron_with_retries.npy', total_spike_counts_with_retries)
    np.save(f'{run_dir}/total_spike_counts_per_neuron_without_retries.npy', total_spike_counts_without_retries)
    
    print(f"[INFO] Total spike data and counts saved for run: {run_dir}.")

def save_spike_data_test(prediction_folder_path: str, image_labels: np.ndarray, all_spike_counts_per_image: np.ndarray,
                                spike_mon_exc_count: np.ndarray) -> None:
    """
    Save and combine spike data and labels from the current run, and optionally load and combine with a previous run.
    
    :param prediction_folder_path: The current run name for saving files.
    :param image_labels: Array of labels for the current run images.
    :param all_spike_counts_per_image: Array of spike counts per image for the current run.
    :param spike_mon_exc_count: Array of spike counts per neuron with retries for the current run.
    """
    
    # Save spike counts without and with retries for current run
    spike_counts_per_neuron_without_retries = np.sum(all_spike_counts_per_image, axis=0)
    np.save(f'{prediction_folder_path}/spike_counts_per_neuron_without_retries.npy', spike_counts_per_neuron_without_retries)
    np.save(f'{prediction_folder_path}/spike_counts_per_neuron_with_retries.npy', spike_mon_exc_count)
    np.savez(f'{prediction_folder_path}/spike_data_with_labels.npz', labels=image_labels, spike_counts=all_spike_counts_per_image)

    print(f"[INFO] Total spike data and counts saved for test : {prediction_folder_path}.")


def plot_spike_counts_with_cbar(run_dir: str, population_exc: int, 
                                load_spike_counts: bool = False, load_run_dir: str = None, 
                                current_data_path: str = None, previous_data_path: str = None) -> None:
    """
    Plot spike counts with a color bar for the current run, and optionally combine and plot with previous run data.

    :param run_dir: The current run name for saving files.
    :param population_exc: The total number of excitatory neurons.
    :param load_spike_counts: Boolean indicating whether to load previous run data.
    :param load_run_dir: The previous run name to load data from if load_spike_counts is True.
    """
    grid_size = int(np.sqrt(population_exc))
    
    # Load current spike data
    if current_data_path is None:
        current_data_path = f'{run_dir}/spike_data_with_labels.npz'
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
    plt.savefig(f'{run_dir}/spike_counts_with_numbers_and_cbar.png')
    plt.show()
    plt.close()

    # If previous data is to be loaded
    if load_spike_counts and load_run_dir:
        if previous_data_path is None:
            previous_data_path = f'{load_run_dir}/total_spike_data_with_labels.npz'
        previous_data = np.load(previous_data_path)
        print(f'[INFO] Loaded from {previous_data_path}')
        previous_spike_counts = previous_data['spike_counts']

        # Combine previous and current spike data
        total_spike_counts = np.concatenate((previous_spike_counts, current_spike_counts), axis=0)
        total_spike_counts_per_neuron_without_retries = np.sum(total_spike_counts, axis=0)

        # Plot total spike counts
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
        with tqdm(total=population_exc, desc="Processing Total Spike Counts", dynamic_ncols=True) as pbar:
            for exc_neuron_idx in range(population_exc):
                neuron_spike_count = total_spike_counts_per_neuron_without_retries[exc_neuron_idx]
                spike_count_matrix = np.array([[neuron_spike_count]])
                row, col = divmod(exc_neuron_idx, grid_size)
                im = axes[row, col].imshow(spike_count_matrix, cmap='hot', vmin=0, vmax=np.max(total_spike_counts_per_neuron_without_retries))
                axes[row, col].axis('off')
                axes[row, col].text(0.5, 0.5, str(neuron_spike_count), color='black', fontsize=12, ha='center', va='center', transform=axes[row, col].transAxes)
                pbar.update(1)
    
        # Add vertical color bar for the total spike counts
        cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])  # Adjust for vertical orientation
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Total Spike Count', fontsize=12)
        cbar.ax.tick_params(labelsize=10)  # Increase color bar tick font size
        plt.savefig(f'{run_dir}/total_spike_counts_with_numbers_and_cbar.png')
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
        
#NEURON LABELING
def get_spike_data_path(
    run_dir: str, 
    load_total_spike_data_with_labels: bool, 
    total_spike_data_with_labels_path: str = None, 
    spike_data_with_labels_path: str = None
) -> str:
    """
    Determines the path for either the total spike data or the regular spike data file based on the flag.
    
    :param run_dir: The directory name for the data.
    :param load_total_spike_data_with_labels: Flag to choose total spike data or regular spike data.
    :param total_spike_data_with_labels_path: Optional custom path for total spike data.
    :param spike_data_with_labels_path: Optional custom path for regular spike data.
    :return: Path to the chosen spike data file.
    """
    # Define default paths if custom paths are not provided
    if total_spike_data_with_labels_path is None:
        total_spike_data_with_labels_path = f'{run_dir}/total_spike_data_with_labels.npz'
    if spike_data_with_labels_path is None:
        spike_data_with_labels_path = f'{run_dir}/spike_data_with_labels.npz'

    # Select path based on the flag
    selected_path = total_spike_data_with_labels_path if load_total_spike_data_with_labels else spike_data_with_labels_path
    print(f'[INFO] Selected path: {selected_path}')

    return selected_path


def crop_spike_data(
    spike_data_path: str,
    run_dir: str,
    start_index: int,
    image_count_spike: int
) -> str:
    """
    Crops the spike data based on the starting index and the number of images, then saves the cropped data.

    :param spike_data_path: Path to the original spike data file.
    :param run_dir: The directory where the cropped data will be saved.
    :param start_index: Starting index for cropping.
    :param image_count_spike: Number of images to include in the cropped data.
    :return: Path to the saved cropped data file.
    """
    # Load the selected spike data
    loaded_data = np.load(spike_data_path)
    print(f'[INFO] Loaded data from {spike_data_path}')
    spike_counts = loaded_data['spike_counts']
    labels = loaded_data['labels']
    
    # Calculate end index based on image count
    end_index = start_index + image_count_spike # when start index is 50k (50001 th image) and image count spike is 10k, end index is 60k
    # thus, spike_counts[50000:60000]
    
    # Validate the indices
    if start_index < 0 or end_index > spike_counts.shape[0]:
        print("The specified range is out of bounds.")
        return None

    # Crop the data
    cropped_spike_counts = spike_counts[start_index:end_index]
    cropped_labels = labels[start_index:end_index]

    # Save the cropped data
    cropped_file_path = f'{run_dir}/cropped_spike_data_with_labels_s{start_index}_c{image_count_spike}.npz'
    os.makedirs(run_dir, exist_ok=True)
    np.savez(cropped_file_path, labels=cropped_labels, spike_counts=cropped_spike_counts)
    print(f"[INFO] Cropped data saved at: {cropped_file_path}")

    return cropped_file_path


def prepare_spike_data_with_labels_folder(
    run_dir: str,
    load_total_spike_data_with_labels: bool,
    use_cropped: bool = False,
    start_index: int = None,
    image_count_spike: int = None,
    total_spike_data_with_labels_path: str = None,
    spike_data_with_labels_path: str = None
) -> str:
    """
    Prepares a folder with the selected spike data (full or cropped) and saves it within the specified run folder.

    :param run_dir: The base directory for saving.
    :param load_total_spike_data_with_labels: Whether to use total spike data or regular spike data.
    :param use_cropped: Whether to crop the spike data.
    :param start_index: Start index for cropping (required if use_cropped is True).
    :param image_count_spike: Number of images to include in the cropped data (required if use_cropped is True).
    :param total_spike_data_with_labels_path: Custom path for total spike data (optional).
    :param spike_data_with_labels_path: Custom path for regular spike data (optional).
    :return: Path to the folder containing the selected spike data.
    """
    
    # Get the path to the spike data file using the custom paths if provided
    spike_data_path = get_spike_data_path(
        run_dir, 
        load_total_spike_data_with_labels, 
        total_spike_data_with_labels_path, 
        spike_data_with_labels_path
    )

    # Determine folder name based on selection criteria
    folder_suffix = "total" if load_total_spike_data_with_labels else "regular"
    folder_suffix += f"_cropped_{start_index}_{image_count_spike}" if use_cropped else "_full"
    spike_data_with_labels_folder_path = f'{run_dir}/{folder_suffix}'
    os.makedirs(spike_data_with_labels_folder_path, exist_ok=True)

    # Load and crop if required, else use full data
    if use_cropped:
        assert start_index is not None and image_count_spike is not None, "Start index and image count must be specified for cropping."
        selected_spike_data_path = crop_spike_data(spike_data_path, run_dir, start_index, image_count_spike)
    else:
        selected_spike_data_path = spike_data_path  # Use the full data

    print(f"[INFO] Prepared data in folder: {spike_data_with_labels_folder_path}")
    print(f"[INFO] Data path for usage: {selected_spike_data_path}")

    return selected_spike_data_path, spike_data_with_labels_folder_path

def load_and_assign_neuron_labels(
    selected_spike_data_path: str,
    spike_data_with_labels_folder_path: str,
    population_exc: int
) -> str:
    """
    Loads spike data, assigns labels based on spike activity, and saves the assigned labels.

    :param selected_spike_data_path: Path to the .npz file with spike data and labels.
    :param population_exc: Number of excitatory neurons.
    :return: Path where the assigned labels are saved.
    """
    # Load spike counts and labels from the .npz file
    loaded_data = np.load(selected_spike_data_path)
    print(f'[INFO] Loaded spike data from {selected_spike_data_path}')
    
    loaded_spike_counts = loaded_data['spike_counts']
    loaded_labels = loaded_data['labels']

    # Initialize assigned labels and max average spike counts
    assigned_labels = np.ones(population_exc, dtype=int) * -1
    maximum_average_spike_counts = [0] * population_exc

    # Ensure shapes match
    assert loaded_spike_counts.shape[0] == loaded_labels.shape[0], \
        "Spike counts and labels must match in number of images."

    # Assign labels based on max average spike counts
    for label in range(10):
        current_label_indices = np.where(loaded_labels == label)[0]
        current_label_count = len(current_label_indices)

        if current_label_count > 0:
            total_spike_counts = np.sum(loaded_spike_counts[current_label_indices], axis=0)
            average_spike_counts = total_spike_counts / current_label_count
            for neuron_idx in range(population_exc):
                if average_spike_counts[neuron_idx] > maximum_average_spike_counts[neuron_idx]:
                    maximum_average_spike_counts[neuron_idx] = average_spike_counts[neuron_idx]
                    assigned_labels[neuron_idx] = label

    assigned_labels_path = f"{spike_data_with_labels_folder_path}/assignments_from_training.npy"
    
    # Save the assigned labels
    np.save(assigned_labels_path, assigned_labels)
    print(f'[INFO] Neuron labels saved at {assigned_labels_path}')

    return assigned_labels_path

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


from collections import Counter

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
    

def save_histogram_data_txt(
    assigned_labels_path: str,
    population_exc: int
) -> None:
    """
    Saves histogram data of neuron labels to a text file.

    :param assigned_labels_path: Path to the file containing assigned neuron labels.
    :param population_exc: Number of excitatory neurons.
    """
    # Load assigned labels
    assigned_labels = np.load(assigned_labels_path)
    print(f'[INFO] Loaded assigned labels from {assigned_labels_path}')

    # Count label occurrences
    label_counts = Counter(assigned_labels)
    
    # Save the histogram data to a text file
    save_folder = os.path.dirname(assigned_labels_path)
    histogram_txt_path = os.path.join(save_folder, 'neuron_labels_histogram.txt')
    
    with open(histogram_txt_path, 'w') as f:
        for label in range(-1, 10):
            count = label_counts.get(label, 0)
            f.write(f'Label {label}: {count}\n')
    
    print(f'[INFO] Neuron label histogram data saved at {histogram_txt_path}')





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


#prediction
def get_predictions_for_current_image(spike_counts_current_image: np.ndarray, assigned_labels_path: str) -> ty.List[int]:
    """
    Predicts the labels for the current image based on spike counts and assigned labels.

    :param spike_counts_current_image: Array of spike counts for the current image.
    :param assigned_labels_path: Path to the assigned labels file.
    :return: List of predicted labels sorted by average spike count in descending order.
    """
    # Load the assigned labels
    assigned_labels = np.load(assigned_labels_path)
    
    predictions = []
    for label in range(10):
        # Get indices of neurons assigned to the current label
        assignment_indices = np.where(assigned_labels == label)[0]
        if len(assignment_indices) > 0:
            # Calculate the total and average spike count for neurons assigned to this label
            total_spike_count = np.sum(spike_counts_current_image[assignment_indices])
            average_spike_count = total_spike_count / len(assignment_indices)
            predictions.append(average_spike_count)
        else:
            # If no neurons are assigned to this label, append 0
            predictions.append(0)

    # Sort labels based on the spike counts in descending order and return the predicted labels
    return list(np.argsort(predictions)[::-1])


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

def get_prediction_folder_path(
    assigned_labels_path: str, 
    image_count_prediction: int, 
    start_index: int = 0, 
    use_test_data_paint: bool = False, 
    use_test_data_mnist: bool = True
) -> str:
    
    """
    Generates and returns the folder path for saving predictions based on dataset type, image count, and start index.
    Ensures the folder is created if it doesn't exist.

    :param assigned_labels_path: Path to the assigned labels .npy file.
    :param image_count_prediction: Total number of images for prediction.
    :param start_index: Starting index for the data selection.
    :param use_test_data_paint: Flag indicating whether the data is from the paint dataset.
    :param use_test_data_mnist: Flag indicating whether to use the MNIST test dataset.
    :return: The folder path where predictions and labels will be saved.
    """
    data_type = "mnist_train_data" if not use_test_data_mnist else "mnist_test_data"
    if use_test_data_paint:
        data_type = "paint_data"
        
    folder_directory = os.path.dirname(assigned_labels_path)
    folder_name = f"prediction_{data_type}_count_{image_count_prediction}_start_{start_index}"
    prediction_folder_path = f'{folder_directory}/{folder_name}'
    
    # Ensure the folder is created
    os.makedirs(prediction_folder_path, exist_ok=True)
    print(f"[INFO] Prediction folder created or already exists at: {prediction_folder_path}")

    return prediction_folder_path

def save_labels_incrementally(prediction_folder_path: str, predicted_label: int, true_label: int,
                              predicted_labels: list, image_labels_in_loop: list,
                              image_index_in_loop: int, image_indexes_in_loop: list, label_predict_range: int = None) -> tuple:
    """
    Appends predicted_label and true_label to lists, and saves them as .npy files at the end 
    when all images have been processed.

    :param prediction_folder_path: Directory path where predictions and labels will be saved.
    :param predicted_label: The current predicted label.
    :param true_label: The current true label.
    :param predicted_labels: List to store predicted labels incrementally.
    :param image_labels_in_loop: List to store true labels incrementally.
    :param image_index_in_loop: The current image index in the loop.
    :param image_indexes_in_loop: List to store image indexes incrementally.
    :param label_predict_range: Specifies the number of last entries to save; saves all if None.
    :return: Updated predicted_labels and image_labels_in_loop lists.
    """
    os.makedirs(prediction_folder_path, exist_ok=True)

    # Append current labels and index to lists
    predicted_labels.append(predicted_label)
    image_labels_in_loop.append(true_label)
    image_indexes_in_loop.append(image_index_in_loop)

    # Define slice range
    range_slice = slice(-label_predict_range, None) if label_predict_range else slice(None)
    
    # Define file paths for saving
    predicted_labels_path = f'{prediction_folder_path}/predicted_labels.npy'
    image_labels_in_loop_path = f'{prediction_folder_path}/image_labels_in_loop.npy'
    image_indexes_in_loop_path = f'{prediction_folder_path}/image_indexes_in_loop.npy'
    
    # Save the specified range of data
    np.save(predicted_labels_path, np.array(predicted_labels[range_slice]))
    np.save(image_labels_in_loop_path, np.array(image_labels_in_loop[range_slice]))
    np.save(image_indexes_in_loop_path, np.array(image_indexes_in_loop[range_slice]))

    return predicted_labels, image_labels_in_loop

def calculate_cumulative_accuracy(prediction_folder_path: str, predicted_labels: list, image_labels_in_loop: list,
                                  cumulative_accuracies: list, label_predict_range: int = None) -> tuple:
    """
    Calculates cumulative accuracy, appends to cumulative_accuracies list, saves as .npy file, and returns updated cumulative_accuracies.

    :param predicted_labels: List of predicted labels.
    :param image_labels_in_loop: List of true labels.
    :param cumulative_accuracies: List to store cumulative accuracies incrementally.
    :return: Updated cumulative_accuracies list and the latest cumulative accuracy.
    """
    # Convert lists to numpy arrays for accuracy calculation
    predicted_labels_np = np.array(predicted_labels)
    image_labels_in_loop_np = np.array(image_labels_in_loop)

    # Calculate cumulative accuracy
    correct_predictions = np.sum(predicted_labels_np == image_labels_in_loop_np)
    cumulative_accuracy = (correct_predictions / len(image_labels_in_loop_np)) * 100

    # Append to cumulative accuracies and save
    cumulative_accuracies.append(cumulative_accuracy)

    range_slice = slice(-label_predict_range, None) if label_predict_range else slice(None)

    cumulative_accuracies_path = f'{prediction_folder_path}/cumulative_accuracies.npy'
    np.save(cumulative_accuracies_path, np.array(cumulative_accuracies[range_slice]))

    return cumulative_accuracies, cumulative_accuracy

def finalize_prediction_report(prediction_folder_path: str, predicted_labels: list, image_labels_in_loop: list,
                               cumulative_accuracies: list, image_indexes_in_loop: list ,image_index_in_loop: int, start_index: int = 0) -> None:

    """
    Combines image_labels_in_loop, predicted_labels, and cumulative_accuracies into a text file report.

    :param prediction_folder_path: Directory path where the final report will be saved.
    :param predicted_labels: List of predicted labels.
    :param image_labels_in_loop: List of true labels.
    :param cumulative_accuracies: List of cumulative accuracy values.
    :param image_index_in_loop: Index of the image in the loop, to be printed for each entry.
    :param start_index: Starting index for the data selection in image labels.
    """
    # Define the save path    
    save_path = f"{prediction_folder_path}/prediction_report.txt"
    tot_ims_seen = [index + 1 for index in image_indexes_in_loop]
    
    with open(save_path, 'w') as f:
        f.write("Total Images Seen\tImage Index in the Loop\tIndex as in Image Labels\tTrue Label\tPredicted Label\tCorrect Classification\tCumulative Accuracy\n")
        for i, (tot_im_seen, image_idx_in_loop, true_label, pred_label, acc) in enumerate(zip(tot_ims_seen, image_indexes_in_loop, image_labels_in_loop, predicted_labels, cumulative_accuracies)):
            is_correct = true_label == pred_label
            current_index = image_idx_in_loop + start_index
            f.write(f"{tot_im_seen}\t{image_idx_in_loop}\t{current_index}\t{true_label}\t{pred_label}\t{is_correct}\t{acc:.2f}%\n")


#ACC
def calculate_accuracy(prediction_folder_path) -> float:
    """
    Calculate accuracy based on true labels and predicted labels, and save to a .txt file.

    :param prediction_folder_path: Directory path where the results will be saved.
    :return: Accuracy as a percentage (float).
    """
    true_labels_path = f'{prediction_folder_path}/image_labels_in_loop.npy'
    true_labels = np.load(true_labels_path)

    predicted_labels_path = f'{prediction_folder_path}/predicted_labels.npy'
    predicted_labels = np.load(predicted_labels_path)

    # Ensure both arrays have the same length
    assert len(true_labels) == len(predicted_labels), "Length of true labels and predicted labels must match."

    # Calculate correct predictions and accuracy
    correct_predictions = np.sum(true_labels == predicted_labels)
    accuracy = (correct_predictions / len(true_labels)) * 100

    # Save results to a .txt file
    save_path = os.path.join(prediction_folder_path, 'accuracy_results.txt')
    with open(save_path, 'w') as f:
        f.write(f'Overall Accuracy: {accuracy:.2f}%\n')
        f.write(f'Total Images: {len(true_labels)}\n')
        f.write(f'Correct Predictions: {correct_predictions}\n')
        f.write(f'Incorrect Predictions: {len(true_labels) - correct_predictions}\n')
    
    print(f"[INFO] Accuracy results saved to {save_path}")
    return accuracy

def calculate_accuracy_per_label(
    prediction_folder_path: str,
    num_labels: int = 10
) -> np.ndarray:
    """
    Calculate accuracy for each label and save to a .txt file.

    :param prediction_folder_path: Path where the results will be saved.
    :param true_labels_path: Path to true labels file.
    :param predicted_labels_path: Path to predicted labels file.
    :param num_labels: Total number of unique labels (default is 10 for digits).
    :return: Accuracy per label as a numpy array.
    """
    true_labels_path = f'{prediction_folder_path}/image_labels_in_loop.npy'
    true_labels = np.load(true_labels_path)

    predicted_labels_path = f'{prediction_folder_path}/predicted_labels.npy'
    predicted_labels = np.load(predicted_labels_path)
    
    # Initialize an array to store accuracy per label
    accuracy_per_label = np.zeros(num_labels)

    # Open the file to save accuracy per label
    save_text_path = os.path.join(prediction_folder_path, 'accuracy_per_label.txt')
    with open(save_text_path, 'w') as f:
        for label in range(num_labels):
            # Get indices where the true label is the current label
            label_indices = np.where(true_labels == label)[0]
            if len(label_indices) > 0:
                # Calculate accuracy for this label
                correct_predictions = np.sum(true_labels[label_indices] == predicted_labels[label_indices])
                accuracy = (correct_predictions / len(label_indices)) * 100
                accuracy_per_label[label] = accuracy

                # Write accuracy for each label to the file
                f.write(f'Label {label} Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(label_indices)})\n')
    
    save_npy_path = os.path.join(prediction_folder_path, 'accuracy_per_label.npy')
    np.save(save_npy_path, accuracy_per_label)
    print(f"[INFO] Accuracy per label results saved to {save_text_path} and {save_npy_path}")

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



def get_spiking_rates_and_labels_from_directory(image_count: int, size_selected: int, directory_path: str):
    """
    Processes PNG images in the given directory, resizes them, extracts labels from filenames, 
    and converts them to spiking rates.
    
    Args:
        image_count (int): The number of images to process.
        size_selected (int): The new size to resize images to (square dimensions).
        directory_path (str): The directory containing PNG images.
        
    Returns:
        image_rates (np.ndarray): Spiking rates for the images.
        image_labels (list[int]): Labels extracted from filenames.
        resized_images (np.ndarray): Resized image data for plotting.
    """
    # new_size = (size_selected, size_selected)
    image_files = sorted(glob(os.path.join(directory_path, "*.png")))[:image_count]  # Get the image files
    
    image_labels = []
    image_rates_list = []
    resized_images_list = []
    
    for png_file in image_files:
        # Extract label from file name (e.g., "image_8.png" -> label 8)
        true_label = int(os.path.splitext(os.path.basename(png_file))[0].split('_')[-1])
        image_labels.append(true_label)

        # Load the PNG image
        image_data = _load_png_image(png_file)  # Assume this function is already defined
        
        # Resize the image
        resized_image = _resize_images([image_data], size_selected)  # Assume this function is defined
        
        # Remove the extra dimension after resizing
        resized_image = resized_image[0]
        
        # Flatten the resized image to match the expected shape (784 elements)
        resized_image_1d = resized_image.flatten()
        
        # Normalize pixel intensities to spiking rates
        image_rates = _convert_to_spiking_rates(resized_image_1d)  # Uses max_rate = 63.75
        image_rates_list.append(image_rates)
        resized_images_list.append(resized_image)  # Keep the resized 2D image for plotting
    
    # Convert lists to numpy arrays for easier manipulation
    image_rates = np.array(image_rates_list)
    resized_images = np.array(resized_images_list)
    
    return image_rates, image_labels, resized_images


# Function to load a PNG image and convert it to MNIST-like format
def _load_png_image(file_path: str):
    img = Image.open(file_path).convert('L')  # Convert image to grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)  # Resize to 28x28 pixels
    img_data = np.array(img)  # Convert to numpy array
    # img_data = 255 - img_data  # Invert colors (MNIST has black background and white digits)
    return img_data


# DEBUG VB CODES

def compare_spike_run_files(run_dir: str):
    # Define file paths
    file_pairs = [
        ("spike_counts_per_neuron_with_retries.npy", "total_spike_counts_per_neuron_with_retries.npy"),
        ("spike_counts_per_neuron_without_retries.npy", "total_spike_counts_per_neuron_without_retries.npy"),
        ("spike_data_with_labels.npz", "total_spike_data_with_labels.npz")
    ]

    # Initialize comparison results
    results = []
    all_identical = True  # Flag to check if all files are identical

    # Compare each pair
    for file1, file2 in file_pairs:
        path1 = os.path.join(run_dir, file1)
        path2 = os.path.join(run_dir, file2)

        # Check if both files exist
        if not (os.path.exists(path1) and os.path.exists(path2)):
            results.append(f"{file1} and {file2} - File missing")
            all_identical = False
            continue

        # Load and compare npy or npz data
        if file1.endswith(".npy"):
            data1 = np.load(path1)
            data2 = np.load(path2)
        elif file1.endswith(".npz"):
            data1 = dict(np.load(path1))
            data2 = dict(np.load(path2))

        # Compare the data
        if np.array_equal(data1, data2):
            results.append(f"{file1} and {file2} - Files are identical")
        else:
            results.append(f"{file1} and {file2} - Files are different")
            all_identical = False

    # Add summary based on comparison results
    if all_identical:
        results.append(
            "All files are identical. This run was either not loaded from a previous run or previous spike counts were not used."
        )
    else:
        results.append(
            "One or more files are different. This indicates that a previous run's spike counts have likely been loaded. "
            "Note: It is possible to select not to use the previous counts in the labeling process."
        )

    # Save results to a text file
    results_path = os.path.join(run_dir, "total_spike_versus_spike_comparison_results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(results))

    print(f"[INFO] Comparison results saved to {results_path}")


def calculate_batch_accuracy(predicted_labels: List[int], true_labels: List[int]) -> float:
    correct_predictions = sum([1 for p, t in zip(predicted_labels, true_labels) if p == t])
    return (correct_predictions / len(predicted_labels)) * 100

def get_batch_accuracies(predicted_labels: List[int], true_labels: List[int], label_predict_range: int) -> List[float]:
    batch_accuracies = []
    for idx in range(0, len(predicted_labels), label_predict_range):
        batch_pred_labels = predicted_labels[idx:idx + label_predict_range]
        batch_true_labels = true_labels[idx:idx + label_predict_range]
        batch_accuracy = calculate_batch_accuracy(batch_pred_labels, batch_true_labels)
        batch_accuracies.append(batch_accuracy)
    return batch_accuracies


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


    
def save_report(report_content: List[str], output_dir: str, filename: str = "training_report.txt") -> None:
    report_path = os.path.join(output_dir, filename)
    with open(report_path, 'w') as report_file:
        for line in report_content:
            report_file.write(line + '\n')
    print(f"[INFO] Saved training report to: {report_path}")

def load_and_analyze_training_data(label_predict_range: int, output_dir: str) -> None:
    """
    Loads training data, performs analysis, and saves report content to a text file.

    Parameters:
    - label_predict_range: Number of images per label prediction batch.
    - output_dir: Directory to save analysis results and report.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data files
    predicted_labels = np.load(f'{output_dir}/predicted_labels.npy')
    image_labels = np.load(f'{output_dir}/image_labels_in_loop.npy')
    # image_indexes_in_loop = np.load(f'{output_dir}/image_indexes_in_loop.npy')
    cumulative_accuracies = np.load(f'{output_dir}/cumulative_accuracies.npy')

    # Analysis and report generation
    last_cumulative_accuracy = cumulative_accuracies[-1]
    total_seen_images = len(predicted_labels) + label_predict_range
    print(f"[INFO] Final cumulative accuracy over all seen images: {last_cumulative_accuracy:.2f}%")
    print(f"[INFO] Total images seen for training: {total_seen_images}")
    
    report_content = [
        f"Final cumulative accuracy over all seen images: {last_cumulative_accuracy:.2f}%",
        f"Total images seen for training: {total_seen_images}"
    ]

    last_batch_pred_labels = predicted_labels[-label_predict_range:]
    last_batch_true_labels = image_labels[-label_predict_range:]
    last_batch_accuracy = calculate_batch_accuracy(last_batch_pred_labels, last_batch_true_labels)
    print(f"[INFO] Accuracy for the last batch of {label_predict_range} images: {last_batch_accuracy:.2f}%")
    report_content.append(f"Accuracy for the last batch of {label_predict_range} images: {last_batch_accuracy:.2f}%")

    batch_accuracies = get_batch_accuracies(predicted_labels, image_labels, label_predict_range)
    for idx, accuracy in enumerate(batch_accuracies, start=1):
        report_content.append(f"Accuracy for batch {idx}: {accuracy:.2f}%")
        print(f"[INFO] Accuracy for batch {idx}: {accuracy:.2f}%")

    # Save batch accuracies for use in plotting
    np.save(f"{output_dir}/batch_accuracies.npy", np.array(batch_accuracies))
    print(f"[INFO] Saved batch accuracies as .npy file to: {output_dir}")

    # Save report content to a text file
    save_report(report_content, output_dir)

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

def count_and_save_synapses(syn_input_exc, syn_exc_inh, syn_inh_exc, run_dir):
    """
    Counts synapses of different types, prints each result separately, and saves the results to a text file.

    Parameters:
    - syn_input_exc: Synapses from input to excitatory neurons
    - syn_exc_inh: Synapses from excitatory to inhibitory neurons
    - syn_inh_exc: Synapses from inhibitory to excitatory neurons
    - run_dir: Directory where the text file with synapse counts will be saved
    """
    # Count each type of synapse
    input_exc_synapse_count = len(syn_input_exc.w_ee)
    exc_inh_synapse_count = len(syn_exc_inh.w_ei)
    inh_exc_synapse_count = len(syn_inh_exc.w_ie)
    total_synapse_count = input_exc_synapse_count + exc_inh_synapse_count + inh_exc_synapse_count

    # Print each synapse count separately
    print("Synapse Counts:")
    print(f"Input to Excitatory: {input_exc_synapse_count}")
    print(f"Excitatory to Inhibitory: {exc_inh_synapse_count}")
    print(f"Inhibitory to Excitatory: {inh_exc_synapse_count}")
    print(f"Total Synapses: {total_synapse_count}")

    # Ensure the run_dir directory exists
    os.makedirs(run_dir, exist_ok=True)

    # Save counts to a text file
    save_path = os.path.join(run_dir, "synapse_counts.txt")
    with open(save_path, 'w') as f:
        f.write("Synapse Counts:\n")
        f.write(f"Input to Excitatory: {input_exc_synapse_count}\n")
        f.write(f"Excitatory to Inhibitory: {exc_inh_synapse_count}\n")
        f.write(f"Inhibitory to Excitatory: {inh_exc_synapse_count}\n")
        f.write(f"Total Synapses: {total_synapse_count}\n")
    print(f"[INFO] Synapse counts saved to '{save_path}'")

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

def plot_final_synapse_weights_rf(run_dir: str, population_exc: int, rf_dimensions_file_path:str, final_weights_path: str = None, save_indiv_figs: bool= False) -> None:
    """
    Plots the final synaptic weights for each excitatory neuron, using the neuron-specific synapse connections
    for a receptive field, and saves a combined plot as well.

    Parameters:
    - run_dir (str): The directory where the plot will be saved.
    - population_exc (int): Total number of excitatory neurons.
    - syn_input_exc: Synapse object connecting input neurons to excitatory neurons.
    - final_weights_path (str): Path to the saved synaptic weights file.
    """
    
    # Load the saved RF dimensions data from the .npz file
     
    rf_data = np.load(rf_dimensions_file_path)
    print(f'[INFO] rf dimensions loaded from {rf_dimensions_file_path}')
    
    # Extract the rf_dimensions array
    rf_dimensions = rf_data['rf_dimensions']
    
    # Access each column separately
    neuron_indices = rf_dimensions[:, 0]  # Neuron Index column
    rf_heights = rf_dimensions[:, 1]      # RF Height column
    rf_widths = rf_dimensions[:, 2]       # RF Width column

    # Define default path if none provided
    if final_weights_path is None:
        final_weights_path = f'{run_dir}/final_synaptic_weights.npy'
    
    # Load the final synaptic weights
    final_weights = np.load(final_weights_path)
    print(f'[INFO] Loaded from {final_weights_path}')

    syn_input_exc_j, syn_input_exc_w_ee = load_synapse_attributes(run_dir)

    # Determine grid size for combined plot
    grid_size = int(np.sqrt(population_exc))
    
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
            
            # Ensure the reshape aligns with (height, width) format
            neuron_weights_reshaped = neuron_weights.reshape(height_calculated, width_calculated)
            normalized_weights = normalize_to_grayscale([neuron_weights_reshaped], vmin=0, vmax=1)[0]
            
            # Save each individual neuron plot with dimensions as (height, width)
            individual_fig, individual_ax = plt.subplots(figsize=(width_calculated, height_calculated))
            individual_ax.imshow(normalized_weights, cmap='gray', vmin=0, vmax=255)
            individual_ax.axis('off')
            plt.tight_layout()
            
            # Define the directory where each neuron's synapse weights will be saved
            os.makedirs(f'{run_dir}/each_neuron_synapse_weights', exist_ok=True)  # Create the directory if it doesn't exist
            save_path = f'{run_dir}/each_neuron_synapse_weights/n_{exc_neuron_idx}_sw_w{width_calculated}_h{height_calculated}.png'
            plt.savefig(save_path) if save_indiv_figs else None
            plt.close(individual_fig)
            
            pbar.update(1)
    
    # Calculate figsize for the combined plot based on neuron dimensions
    max_width = max(width for width, _ in neuron_dimensions)
    max_height = max(height for _, height in neuron_dimensions)
    
    combined_fig, combined_axes = plt.subplots(grid_size, grid_size, figsize=(max_width, max_height))
    
    # Plot each neuron's weights on the combined plot grid
    with tqdm(total=population_exc, desc="Creating Combined Plot", dynamic_ncols=True) as pbar:
        for exc_neuron_idx in range(population_exc):
            row, col = divmod(exc_neuron_idx, grid_size)
            width_calculated, height_calculated = neuron_dimensions[exc_neuron_idx]
            synapse_indices = np.where(syn_input_exc_j == exc_neuron_idx)[0]
            neuron_weights = final_weights[synapse_indices]
            neuron_weights_reshaped = neuron_weights.reshape(height_calculated, width_calculated)
            normalized_weights = normalize_to_grayscale([neuron_weights_reshaped], vmin=0, vmax=1)[0]
            combined_axes[row, col].imshow(normalized_weights, cmap='gray', vmin=0, vmax=255)   
            combined_axes[row, col].axis('off')
            pbar.update(1)
    
    # Save the combined final synapse weights plot
    plt.tight_layout()
    combined_save_path = os.path.join(run_dir, "final_synapse_weights.png")
    plt.savefig(combined_save_path)
    plt.show()
    plt.close(combined_fig)
    print(f"[INFO] Combined synapse weights plot saved as '{combined_save_path}'.")

def divisive_weight_normalization_rf(synapse, population_exc, normalization_dict) -> None:
    """
    Normalizes the synaptic weights of the connections to each post-synaptic neuron,
    using a unique normalization value for each neuron from the provided normalization dictionary.

    Parameters:
    - synapse: The Synapses object containing synaptic weights and connections.
    - population_exc: The number of excitatory neurons (i.e., post-synaptic population size).
    - normalization_dict: A dictionary where each post-synaptic neuron index maps to its normalization value.
    
    Process:
    - For each post-synaptic neuron, the function identifies all synapses connecting to it.
    - It retrieves the normalization value for the current neuron.
    - The sum of synaptic weights for that neuron is calculated, and a normalization factor is determined.
    - The weights are scaled by this factor, ensuring their sum matches the specified normalization value for that neuron.
    """
    for post_idx in range(population_exc):
        # Extract indices of synapses that connect to the current post-synaptic neuron
        target_indices = np.where(synapse.j == post_idx)[0]

        # Extract weights of these synapses
        weights_to_same_post = synapse.w_ee[target_indices]

        # Retrieve the normalization value for the current neuron
        normalization_value = normalization_dict.get(post_idx, 1)  # Default to 1 if not found

        # Calculate sum of weights connected to the current post-synaptic neuron
        sum_of_weights = np.sum(weights_to_same_post)

        # Calculate normalization factor based on the neuron-specific normalization value
        normalization_factor = normalization_value / sum_of_weights if sum_of_weights != 0 else 0
        
        # Update the weights in the Synapses object
        synapse.w_ee[target_indices] *= normalization_factor

    # print(f"[INFO] Weights normalized for each neuron using the provided normalization dictionary.")
    
    