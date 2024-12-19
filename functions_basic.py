# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:10:32 2024

@author: ege-demir
"""
import struct
from PIL import Image
import os
import numpy as np
from collections import Counter
import typing as ty
from typing import List


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

def increase_spiking_rates(image, current_max_rate):
    new_maximum_rate = current_max_rate + 32
    return (image * new_maximum_rate) / current_max_rate

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

    np.save(f"{output_dir}/batch_accuracies.npy", np.array(batch_accuracies))
    print(f"[INFO] Saved batch accuracies as .npy file to: {output_dir}")

    # Save report content to a text file
    save_report(report_content, output_dir)
    

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

    

def normalize_to_grayscale(kernels, vmin=0, vmax=1):
    """
    Normalize weights to a grayscale range from 0 to 255.
    """
    grayscaled_kernels = []
    for kernel in kernels:
        grayscaled_kernels.append(np.floor(np.clip(255 * (kernel - vmin) / (vmax - vmin), 0, 255)))
    return grayscaled_kernels



def save_spike_data(run_dir: str, image_labels: np.ndarray, all_spike_counts_per_image: np.ndarray, spike_mon_exc_count: np.ndarray) -> None:
    """
    Save spike data and labels for the current run.

    :param run_dir: The current run name for saving files.
    :param image_labels: Array of labels for the current run images.
    :param all_spike_counts_per_image: Array of spike counts per image for the current run.
    :param spike_mon_exc_count: Array of spike counts per neuron with retries for the current run.
    """
    # Save spike counts without and with retries for the current run
    spike_counts_per_neuron_without_retries = np.sum(all_spike_counts_per_image, axis=0)
    np.save(f'{run_dir}/spike_counts_per_neuron_without_retries.npy', spike_counts_per_neuron_without_retries)
    np.save(f'{run_dir}/spike_counts_per_neuron_with_retries.npy', spike_mon_exc_count)
    
    # Save the spike data and labels for the current run
    np.savez(f'{run_dir}/spike_data_with_labels.npz', labels=image_labels, spike_counts=all_spike_counts_per_image)
    
    print(f"[INFO] Spike data and counts saved for run: {run_dir}.")

def get_spike_data_path(run_dir: str) -> str:
    """
    Determines the path for the regular spike data file.

    :param run_dir: The directory name for the data.
    :return: Path to the regular spike data file.
    """
    # Define the default path for regular spike data
    spike_data_with_labels_path = f'{run_dir}/spike_data_with_labels.npz'

    print(f'[INFO] Selected path: {spike_data_with_labels_path}')

    return spike_data_with_labels_path

def prepare_spike_data_with_labels_folder(run_dir: str) -> str:
    """
    Prepares a folder with the full spike data and saves it within the specified run folder.

    :param run_dir: The base directory for saving.
    :return: Path to the folder containing the spike data.
    """
    
    # Get the path to the regular spike data file
    spike_data_path = get_spike_data_path(run_dir)
    
    # Determine folder name for full data
    folder_suffix = "regular_full"
    spike_data_with_labels_folder_path = f'{run_dir}/{folder_suffix}'
    os.makedirs(spike_data_with_labels_folder_path, exist_ok=True)

    print(f"[INFO] Prepared data in folder: {spike_data_with_labels_folder_path}")
    print(f"[INFO] Data path for usage: {spike_data_path}")

    return spike_data_path, spike_data_with_labels_folder_path

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


def get_prediction_folder_path(
    assigned_labels_path: str, 
    image_count_prediction: int, 
    start_index: int = 0, 
    use_test_data_mnist: bool = True
) -> str:
    
    """
    Generates and returns the folder path for saving predictions based on dataset type, image count, and start index.
    Ensures the folder is created if it doesn't exist.

    :param assigned_labels_path: Path to the assigned labels .npy file.
    :param image_count_prediction: Total number of images for prediction.
    :param start_index: Starting index for the data selection.
    :param use_test_data_mnist: Flag indicating whether to use the MNIST test dataset.
    :return: The folder path where predictions and labels will be saved.
    """
    data_type = "mnist_train_data" if not use_test_data_mnist else "mnist_test_data"
        
    folder_directory = os.path.dirname(assigned_labels_path)
    folder_name = f"prediction_{data_type}_count_{image_count_prediction}_start_{start_index}"
    prediction_folder_path = f'{folder_directory}/{folder_name}'
    
    # Ensure the folder is created
    os.makedirs(prediction_folder_path, exist_ok=True)
    print(f"[INFO] Prediction folder created or already exists at: {prediction_folder_path}")

    return prediction_folder_path



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
