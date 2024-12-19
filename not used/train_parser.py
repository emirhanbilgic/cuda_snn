# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:57:54 2024

@author: ege-demir

to run directly from terminal
python train_parser.py --image_count 60000 --u 1 --exc_neuron_num 400 --tr_label_pred True --label_predict_range 10000 --save_state True --save_state_range 1000 --rf True --rf_config '{"input_size": 28, "filter_size": 27, "stride": 1, "padding": 9, "output_size": 20, "directory_of_mapping": "input_to_output_mapping/refined_rf_trials", "norm_neuron_spesific": true, "rf_inh2exc": false}' --notes "Experiment 1"

"""
import functions
import classes
import pygame  # Import pygame for sound playback
import brian2
import numpy as np
from tqdm import tqdm
import os
import argparse
import json


# Utility function to handle boolean arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Parse arguments dynamically for RunConfiguration
parser = argparse.ArgumentParser()

# Add all possible arguments
parser.add_argument("--image_count", type=int, required=True, help="Number of images to process (required)")
parser.add_argument("--create_dir", type=str2bool, default=True, help="Whether to create a new directory for the run")
parser.add_argument("--user_run_dir", type=str, default=None, help="Path to a user-specified directory for the run")
parser.add_argument("--load_from_a_file", type=str2bool, default=False, help="Whether to load a run configuration from a file")
parser.add_argument("--load_run_dir", type=str, default=None, help="Directory to load run configuration from")
parser.add_argument("--load_weights", type=str2bool, default=False, help="Whether to load pre-trained weights")
parser.add_argument("--load_theta_values", type=str2bool, default=False, help="Whether to load pre-trained theta values")
parser.add_argument("--load_spike_counts", type=str2bool, default=False, help="Whether to load pre-trained spike counts")
parser.add_argument("--save_state", type=str2bool, default=False, help="Whether to save the simulation state during the run")
parser.add_argument("--save_state_range", type=int, default=None, help="Interval for saving simulation states")
parser.add_argument("--tr_label_pred", type=str2bool, default=False, help="Whether to perform training with label prediction")
parser.add_argument("--label_predict_range", type=int, default=None, help="Range for label prediction")
parser.add_argument("--size_selected", type=int, default=28, help="Size of input images")
parser.add_argument("--g_e_multiplier", type=float, default=1, help="Multiplier for excitatory synaptic conductance")
parser.add_argument("--normalization_val", type=float, default=78, help="Value to normalize synaptic weights to")
parser.add_argument("--exc_neuron_num", type=int, default=100, help="Number of excitatory neurons")
parser.add_argument("--skip_norm", type=str2bool, default=False, help="Whether to skip normalization during training")
parser.add_argument("--max_rate", type=float, default=63.75, help="Maximum rate for Poisson input neurons (Hz)")
parser.add_argument("--start_index", type=int, default=0, help="Start index for image dataset")
parser.add_argument("--seed_train", type=str2bool, default=True, help="Whether to seed the training data")
parser.add_argument("--seed_test", type=str2bool, default=True, help="Whether to seed the test data")
parser.add_argument("--file_suffix_optional", type=str, default="", help="Optional suffix for filenames")
parser.add_argument("--rf", type=str2bool, default=False, help="Whether to use receptive field configurations")
parser.add_argument("--rf_config", type=str, default=None, help="JSON string containing receptive field configuration")
parser.add_argument("--u", type=float, default=1, help="User-defined parameter (default=1)")
parser.add_argument("--notes", type=str, default="", help="Notes about the run")

args = parser.parse_args()

# Parse receptive field configuration if provided
rf_config = None
if args.rf_config:
    try:
        rf_config = json.loads(args.rf_config)
        print("[DEBUG] Parsed rf_config:", rf_config)
    except json.JSONDecodeError as e:
        print("[ERROR] Failed to parse rf_config:", e)
        raise

# Check if the directory for RF mapping exists
if rf_config and not os.path.exists(rf_config["directory_of_mapping"]):
    raise FileNotFoundError(f"Directory '{rf_config['directory_of_mapping']}' does not exist.")

# Initialize RunConfiguration dynamically
run_config = classes.RunConfiguration(
    image_count=args.image_count,
    create_dir=args.create_dir,
    user_run_dir=args.user_run_dir,
    load_from_a_file=args.load_from_a_file,
    load_run_dir=args.load_run_dir,
    load_weights=args.load_weights,
    load_theta_values=args.load_theta_values,
    load_spike_counts=args.load_spike_counts,
    save_state=args.save_state,
    save_state_range=args.save_state_range,
    tr_label_pred=args.tr_label_pred,
    label_predict_range=args.label_predict_range,
    size_selected=args.size_selected,
    g_e_multiplier=args.g_e_multiplier,
    normalization_val=args.normalization_val,
    exc_neuron_num=args.exc_neuron_num,
    skip_norm=args.skip_norm,
    max_rate=args.max_rate,
    start_index=args.start_index,
    seed_train=args.seed_train,
    seed_test=args.seed_test,
    file_suffix_optional=args.file_suffix_optional,
    rf=args.rf,
    rf_config=rf_config,
    u=args.u,
    notes=args.notes
)


# Add this line to output `run_dir` for external scripts
print(f"[RUN_DIR] {run_config.run_dir}")


# Use run_config for the rest of your script
print("Initialized RunConfiguration:")
for key, value in run_config.__dict__.items():
    print(f"  {key}: {value}")

#TRAINNG CODE WITH LABELING PRED ACC
if 'i' in globals():
    del i
if 'j' in globals():
    del j

sum_sp_max = 5 #not iterative yet, in order not to mess it up, 5.

# Parameters for excitatory and inhibitory neurons
E_rest_exc  = -65 * brian2.mV
E_rest_inh  = -60 * brian2.mV
E_exc_for_exc = 0 * brian2.mV
E_inh_for_exc = -100 * brian2.mV
E_exc_for_inh = 0 * brian2.mV
E_inh_for_inh = -85 * brian2.mV
tau_lif_exc = 100 * brian2.ms
tau_lif_inh = 10 * brian2.ms
tau_ge  = 1 * brian2.ms
tau_gi  = 2 * brian2.ms
tau_theta =  1e7 * brian2.ms
theta_inc_exc =  0.05 * brian2.mV
v_threshold_exc = -52 * brian2.mV
v_threshold_inh = -40 * brian2.mV
v_offset_exc = 20 * brian2.mV
v_reset_exc = -65 * brian2.mV
v_reset_inh = -45 * brian2.mV
population_exc = run_config.exc_neuron_num  # Excitatory neuron population
population_inh = population_exc  # Inhibitory neuron population

# Synapse Parameters
tau_Apre_ee = 20 * brian2.ms
tau_Apost1_ee = 20 * brian2.ms
eta_pre_ee  = 0.0001
eta_post_ee = 0.01
w_min_ee = 0
w_max_ee = 1
w_ei_ = 10.4
w_ie_ = 17

g_e_multiplier = run_config.g_e_multiplier

# Neuron equations for excitatory and inhibitory populations
ng_eqs_exc = """
dv/dt = ((E_rest_exc - v) + g_e*(E_exc_for_exc - v) + g_i*(E_inh_for_exc - v))/tau_lif_exc : volt (unless refractory)
dg_e/dt = -g_e/tau_ge : 1
dg_i/dt = -g_i/tau_gi : 1
dtheta/dt = -theta/tau_theta  : volt
"""

ng_eqs_inh = """
dv/dt = ((E_rest_inh - v) + g_e*(E_exc_for_inh - v) + g_i*(E_inh_for_inh - v))/tau_lif_inh : volt (unless refractory)
dg_e/dt = -g_e/tau_ge : 1
dg_i/dt = -g_i/tau_gi : 1
"""

# Threshold and reset equations for both populations
ng_threshold_exc = "v > v_threshold_exc - v_offset_exc + theta"
ng_reset_exc = "v = v_reset_exc; theta += theta_inc_exc"

ng_threshold_inh = "v > v_threshold_inh"
ng_reset_inh = "v = v_reset_inh"

# Synapse equations for excitatory -> excitatory (with STDP)
syn_eqs_ee_training = """
w_ee : 1
dApre_ee/dt = -Apre_ee/tau_Apre_ee : 1 (event-driven)
dApost1_ee/dt = -Apost1_ee/tau_Apost1_ee : 1 (event-driven)
"""

syn_on_pre_ee_training = """
Apre_ee = 1
w_ee = clip(w_ee + (-eta_pre_ee * Apost1_ee), w_min_ee, w_max_ee)
g_e_post += w_ee * g_e_multiplier
"""

syn_on_post_ee_training = """
w_ee = clip(w_ee + (eta_post_ee * Apre_ee), w_min_ee, w_max_ee)
Apost1_ee = 1
"""

# Create neuron groups for excitatory and inhibitory neurons
neuron_group_exc = brian2.NeuronGroup(N=population_exc, model=ng_eqs_exc, threshold=ng_threshold_exc, reset=ng_reset_exc, refractory=5*brian2.ms, method="euler")
neuron_group_inh = brian2.NeuronGroup(N=population_inh, model=ng_eqs_inh, threshold=ng_threshold_inh, reset=ng_reset_inh, refractory=2*brian2.ms, method="euler")

# Set initial values
neuron_group_exc.v = E_rest_exc - 40 * brian2.mV
neuron_group_inh.v = E_rest_inh - 40 * brian2.mV

if run_config.load_theta_values and os.path.exists(f'{run_config.load_run_dir}/neuron_group_exc_theta.npy'):
    theta_values = np.load(f"{run_config.load_run_dir}/neuron_group_exc_theta.npy")
    neuron_group_exc.theta = theta_values * brian2.volt
    print(f'[INFO] Loaded saved theta values from run name {run_config.load_run_dir}.')  
else:
    neuron_group_exc.theta = 20 * brian2.mV
    print("[INFO] Theta values are set at 20 mV.")

# Define PoissonGroup for input image (MNIST)
tot_input_num = run_config.size_selected * run_config.size_selected
image_input = brian2.PoissonGroup(N=tot_input_num, rates=0*brian2.Hz)  # Will set the rates based on the image

# Synapse connecting input neurons to excitatory neurons
syn_input_exc = brian2.Synapses(image_input, neuron_group_exc, model=syn_eqs_ee_training, on_pre=syn_on_pre_ee_training, on_post=syn_on_post_ee_training, method="euler")
run_config.connect_synapses_input2exc(syn_input_exc)

if run_config.load_weights and os.path.exists(f'{run_config.load_run_dir}/final_synaptic_weights.npy'):
    saved_weights = np.load(f'{run_config.load_run_dir}/final_synaptic_weights.npy')
    syn_input_exc.w_ee[:] = saved_weights  # Apply the saved weights to the synapses
    
    print(f'[INFO] Loaded saved synaptic weights from run name {run_config.load_run_dir}.')    
else:
    syn_input_exc.w_ee[:] = "rand() * 0.3"
    print("[INFO] Initialized synaptic weights randomly.")

syn_input_exc.delay = 10 * brian2.ms

# Synapse connecting excitatory -> inhibitory neurons (one-to-one)
syn_exc_inh = brian2.Synapses(neuron_group_exc, neuron_group_inh, model="w_ei : 1", on_pre="g_e_post += w_ei", method="euler")
syn_exc_inh.connect(j='i')  # One-to-one connection
syn_exc_inh.w_ei = w_ei_

# Synapse connecting inhibitory -> excitatory neurons (all-to-all except same index)
syn_inh_exc = brian2.Synapses(neuron_group_inh, neuron_group_exc, model="w_ie : 1", on_pre="g_i_post += w_ie", method="euler")
#syn_inh_exc.connect("i != j")  # Inhibitory neurons connect to all excitatory neurons except the one with the same index
run_config.connect_synapses_inh2exc(syn_inh_exc)
syn_inh_exc.w_ie = w_ie_


functions.count_and_save_synapses(syn_input_exc, syn_exc_inh, syn_inh_exc, run_config.run_dir)

#weight_mon = StateMonitor(syn_input_exc, 'w_ee', record=False)
spike_mon_exc = brian2.SpikeMonitor(neuron_group_exc)
net = brian2.Network(neuron_group_exc, neuron_group_inh, image_input, syn_input_exc, syn_exc_inh, syn_inh_exc, spike_mon_exc)
net.store('initialized')  # Store the initialized state for later reset
print(f'[INFO] Processing total of {run_config.image_count} images starting from index {run_config.start_index}...')

image_input_rates, image_labels, image_intensities = functions.get_spiking_rates_and_labels(use_test_data_mnist = False,
                                                                                  image_count=run_config.image_count,
                                                                                  seed_data=run_config.seed_train,
                                                                                  size_selected=run_config.size_selected,
                                                                                  start_index=run_config.start_index)

all_spike_counts_per_image = []
max_rate_current_image = run_config.max_rate
predicted_labels = []
image_labels_in_loop = []
cumulative_accuracies = []
image_indexes_in_loop = []
spike_data_within_training = []

with tqdm(total=run_config.image_count, desc="Processing Images", dynamic_ncols=True) as pbar:
    previous_spike_counts = np.zeros(population_exc, dtype=int)
    for image_index_in_loop in range(run_config.image_count):
        current_image_index = run_config.start_index + image_index_in_loop
        tot_seen_images = image_index_in_loop + 1
        image_retries = 0
        successful_training = False
        while not successful_training:
            previous_spike_counts = spike_mon_exc.count[:]
            image_input.rates = image_input_rates[image_index_in_loop] * brian2.Hz
            if not run_config.skip_norm:
                run_config.apply_normalization(syn_input_exc, population_exc)
                
            net.run(350 * brian2.ms)
            
            current_spike_counts = spike_mon_exc.count[:]
            spike_counts_current_image = current_spike_counts - previous_spike_counts
            max_spike_count = np.max(spike_counts_current_image)
            neurons_with_max_spikes = np.where(spike_counts_current_image == max_spike_count)[0]
            sum_spike_count = np.sum(spike_counts_current_image)
            
            if (run_config.tr_label_pred == False or run_config.label_predict_range is None) or (tot_seen_images <= run_config.label_predict_range):
                pbar.set_description(
                    f"Image #{current_image_index}, #{image_index_in_loop} in loop,"
                    f"(label: {image_labels[image_index_in_loop]}), Retry: {image_retries}, Sum Spikes: {sum_spike_count}, "
                    f"Max Spikes: {max_spike_count}, Neurons: {neurons_with_max_spikes.tolist()}, Current Max Rate: {max_rate_current_image}"
                )
            
            if sum_spike_count >= sum_sp_max:
                successful_training = True
                all_spike_counts_per_image.append(spike_counts_current_image.copy())
                
                if run_config.save_state and run_config.save_state_range is not None and (tot_seen_images % run_config.save_state_range == 0):
                    print("Running Task save_state")
                    save_simulation_state_folder = f'{run_config.run_dir}/save_state_totim_{tot_seen_images}'
                    last_image_index = current_image_index
                    functions.save_simulation_state(run_config.run_dir, last_image_index, syn_input_exc, neuron_group_exc, neuron_group_inh, save_simulation_state_folder)
                    
                if run_config.tr_label_pred and run_config.label_predict_range is not None and (tot_seen_images % run_config.label_predict_range == 0):
                    print("Running Task X: save labels and images for every label_predict_range images")
                    start_index_train_labeling = tot_seen_images - run_config.label_predict_range # 0 for 10k tot seen images
                    end_index_train_labeling = tot_seen_images # 10k for 10k tot seen images
                    
                    image_labels_within_training = image_labels[start_index_train_labeling:end_index_train_labeling] #[0:10k]
                    spike_data_within_training = all_spike_counts_per_image[-run_config.label_predict_range:]
                    
                    spike_data_within_training_path = f'{run_config.run_dir}/tr_label_pred/spike_data_{int(start_index_train_labeling+1)}_{end_index_train_labeling}.npz'
                    os.makedirs(os.path.dirname(spike_data_within_training_path), exist_ok=True)
                    
                    # Save the spike data for the current range
                    np.savez(spike_data_within_training_path, labels=image_labels_within_training, spike_counts=spike_data_within_training)
                    spike_data_with_labels_folder_path = f'{run_config.run_dir}/tr_label_pred/spike_data_w_labels_{int(start_index_train_labeling+1)}_{end_index_train_labeling}'
                    os.makedirs(spike_data_with_labels_folder_path, exist_ok=True)
                    
                    assigned_labels_within_training_path = functions.load_and_assign_neuron_labels(selected_spike_data_path = spike_data_within_training_path, 
                                                                                         spike_data_with_labels_folder_path = spike_data_with_labels_folder_path,
                                                                                         population_exc = population_exc)

                if run_config.tr_label_pred and run_config.label_predict_range is not None and (tot_seen_images > run_config.label_predict_range):  # e.g., 10001st image, 10002nd image etc
                    # print("Running Task Y: using the assigned_labels_path from task X, do predictions, calculate acc etc")

                    start_index_train_predict = tot_seen_images - 1 #index is 10k, meaning start with 10001th image
                    end_index_train_predict = tot_seen_images -1 + run_config.label_predict_range #index is 20k, meaning end with 20k th image (as [x:y] means include xth index but not yth index)
                    
                    predictions = functions.get_predictions_for_current_image(spike_counts_current_image = spike_counts_current_image,
                                                            assigned_labels_path = assigned_labels_within_training_path)

                    true_label = image_labels[image_index_in_loop]
                    predicted_label = predictions[0]  # The top predicted label

                    if (tot_seen_images - 1) % run_config.label_predict_range == 0:
                        prediction_folder_within_training_path = f'{spike_data_with_labels_folder_path}/predictions_{int(start_index_train_predict+1)}_to_{end_index_train_predict}'
                        os.makedirs(prediction_folder_within_training_path, exist_ok=True)
                        
                    predicted_labels, image_labels_in_loop = functions.save_labels_incrementally(prediction_folder_within_training_path,
                                                                                             predicted_label,
                                                                                             true_label,
                                                                                             predicted_labels,
                                                                                             image_labels_in_loop,
                                                                                             image_index_in_loop,
                                                                                             image_indexes_in_loop,
                                                                                             run_config.label_predict_range)

                    cumulative_accuracies, cumulative_accuracy = functions.calculate_cumulative_accuracy(prediction_folder_within_training_path,
                                                                                               predicted_labels,
                                                                                               image_labels_in_loop,
                                                                                               cumulative_accuracies,
                                                                                               run_config.label_predict_range)
                    
                    functions.finalize_prediction_report(prediction_folder_within_training_path,
                                               predicted_labels[-run_config.label_predict_range:],
                                               image_labels_in_loop[-run_config.label_predict_range:],
                                               cumulative_accuracies[-run_config.label_predict_range:],
                                               image_indexes_in_loop[-run_config.label_predict_range:],
                                               run_config.start_index)

                    pbar.set_description(
                        f"Image #{current_image_index} (Loop Index: {image_index_in_loop}, True Label: {true_label}, "
                        f"Predicted: {predicted_label}) | Cumulative Accuracy: {cumulative_accuracy:.2f}% | "
                        f"Retry: {image_retries} | Sum Spikes: {sum_spike_count} | Max Spikes: {max_spike_count} | "
                        f"Neurons with Max Spikes: {neurons_with_max_spikes.tolist()} | Current Max Rate: {max_rate_current_image}"
                    )

                image_input.rates = 0 * brian2.Hz
                net.run(150 * brian2.ms)
                max_rate_current_image = run_config.max_rate
                pbar.update(1)
            else:
                image_retries += 1
                max_rate_current_image += 32  # Increase the rate
                image_input_rates[image_index_in_loop] = functions.increase_spiking_rates(image_input_rates[image_index_in_loop], max_rate_current_image)
                image_input.rates = 0 * brian2.Hz
                net.run(150 * brian2.ms)


grid_size = int(np.sqrt(run_config.exc_neuron_num))
print(f"[INFO] Exc neuron group size is {grid_size}x{grid_size} = {run_config.exc_neuron_num}")

last_image_index = current_image_index
functions.save_simulation_state(run_config.run_dir, last_image_index, syn_input_exc, neuron_group_exc, neuron_group_inh)
all_spike_counts_per_image = np.array(all_spike_counts_per_image)
spike_counts_per_neuron_with_retries = spike_mon_exc.count[:]

functions.save_and_combine_spike_data(run_dir=run_config.run_dir, image_labels=image_labels, all_spike_counts_per_image=all_spike_counts_per_image,
                                spike_mon_exc_count=spike_counts_per_neuron_with_retries, load_spike_counts=run_config.load_spike_counts, 
                                load_run_dir=run_config.load_run_dir, previous_data_path = None, previous_spike_counts_with_retries_path = None)

functions.compare_spike_run_files(run_config.run_dir)


if run_config.tr_label_pred and run_config.label_predict_range is not None:
    
    output_dir = f"{run_config.run_dir}/tr_label_pred"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths for saving
    predicted_labels_path = f'{output_dir}/predicted_labels.npy'
    image_labels_in_loop_path = f'{output_dir}/image_labels_in_loop.npy'
    image_indexes_in_loop_path = f'{output_dir}/image_indexes_in_loop.npy'
    cumulative_accuracies_path = f'{output_dir}/cumulative_accuracies.npy'
    
    # Save the specified range of data
    np.save(predicted_labels_path, np.array(predicted_labels))
    np.save(image_labels_in_loop_path, np.array(image_labels_in_loop))
    np.save(image_indexes_in_loop_path, np.array(image_indexes_in_loop))
    np.save(cumulative_accuracies_path, np.array(cumulative_accuracies))


    functions.finalize_prediction_report(output_dir,
                            predicted_labels,
                            image_labels_in_loop,
                            cumulative_accuracies,
                            image_indexes_in_loop,
                            run_config.start_index)
    
    
    functions.load_and_analyze_training_data(run_config.label_predict_range, output_dir)
    functions.plot_training_results(output_dir, run_config.label_predict_range)

run_config.plot_synapse_weights(run_config.run_dir, population_exc, save_indiv_figs = False)
functions.plot_spike_counts_with_cbar(run_dir=run_config.run_dir, population_exc=population_exc, 
                            load_spike_counts=run_config.load_spike_counts,
                            load_run_dir=run_config.load_run_dir) # not writing the None values from now on


pygame.mixer.init()
pygame.mixer.music.load("butun.mp3")
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(5)

pygame.mixer.init()
pygame.mixer.music.load("horn.mp3")
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(5)
    

