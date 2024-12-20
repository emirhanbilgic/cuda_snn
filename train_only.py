# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:57:54 2024

@author: ege-demir
"""
import functions_basic
import functions_basic_plots

import pygame
import numpy as np
from tqdm import tqdm
import os
# Standard Brian2 import
from brian2 import *
import brian2
from brian2 import set_device

# Enable GPU usage via Brian2CUDA
import brian2cuda
set_device("cuda_standalone")

# RUN CONFIG
run_dir = "deneme5"
image_count = 100
exc_neuron_num = 400
tr_label_pred = True
label_predict_range = 10
size_selected = 28 # input is 28x28
max_rate = 63.75
seed_train = True
start_index = 0 # start index of train
skip_norm = False # skips the normalization algorithm if True
normalization_val = 78

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
population_exc = exc_neuron_num  # Excitatory neuron population
population_inh = population_exc  # Inhibitory neuron population

# Synapse Parameters
tau_apre_ee = 20 * brian2.ms
tau_apost1_ee = 20 * brian2.ms
eta_pre_ee  = 0.0001
eta_post_ee = 0.01
w_min_ee = 0
w_max_ee = 1
w_ei_ = 10.4
w_ie_ = 17
tau_apost2_ee = 40 * brian2.ms  # Time constant for apost2_ee

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

syn_eqs_ee_training = """
w_ee : 1                                               
apost2_prev_ee : 1
dapre_ee/dt = -apre_ee/tau_apre_ee : 1 (event-driven)
dapost1_ee/dt = -apost1_ee/tau_apost1_ee : 1 (event-driven)
dapost2_ee/dt = -apost2_ee/tau_apost2_ee : 1 (event-driven)
"""

syn_on_pre_ee_training = """
apre_ee = 1
w_ee = clip(w_ee + (-eta_pre_ee * apost1_ee), w_min_ee, w_max_ee)
g_e_post += w_ee
"""

syn_on_post_ee_training = """
apost2_prev_ee = apost2_ee
w_ee = clip(w_ee + (eta_post_ee * apre_ee * apost2_prev_ee), w_min_ee, w_max_ee)
apost1_ee = 1 
apost2_ee = 1
"""

# Create neuron groups for excitatory and inhibitory neurons
neuron_group_exc = brian2.NeuronGroup(N=population_exc, model=ng_eqs_exc, threshold=ng_threshold_exc, reset=ng_reset_exc, refractory=5*brian2.ms, method="euler")
neuron_group_inh = brian2.NeuronGroup(N=population_inh, model=ng_eqs_inh, threshold=ng_threshold_inh, reset=ng_reset_inh, refractory=2*brian2.ms, method="euler")

# Set initial values
neuron_group_exc.v = E_rest_exc - 40 * brian2.mV
neuron_group_inh.v = E_rest_inh - 40 * brian2.mV
neuron_group_exc.theta = 20 * brian2.mV

# Define PoissonGroup for input image (MNIST)
tot_input_num = size_selected * size_selected
image_input = brian2.PoissonGroup(N=tot_input_num, rates=0*brian2.Hz)  # Will set the rates based on the image

# Synapse connecting input neurons to excitatory neurons
syn_input_exc = brian2.Synapses(image_input, neuron_group_exc, model=syn_eqs_ee_training, on_pre=syn_on_pre_ee_training, on_post=syn_on_post_ee_training, method="euler")
syn_input_exc.connect()
syn_input_exc.w_ee[:] = "rand() * 0.3"
syn_input_exc.delay = 10 * brian2.ms

# Synapse connecting excitatory -> inhibitory neurons (one-to-one)
syn_exc_inh = brian2.Synapses(neuron_group_exc, neuron_group_inh, model="w_ei : 1", on_pre="g_e_post += w_ei", method="euler")
syn_exc_inh.connect(j='i')  # One-to-one connection
syn_exc_inh.w_ei = w_ei_

# Synapse connecting inhibitory -> excitatory neurons (all-to-all except same index)
syn_inh_exc = brian2.Synapses(neuron_group_inh, neuron_group_exc, model="w_ie : 1", on_pre="g_i_post += w_ie", method="euler")
syn_inh_exc.connect("i != j")
syn_inh_exc.w_ie = w_ie_

# Create monitors
spike_mon_exc = brian2.SpikeMonitor(neuron_group_exc)
net = brian2.Network(neuron_group_exc, neuron_group_inh, image_input, syn_input_exc, syn_exc_inh, syn_inh_exc, spike_mon_exc)

# Load image input rates and labels
image_input_rates, image_labels, image_intensities = functions_basic.get_spiking_rates_and_labels(
    use_test_data_mnist=False,
    image_count=image_count,
    seed_data=seed_train,
    size_selected=size_selected,
    start_index=start_index
)

# **Perform optional normalization before running the simulation**
if not skip_norm:
    functions_basic.divisive_weight_normalization(syn_input_exc, population_exc, normalization_value=normalization_val)

all_spike_counts_per_image = []
max_rate_current_image = max_rate
predicted_labels = []
image_labels_in_loop = []
cumulative_accuracies = []
image_indexes_in_loop = []
spike_data_within_training = []

# Use tqdm to display progress
with tqdm(total=image_count, desc="Processing Images", dynamic_ncols=True) as pbar:
    previous_spike_counts = np.zeros(population_exc, dtype=int)
    
    # Loop through each image
    for image_index_in_loop in range(image_count):
        current_image_index = start_index + image_index_in_loop
        tot_seen_images = image_index_in_loop + 1
        image_retries = 0
        successful_training = False
        
        while not successful_training:
            previous_spike_counts = spike_mon_exc.count[:]
            image_input.rates = image_input_rates[image_index_in_loop] * brian2.Hz
            
            # Note: No normalization here - done before the simulation started
            
            # Run the simulation for this image
            net.run(350 * brian2.ms)
            
            # Calculate spike counts
            current_spike_counts = spike_mon_exc.count[:]
            spike_counts_current_image = current_spike_counts - previous_spike_counts
            max_spike_count = np.max(spike_counts_current_image)
            neurons_with_max_spikes = np.where(spike_counts_current_image == max_spike_count)[0]
            sum_spike_count = np.sum(spike_counts_current_image)
            
            if (tr_label_pred == False or label_predict_range is None) or (tot_seen_images <= label_predict_range):
                pbar.set_description(
                    f"Image #{current_image_index}, #{image_index_in_loop} in loop,"
                    f"(label: {image_labels[image_index_in_loop]}), Retry: {image_retries}, Sum Spikes: {sum_spike_count}, "
                    f"Max Spikes: {max_spike_count}, Neurons: {neurons_with_max_spikes.tolist()}, Current Max Rate: {max_rate_current_image}"
                )
            
            if sum_spike_count >= 5:
                successful_training = True
                all_spike_counts_per_image.append(spike_counts_current_image.copy())
                
                # Handle label prediction and saving
                if tr_label_pred and label_predict_range is not None:
                    if (tot_seen_images % label_predict_range == 0):
                        print("Running Task X: save labels and images for every label_predict_range images")
                        start_index_train_labeling = tot_seen_images - label_predict_range
                        end_index_train_labeling = tot_seen_images
                        
                        image_labels_within_training = image_labels[start_index_train_labeling:end_index_train_labeling]
                        spike_data_within_training = all_spike_counts_per_image[-label_predict_range:]
                        
                        spike_data_within_training_path = f'{run_dir}/tr_label_pred/spike_data_{int(start_index_train_labeling+1)}_{end_index_train_labeling}.npz'
                        os.makedirs(os.path.dirname(spike_data_within_training_path), exist_ok=True)
                        np.savez(spike_data_within_training_path, labels=image_labels_within_training, spike_counts=spike_data_within_training)
                        
                        spike_data_with_labels_folder_path = f'{run_dir}/tr_label_pred/spike_data_w_labels_{int(start_index_train_labeling+1)}_{end_index_train_labeling}'
                        os.makedirs(spike_data_with_labels_folder_path, exist_ok=True)
                        assigned_labels_within_training_path = functions_basic.load_and_assign_neuron_labels(
                            selected_spike_data_path=spike_data_within_training_path,
                            spike_data_with_labels_folder_path=spike_data_with_labels_folder_path,
                            population_exc=population_exc
                        )
                    
                    if tot_seen_images > label_predict_range:
                        start_index_train_predict = tot_seen_images - 1
                        end_index_train_predict = tot_seen_images - 1 + label_predict_range
                        
                        predictions = functions_basic.get_predictions_for_current_image(
                            spike_counts_current_image=spike_counts_current_image,
                            assigned_labels_path=assigned_labels_within_training_path
                        )

                        true_label = image_labels[image_index_in_loop]
                        predicted_label = predictions[0]

                        if (tot_seen_images - 1) % label_predict_range == 0:
                            prediction_folder_within_training_path = f'{spike_data_with_labels_folder_path}/predictions_{int(start_index_train_predict+1)}_to_{end_index_train_predict}'
                            os.makedirs(prediction_folder_within_training_path, exist_ok=True)
                        
                        predicted_labels, image_labels_in_loop = functions_basic.save_labels_incrementally(
                            prediction_folder_within_training_path,
                            predicted_label,
                            true_label,
                            predicted_labels,
                            image_labels_in_loop,
                            image_index_in_loop,
                            image_indexes_in_loop,
                            label_predict_range
                        )

                        cumulative_accuracies, cumulative_accuracy = functions_basic.calculate_cumulative_accuracy(
                            prediction_folder_within_training_path,
                            predicted_labels,
                            image_labels_in_loop,
                            cumulative_accuracies,
                            label_predict_range
                        )
                        functions_basic.finalize_prediction_report(
                            prediction_folder_within_training_path,
                            predicted_labels[-label_predict_range:],
                            image_labels_in_loop[-label_predict_range:],
                            cumulative_accuracies[-label_predict_range:],
                            image_indexes_in_loop[-label_predict_range:],
                            start_index
                        )

                        pbar.set_description(
                            f"Image #{current_image_index} (Loop Index: {image_index_in_loop}, True Label: {true_label}, "
                            f"Predicted: {predicted_label}) | Cumulative Accuracy: {cumulative_accuracy:.2f}% | "
                            f"Retry: {image_retries} | Sum Spikes: {sum_spike_count} | Max Spikes: {max_spike_count} | "
                            f"Neurons with Max Spikes: {neurons_with_max_spikes.tolist()} | Current Max Rate: {max_rate_current_image}"
                        )

                image_input.rates = 0 * brian2.Hz
                net.run(150 * brian2.ms)
                max_rate_current_image = max_rate
                pbar.update(1)
            else:
                image_retries += 1
                max_rate_current_image += 32
                image_input_rates[image_index_in_loop] = functions_basic.increase_spiking_rates(
                    image_input_rates[image_index_in_loop], max_rate_current_image
                )
                image_input.rates = 0 * brian2.Hz
                net.run(150 * brian2.ms)

last_image_index = current_image_index
functions_basic.save_simulation_state(run_dir,
                                      last_image_index,
                                      syn_input_exc,
                                      neuron_group_exc,
                                      neuron_group_inh)

all_spike_counts_per_image = np.array(all_spike_counts_per_image)
spike_counts_per_neuron_with_retries = spike_mon_exc.count[:]

functions_basic.save_spike_data(run_dir=run_dir,
                                image_labels=image_labels,
                                all_spike_counts_per_image=all_spike_counts_per_image,
                                spike_mon_exc_count=spike_counts_per_neuron_with_retries)

if tr_label_pred and label_predict_range is not None:
    output_dir = f"{run_dir}/tr_label_pred"
    os.makedirs(output_dir, exist_ok=True)
    
    predicted_labels_path = f'{output_dir}/predicted_labels.npy'
    image_labels_in_loop_path = f'{output_dir}/image_labels_in_loop.npy'
    image_indexes_in_loop_path = f'{output_dir}/image_indexes_in_loop.npy'
    cumulative_accuracies_path = f'{output_dir}/cumulative_accuracies.npy'
    
    np.save(predicted_labels_path, np.array(predicted_labels))
    np.save(image_labels_in_loop_path, np.array(image_labels_in_loop))
    np.save(image_indexes_in_loop_path, np.array(image_indexes_in_loop))
    np.save(cumulative_accuracies_path, np.array(cumulative_accuracies))

    functions_basic.finalize_prediction_report(output_dir,
                                               predicted_labels,
                                               image_labels_in_loop,
                                               cumulative_accuracies,
                                               image_indexes_in_loop,
                                               start_index)
    
    functions_basic.load_and_analyze_training_data(label_predict_range, output_dir)
    # PLOT
    functions_basic_plots.plot_training_results(output_dir, label_predict_range)

# PLOT final synapse weights
functions_basic_plots.plot_final_synapse_weights(
    run_dir=run_dir,
    population_exc=population_exc,
    size_selected=size_selected)

# PLOT spike counts
functions_basic_plots.plot_spike_counts_with_cbar(run_dir=run_dir,
                                                  population_exc=population_exc)

pygame.mixer.init()
pygame.mixer.music.load("sesler/horn.mp3")
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(5)
