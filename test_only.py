# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:17:00 2024

@author: ege-demir
"""
# %%
import functions_basic
import functions_basic_plots
import pygame  # Import pygame for sound playback 
import brian2
import numpy as np
from tqdm import tqdm

#LABELING NEURONS FOR EVAL TESTING, USING SPIKE DATA AND IMAGE LABELS MONITORED IN TRAINING

#run config - general
run_dir = "deneme5"
image_count=100
exc_neuron_num = 400
tr_label_pred=True
label_predict_range=10
size_selected = 28 #input is 28x28
max_rate = 63.75
seed_train = True
start_index = 0 # start index of train
skip_norm = False # skips the normalization algoritm if True
normalization_val = 78

# Labeling config

#labeling config is fixed for this basic code, you label the neurons using all the spiking data from the training

selected_spike_data_path, spike_data_with_labels_folder_path = functions_basic.prepare_spike_data_with_labels_folder(run_dir = run_dir)

assigned_labels_path = functions_basic.load_and_assign_neuron_labels(selected_spike_data_path = selected_spike_data_path,
                                                     spike_data_with_labels_folder_path = spike_data_with_labels_folder_path,
                                                     population_exc = exc_neuron_num)


functions_basic_plots.plot_neurons_with_label_spike_counts(selected_spike_data_path = selected_spike_data_path,
                                    spike_data_with_labels_folder_path = spike_data_with_labels_folder_path,
                                    population_exc = exc_neuron_num)

functions_basic_plots.plot_neurons_with_labels(assigned_labels_path = assigned_labels_path,
                        population_exc = exc_neuron_num)

functions_basic.save_histogram_data_txt(assigned_labels_path = assigned_labels_path,
                        population_exc = exc_neuron_num)

functions_basic_plots.plot_neurons_with_labels_detailed(assigned_labels_path = assigned_labels_path,
                        population_exc = exc_neuron_num)


# %%

#PREDICTION WITH THE TEST DATASET
#run config - general
run_dir = "deneme5"
image_count=100
exc_neuron_num = 400
tr_label_pred=True
label_predict_range=10
size_selected = 28 #input is 28x28
max_rate = 63.75
seed_train = True
start_index = 0 # start index of train
skip_norm = False # skips the normalization algoritm if True
normalization_val = 78


#pred config
start_index_prediction = 0
image_count_prediction = 98
seed_test = True
plots_off = True

prediction_folder_path = functions_basic.get_prediction_folder_path(assigned_labels_path = assigned_labels_path,
                        image_count_prediction = image_count_prediction,
                        start_index = start_index_prediction,
                        use_test_data_mnist = True)


checkpoint_interval = max(1, round(image_count_prediction * 0.1))  # Ensures at least 1 image if image_count_prediction is small

if 'i' in globals():
    del i
if 'j' in globals():
    del j

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
tau_Apre_ee = 20 * brian2.ms
tau_Apost1_ee = 20 * brian2.ms
eta_pre_ee  = 0.0001
eta_post_ee = 0.01
w_min_ee = 0
w_max_ee = 1
w_ei_ = 10.4
w_ie_ = 17

# Neuron equations for excitatory and inhibitory populations
ng_eqs_exc = """
dv/dt = ((E_rest_exc - v) + g_e*(E_exc_for_exc - v) + g_i*(E_inh_for_exc - v))/tau_lif_exc : volt (unless refractory)
dg_e/dt = -g_e/tau_ge : 1
dg_i/dt = -g_i/tau_gi : 1
theta : volt
"""

ng_eqs_inh = """
dv/dt = ((E_rest_inh - v) + g_e*(E_exc_for_inh - v) + g_i*(E_inh_for_inh - v))/tau_lif_inh : volt (unless refractory)
dg_e/dt = -g_e/tau_ge : 1
dg_i/dt = -g_i/tau_gi : 1
"""

# Threshold and reset equations for both populations
ng_threshold_exc = "v > v_threshold_exc - v_offset_exc + theta"
ng_reset_exc = "v = v_reset_exc"

ng_threshold_inh = "v > v_threshold_inh"
ng_reset_inh = "v = v_reset_inh"


# Synapse equations for exc. -> exc. connections (test phase)
syn_eqs_ee_test = """
w_ee : 1
"""

syn_on_pre_ee_test = """
g_e_post += w_ee
"""
    
# Create neuron groups for excitatory and inhibitory neurons
neuron_group_exc = brian2.NeuronGroup(N=population_exc, model=ng_eqs_exc, threshold=ng_threshold_exc, reset=ng_reset_exc, refractory=5*brian2.ms, method="euler")
neuron_group_inh = brian2.NeuronGroup(N=population_inh, model=ng_eqs_inh, threshold=ng_threshold_inh, reset=ng_reset_inh, refractory=2*brian2.ms, method="euler")

# Set initial values
neuron_group_exc.v = E_rest_exc - 40 * brian2.mV
neuron_group_inh.v = E_rest_inh - 40 * brian2.mV #bu orijinal kodda neden commentli?

theta_values = np.load(f"{run_dir}/neuron_group_exc_theta.npy")
neuron_group_exc.theta = theta_values * brian2.volt


# Define PoissonGroup for input image (MNIST)
tot_input_num = size_selected * size_selected
image_input = brian2.PoissonGroup(N=tot_input_num, rates=0*brian2.Hz)  # Will set the rates based on the image

syn_input_exc = brian2.Synapses(image_input, neuron_group_exc, model=syn_eqs_ee_test, on_pre=syn_on_pre_ee_test, method="euler")
syn_input_exc.connect()
saved_weights = np.load(f'{run_dir}/final_synaptic_weights.npy')
syn_input_exc.w_ee[:] = saved_weights
print(f'Loaded saved synaptic weights from run name {run_dir}.')

syn_input_exc.delay = 10 * brian2.ms

# Synapse connecting excitatory -> inhibitory neurons (one-to-one)
syn_exc_inh = brian2.Synapses(neuron_group_exc, neuron_group_inh, model="w_ei : 1", on_pre="g_e_post += w_ei", method="euler")
syn_exc_inh.connect(j='i')  # One-to-one connection
syn_exc_inh.w_ei = w_ei_

# Synapse connecting inhibitory -> excitatory neurons (all-to-all except same index)
syn_inh_exc = brian2.Synapses(neuron_group_inh, neuron_group_exc, model="w_ie : 1", on_pre="g_i_post += w_ie", method="euler")
syn_inh_exc.connect("i != j")
syn_inh_exc.w_ie = w_ie_

# Create a SpikeMonitor to record the spikes of excitatory neurons
spike_mon_exc = brian2.SpikeMonitor(neuron_group_exc)

# Create a network to encapsulate all components
net = brian2.Network(neuron_group_exc, neuron_group_inh, image_input, syn_input_exc, syn_exc_inh, syn_inh_exc, spike_mon_exc)

# Reset the network (reset all components to initial values)
net.store('initialized')  # Store the initialized state for later reset


image_input_rates, image_labels, image_intensities = functions_basic.get_spiking_rates_and_labels(use_test_data_mnist = True,
                                                                                  image_count=image_count_prediction,
                                                                                  seed_data=seed_test,
                                                                                  size_selected=size_selected,
                                                                                  start_index=start_index_prediction)
# Initialize a list to store spike counts for each image
all_spike_counts_per_image = []
predicted_labels = []
image_labels_in_loop = []
cumulative_accuracies = []
image_indexes_in_loop = []
max_rate_current_image = max_rate  # Set the initial maximum rate

# Process each test image
with tqdm(total=image_count_prediction, desc="Processing Test Images", dynamic_ncols=True) as pbar:
    previous_spike_counts = np.zeros_like(spike_mon_exc.count[:])  # Initialize previous spike counts to zero
    max_rate_current_image = max_rate  # Set initial rate

    for image_index_in_loop in range(image_count_prediction):
        current_image_index = start_index_prediction + image_index_in_loop  # Track the correct image index
        image_retries = 0  # Track retries per image
        successful_test = False

        while not successful_test:
            previous_spike_counts = spike_mon_exc.count[:]  # Spike counts before running the network

            # Apply the input rates for 350 ms for the current image
            image_input.rates = image_input_rates[image_index_in_loop] * brian2.Hz

            # Run the network for 350 ms
            net.run(350 * brian2.ms)

            # Get the current total spike counts after processing this image
            current_spike_counts = spike_mon_exc.count[:]
            spike_counts_current_image = current_spike_counts - previous_spike_counts  # Get spikes only for this image
            max_spike_count = np.max(spike_counts_current_image)
            neurons_with_max_spikes = np.where(spike_counts_current_image == max_spike_count)[0]
            sum_spike_count = np.sum(spike_counts_current_image)
        
            # Check if any neuron spiked 5 or more times
            if sum_spike_count >= 5:
                successful_test = True
                all_spike_counts_per_image.append(spike_counts_current_image.copy())  # Copy current spikes to avoid overwriting later

                # Get the true label and predictions
                true_label = image_labels[image_index_in_loop]
                predictions = functions_basic.get_predictions_for_current_image(spike_counts_current_image = spike_counts_current_image,
                                                                assigned_labels_path = assigned_labels_path)
                predicted_label = predictions[0]  # The top predicted label
                predicted_labels, image_labels_in_loop = functions_basic.save_labels_incrementally(prediction_folder_path,
                                                                                         predicted_label,
                                                                                         true_label,
                                                                                         predicted_labels,
                                                                                         image_labels_in_loop,
                                                                                         image_index_in_loop,
                                                                                         image_indexes_in_loop
                                                                                         )

                cumulative_accuracies, cumulative_accuracy = functions_basic.calculate_cumulative_accuracy(prediction_folder_path,
                                                                                           predicted_labels,
                                                                                           image_labels_in_loop,
                                                                                           cumulative_accuracies)

                if (image_index_in_loop + 1) % checkpoint_interval == 0 or (image_index_in_loop + 1) == image_count_prediction:
                    functions_basic.finalize_prediction_report(prediction_folder_path, predicted_labels, image_labels_in_loop,
                                               cumulative_accuracies,image_indexes_in_loop, image_index_in_loop, start_index_prediction)

                if not plots_off:
                    #PLOT
                    functions_basic_plots.plot_test_image_with_prediction(image_intensities[image_index_in_loop], true_label, predicted_label)
                    functions_basic_plots.plot_prediction_process(spike_counts_current_image, assigned_labels_path)
    
                # Update the progress bar with the true label, prediction, and cumulative accuracy
                pbar.set_description(
                    f"Image #{current_image_index} (Loop Index: {image_index_in_loop}, True Label: {true_label}, "
                    f"Predicted: {predicted_label}) | Cumulative Accuracy: {cumulative_accuracy:.2f}% | "
                    f"Retry: {image_retries} | Sum Spikes: {sum_spike_count} | Max Spikes: {max_spike_count} | "
                    f"Neurons with Max Spikes: {neurons_with_max_spikes.tolist()} | Current Max Rate: {max_rate_current_image}"
                )
                    
                # Reset max_rate_current_image before presenting new image.
                max_rate_current_image = max_rate  # Reset to the original max_rate after successful image processing
                pbar.update(1)

                # Stop the input for 150 ms
                image_input.rates = 0 * brian2.Hz
                net.run(150 * brian2.ms)  # Simulate the network without input for 100 ms
            else:
                image_retries += 1
                max_rate_current_image += 32  # Increase input rate if spikes are not enough
                image_input_rates[image_index_in_loop] = functions_basic.increase_spiking_rates(image_input_rates[image_index_in_loop], max_rate_current_image)
                image_input.rates = 0 * brian2.Hz
                net.run(150 * brian2.ms)  # Simulate the network without input for 100 ms


print(f"[INFO] Test phase complete. Predictions made for {image_count_prediction} images with the UNSEEN TEST data.")

all_spike_counts_per_image = np.array(all_spike_counts_per_image)
spike_counts_per_neuron_with_retries = spike_mon_exc.count[:]
functions_basic.save_spike_data_test(prediction_folder_path = prediction_folder_path,
                     image_labels = image_labels,
                     all_spike_counts_per_image = all_spike_counts_per_image,
                     spike_mon_exc_count = spike_counts_per_neuron_with_retries)

accuracy = functions_basic.calculate_accuracy(prediction_folder_path = prediction_folder_path)
print(f"[INFO] Test Accuracy: {accuracy:.2f}%")

image_indexes_in_loop_path = f'{prediction_folder_path}/image_indexes_in_loop.npy'
cumulative_accuracies_path = f'{prediction_folder_path}/cumulative_accuracies.npy'
np.save(image_indexes_in_loop_path, np.array(image_indexes_in_loop))
np.save(cumulative_accuracies_path, np.array(cumulative_accuracies))

# Calculate accuracy per label
accuracy_per_label = functions_basic.calculate_accuracy_per_label(prediction_folder_path = prediction_folder_path)


#PLOTS PLOTS PLOTS
functions_basic_plots.plot_spike_counts_with_cbar_test(prediction_folder_path = prediction_folder_path,
                                 population_exc = exc_neuron_num) 

functions_basic_plots.plot_accuracy_per_label(prediction_folder_path = prediction_folder_path)

functions_basic_plots.plot_confusion_matrix(prediction_folder_path = prediction_folder_path)

functions_basic_plots.plot_cumulative_test_accuracy(image_indexes_in_loop_path = image_indexes_in_loop_path,
                              prediction_folder_path = prediction_folder_path)

pygame.mixer.init()
pygame.mixer.music.load("sesler/horn.mp3") 
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(5)