# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:23:33 2024

@author: ege-demir
"""
import os
import time
import json
import numpy as np
from datetime import datetime
from functions import divisive_weight_normalization_rf, divisive_weight_normalization
from functions import plot_final_synapse_weights_rf, plot_final_synapse_weights



class RunConfiguration:
    class ReceptiveFieldConfig:
        def __init__(self, input_size, filter_size, stride, padding, output_size, directory_of_mapping, norm_neuron_spesific, rf_inh2exc):
            self.input_size = input_size
            self.filter_size = filter_size
            self.stride = stride
            self.padding = padding
            self.output_size = output_size
            self.directory_of_mapping = directory_of_mapping
            self.norm_neuron_spesific = norm_neuron_spesific
            self.rf_inh2exc = rf_inh2exc

    def __init__(self, image_count, create_dir=True, user_run_dir=None, 
                 load_from_a_file=False, load_run_dir=None, load_weights=False, load_theta_values=False, load_spike_counts=False,
                 save_state=False, save_state_range=None,
                 tr_label_pred=False, label_predict_range=None,
                 size_selected=28, g_e_multiplier=1, normalization_val=78, exc_neuron_num=100,
                 skip_norm=False, max_rate=63.75, start_index=0, seed_train=True, seed_test=True,
                 file_suffix_optional='', rf=False, rf_config=None, u=1, notes=''):
        
        # Initialize parameters
        self.size_selected = size_selected
        self.g_e_multiplier = g_e_multiplier
        self.image_count = image_count
        self.skip_norm = skip_norm
        self.normalization_val = normalization_val
        self.load_from_a_file = load_from_a_file
        self.load_run_dir = load_run_dir
        self.load_weights = load_weights
        self.load_theta_values = load_theta_values
        self.load_spike_counts = load_spike_counts
        self.exc_neuron_num = exc_neuron_num
        self.max_rate = max_rate
        self.start_index = start_index
        self.u = u
        self.seed_train = seed_train
        self.seed_test = seed_test
        self.tr_label_pred = tr_label_pred
        self.label_predict_range = label_predict_range
        self.save_state = save_state
        self.save_state_range = save_state_range
        self.file_suffix_optional = file_suffix_optional
        self.notes = notes
        self.rf = rf
        self.create_dir = create_dir
        self.user_run_dir = user_run_dir

        # Print the initialized parameters
        print("RunConfiguration initialized with parameters:")
        for key, value in self.__dict__.items():
            print(f"  {key}: {value}")
            

        # Handle receptive field configuration if enabled
        if rf and rf_config is not None:
            self.rf_config = self.ReceptiveFieldConfig(**rf_config)
            self.rf_name = (f"rf_{self.rf_config.input_size}_{self.rf_config.filter_size}_{self.rf_config.stride}_"
                            f"{self.rf_config.padding}_{self.rf_config.output_size}_nns{int(self.rf_config.norm_neuron_spesific)}"
                            f"_rfie{int(self.rf_config.rf_inh2exc)}")
            self.load_mapping_normalization_and_rfdim_data()
        else:
            self.rf_config = None
            self.rf_name = "rf_dis"
            # Set attributes to None when RF is disabled
            self.pre_neuron_idx_input = None
            self.post_neuron_idx_exc = None
            self.pre_neuron_idx_inh = None
            self.post_neuron_idx_exc_2 = None
            self.normalization_dict = None
            self.rf_dimensions_file_path = None
            self.norm_neuron_spesific = False
            self.rf_inh2exc = False

        # Generate directory and filenames
        trlbpr_str = f"trlbpr{self.label_predict_range}" if self.tr_label_pred and self.label_predict_range is not None else "trlbpr_dis"
        ss_str = f"ss{self.save_state_range}" if self.save_state and self.save_state_range is not None else "ss_dis"
        
        self.run_name = (
            f"lf{int(self.load_from_a_file)}_g{self.g_e_multiplier}_sz{self.size_selected}_nor{self.normalization_val}"
            f"_ic{self.image_count}_sk{int(self.skip_norm)}_mr{int(self.max_rate)}"
            f"_th{int(self.load_theta_values)}_we{int(self.load_weights)}"
            f"_sp{int(self.load_spike_counts)}_exc{self.exc_neuron_num}"
            f"_st{self.start_index}_u{self.u}_str{int(self.seed_train)}_ste{int(self.seed_test)}"
            f"_{trlbpr_str}_{ss_str}_{self.rf_name}_{self.file_suffix_optional}"
        )
        



        # Directory creation logic
        if self.create_dir:
            # Create a timestamp-based directory name
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            timestamped_dir_name = f"run_{timestamp}"
            self.run_dir = timestamped_dir_name
            success = False
            retries = 3
            while retries > 0 and not success:
                try:
                    os.makedirs(self.run_dir, exist_ok=True)
                    success = True
                except Exception as e:
                    print(f"Attempt to create directory failed: {e}")
                    retries -= 1
                    time.sleep(1)  # Wait before retrying
            
            if not success:
                raise FileNotFoundError(f"Failed to create directory '{self.run_dir}' after multiple attempts.")
            
            print(f"[INFO] Folder '{self.run_dir}' created (or already exists).")
            config_file_path = os.path.join(self.run_dir, "run_name_mapping.txt")
            with open(config_file_path, "w") as f:
                f.write(f"Original run name: {self.run_name}\nTimestamped directory name: {self.run_dir}\n")
            
            print(f"[INFO] Configuration saved in '{config_file_path}'.")
            self.save_config(self.run_dir)  # Save configuration

        else:
            if self.user_run_dir and os.path.isdir(self.user_run_dir):
                self.run_dir = self.user_run_dir
            else:
                self.create_dir = False

    def save_config(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        config_path = os.path.join(directory, 'run_config.json')
        # Convert the configuration to a serializable dictionary
        config_data = self.__dict__.copy()
        
        # Ensure create_dir is always set to False
        config_data["create_dir"] = False
        
        # Convert rf_config to a dictionary if it exists
        if isinstance(config_data.get("rf_config"), RunConfiguration.ReceptiveFieldConfig):
            config_data["rf_config"] = config_data["rf_config"].__dict__
        
        # Convert any numpy arrays to lists and keys to standard Python types
        def convert_keys_and_values(data):
            if isinstance(data, dict):
                return {str(key): convert_keys_and_values(value) for key, value in data.items()}
            elif isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, np.integer):
                return int(data)
            elif isinstance(data, np.floating):
                return float(data)
            elif isinstance(data, np.bool_):
                return bool(data)
            return data
        
        config_data = convert_keys_and_values(config_data)
        
        # Save as JSON
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"[INFO] Run configuration saved to {config_path}")
    
    @classmethod
    def load_config(cls, directory: str):
        config_path = os.path.join(directory, 'run_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No configuration file found at {config_path}")
    
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    
        # Filter out non-init parameters to avoid errors during instantiation
        init_params = cls.__init__.__code__.co_varnames
        config_data = {k: v for k, v in config_data.items() if k in init_params}
    
        # Convert rf_config to a dictionary if it's an instance of ReceptiveFieldConfig
        if "rf_config" in config_data and isinstance(config_data["rf_config"], cls.ReceptiveFieldConfig):
            config_data["rf_config"] = config_data["rf_config"].__dict__
    
        # Instantiate the configuration with run_dir set
        instance = cls(**config_data)
        instance.run_dir = directory
        return instance


        
    def load_mapping_normalization_and_rfdim_data(self):
        # Construct the mapping file path
        # Construct the mapping file path
        mapping_file = f"{self.rf_config.directory_of_mapping}/output_to_input_mapping_{self.rf_config.input_size}_" \
                       f"{self.rf_config.filter_size}_{self.rf_config.stride}_{self.rf_config.padding}_" \
                       f"{self.rf_config.output_size}.npz"
        # Load mapping data
        mapping_data = np.load(mapping_file)
        print(f"[INFO] Loaded mapping data from '{mapping_file}'")

        input_neuron_idx = mapping_data['input_neuron_idx']
        output_neuron_idx = mapping_data['output_neuron_idx']

        print(f"[INFO] input_neuron_idx len '{len(input_neuron_idx)}'")
        print(f"[INFO] output_neuron_idx len '{len(output_neuron_idx)}'")
        
        self.pre_neuron_idx_input = input_neuron_idx
        self.post_neuron_idx_exc = output_neuron_idx

        print(f"[INFO] self.pre_neuron_idx_input len '{len(self.pre_neuron_idx_input)}'")
        print(f"[INFO] self.post_neuron_idx_exc len '{len(self.post_neuron_idx_exc)}'")

        
        mapping_file_2 = f"{self.rf_config.directory_of_mapping}/output_to_input_mapping_{self.rf_config.input_size}_" \
                       f"{self.rf_config.filter_size}_{self.rf_config.stride}_{self.rf_config.padding}_" \
                       f"{self.rf_config.output_size}.npz"
                       
        # Load mapping data
        mapping_data_2 = np.load(mapping_file_2)
        print(f"[INFO] Loaded mapping 2 data from '{mapping_file_2}'")

        input_neuron_2_idx = mapping_data_2['input_neuron_idx']
        output_neuron_2_idx = mapping_data_2['output_neuron_idx']

        print(f"[INFO] input_neuron_2_idx len '{len(input_neuron_2_idx)}'")
        print(f"[INFO] output_neuron_2_idx len '{len(output_neuron_2_idx)}'")

        # Filter out indices where input and output neuron indices are the same
        mask = input_neuron_2_idx != output_neuron_2_idx
        self.pre_neuron_idx_inh = output_neuron_2_idx[mask]
        self.post_neuron_idx_exc_2 = input_neuron_2_idx[mask]

        print(f"[INFO] self.pre_neuron_idx_inh len '{len(self.pre_neuron_idx_inh)}'")
        print(f"[INFO] self.post_neuron_idx_exc_2 len '{len(self.post_neuron_idx_exc_2)}'")

        # Construct the normalization file path
        normalization_file = f"{self.rf_config.directory_of_mapping}/output_neuron_intensity_sums_rounded_{self.rf_config.input_size}_" \
                             f"{self.rf_config.filter_size}_{self.rf_config.stride}_{self.rf_config.padding}_" \
                             f"{self.rf_config.output_size}.npz"
        # Load normalization data
        normalization_data = np.load(normalization_file)
        normalization_output_neuron_idx = normalization_data['output_neuron_idx']
        normalization_values = normalization_data['intensity_sum']
        # Create a normalization dictionary
        self.normalization_dict = dict(zip(normalization_output_neuron_idx, normalization_values))
         
        # Define path for RF dimensions file
        self.rf_dimensions_file_path = (
            f"{self.rf_config.directory_of_mapping}/{self.rf_config.input_size}_"
            f"{self.rf_config.filter_size}_{self.rf_config.stride}_"
            f"{self.rf_config.padding}_{self.rf_config.output_size}_rf_dimensions.npz"
        )
        print(f"[INFO] plot_final_synapse_weights_rf is called; RF dimensions path: {self.rf_dimensions_file_path}")

    def connect_synapses_input2exc(self, syn_input_exc):
        """Connect synapses based on the receptive field configuration."""
        if self.rf and self.rf_config is not None:
            syn_input_exc.connect(i=self.pre_neuron_idx_input, j=self.post_neuron_idx_exc)
            print("[INFO] input2exc connected based on RF.")
        else:
            syn_input_exc.connect()
            print("[INFO] input2exc connected all2all.")

    def connect_synapses_inh2exc(self, syn_inh_exc):
        """Connect synapses based on the receptive field configuration."""
        if self.rf and self.rf_config is not None:
            if self.rf_config.rf_inh2exc:
                syn_inh_exc.connect(i=self.pre_neuron_idx_inh, j=self.post_neuron_idx_exc_2)
                print("[INFO] inh2exc connected based on RF & i !=j.")
            else:
                syn_inh_exc.connect("i != j")
                print("[INFO] inh2exc connected based on only i !=j. This was selected eventough RF is on.")
        else:
            syn_inh_exc.connect("i != j")
            print("[INFO] inh2exc connected based on only i !=j.")


    def apply_normalization(self, syn_input_exc, population_exc):
        """Apply divisive weight normalization based on the rf configuration."""
        # If rf is enabled and rf_config is available, check norm_neuron_specific
        if self.rf and self.rf_config is not None:
            if self.rf_config.norm_neuron_spesific:
                # If neuron-specific normalization is required, use normalization dictionary
                divisive_weight_normalization_rf(syn_input_exc, population_exc, self.normalization_dict)
            else:
                # If neuron-specific normalization is not required, use the uniform normalization value
                divisive_weight_normalization(syn_input_exc, population_exc, normalization_value=self.normalization_val)
        else:
            # If rf is disabled, apply the default divisive weight normalization
            divisive_weight_normalization(syn_input_exc, population_exc, normalization_value=self.normalization_val)

    def plot_synapse_weights(self, run_dir, population_exc, save_indiv_figs):
        """Plot the synapse weights based on the rf configuration."""
        if self.rf and self.rf_config is not None:
            print('[INFO] plot_final_synapse_weights_rf is called')
            plot_final_synapse_weights_rf(
                run_dir=run_dir,
                population_exc=population_exc,
                rf_dimensions_file_path=self.rf_dimensions_file_path,
                save_indiv_figs=save_indiv_figs)
        else:
            print("[INFO] Receptive field configuration is disabled.")
            plot_final_synapse_weights(
                run_dir=run_dir,
                population_exc=population_exc,
                size_selected=self.size_selected,
                final_weights_path=None)

