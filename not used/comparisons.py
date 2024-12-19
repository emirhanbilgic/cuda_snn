# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:16:59 2024

@author: ege-demir
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Define the directory structure and range of tot_im
run_dir = "run_20241112104344"  # Replace with actual directory path
tot_im_range = range(1000, 20001, 1000)

# Initialize lists to store data
tot_im_list = []
accuracy_list = []

# Loop through tot_im and read the cumulative_accuracies.npy
for tot_im in tot_im_range:
    file_path = f"{run_dir}/regular_cropped_0_{tot_im}/prediction_mnist_test_data_count_5000_start_0/cumulative_accuracies.npy"
    if os.path.exists(file_path):
        cumulative_accuracies = np.load(file_path)
        last_accuracy = cumulative_accuracies[-1]  # Get the last element
        tot_im_list.append(tot_im)
        accuracy_list.append(last_accuracy)
    else:
        print(f"File not found: {file_path}")

# Create a DataFrame for table
data = {'Total Trained Images': tot_im_list, 'Accuracy': accuracy_list}
df = pd.DataFrame(data)

# Print the table
print(df)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(tot_im_list, accuracy_list, marker='o', linestyle='-', label='Accuracy vs. Total Trained Images')
plt.title('Accuracy vs. Total Trained Images')
plt.xlabel('Total Trained Images')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#prefix = "tr"
prefix = "test"
pred_count = 2500
# Define the directory structures and range of tot_im
run_dirs = [
    "run_20241111111111",  # Replace with actual path for Run 1
    "run_20241108131041",  # Replace with actual path for Run 2
    "run_20241112104344",  # Replace with actual path for Run 3
]
run_names = ["Run 1 nns1", "Run 2 nss0", "Run 3 rf off"]  # Names for labeling
colors = ["blue", "green", "red"]  # Colors for each run
tot_im_range = range(1000, 60001, 1000)

# Dynamically create the comparison folder name
comparison_dir = f"{prefix}_comparisons_{run_dirs[0]}_{run_dirs[1]}_{run_dirs[2]}"
os.makedirs(comparison_dir, exist_ok=True)

# Store combined results for the final plot and table
combined_data = []

# Initialize a dictionary for the final table
final_table = {"Total Trained Images": []}
for run_name in run_names:
    final_table[run_name] = []

# Loop through each run_dir
for run_dir, run_name, color in zip(run_dirs, run_names, colors):
    tot_im_list = []
    accuracy_list = []
    
    # Loop through tot_im and read cumulative_accuracies.npy
    for tot_im in tot_im_range:
        file_path = f"{run_dir}/regular_cropped_0_{tot_im}/prediction_mnist_test_data_count_{pred_count}_start_0/cumulative_accuracies.npy"
        if os.path.exists(file_path):
            cumulative_accuracies = np.load(file_path)
            last_accuracy = cumulative_accuracies[-1]  # Get the last element
            tot_im_list.append(tot_im)
            accuracy_list.append(last_accuracy)
        else:
            print(f"File not found: {file_path}")
    
    # Store data for combined analysis
    combined_data.append((run_name, tot_im_list, accuracy_list, color))
    
    # Add data to the final table dictionary
    if not final_table["Total Trained Images"]:
        final_table["Total Trained Images"] = tot_im_list
    final_table[run_name] = [f"{acc:.2f}" for acc in accuracy_list]  # Format to 2 decimals
    
    # Create a DataFrame for table
    data = {'Total Trained Images': tot_im_list, 'Accuracy': accuracy_list}
    df = pd.DataFrame(data)
    
    # Save the table for each run
    table_file = os.path.join(comparison_dir, f"{run_name.replace(' ', '_')}_table.txt")
    df.to_csv(table_file, index=False, sep="\t", float_format="%.2f")
    print(f"Table for {run_name} saved to {table_file}")
    
    # Save and show the plot for each run
    plt.figure(figsize=(10, 6))
    plt.plot(tot_im_list, accuracy_list, marker='o', linestyle='-', color=color, label=f'{run_name}')
    plt.title(f'Accuracy vs. Total Trained Images ({run_name})')
    plt.xlabel('Total Trained Images')
    plt.ylabel('Accuracy')
    plt.yticks(range(0, 101, 5))  # Set y-ticks
    plt.grid(True)
    plt.legend()
    plot_file = os.path.join(comparison_dir, f"{run_name.replace(' ', '_')}_plot.png")
    plt.savefig(plot_file)
    plt.show()
    print(f"Plot for {run_name} saved to {plot_file}")

# Add a column for the highest accuracy
final_table["Highest Accuracy"] = [
    max(
        (float(final_table[run_names[0]][i]), run_names[0]),
        (float(final_table[run_names[1]][i]), run_names[1]),
        (float(final_table[run_names[2]][i]), run_names[2]),
    )[1]
    for i in range(len(final_table["Total Trained Images"]))
]

# Create the final combined plot
plt.figure(figsize=(12, 8))
for run_name, tot_im_list, accuracy_list, color in combined_data:
    plt.plot(tot_im_list, accuracy_list, marker='o', linestyle='-', color=color, label=f'{run_name}')
plt.title(f'Test Accuracy vs. Total Trained Images Tested on 2500 Images')
plt.xlabel('Total Trained Images')
plt.ylabel('Accuracy')
plt.yticks(range(0, 101, 5))  # Set y-ticks
plt.grid(True)
plt.legend()
combined_plot_file = os.path.join(comparison_dir, "combined_plot.png")
plt.savefig(combined_plot_file)
plt.show()
print(f"Combined plot saved to {combined_plot_file}")

# Create and save the final DataFrame for the combined table
final_df = pd.DataFrame(final_table)
final_table_file = os.path.join(comparison_dir, "final_combined_table.txt")
final_df.to_csv(final_table_file, index=False, sep="\t")
print(f"Final combined table saved to {final_table_file}")

# Display the final table
print("Final Combined Table:")
print(final_df)



