import os
import json
import numpy as np
import pandas as pd

def calculate_rmse(original_dir, imputed_dir, output_json="results/rmse_results.json"):
    """Calculates RMSE between original and imputed files using only the first part of the filename as a key."""
    
    # Ensure directories exist
    if not os.path.exists(original_dir):
        print(f"Original directory {original_dir} does not exist.")
        return
    
    if not os.path.exists(imputed_dir):
        print(f"Imputed directory {imputed_dir} does not exist.")
        return

    # Dictionary to store RMSE results
    rmse_results = {}

    # Map original files by their base name (first part before the first space)
    original_files = {}
    for file in os.listdir(original_dir):
        if file.endswith(".csv"):
            base_name = file.split(" ")[0]  # Extract base identifier
            original_files[base_name] = os.path.join(original_dir, file)

    # Iterate through imputation techniques
    for technique in os.listdir(imputed_dir):
        technique_path = os.path.join(imputed_dir, technique)
        if not os.path.isdir(technique_path):
            continue  # Skip if not a directory

        # Iterate through files in the technique's directory
        for imputed_file in os.listdir(technique_path):
            if not imputed_file.endswith(".csv"):
                continue  # Skip non-CSV files

            file_key = imputed_file.split(" ")[0]  # Extract base name

            # Check if the corresponding original file exists
            if file_key not in original_files:
                print(f"Warning: No matching original file found for {imputed_file}")
                continue

            # Read the files
            original_df = pd.read_csv(original_files[file_key])
            imputed_df = pd.read_csv(os.path.join(technique_path, imputed_file))

            # Ensure 'Throughput' column exists in both files
            if 'Throughput' not in original_df.columns or 'Throughput' not in imputed_df.columns:
                print(f"Skipping {imputed_file} due to missing 'Throughput' column.")
                continue

            # Compute RMSE
            common_indices = original_df.index.intersection(imputed_df.index)
            original_values = original_df.loc[common_indices, 'Throughput']
            imputed_values = imputed_df.loc[common_indices, 'Throughput']
            rmse = np.sqrt(np.mean((original_values - imputed_values) ** 2))

            # Store results in dictionary
            if file_key not in rmse_results:
                rmse_results[file_key] = {}

            rmse_results[file_key][technique] = {"rmse": rmse}

    # Ensure the results directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # Save results to JSON file
    with open(output_json, "w") as json_file:
        json.dump(rmse_results, json_file, indent=4)

    print(f"RMSE results saved to {output_json}")
