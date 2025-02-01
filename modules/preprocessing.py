import os
import pandas as pd
import numpy as np
from datetime import date, datetime

# This function divides original throughput data in intervals of 6 hours
## OKAY
def data_treatment_intervals(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(source_dir)
    sorted_files = sorted(files)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in sorted_files:
        print(file_name)
        dataset = pd.read_csv(os.path.join(source_dir, file_name))

        if dataset.empty:
            print(f"No data in {file_name}. Skipping...")
            continue
            
        dataset = dataset.rename(columns={' Vazao': 'Throughput'})

        # Ensure all Timestamp values are numeric before conversion
        dataset = dataset[pd.to_numeric(dataset['Timestamp'], errors='coerce').notna()]
        dataset['Timestamp'] = dataset['Timestamp'].astype(float)
        dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'], unit='s')
            
        # Define function to get interval start time
        def get_interval_start(dt):
            hour = dt.hour
            if hour < 6:
                return dt.replace(hour=0, minute=0, second=0, microsecond=0)
            elif hour < 12:
                return dt.replace(hour=6, minute=0, second=0, microsecond=0)
            elif hour < 18:
                return dt.replace(hour=12, minute=0, second=0, microsecond=0)
            else:
                return dt.replace(hour=18, minute=0, second=0, microsecond=0)

        # Apply the function to create 'Interval' column
        dataset['Timestamp'] = dataset['Timestamp'].apply(get_interval_start)

        def calculate_mean_without_outliers(data):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_outlier_cut = Q1 - 1.5 * IQR
            upper_outlier_cut = Q3 + 1.5 * IQR
            filtered_data = data[(data >= lower_outlier_cut) & (data <= upper_outlier_cut)]
            return filtered_data.mean()

        # Group by 'Interval' and calculate mean of 'Throughput'
        grouped = dataset.groupby('Timestamp')['Throughput'].apply(calculate_mean_without_outliers).reset_index()

        start_time = dataset['Timestamp'].min().replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = dataset['Timestamp'].max().replace(hour=18, minute=0, second=0, microsecond=0)
        all_intervals = pd.date_range(start=start_time, end=end_time, freq='6h')

        grouped = grouped.set_index('Timestamp').reindex(all_intervals).reset_index()
        grouped.columns = ['Timestamp', 'Throughput']  # Rename columns after reindex
        grouped['Throughput'] = grouped['Throughput'].fillna(np.nan)

        # Format 'Interval' to 'dd-mm-aa hh:mm:ss'
        grouped['Timestamp'] = grouped['Timestamp'].dt.strftime('%d-%m-%y %H:%M:%S')

        output_file = os.path.join(output_dir, file_name.split(" ")[0] + " PROCESSED.csv")
        grouped.to_csv(output_file, index=False)

def outlier_removal(df, column):
    # # Replace -1 with NaN in the specified column
    # df[column] = df[column].replace(-1, np.nan)

    # Extract non-NaN values from the column
    r = df[column].dropna().to_numpy()
    
    # Check if there are enough values for analysis
    if r.size == 0:
        print("Column does not contain enough values for analysis.")
        return df

    # Normalize the values by dividing by the maximum value
    r_max = np.max(r) 
    r = r / r_max  

    # Calculate the minimum threshold
    perc_min = []
    p_min = np.linspace(0.1, 2, 20)
    for i in p_min:
        perc_min.append(np.percentile(r, i))
    diff_perc_min = np.diff(perc_min)
    index_min = np.argmax(diff_perc_min)  
    thres_min = np.mean(perc_min[index_min:index_min + 2])

    # Calculate the maximum threshold
    perc_max = []
    p_max = np.linspace(98, 100, 20)
    for i in p_max:
        perc_max.append(np.percentile(r, i))
    diff_perc_max = np.diff(perc_max)
    index_max = np.argmax(diff_perc_max)  
    thres_max = np.mean(perc_max[index_max:index_max + 2])

    # Filter values outside the thresholds and set them to NaN
    r_filtered = np.where((r < thres_min) | (r > thres_max), np.nan, r)

    # Restore the original scale by multiplying by the maximum value
    r_filtered = r_filtered * r_max  

    # Create a copy of the DataFrame and update the column with filtered values
    df_filtered = df.copy()
    df_filtered.loc[~df[column].isna(), column] = r_filtered

    return df_filtered

# This function gets the greater longest interval (pre defined in quantity parameter)
def get_files_size_failure_rate(percentage, path, protocols, quantity=10):
    archive_data = {}
    file_list = []
    file_list_names = []

    # Loop through all relevant CSV files in the directory
    for protocol in protocols:
        full_path = os.path.join(path, protocol)
        for arquivo in os.listdir(full_path):
            if arquivo.endswith('.csv'):
                caminho_arquivo = os.path.join(full_path, arquivo)
                df = pd.read_csv(caminho_arquivo)

                total_rows = len(df)
                # Check if the dataset has more than 700 lines
                if total_rows <= 700:
                    continue  # Skip the file if it has 700 lines or less

                num_failures = df['Throughput'].isna().sum()
                failure_percentage = (num_failures / total_rows) * 100 if total_rows > 0 else 0
                
                # Include only archives with failure percentage below specified percentage
                if failure_percentage < percentage:
                    # Store the DataFrame, total rows, and failure percentage in the dictionary
                    archive_data[f'{protocol} {arquivo}'] = {'df': df, 'total_rows': total_rows, 'failure_percentage': failure_percentage}
                    file_list_names.append(arquivo)

    # Check if any archives meet the criteria
    if not archive_data:
        print(f"No archives with failure percentage below {percentage}% were found.")
        return []

    # Sort the archives by line count (descending) and then by failure percentage (ascending)
    sorted_archives = sorted(
        archive_data.items(),
        key=lambda x: (-x[1]['total_rows'], x[1]['failure_percentage'])
    )

    top_archives = sorted_archives[:quantity]

    print(f"Top {quantity} archives with failure percentage below {percentage}% and more than 700 lines:")
    for archive, data in top_archives:
        failure_percentage = data['failure_percentage']
        total_rows = data['total_rows']
        file_list.append(f"{archive}: {total_rows} lines, Failure Percentage: {failure_percentage:.2f}%")
        print(f"{archive}: {total_rows} lines, Failure Percentage: {failure_percentage:.2f}%")

    return top_archives, file_list, file_list_names

#This function get the longest interval of some directory
## OKAY
def get_longest_interval(original_dir, saving_dir):
    os.makedirs(saving_dir, exist_ok=True)

    for file in os.listdir(original_dir):
        if file.endswith('.csv'):
            print(f"Processing file: {file}")
            file_path = os.path.join(original_dir, file)
            df = pd.read_csv(file_path)

            longest_interval = []
            current_interval = []

            if 'Throughput' not in df.columns:
                raise ValueError(f"'Throughput' not found in {file}.")

            for index, row in df.iterrows():
                if pd.isna(row['Throughput']):  # Check if the value is NaN
                    # Update longest_interval if current_interval is greater
                    if len(current_interval) > len(longest_interval):
                        longest_interval = current_interval
                    current_interval = []
                else:
                    current_interval.append(row)  # Add the row to the current range

            # Check again at the end if the largest range ends at the end of the file
            if len(current_interval) > len(longest_interval):
                longest_interval = current_interval

            if longest_interval:
                print(f"Largest gap-free interval found in file {file} with {len(longest_interval)} lines.")
                longest_df = pd.DataFrame(longest_interval)
                output_file = file.split(" ")[0] + ' CONTINUOUS INTERVAL.csv'
                longest_df.to_csv(os.path.join(saving_dir, output_file), index=False)
            else:
                print(f"No continuous gapless intervals were found in the file {file}.")

# This function checks the original dataset failure rate and applies in his respective longest interval
## OKAY
def apply_original_failure_rate_on_longest(original_dir, longest_interval_dir, saving_dir):
    os.makedirs(saving_dir, exist_ok=True)

    for arquivo in os.listdir(original_dir):
        path = os.path.join(original_dir, arquivo)
        df = pd.read_csv(path)

        new_name = arquivo.replace('PROCESSED.csv', 'CONTINUOUS INTERVAL.csv')
        path2 = os.path.join(longest_interval_dir, new_name)
        df2 = pd.read_csv(path2)

        # Calculate the percentage of missing data for the original dataset
        failures_percentual = df.isna().sum().sum() / (df.shape[0] * df.shape[1])

        # Calculate the number of values ​​to be made NaN in the 'Throughput' column
        total_celulas = df2.shape[0]  # Total rows in 'Throughput' column
        missing_num = int(total_celulas * failures_percentual)

        # Choose random indexes only on the 'Throughput' column
        nan_indices = np.random.choice(df2.index, missing_num, replace=False)
        df2.loc[nan_indices, 'Throughput'] = np.nan  

        path_salvar = os.path.join(saving_dir, new_name.split(" ")[0] + " GAPPED INTERVAL.csv")
        df2.to_csv(path_salvar, index=False)
        print(f"{new_name} processed with NaNs applied in the column 'Throughput'")