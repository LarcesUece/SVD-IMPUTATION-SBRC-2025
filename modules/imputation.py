import pandas as pd
import numpy as np
import os 

from sklearn.impute import KNNImputer

import matplotlib.pyplot as plt

from modules.preprocessing import outlier_removal

def impute_rolling_median(df, window_size=3):
    df['Throughput'] = df['Throughput'].fillna(df['Throughput'].rolling(window=window_size, min_periods=1).median())
    global_median = df['Throughput'].median()
    df['Throughput'] = df['Throughput'].fillna(global_median)
    return df

def impute_rolling_average(df, window_size=3):
    df['Throughput'] = df['Throughput'].fillna(df['Throughput'].rolling(window=window_size, min_periods=1).mean())
    global_mean = df['Throughput'].mean()
    df['Throughput'] = df['Throughput'].fillna(global_mean)
    return df

def linear_interpolation(df, limit_direction='both', method='linear'):
    df_imputed = df.interpolate(method=method, limit_direction=limit_direction)
    df['Throughput'] = df['Throughput'].fillna(df_imputed['Throughput'])
    return df

def impute_knn(df, k=5):
    imputer = KNNImputer(n_neighbors=k)
    df_copy = df.copy()
    df_copy['Throughput'] = imputer.fit_transform(df_copy[['Throughput']])
    return df_copy

def svd_decomposition(df):
    U, S, Vt = np.linalg.svd(df, full_matrices=True)
    return U, S, Vt

def variability_plot(variability, S):
    plt.plot(range(1, len(variability) + 1), variability, marker='o', markersize=1, markerfacecolor='teal', markeredgecolor='teal', color='darkturquoise')
    plt.xlabel('Number of Singular Values')
    plt.ylabel('Cumulative Variability')
    plt.title('Singular Values by Cumulative Variability')
    plt.grid(color='lightgray', alpha=0.7)
    plt.show()

def principal_components(r, U, S, Vt):
    U_reduced = U[:, :r]
    S_reduced = S[:r]
    Vt_reduced = Vt[:r, :]
    return U_reduced, S_reduced, Vt_reduced

# Converts the matrix back into a dataframe
def matrix_to_df(dataframes):
    df = pd.DataFrame(dataframes)
    q = df.shape[0] * df.shape[1]
    df = df.transpose()
    df = df.to_numpy().reshape(-1, q)
    df = df.transpose()
    df = pd.DataFrame(df)
    if 0 in df.columns:
        df = df.rename(columns={0: 'Throughput'})
    return df

# Calculates RMSE of the values generated in SVD with the 'original' values
def calculate_rmse(df1, df2, column):
    common_indices = df1.index.intersection(df2.index)
    values_df1 = df1.loc[common_indices, column]
    values_df2 = df2.loc[common_indices, column]
    rmse = np.sqrt(np.mean((values_df1 - values_df2) ** 2))
    return rmse

# Generates the final CSV file with the imputation
def generate_csv_file(df1, df2, base_path, csv_file_name):
    df1['Throughput'] = pd.NA
    df1['Throughput'] = df2['Throughput']
    
    # Drops the last rows of the file where imputation was not performed
    df1 = df1.dropna(subset=['Throughput'])
    svd_path = os.path.join(base_path, 'svd')
    if not os.path.exists(svd_path):
        os.makedirs(svd_path)
    csv_file_path = os.path.join(svd_path, csv_file_name)

    # Saves the resulting DataFrame to a CSV file
    df1.to_csv(csv_file_path, index=False)
    print(f"CSV file '{csv_file_path}' generated!")
    return df1

def matrix(path):
    df = pd.read_csv(path)
    df = outlier_removal(df, 'Throughput')
    df_datetime = df.copy()
    df_datetime.drop(columns=['Throughput'], inplace=True)
    df['Throughput'] = df['Throughput'].replace(-1, np.nan)
    
    Throughput = df['Throughput'].values
    num_data = len(Throughput)
    num_columns = num_data // 28
    matrix = Throughput[:num_columns * 28].reshape(num_columns, 28).T
    original_matrix = pd.DataFrame(matrix)
    df_interpolated = df["Throughput"].interpolate(method='linear', limit_direction='both')
    Throughput_ = df_interpolated.values
    interpolated_matrix = Throughput_[:num_columns * 28].reshape(num_columns, 28).T
    interpolated_matrix = pd.DataFrame(interpolated_matrix)
    mask = np.isnan(original_matrix.values)
    mask_matrix = pd.DataFrame(mask)
    return original_matrix, mask_matrix, interpolated_matrix, df_datetime


def apply_svd_imputation(source_dir, destination_path):
    results = {}

    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    # Get all CSV files from source_dir (including subdirectories)
    csv_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("No CSV files found in the source directory.")
        return

    for csv_path in csv_files:

        df = pd.read_csv(csv_path)

        if df.shape[0] < 28:
            print(f'The {csv_path} file does not have sufficient quantity of lines for imputation (28)')
            continue

        file_name = os.path.basename(csv_path)
        
        results[file_name] = {'linear_interpolation': None, 'final_svd': None}
        df_matrix, df_mask, df_interpolated, df_datetime = matrix(csv_path)
        results[file_name]['linear_interpolation'] = df_interpolated.copy()

        A_previous = df_interpolated.values.copy()
        rmse = float('inf') 
        max_iter = 300
        n_iter = 0

        while rmse >= 1e-3 and n_iter <= max_iter:  
            U, S, Vt = svd_decomposition(df_interpolated)
            variability = np.cumsum(S**2) / np.sum(S**2)

            variability_percentage = 0.95
            r = np.where(variability >= variability_percentage)[0][0] + 1
            
            U_reduced, S_reduced, Vt_reduced = principal_components(r, U, S, Vt)
            S_reduced_matrix = np.diag(S_reduced)

            A_approximated = np.dot(np.dot(U_reduced, S_reduced_matrix), Vt_reduced)
            A_approximated_df = pd.DataFrame(A_approximated)

            df_matrix_filled = df_matrix.fillna(A_approximated_df)
            
            results[file_name]['final_svd'] = df_matrix_filled

            df_interpolated = df_matrix_filled.values
            rmse = np.sqrt(np.mean((A_approximated - A_previous) ** 2))

            A_previous = A_approximated.copy()
            n_iter += 1

        svd = matrix_to_df(results[file_name]["final_svd"])
        interpolation = matrix_to_df(results[file_name]["linear_interpolation"])
        mask = matrix_to_df(df_mask)

        reshaped_dfs = [interpolation, svd, mask]
        
        generate_csv_file(df_datetime, reshaped_dfs[1], destination_path, file_name.split(" ")[0] + " IMPUTED SVD.csv")

        print(f"Processed file: {csv_path}")


def apply_basic_imputations(source_dir, destination_path):
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    csv_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("No CSV files found in the source directory.")
        return

    for path in csv_files:
        df = pd.read_csv(path)

        if df.shape[0] < 28:
            print(f'The file {path} does not have sufficient lines for imputation (28)')
            continue

        df = outlier_removal(df, 'Throughput')

        df_knn = impute_knn(df.copy())
        df_rolling_median = impute_rolling_median(df.copy())
        df_rolling_average = impute_rolling_average(df.copy())
        df_interpolation = linear_interpolation(df.copy())

        techniques = {
            'knn': df_knn,
            'rolling_median': df_rolling_median,
            'rolling_average': df_rolling_average,
            'linear_interpolation': df_interpolation
        }

        original_filename = os.path.basename(path).split(" ")[0] 

        for technique, df_imputed in techniques.items():
            technique_dir = os.path.join(destination_path, technique)

            # Ensure the technique directory exists
            os.makedirs(technique_dir, exist_ok=True)

            new_filename = f"{original_filename} IMPUTED {technique.upper()}.csv"
            output_file = os.path.join(technique_dir, new_filename)

            # Save the imputed file
            df_imputed.to_csv(output_file, index=False)

        print(f"Processed file: {path}")
