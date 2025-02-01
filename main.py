import os
import pandas as pd
from modules.imputation import apply_svd_imputation, apply_basic_imputations
from modules.utils import calculate_rmse
from modules.prediction import gru_prediction
from modules.preprocessing import data_treatment_intervals, get_longest_interval, apply_original_failure_rate_on_longest

def main():
    raw_data_path = 'data/raw/'
    processed_data_path = 'data/processed/'
    data_treatment_intervals(raw_data_path, processed_data_path)

    processed_data_path = 'data/processed/'
    continuous_interval_path = 'data/continuous_intervals/'
    get_longest_interval(processed_data_path, continuous_interval_path)
 
    processed_data_path = 'data/processed/'
    continuous_interval_path = 'data/continuous_intervals/'
    get_longest_interval(processed_data_path, continuous_interval_path)

    processed_data_path = 'data/processed/'
    continuous_interval_path = 'data/continuous_intervals/'
    gapped_interval_path = 'data/gapped_intervals/'
    apply_original_failure_rate_on_longest(processed_data_path, continuous_interval_path, gapped_interval_path)

    gapped_interval_path = 'data/gapped_intervals/'
    imputed_interval_path = 'data/imputed_intervals/'
    apply_basic_imputations(gapped_interval_path, imputed_interval_path)
    apply_svd_imputation(gapped_interval_path, imputed_interval_path)

    continuous_interval_path = 'data/continuous_intervals/'
    imputed_interval_path = 'data/imputed_intervals/'
    output_json = 'results/intervals_imputation_rmse_results.json'
    calculate_rmse(continuous_interval_path, imputed_interval_path, output_json)

    data_with_failures_path = 'data/processed/'
    imputed_interval_path = 'data/imputed/'
    apply_basic_imputations(data_with_failures_path, imputed_interval_path)
    apply_svd_imputation(data_with_failures_path, imputed_interval_path)

    imputed_interval_path = 'data/imputed/'
    gru_prediction(imputed_interval_path)

if __name__ == "__main__":
    main()