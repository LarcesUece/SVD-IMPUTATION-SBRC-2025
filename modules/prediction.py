import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
import json

def create_dataset(X, look_back=3):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i+look_back])
        ys.append(X[i+look_back])
    return np.array(Xs), np.array(ys)

def create_gru(units, train, learning_rate):
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=[train.shape[1], train.shape[2]]))
    model.add(GRU(units=units))
    model.add(Dense(1))
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    return model

def fit_model_with_cross_validation(model, xtrain, ytrain, patience, epochs, batch_size):
    tscv = TimeSeriesSplit(n_splits=4)
    histories = []
    for train_index, val_index in tscv.split(xtrain):
        x_train_fold, x_val_fold = xtrain[train_index], xtrain[val_index]
        y_train_fold, y_val_fold = ytrain[train_index], ytrain[val_index]
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = model.fit(x_train_fold, y_train_fold, epochs=epochs, validation_data=(x_val_fold, y_val_fold), batch_size=batch_size, callbacks=[early_stop], verbose=1)
        histories.append(history)
    return histories

def calculate_mean_history(histories):
    mean_history = {'loss': [], 'val_loss': []}
    for fold_history in histories:
        for key in mean_history.keys():
            mean_history[key].append(fold_history.history[key])
    for key, values in mean_history.items():
        max_len = max(len(val) for val in values)
        for i in range(len(values)):
            if len(values[i]) < max_len:
                values[i] += [values[i][-1]] * (max_len - len(values[i]))
    for key, values in mean_history.items():
        mean_history[key] = [sum(vals) / len(vals) for vals in zip(*values)]
    return mean_history

def save_model(model, directory, substring_desejada):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'{substring_desejada} - GRU.keras')
    model.save(file_path)
    print(f"Model saved in '{file_path}'")

def prediction(model, xtest, myscaler):
    prediction = model.predict(xtest)
    return myscaler.inverse_transform(prediction)

def evaluate_prediction(predictions, actual):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}')
    return rmse, mae

def gru_prediction(source_dir):
    evaluation = {}
    for pasta_raiz, _, arquivos in os.walk(source_dir):
        for arquivo in arquivos:
            if arquivo.endswith('.csv'):
                caminho_arquivo = os.path.join(pasta_raiz, arquivo)
                try:
                    df = pd.read_csv(caminho_arquivo, index_col='Timestamp')
                    if '0' in df.columns:
                        df.drop(columns=['0'], inplace=True)
                    tamanho = int(len(df.index) * 0.8)
                    train_data = df[:tamanho]['Throughput'].values.reshape(-1, 1)
                    test_data = df[tamanho:]['Throughput'].values.reshape(-1, 1)
                    scaler = MinMaxScaler().fit(train_data)
                    train_scaled = scaler.transform(train_data)
                    test_scaled = scaler.transform(test_data)
                    X_train, y_train = create_dataset(train_scaled)
                    X_test, y_test = create_dataset(test_scaled)
                    best_params_gru = {'learning_rate': 0.0001, 'epochs': 100, 'batch_size': 32, 'patience': 5}
                    model_gru = create_gru(64, X_train, best_params_gru['learning_rate'])
                    prev_history_gru = fit_model_with_cross_validation(model_gru, X_train, y_train, best_params_gru['patience'], best_params_gru['epochs'], best_params_gru['batch_size'])
                    history_gru = calculate_mean_history(prev_history_gru)
                    # save_model(model_gru, '../../modelo_salvo', arquivo)
                    y_test = scaler.inverse_transform(y_test)
                    prediction_gru = prediction(model_gru, X_test, scaler)
                    evaluation[arquivo] = evaluate_prediction(prediction_gru, y_test)
                except Exception as e:
                    print(f"Error proccessing {arquivo}: {e}")

    with open('evaluation_rmse_mae.json', 'w') as f:
        json.dump(evaluation, f, indent=4)
