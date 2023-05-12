from config import CONFIG

from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path, is_train):
    df_full = pd.read_csv(path).fillna(0).replace(np.inf, 1e18).replace(-np.inf, -1e18)
    values, unique = [], []
    if CONFIG.GENERAL.target_column in df_full.columns:
        values = df_full[CONFIG.GENERAL.target_column]
        df_full = df_full.drop(CONFIG.GENERAL.target_column, axis=1)
        unique = values.unique()
        values = np.array([np.where(unique == values.to_numpy()[i])[0][0] for i in range(values.shape[0])])
    df_numpy = df_full.to_numpy()
    lookback = CONFIG.GENERAL.lookback
    columns = df_full.columns.tolist()
    if lookback > 0:
        delete_indexes = []
        new_numpy = []
        for i in range(0, df_numpy.shape[0]):
            if (values[i] == 0 and i % 6 != 0 and is_train):
                delete_indexes.append(i)
                continue

            left_border = max(0, i - lookback)
            new_features = df_numpy[left_border:i + 1].flatten()
            if new_features.shape[0] < (lookback + 1) * df_numpy.shape[1]:
                new_features = np.concatenate([np.zeros(((lookback + 1) * df_numpy.shape[1] - new_features.shape[0])), new_features])
            new_numpy.append(new_features)
        new_numpy = np.stack(new_numpy)
        df_numpy = new_numpy
        print(df_numpy.size * df_numpy.itemsize)

        columns_copy = columns.copy()
        for i in range(lookback):
            new_columns = columns_copy.copy()
            for j in range(len(new_columns)):
                new_columns[j] += ("_" + str(i))
            columns.extend(new_columns)

        values = np.delete(values, delete_indexes)

    return df_numpy, values, columns, unique

def get_result_dir():
    return CONFIG.GENERAL.results_dir.replace('{model}', CONFIG.MODEL.model_name).replace('{upsampling}', CONFIG.UPSAMPLING.upsampling_name)

def create_result_dir():
    path = get_result_dir()
    Path(path).mkdir(parents=True, exist_ok=True)

def get_weights_path():
    return CONFIG.GENERAL.weights_path.replace('{model}', CONFIG.MODEL.model_name).replace('{upsampling}', CONFIG.UPSAMPLING.upsampling_name)

def get_upsampling_name(model_number):
    return CONFIG.UPSAMPLING.upsampling_name if model_number == CONFIG.GENERAL.upsampled_model_number else ""

def get_upsampled_data_path():
    return CONFIG.UPSAMPLING.upsampled_data_path.replace("{upsampling}", CONFIG.UPSAMPLING.upsampling_name).replace("{lookback}", str(CONFIG.GENERAL.lookback))

def get_confusion_matrix_path(model_number):
    return CONFIG.GENERAL.confusion_matrix_path.replace('{model}', CONFIG.MODEL.model_name).replace('{upsampling}', get_upsampling_name(model_number))

def get_normalized_confusion_matrix_path(model_number):
    return CONFIG.GENERAL.normalized_confusion_matrix_path.replace('{model}', CONFIG.MODEL.model_name).replace('{upsampling}', get_upsampling_name(model_number))
