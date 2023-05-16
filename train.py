import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import imblearn.over_sampling as ups

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

import models
import utils
import upsampling

from plotter import save_df_as_image
from config import CONFIG


def save_data(X, y, columns, path):
    df_result = pd.DataFrame(X, columns=columns)
    df_result[CONFIG.GENERAL.target_column] = y
    df_result.to_csv(path, index=False)

def save_values(values, path):
    np.save(path, values)
    print('Values saved.')

def upsample(X, y, columns):
    if CONFIG.UPSAMPLING.load_upsampled:
        df_train = pd.read_csv(utils.get_upsampled_data_path())
        values = df_train[CONFIG.GENERAL.target_column]
        df_train = df_train.drop(CONFIG.GENERAL.target_column, axis=1).to_numpy()
        print('Upsampling loaded')
        return df_train, values.to_numpy()

    unique, counts = np.unique(y, return_counts=True)
    sizes_dict = {}
    for i in range(len(unique)):
        if (counts[i] < CONFIG.UPSAMPLING.min_samples_per_class and counts[i] > CONFIG.UPSAMPLING.min_samples_to_upsample):
            sizes_dict[unique[i]] = CONFIG.UPSAMPLING.min_samples_per_class
    X_smoted, y_smoted = upsampling.create_upsampling(sampling_strategy=sizes_dict).fit_resample(X, y.astype('int'))
    X_smoted, y_smoted = shuffle(X_smoted, y_smoted, random_state=CONFIG.GENERAL.random_state)
    save_data(X_smoted, y_smoted, columns, utils.get_upsampled_data_path())
    print('Upsampling successful')
    return X_smoted, y_smoted

def measure_accuracy(y_pred, y_test, values):
    unique = np.unique(y_test)
    for val in unique:
        cnt = (y_test == val).sum()
        correct_cnt = ((y_test == val) * (y_test == y_pred)).sum()
        print(f"Accuracy for value {values[val]} is {correct_cnt / cnt}")

    return (y_test == y_pred).mean()

def save_confusion_matrix(result_np, values, path):
    result_df = pd.DataFrame(result_np, columns=values)
    result_df.insert(0, "True class", values)
    result_df = result_df.set_index("True class")
    save_df_as_image(result_df, path)

def get_confusion_matrices(y_pred, y_test, values, model_number):
    num_classes = values.shape[0]
    result_np = confusion_matrix(y_pred, y_test, labels=np.arange(0, num_classes, 1))
    result_np_norm = confusion_matrix(y_pred, y_test, labels=np.arange(0, num_classes, 1), normalize='pred')
    save_confusion_matrix(result_np, values, utils.get_confusion_matrix_path(model_number))
    save_confusion_matrix(result_np_norm, values, utils.get_normalized_confusion_matrix_path(model_number))

def calculate_and_print_metrics(y_pred, y_test, values, model_number):
    print(f'Accuracies of model number {model_number}:')
    print(f'Overall accuracy of model number {model_number}:', measure_accuracy(y_pred, y_test, values), '\n')
    utils.create_result_dir()
    get_confusion_matrices(y_pred, y_test, values, model_number)

def prepare_data():
    X, y, columns, values = utils.load_data(CONFIG.GENERAL.full_data_path, is_train=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG.GENERAL.test_size, random_state=CONFIG.GENERAL.random_state)
    print(X_train.size * X_train.itemsize)
    X_train_upsampled, y_train_upsampled = upsample(X_train, y_train, columns)
    save_values(values, CONFIG.GENERAL.values_path)
    return X_train, X_test, X_train_upsampled, y_train, y_test, y_train_upsampled, values

def train_model_and_print_metrics(X_train, X_test, y_train, y_test, values, model_number):
    model = models.create_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    calculate_and_print_metrics(y_pred, y_test, values, model_number)
    return model

def save_model(model):
    path = utils.get_weights_path()
    model.save(path)
    print('Model saved')

def main():
    X_train, X_test, X_train_upsampled, y_train, y_test, y_train_upsampled, values = prepare_data()
    if CONFIG.GENERAL.train_model_without_upsampling:
        basic_model = train_model_and_print_metrics(X_train, X_test, y_train, y_test, values, CONFIG.GENERAL.basic_model_number)
    upsampled_model = train_model_and_print_metrics(X_train_upsampled, X_test, y_train_upsampled, y_test, values, CONFIG.GENERAL.upsampled_model_number)
    save_model(upsampled_model)

main()
