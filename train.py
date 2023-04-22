import numpy as np
import pandas as pd
import lightgbm as lgb
import imblearn.over_sampling as ups
import joblib

from sklearn.model_selection import train_test_split

from config import CONFIG
from utils import load_data, get_weights_path, train, inference

def save_data(X, y, columns, path):
    df_result = pd.DataFrame(X, columns=columns)
    df_result[CONFIG.GENERAL.target_column] = y
    df_result.to_csv(path, index=False)

def save_values(values, path):
    np.save(path, values)
    print('Values saved.')

def upsample(X, y, columns):
    if CONFIG.UPSAMPLING.load_upsampled:
        df_train = pd.read_csv(CONFIG.UPSAMPLING.upsampled_data_path)
        values = df_train[CONFIG.GENERAL.target_column]
        df_train = df_train.drop(CONFIG.GENERAL.target_column, axis=1).to_numpy()
        print('Upsampling loaded')
        return df_train, values

    unique, counts = np.unique(y, return_counts=True)
    sizes_dict = {}
    for i in range(len(unique)):
        if (counts[i] < CONFIG.UPSAMPLING.min_samples_per_class):
            sizes_dict[unique[i]] = CONFIG.UPSAMPLING.min_samples_per_class
    X_smoted, y_smoted = ups.SMOTE(sampling_strategy=sizes_dict).fit_resample(X, y.astype('int'))
    save_data(X_smoted, y_smoted, columns, CONFIG.UPSAMPLING.upsampled_data_path)
    print('Upsampling successful')
    return X_smoted, y_smoted

def create_model():
    return lgb.LGBMClassifier(num_leaves=CONFIG.MODEL.LGBM.num_leaves, n_estimators=CONFIG.MODEL.LGBM.num_trees)

def measure_accuracy(y_pred, y_test, values):
    unique = np.unique(y_test)
    for val in unique:
        cnt = (y_test == val).sum()
        correct_cnt = ((y_test == val) * (y_test == y_pred)).sum()
        print(f"Accuracy for value {values[val]} is {correct_cnt / cnt}")

    return (y_test == y_pred).mean()

def calculate_and_print_metrics(y_pred, y_test, values, model_number):
    print(f'Accuracies of model number {model_number}:')
    print(f'Overall accuracy of model number {model_number}:', measure_accuracy(y_pred, y_test, values), '\n')

def prepare_data():
    X, y, columns, values = load_data(CONFIG.GENERAL.full_data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG.GENERAL.test_size, random_state=CONFIG.GENERAL.random_state)
    X_train_upsampled, y_train_upsampled = upsample(X_train, y_train, columns)
    save_values(values, CONFIG.GENERAL.values_path)
    return X_train, X_test, X_train_upsampled, y_train, y_test, y_train_upsampled, values

def train_model_and_print_metrics(X_train, X_test, y_train, y_test, values, model_number):
    model = create_model()
    model = train(model, X_train, y_train)
    y_pred = inference(model, X_test)
    calculate_and_print_metrics(y_pred, y_test, values, model_number)
    return model

def save_model(model):
    path = get_weights_path()
    joblib.dump(model, path)
    print('Model saved')

def main():
    X_train, X_test, X_train_upsampled, y_train, y_test, y_train_upsampled, values = prepare_data()
    basic_model = train_model_and_print_metrics(X_train, X_test, y_train, y_test, values, CONFIG.GENERAL.basic_model_number)
    upsampled_model = train_model_and_print_metrics(X_train_upsampled, X_test, y_train_upsampled, y_test, values, CONFIG.GENERAL.upsampled_model_number)
    save_model(upsampled_model)

main()
