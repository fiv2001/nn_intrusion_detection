from config import CONFIG

import numpy as np
import pandas as pd
import lightgbm as lgb
import imblearn.over_sampling as ups

from sklearn.model_selection import train_test_split

def load_data():
    df_full = pd.read_csv(CONFIG.GENERAL.full_data_path).fillna(0).replace(np.inf, 1e18).replace(-np.inf, -1e18)
    values = df_full[CONFIG.GENERAL.target_column]
    df_full = df_full.drop(CONFIG.GENERAL.target_column, axis=1)
    unique = values.unique()
    values = np.array([np.where(unique == values.to_numpy()[i])[0][0] for i in range(values.shape[0])])
    df_numpy = df_full.to_numpy()
    return df_numpy, values, df_full.columns, unique

def save_data(X, y, columns, path):
    df_result = pd.DataFrame(X, columns=columns)
    df_result[CONFIG.GENERAL.target_column] = y
    df_result.to_csv(path, index=False)

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

def measure_accuracy(X_train, X_test, y_train, y_test, values):
    model = create_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    unique = np.unique(y_test)
    for val in unique:
        cnt = (y_test == val).sum()
        correct_cnt = ((y_test == val) * (y_test == y_pred)).sum()
        print(f"Accuracy for value {values[val]} is {correct_cnt / cnt}")

    return (y_test == y_pred).mean()

def main():
    X, y, columns, values = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=CONFIG.GENERAL.random_state)
    X_train_upsampled, y_train_upsampled = upsample(X_train, y_train, columns)
    print('Accuracies without upsampling:')
    print('Overall accuracy without upsampling:', measure_accuracy(X_train, X_test, y_train, y_test, values), '\n')
    print('Accuracies with upsampling:')
    print('Overall accuracy with upsamping:', measure_accuracy(X_train_upsampled, X_test, y_train_upsampled, y_test, values), '\n')

main()
