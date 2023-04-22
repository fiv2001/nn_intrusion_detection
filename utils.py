from config import CONFIG

import numpy as np
import pandas as pd

def load_data(path):
    df_full = pd.read_csv(path).fillna(0).replace(np.inf, 1e18).replace(-np.inf, -1e18)
    values, unique = [], []
    if CONFIG.GENERAL.target_column in df_full.columns:
        values = df_full[CONFIG.GENERAL.target_column]
        df_full = df_full.drop(CONFIG.GENERAL.target_column, axis=1)
        unique = values.unique()
        values = np.array([np.where(unique == values.to_numpy()[i])[0][0] for i in range(values.shape[0])])
    df_numpy = df_full.to_numpy()
    return df_numpy, values, df_full.columns, unique

def get_weights_path():
    return CONFIG.GENERAL.weights_path.replace('{model}', CONFIG.MODEL.model_name)

def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def inference(model, X_test):
    return model.predict(X_test)
