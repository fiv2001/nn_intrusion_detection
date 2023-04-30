import numpy as np
import joblib

import utils

from config import CONFIG

def load_model(path):
    model = joblib.load(path)
    return model

def get_labels_from_indexes(indexes):
    values = np.load(CONFIG.GENERAL.values_path, allow_pickle=True)
    return values[indexes]

def save_result(result):
    result_path = CONFIG.GENERAL.result_path
    with open(result_path, 'w') as f:
        for value in result:
            f.write(value + '\n')
    print(f'Result saved to {result_path}')

def inference():
    model = load_model(utils.get_weights_path())
    X_test, _, _, _ = utils.load_data(CONFIG.GENERAL.test_data_path)
    result = get_labels_from_indexes(model.predict(X_test))
    save_result(result)

inference()
