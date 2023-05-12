# This is a utility script for training all models with all upsamplings at once

import subprocess

def change_config(field_name, old_value, new_value):
    if type(old_value) == str:
        old_value = '"' + old_value + '"'
        new_value = '"' + new_value + '"'
    else:
        old_value = str(old_value)
        new_value = str(new_value)

    with open("config.py", 'r') as file:
        data = file.read()
        data = data.replace(field_name + " = " + old_value, field_name + " = " + new_value)

    with open("config.py", 'w') as file:
        file.write(data)

def run_train():
    subprocess.run(["python3", "train.py"])

def run_all_trains():
    normal_models = ["lgbm", "perceptron"]
    recurrent_models = ["rnn", "gru", "lstm"]
    upsamplings = ["none", "random", "smote", "borderline_smote", "svm_smote", "kmeans_smote", "adasyn"]

    cur_upsampling = "none"
    cur_model = "lgbm"

    for upsampling in ["adasyn"]:
        change_config("load_upsampled", True, False)
        change_config("upsampling_name", cur_upsampling, upsampling)
        cur_upsampling = upsampling
        for model in normal_models:
            change_config("model_name", cur_model, model)
            cur_model = model
            run_train()
            print(f"Successfully trained model {model} with upsampling {upsampling}")
            change_config("load_upsampled", False, True)

    change_config("lookback", 0, 2)
    for upsampling in ["none", "random", "smote", "borderline_smote", "adasyn"]:
        change_config("load_upsampled", True, False)
        change_config("upsampling_name", cur_upsampling, upsampling)
        cur_upsampling = upsampling
        for model in recurrent_models:
            change_config("model_name", cur_model, model)
            cur_model = model
            run_train()
            print(f"Successfully trained model {model} with upsampling {upsampling}")
            change_config("load_upsampled", False, True)

run_all_trains()
