class CONFIG:
    class GENERAL:
        full_data_path = 'data/full_cicids2017.csv'
        test_data_path = 'data/test_data.csv'
        result_path = 'results/result.txt'
        values_path = 'results/values.npy'
        test_size = 0.2
        random_state = 998244353
        target_column = ' Label'
        weights_path = 'results/{model}_weights.pkl'
        basic_model_number = 1
        upsampled_model_number = 2
        confusion_matrix_path = 'results/{model}_confusion_matrix_{model_number}.jpg'
        normalized_confusion_matrix_path = 'results/{model}_confusion_matrix_normalized_{model_number}.jpg'
    class UPSAMPLING:
        min_samples_per_class = 100000
        load_upsampled = True
        upsampled_data_path = "data/smoted.csv"
    class MODEL:
        model_name = 'lgbm'
        class LGBM:
            num_leaves = 15
            num_trees = 10
