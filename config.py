class CONFIG:
    class GENERAL:
        full_data_path = "data/full_cicids2017.csv"
        test_data_path = "data/test_data.csv"
        result_path = "results/result.txt"
        values_path = "results/values.npy"
        test_size = 0.2
        random_state = 998244353
        target_column = " Label"
        weights_path = "results/{model}_{upsampling}_weights.pkl"
        basic_model_number = 1
        upsampled_model_number = 2
        confusion_matrix_path = "results/{model}_{upsampling}_confusion_matrix.jpg"
        normalized_confusion_matrix_path = "results/{model}_{upsampling}_confusion_matrix_normalized.jpg"
        shuffle = False
    class UPSAMPLING:
        upsampling_name = "smote"
        min_samples_per_class = 100000
        load_upsampled = False
        upsampled_data_path = "data/{upsampling}.csv"
    class MODEL:
        model_name = "lgbm"
        class LGBM:
            num_leaves = 15
            num_trees = 10
        class PERCEPTRON:
            input_size = 78
            output_size = 15
            hidden_layers = [100, 150, 150, 100]
            loss_function = "CrossEntropy"
            optimizer = "Adam"
            n_epochs = 20
            batch_size = 64
            lr = 1e-4
