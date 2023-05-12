class CONFIG:
    class GENERAL:
        full_data_path = "data/full_cicids2017.csv"
        test_data_path = "data/test_data.csv"
        result_path = "results/result.txt"
        values_path = "results/values.npy"
        test_size = 0.2
        random_state = 998244353
        target_column = " Label"
        weights_path = "results/{model}/{upsampling}/weights.pkl"
        results_dir = "results/{model}/{upsampling}"
        basic_model_number = 1
        upsampled_model_number = 2
        confusion_matrix_path = "results/{model}/{upsampling}/confusion_matrix.jpg"
        normalized_confusion_matrix_path = "results/{model}/{upsampling}/confusion_matrix_normalized.jpg"
        lookback = 0
        train_model_without_upsampling = False
    class UPSAMPLING:
        upsampling_name = "none" # options: "none", "random", "smote", "borderline_smote", "svm_smote", "kmeans_smote", "adasyn"
        min_samples_per_class = 100000
        min_samples_to_upsample = 50
        load_upsampled = True
        upsampled_data_path = "data/{upsampling}_lookback={lookback}.csv"
        kmeans_cluster_balance_threshold = 3
        kmeans_n_init = 1
    class MODEL:
        model_name = "lgbm" # options: "lgbm", "perceptron", "rnn", "gru", "lstm"
        class LGBM:
            num_leaves = 15
            num_trees = 10
        class PERCEPTRON:
            input_size = 78
            output_size = 15
            hidden_layers = [100, 150, 150, 100]
            loss_function = "CrossEntropy"
            optimizer = "Adam"
            n_epochs = 10
            batch_size = 64
            lr = 1e-4
        class RECURRENT:
            input_size = 78
            output_size = 15
            hidden_size = 100
            num_layers = 1
            loss_function = "CrossEntropy"
            optimizer = "Adam"
            n_epochs = 10
            batch_size = 64
            lr = 1e-4

