class CONFIG:
    class GENERAL:
        full_data_path = 'data/full_cicids2017.csv'
        random_state = 998244353
        target_column = ' Label'
    class UPSAMPLING:
        min_samples_per_class = 100000
        load_upsampled = False
        upsampled_data_path = "data/smoted.csv"
    class MODEL:
        class LGBM:
            num_leaves = 15
            num_trees = 10
