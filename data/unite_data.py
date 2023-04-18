import os
import pandas as pd

def build_dataset(work_dir):
    file_names = os.listdir(work_dir)
    dataframes = []
    for file_name in file_names:
      path = os.path.join(work_dir, file_name)
      d_df = pd.read_csv(path)
      dataframes.append(d_df.copy())
    return pd.concat(dataframes)

df_full = build_dataset('MachineLearningCSV/MachineLearningCVE/')
df_full.to_csv('full_cicids2017.csv', index=False)
