import pandas as pd

df = pd.read_csv('random_lookback=2.csv')
print(df[' Label'].value_counts())
