import pandas as pd
data = pd.read_csv('spam.csv', encoding='latin1')
print(data.head())