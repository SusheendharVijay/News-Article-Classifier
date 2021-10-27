import pandas as pd
train = pd.read_csv('combined.csv')


print(train.category.value_counts())
