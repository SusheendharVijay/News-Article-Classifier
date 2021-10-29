import pandas as pd
train = pd.read_csv('combined.csv')
new_data = pd.read_csv("fastapi/live_data.csv")


print(train.head())
print("-------------------------------")
print(new_data.head())
