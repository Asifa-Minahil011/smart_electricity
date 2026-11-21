import pandas as pd

df = pd.read_csv("data/household_power_consumption.txt",
                 sep=";", na_values=["?"], low_memory=False, nrows=5)
print(df)