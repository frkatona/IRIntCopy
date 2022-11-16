import pandas as pd
import matplotlib.pyplot as plt

x_names = ["this thing", "that thing", "the other thing"]
y_values = [1, 5, 10]
plt.bar(x_names, y_values)
# df = pd.read_csv(r'CSVs\220921_apk_PDMSCompare_uncured-7A30s808nm\s182_0s_3.csv', skiprows = 1)
# print(df['A'][6400])

plt.show()