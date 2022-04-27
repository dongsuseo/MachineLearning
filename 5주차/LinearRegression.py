import pandas as pd
data = pd.read_csv(".././health.csv")
h = data[['H']]
w = data[['W']]

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    h, w, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression # 선형회귀
lr = LinearRegression()
lr.fit(train_data, train_target)
print("Train-Eval:", lr.score(train_data,train_target))
print("Test-Eval:", lr.score(test_data, test_target))

import numpy as np  # 다항회귀
train_poly = np.column_stack((train_data ** 2, train_data))
test_poly = np.column_stack((test_data ** 2, test_data))

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print("Train-Eval:", lr.score(train_data, train_target))
print("Test-Eval:", lr.score(test_data, test_target))