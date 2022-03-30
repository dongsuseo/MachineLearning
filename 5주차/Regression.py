#이웃 회귀

import pandas as pd
data = pd.read_csv(".././health.csv")

h = data[['H']]
w = data[['W']]

#데이터 시각화

# import matplotlib.pyplot as plt
# plt.scatter(h,w)
# plt.xlabel("Height")
# plt.ylabel("Weight")
# plt.title('Health Data')
# plt.show()

# 예측

from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    h,w,test_size=0.2, random_state=42
)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor() #이웃을 default로 두면 5로 자동 설정된다.
knr.fit(train_data, train_target)
print("Train-Eval:", knr.score(train_data,train_target))
print('Test-Eval:', knr.score(test_data, test_target))

test_pred = knr.predict(test_data)

print(test_data)
print(test_target)
print(test_pred)

import matplotlib.pyplot as plt # 그래프로 보여줄려고 임포트

plt.scatter(test_data,test_target) # x축 test_data y축 test_target으로 점 찍어라
plt.scatter(test_data, test_pred) # x축 test_data y축 test_pred으로 점 찍어라
plt.xlabel("Height")
plt.ylabel('Weight')
plt.title('Predicted Data')
plt.show()