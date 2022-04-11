import pandas as pd
data3 = pd.read_csv('.././health.csv')
data = data3[['H', 'W']]
t = data3['T']

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split( #스플릿을 할 떄는 데이터 타켓 순으로 한다.
    data, t, test_size=0.3, random_state=42, stratify=t
)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_data, train_target)
print("Train-Eval:", lr.score(train_data, train_target))
print("Test_Eval:", lr.score(test_data,test_target))
print(lr.coef_, lr.intercept_)

import numpy as np
print(np.round(lr.predict_proba(test_data), 3)) # 로지스틱 회귀 예측 확률 값 출력
print(lr.predict(test_data)) # 로지스틱 회귀 예측 값 출력

print(lr.coef_[0][0]*test_data['H']+lr.coef_[0][1]*test_data['W']+lr.intercept_[0])
z = lr.decision_function(test_data) #선형 방정식의 결과 값 출력 결과값이 큰 음수일 경우 예측값은 0으로 근접
print(z)
