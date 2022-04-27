import seaborn as sns
iris = sns.load_dataset('iris')

data = iris.drop('species', axis=1)
t = iris[['species']].copy()

t[t['species']=='setosa']=0  # 데이터 실수화
t[t['species']=='versicolor']=1
t[t['species']=='virginica']=2
t = t['species'].astype('int')

from sklearn.model_selection import train_test_split #데이터 분류
train_data, test_data, train_target, test_target = train_test_split(
    data, t, random_state=42, stratify=t, test_size=0.3
)

from sklearn.preprocessing import StandardScaler # 데이터 변환(표준화)
scaler = StandardScaler()
scaler.fit(train_data)
train_data_scaled = scaler.transform(train_data)
test_data_scaled = scaler.transform(test_data)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis(n_components=2) # 몇 차원으로 줄일지
model.fit(train_data_scaled, train_target)
print("Train-Eval:", model.score(train_data_scaled, train_target))
print("Test-Eval:", model.score(test_data_scaled, test_target))


from sklearn.metrics import confusion_matrix
conf = confusion_matrix(test_target, model.predict(test_data_scaled))
print(conf)

train_data_lda = model.transform(train_data_scaled) #차원 축소
print(model.coef_)
print(model.intercept_)

import matplotlib.pyplot as plt
plt.xlabel("LDA1")
plt.ylabel("LDA2")
# print(train_data_lda)
plt.scatter(train_data_lda[:,0],train_data_lda[:,1], c= train_target)
plt.show()

# QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
model2 = QuadraticDiscriminantAnalysis()
model2.fit(train_data_scaled, train_target)
print("Train-Eval:", model2.score(train_data_scaled, train_target))
print("Test-Eval:", model2.score(test_data_scaled, test_target))

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(test_target, model2.predict(test_data_scaled))
print(conf)

