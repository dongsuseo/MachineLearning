import seaborn as sns
iris = sns.load_dataset('iris')
data = iris.drop('species', axis=1)
t = iris[['species']].copy()
t[t['species']=='setosa'] = 0
t[t['species']=='versicolor']= 1
t[t['species']=='virginica']= 2
t = t['species'].astype('int')

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    data, t, test_size=0.3, random_state=42, stratify=t
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# scaler.fit
scaler.fit(train_data)
train_data_scaled = scaler.transform(train_data)
test_data_scaled = scaler.transform(test_data)

from sklearn.preprocessing import MinMaxScaler
scaler2 = MinMaxScaler()
# scaler2.fit
scaler2.fit(train_data)
train_data_scaled2 = scaler2.transform(train_data)
test_data_scaled2 = scaler2.transform(test_data)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# train_data -> train_data_scaled
lr.fit(train_data_scaled, train_target)
print("표준화")
print("Train-Eval: ", lr.score(train_data_scaled, train_target))
print("Test-Eval: ", lr.score(test_data_scaled, test_target))
print("정규화")
# lr 에는 이미 train_data_scaled로 적용된 모델이 있으므로 다시 초기화해서 학습시켜줘야함.
# train_data -> train_data_scaled2
lr.fit(train_data_scaled2, train_target)
print("Train-Eval: ", lr.score(train_data_scaled2, train_target))
print("Test-Eval: ", lr.score(test_data_scaled2, test_target))