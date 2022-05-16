# print("이 코드창은 수정하지 마세요, 수정 시 0점")
import seaborn as sns
iris = sns.load_dataset("iris")

idx = iris[iris["species"] == "setosa"].index
iris.drop(idx, inplace = True)
data1 = iris.drop(["petal_length", "species"], axis = 1)
t = iris[["species"]].copy()
t[t["species"] == "versicolor"] = 0
t[t["species"] == "virginica"] = 1
t = t["species"].astype("int")

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    data1, t, test_size = 0.2, random_state = 42, stratify = t)

print(data1)

from sklearn.preprocessing import MinMaxScaler # 데이터 변환(표준화)
scaler = MinMaxScaler()
scaler.fit(train_data)
train_data_scaled = scaler.transform(train_data)
test_data_scaled = scaler.transform(test_data)

from sklearn.tree import DecisionTreeClassifier #의사결정나무 분류
dt = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=4, min_samples_split=3)
dt.fit(train_data, train_target)
print('Train-Eval:', dt.score(train_data, train_target))
print('Test-Eval:', dt.score(test_data, test_target))