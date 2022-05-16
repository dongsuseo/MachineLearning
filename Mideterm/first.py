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


#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis(n_components=1) # 몇 차원으로 줄일지
model.fit(train_data, train_target)
print("Train-Eval:", model.score(train_data, train_target))
print("Test-Eval:", model.score(test_data, test_target))
