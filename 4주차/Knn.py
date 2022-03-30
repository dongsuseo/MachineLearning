# KNN

import seaborn as sns

iris = sns.load_dataset("iris")
data = iris.drop("species", axis =1)
t = iris[['species']].copy()


from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, t, test_size = 0.3, random_state=42, stratify=t
) # stratify => 전체 집단의 답 비율과 나뉘어진 섹션들의 답 비율을 일치하게 분리해줌

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_data, train_target)
print("Train-Eval", kn.score(train_data, train_target))
print("Test-Eval", kn.score(test_data, test_target))

from sklearn.metrics import confusion_matrix

conf = confusion_matrix(test_target, kn.predict(test_data))
print(conf)
