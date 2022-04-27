import seaborn as sns
iris = sns.load_dataset('iris')

data = iris.drop('species', axis=1)  #데이터 실수화
t = iris[['species']].copy()
t[t['species']=='setosa'] = 0
t[t['species']=='versicolor'] = 1
t[t['species']=='virginica'] = 2
t = t['species'].astype('int')

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    data, t, test_size=0.3, random_state=42, stratify=t
)

from sklearn.tree import DecisionTreeClassifier #의사결정나무 분류
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_data, train_target)
print('Train-Eval:', dt.score(train_data, train_target))
print('Test-Eval:', dt.score(test_data, test_target))

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(test_target,dt.predict(test_data))
print(conf)


import matplotlib.pyplot as plt
from sklearn import tree  # 의사결정나무 시각화
tree.plot_tree(dt , max_depth=5, filled=True) # filled : 색상을 넣을 것인가 # max_depth : 최대 깊이
plt.show()