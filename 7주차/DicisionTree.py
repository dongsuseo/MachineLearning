import seaborn as sns
iris = sns.load_dataset('iris')

data = iris.drop('species', axis=1)
t = iris[['species']].copy()
t[t['species']=='setosa'] = 0
t[t['species']=='versicolor'] = 1
t[t['species']=='virginica'] = 2
t = t['species'].astype('int')

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    data, t, test_size=0.3, random_state=42, stratify=t
)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_data, train_target)
print('Train-Eval:', dt.score(train_data, train_target))
print('Test-Eval:', dt.score(test_data, test_target))

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(test_target,dt.predict(test_data))
print(conf)


import matplotlib.pyplot as plt

from sklearn import tree
tree.plot_tree(dt , max_depth=5, class_names= t) #아직 구현이 안된다
plt.show()