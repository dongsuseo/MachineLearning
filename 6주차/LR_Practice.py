import seaborn as sns
iris = sns.load_dataset("iris")

idx = iris[iris["species"]=="virginica"].index
iris.drop(idx, inplace=True)
data = iris.drop('species', axis=1)

t = iris[['species']].copy()

t[t['species']=='setosa'] = 0
t[t['species']=='versicolor'] = 1
t = t['species'].astype('int')

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    data, t, test_size=0.3, random_state=42, stratify=t
)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_data, train_target)
print("Train-Eval:", lr.score(train_data, train_target))
print("Test-Eval:", lr.score(test_data, test_target))
print(lr.coef_, lr.intercept_)

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(test_target, lr.predict(test_data))
print(conf)

