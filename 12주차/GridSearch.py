import seaborn as sns
titanic = sns.load_dataset('titanic')
data = titanic[['sex', 'age', 'sibsp', 'adult_male', 'parch']].copy()
t = titanic['survived']
data['age'].fillna(30,inplace=True)
data['sex'].replace('male', 1, inplace=True)
data['sex'].replace('female', 0, inplace=True)

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    data, t, test_size=0.3, random_state=42, stratify=t
)

from sklearn.neighbors import KNeighborsClassifier
params = {'n_neighbors': [1,3,5,7,10], 'p':[1,2]}

from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(KNeighborsClassifier(), params, cv = 10)

gs.fit(train_data, train_target)
print(gs.best_estimator_) # 하이퍼파라미터 출력
kn = gs.best_estimator_

print(kn.score(train_data, train_target))
print(kn.score(test_data, test_target))

# import matplotlib.pyplot as plt
# plt.plot()