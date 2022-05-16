import seaborn as sns
titanic = sns.load_dataset('titanic')
data = titanic[['sex', 'age', 'sibsp', 'adult_male', 'parch']].copy()
t = titanic['survived']
data['age'].fillna(30, inplace = True)
data['sex'].replace('male', 1, inplace = True)
data['sex'].replace('female',0, inplace = True)

from sklearn.model_selection import train_test_split
data2, test_data, t2, test_target = train_test_split(
    data, t, test_size=0.3, random_state=42, stratify=t
)
train_data, val_data, train_target, val_target = train_test_split(
    data2, t2, test_size=0.2, random_state=42, stratify=t2
)

from sklearn.neighbors import KNeighborsClassifier
best_score = best_n = best_m = 0
score1 = []
score2 = []
list1 = [1,2]
list2 = [1,3,5,7,10]

for m in list1:
    for n in list2:
        kn = KNeighborsClassifier(n_neighbors=n, p = m)
        kn.fit(train_data, train_target)

        val_score = kn.score(val_data,val_target)
        if m == 1 :
            score1.append(val_score)
        else:
            score2.append(val_score)

        if best_score < val_score:
            best_score = val_score
            best_n = n
            best_m = m
print(score1)
print(score2)
print(best_score, best_n, best_m)

import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.xlabel('n_neighbors')
plt.ylabel('val_score')
plt.title('p = 1')
plt.scatter(list2,score1)

plt.subplot(2,1,2)
plt.xlabel('n_neighbors')
plt.ylabel('val_score')
plt.title('p = 2')
plt.scatter(list2,score2)

plt.show()