import seaborn as sns
titanic = sns.load_dataset('titanic')
data = titanic[['sex', 'age', 'sibsp','adult_male', 'parch']].copy()
t = titanic['survived']
data['age'].fillna(30, inplace = True)
data['sex'].replace('male', 1, inplace=True)
data['sex'].replace('female', 0, inplace=True)

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    data, t, test_size = 0.3, random_state=42, stratify=t
)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
#의사결정나무
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
#로지스틱 회귀
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

#일반적인 KNN 앙상블-보팅 적용
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[('DT', dt), ('KNN', kn), ('LR', lr)])
model.fit(train_data, train_target)

print("Train-Eval:", model.score(train_data,train_target))
print("Test-Eval:", model.score(test_data,test_target))

# 그리드서치를 사용하여 하이퍼파라미터를 찾은 다음 최적의 Kn모델 생성
from sklearn.neighbors import KNeighborsClassifier
params = {'n_neighbors': [1,3,5,7,10], 'p':[1,2]}
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(KNeighborsClassifier(), params, cv = 10)
gs.fit(train_data, train_target)
print(gs.best_estimator_) # 하이퍼파라미터 출력
kn = gs.best_estimator_

#그리드서치 + 앙상블-보팅 적용
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[('DT', dt), ('KNN', kn), ('LR', lr)])
model.fit(train_data, train_target)

print("Train-Eval2:", model.score(train_data,train_target))
print("Test-Eval2:", model.score(test_data,test_target))
# print(model.estimators_)  # 어떤 모델을 적용했는지 출력
# print(model.feature_names_in_) # 적용한 특징 출력
