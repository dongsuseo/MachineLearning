# 공개데이터명 : penguins
# 사용모델명 : KNN, 의사결정나무, 로지스틱회귀
# 특징 수 : 4개
# 클래스 수 : 3개

# 데이터 로드
import seaborn as sns
penguins = sns.load_dataset('penguins')

print(penguins)

# 데이터 정제(NaN값 제거)
penguins.dropna(axis=0, inplace=True)

#데이터 확인
print(penguins)

# 특징 추출
data = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].copy()

# 데이터 실수화
penguins['species'].replace('Adelie', 0, inplace=True)
penguins['species'].replace('Chinstrap', 1, inplace=True)
penguins['species'].replace('Gentoo', 2, inplace=True)

# 종속변수 추출
t = penguins['species']

# 학습 데이터 테스트 데이터 분류
from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    data, t, test_size = 0.4, random_state=42, stratify=t
)

#KNN
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
#의사결정나무
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
#로지스틱 회귀
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=500)

#일반적인 KNN 앙상블-보팅 적용
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[('DT', dt), ('KNN', kn), ('LR', lr)])
model.fit(train_data, train_target)

print("Train-Eval:", model.score(train_data,train_target))
print("Test-Eval:", model.score(test_data,test_target))

# 그리드서치를 사용하여 하이퍼파라미터를 찾은 다음 최적의 KNN모델 생성
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