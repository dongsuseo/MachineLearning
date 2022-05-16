# 학번 : 18011878
# 분류모델 : 의사결정나무
# 특징조합1 : pclass, sex, age, sibsp, embarked
# 조합이유1 : 일단 데이터를 정제 축소하면서 남은 특징들 모두를 사용해봤습니다.
# 특징조합2 : pclass, sex, age, embarked
# 조합이유2 : 데이터 중요도를 확인하여 가장 중요도가 낮은 특징을 제거해봤습니다.
# 특징조합3 : pclass, sex, age
# 조합이유3 : 조합이유2와 동일합니다. 중요도가 높은 3개의 특징이 제가 상식적으로 생각했을 때 생존과 관련된 특징과 같았습니다.

import copy
import seaborn as sns

# 데이터 로드
titanic = sns.load_dataset('titanic')

# 데이터 확인
# print(titanic.info())

titanic.drop(columns='deck', inplace=True) # 203개의 데이터만 값을 가지고 있어서 데이터 정제
titanic.drop(columns="alive", inplace=True) # survived와 의미가 겹쳐 축소
titanic.drop(columns="class", inplace=True) # pclass와 의미가 겹쳐 축소
titanic.drop(columns="fare", inplace=True) # pclass와 의마 겹쳐 축소
titanic.drop(columns="who", inplace=True) # sex와 의미가 겹쳐 축소
titanic.drop(columns="adult_male", inplace=True) # sex,age와 의미가 겹쳐 축소
titanic.drop(columns="alone", inplace=True)  # sibsp와 의미가 겹쳐 축소
titanic.drop(columns="parch", inplace=True) # sibsp와 의미가 겹쳐 축소
titanic.drop(columns="embark_town", inplace=True) # embarked와 의미가 겹쳐 축소

# 데이터 확인
# print(titanic.info())

# survived: 생존여부
# pclass: 객실등급
# sex: 성별
# sibsp: 형제자매 및 배우자 수
# age: 나이
# embarked: 탑승 정보
# 이렇게 6개의 특징만 남게 됐다.

titanic["age"].fillna(30,inplace=True) # age값이 NaN인 값에 30씩 넣어준다.

data = titanic.dropna(axis=0) #info에서 확인한 바 embarked값이 NaN인 데이터 2개 제거

t = data['survived'].copy()
data2 = data.drop(columns='survived') # 종속 변수로 빠졌으니 제거해준다.

# 데이터 및 종속 변수 확인.
# print(data2.info())
# print(t)

from sklearn.model_selection import train_test_split

print(data2)

cnt = 0
# 데이터 실수화
data2['sex'].replace('male',0,inplace=True)
data2['sex'].replace('female',1,inplace=True)
data2['embarked'].replace('S',0,inplace=True)
data2['embarked'].replace('C',1,inplace=True)
data2['embarked'].replace('Q',2,inplace=True)

print(data2)

data3 = copy.deepcopy(data2) # 초기화용으로 복제본 생성

import matplotlib.pyplot as plt

# 데이터 특징별 상관관계 출력
corr_df = data3.corr()
ax = sns.heatmap(corr_df, annot=True, annot_kws=dict(color='g'), cmap='Greys')
plt.show()
# 출력해본 결과 상관관계가 거의 없음을 확인하였다. 더이상 특징을 제거할 필요는 없어보임

# 데이터 분류
train_data, test_data, train_target, test_target = train_test_split(
  data2, t, test_size=0.3, random_state=42, stratify=t
)

# 의사결정나무 모델을 사용
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_data, train_target)

#데이터 중요도 출력
plt.subplot(3,1,1)
plt.bar(dt.feature_names_in_,dt.feature_importances_,color=['red','blue','yellow','green','black'])
plt.xlabel('Feature name')
plt.ylabel('Importance')
plt.title('Set 1: Data Importance')

# 정확도 1
print("조합 1")
print(dt.feature_names_in_) #데이터 특징 출력
print(dt.feature_importances_) #데이터 중요도 출력

print('Train-Eval:', dt.score(train_data, train_target))
print('Test-Eval:', dt.score(test_data, test_target))


data2.drop(columns='embarked', inplace=True)

train_data, test_data, train_target, test_target = train_test_split(
  data2, t, test_size=0.3, random_state=42, stratify=t
)
dt.fit(train_data, train_target)

# subplot을 사용하여 조합 3개의 특징 중요도를 한눈에 파악하게 사용했습니다.
plt.subplot(3,1,2)
plt.bar(dt.feature_names_in_,dt.feature_importances_,color=['red','blue','yellow','green'])
plt.xlabel('Feature name')
plt.ylabel('Importance')
plt.title('Set 2: Data Importance')

# 정확도 2
print("조합 2")
print(dt.feature_names_in_)
print(dt.feature_importances_)
print('Train-Eval:', dt.score(train_data, train_target))
print('Test-Eval:', dt.score(test_data, test_target))

data2.drop(columns='sibsp', inplace=True)

train_data, test_data, train_target, test_target = train_test_split(
  data2, t, test_size=0.3, random_state=42, stratify=t
)

dt.fit(train_data, train_target)

plt.subplot(3,1,3)
plt.bar(dt.feature_names_in_,dt.feature_importances_,color=['red','blue','yellow'])
plt.xlabel('Feature name')
plt.ylabel('Importance')
plt.title('Set 3: Data Importance')
plt.subplots_adjust(wspace=0.35, hspace=0.95)

# 정확도 3
print("조합 3")
print(dt.feature_names_in_)
print(dt.feature_importances_)
print('Train-Eval:', dt.score(train_data, train_target))
print('Test-Eval:', dt.score(test_data, test_target))

# 데이터 중요도 시각화를 하여 중요도를 파악하고 중요도가 가장 낮은 특징을 제거해가면서 조합을 설정해봤습니다.
# 여기서는 출력을 안했지만 조합3에서 특징을 하나 더 제거하면 오히려 정확도가 내려감을 확인 할 수 있었습니다.
# 제가 조합을 했을 때는 조합3이 Test-Eval값이 가장 크게 나왔습니다.

# 특징조합 설명을 위한 데이터 시각화
plt.show()


