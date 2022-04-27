# 데이터 전처리

import seaborn as sns # seaborn에서 sns로 임포트

iris = sns.load_dataset("iris") # seaborn에서 "iris"데이터 가져오기

data = iris.drop("species", axis=1) # iris 데이터에서 species 변수 열 삭제 만약 aixs=0이면 행삭제
data2 = iris.drop(0, axis=0) #행을 삭제할떄는 인덱스이므로 ""를 쓰지않고 int로 기입해준다 "0"이라 하면 오류 생김


t = iris[['species']].copy() # iris 데이터에서 species변수 복사

print(t['species'].unique()) # t의 species값 중복없이 출력

t[t['species']=='setosa'] = 0 # t의 species에서 setosa값은 0으로 바꾸기
t[t['species']=='versicolor'] = 1 # t의 species에서 versicolor값은 1으로 바꾸기
t[t['species']=='virginica'] = 2 # t의 species에서 virginica값은 2으로 바꾸기

print(t['species'].unique())

t = t['species'].astype('int') #t의 species 타입을 int로 바꾸기
# print(t.dtypes) t의 데이터타입 출력

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