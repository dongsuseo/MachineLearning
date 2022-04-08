import seaborn as sns

penguins = sns.load_dataset('penguins') # seaborn에서 penguins 데이터 가져오기

data5 = penguins.dropna(axis=0) # 데이터에서 NaN값 가지는 행 제거 (데이터 정제)
data4 = data5.drop("bill_depth_mm", axis=1) # bill_depth_mm 열 제거 (데이터 축소)
data3 = data4.drop("body_mass_g", axis=1) # body_mass_g 열 제거 (데이터 죽소)
data2 = data3.drop("sex", axis=1) # sex 열 제거 (데이터 축소)
data1 = data2.drop("island", axis=1) # island 열 제거 (데이터 축소)
data = data1.drop('species', axis=1) # species 열 제거 (데이터 축소)
t = data1[['species']].copy() # 답에다가 species 값 복사

print(t['species'].unique()) # 데이터 species 확인

# 데이터 실수화
t[t['species'] == 'Adelie'] = 0
t[t['species'] == 'Chinstrap'] = 1
t[t['species'] == 'Gentoo'] = 2
print(t['species'].unique())
t = t['species'].astype("int")


print(data.head(n=345)) #데이터 확인

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    data, t, test_size=0.3, random_state= 42, stratify=t
) # 학습 데이터 테스트 데이터 분류 비율은 7:3 / train_data,test_data 데이터 분포도 일정

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_data,train_target)
print("Train-Eval:", kn.score(train_data, train_target))
print("Test-Eval:", kn.score(test_data, test_target))

# 데이터 시각화
import matplotlib.pyplot as plt
bill_length = data[['bill_length_mm']]
flipper_length = data[['flipper_length_mm']]

plt.scatter(bill_length,flipper_length)

# x축,y축 및 title 이름 변경
plt.xlabel("bill_length_mm")
plt.ylabel("flipper_length_mm")
plt.title("Penguins Data")

# 표 출력
plt.show()

#sns.set_theme()
#sns.pairplot(flight, hue = 'passengers')
#import matplotlib.pyplot as plt
#plt.show()
