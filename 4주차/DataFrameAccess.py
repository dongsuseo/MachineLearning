# 데이터 프레임 접근

players = ['Mbappe', 'Holland', 'Salah', 'Messi'] #player 리스트 생성
country = ['France', 'Norway', 'Egypt', 'Argentina'] #country 리스트 생성
dict_data = {'Players': players, 'Country': country} #딕셔너리 생성

import pandas as pd # pandas pd로 임포트

df = pd.DataFrame(dict_data) #pandas의 데이터프레임만들어주는 함수 사용

print(dict_data)
print(df)

df.columns = ['Top-Player', 'Nationallity'] #데이터프레임 열 변수 바꿔주기
df.index = ['1st', '2nd', '3rd', '4th'] #데이터프레임 인덱스 바꿔주기

print(df.loc['2nd', 'Nationallity'])
print(df.loc[:,"Top-Player"])
#print(df.loc["Top-Player"] 안된다
#print(df.loc[:1, "Top-Player"] 이것도 안된다.

print(df.loc["2nd"]) #index명으로 찾기
print(df.iloc[1]) #index값으로 찾기

#print(df.iloc["2nd"] iloc을 사용할때는 인덱스 값으로만 가능하다
