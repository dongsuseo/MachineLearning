# 데이터 프레임

players = ['Mbappe', 'Holland', 'Salah', 'Messi'] #player 리스트 생성
country = ['France', 'Norway', 'Egypt', 'Argentina'] #country 리스트 생성
dict_data = {'Players': players, 'Country': country} #딕셔너리 생성

import pandas as pd # pandas pd로 임포트

df = pd.DataFrame(dict_data) #pandas의 데이터프레임만들어주는 함수 사용

print(dict_data)
print(df)