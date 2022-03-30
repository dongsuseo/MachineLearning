# 데이터 전처리
# import numpy as np
# import matplotlib.pyplot as plt
#
# t = np.arange(0,100)*0.01
# s = np.sin(2*np.pi*t)
# c = np.cos(2*np.pi*t)
#
# plt.subplot(2,1,1);plt.plot(t,s);plt.grid()
# plt.subplot(2,1,2); plt.plot(t,c); plt.grid()
# plt.show()


#################################################################
# 데이터 시각화

import matplotlib.pyplot as plt

w=[];h=[];t=[]


with open("health.csv", "r") as file:
    lines = file.readlines()[1:]
    for line in lines:
        x,y,z = line.strip().split(',')
        w.append(float(x))
        h.append(float(y))
        t.append(int(z))

data = [[x,y]for x,y in zip(w,h)]

data3 = [[x,y,z] for x,y,z in zip(w,h,t)]
ch_w = [x for x,y,z in data3 if z==1]
ch_h = [y for x,y,z in data3 if z==1]
ad_w = [x for x,y,z in data3 if z==0]
ad_h = [y for x,y,z in data3 if z==0]

plt.scatter(ch_w,ch_h)
plt.scatter(ad_w,ad_h)

from sklearn.neighbors import KNeighborsClassifier
neighbor = 7
kn = KNeighborsClassifier(n_neighbors=neighbor,p=2)
kn.fit(data,t)

plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Health Data")
test_h = 150; test_w = 29
plt.scatter(test_h, test_w, marker = "^", c = "black")
# plt.show()
test_h = 150; test_w =29
print("Test:", test_w, test_h,"=>", kn.predict([[test_h, test_w]]))
print("Prob:", kn.predict_proba([[test_h, test_w]]))
dist, idx = kn.kneighbors([[test_h,test_w]], n_neighbors = neighbor)
print(dist,idx)

