x = [[0],[1],[2],[3]]
y = [0,0,1,1]

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x,y)

print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))

import pandas as pd
health = pd.read_csv(".././health.csv")
print(health.value)

data = health[["H", 'W']]
t = health['T']

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(data, t)
print("Eval:" kn.score(data, t))

test_h =150; test_w = 29
test = pd.DataFrame([[test_h, test_w]], columns = ['H', 'W'])
print("Test:", test_w, test_h, "=>", kn.predict(test))
print("Prob", kn.predict_proba(test))

