import pandas as pd
data3 = pd.read_csv("health.csv")
data = data3[["H","W"]]
t = data3["T"]

from sklearn.neighbors import KNeighborsClassifier
neighbor = 3
kn = KNeighborsClassifier(n_neighbors = neighbor, p=2)
kn.fit(data, t)
print("Eval: ", kn.score(data, t))
test_h = 150; test_w =29
test = pd.DataFrame([[test_h,test_w]], columns = ["H", "W"])
print("Test: ", test_w, test_h, "=>", kn.predict(test))
print("Prob: ", kn.predict_proba(test))


