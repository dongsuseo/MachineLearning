import seaborn as sns

penguins = sns.load_dataset('penguins')
data5 = penguins.dropna(axis=0)
data4 = data5.drop("bill_depth_mm", axis=1)
data3 = data4.drop("body_mass_g", axis=1)
data2 = data3.drop("sex", axis=1)
data1 = data2.drop("island", axis=1)
data = data1.drop('species', axis=1)
t = data1[['species']].copy()
print(t['species'].unique())
t[t['species'] == 'Adelie'] = 0
t[t['species'] == 'Chinstrap'] = 1
t[t['species'] == 'Gentoo'] = 2
print(t['species'].unique())
t = t['species'].astype("int")


print(data.head(n=345))

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    data, t, test_size=0.3, random_state= 42, stratify=t
)
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_data,train_target)
print("Train-Eval:", kn.score(train_data, train_target))
print("Test-Eval:", kn.score(test_data, test_target))

#sns.set_theme()
#sns.pairplot(flight, hue = 'passengers')
#import matplotlib.pyplot as plt
#plt.show()
