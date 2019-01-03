from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
#print(X.shape) (178,13)
#print(y.shape) (178,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
 
print(neigh.score(X_test, y_test))
#最好的ｋ是３，准确率是0.69