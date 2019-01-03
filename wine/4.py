from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
#print(X.shape) (178,13)
#print(y.shape) (178,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
#0.97,决策树的分类效果比KNN好