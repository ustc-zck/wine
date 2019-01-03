from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

X, y = load_wine(return_X_y=True)
#print(X.shape) (178,13)
#print(y.shape) (178,)

names = load_wine().feature_names

plt.figure(figsize = (50,30))

for i in range(13):
    plt.subplot(4,4, i+1)
    plt.scatter(X[:, i], y)
    plt.xlabel(names[i])
    plt.ylabel('Classification')
plt.show()

