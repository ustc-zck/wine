from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

X, y = load_wine(return_X_y=True)
#print(X.shape) (178,13)
#print(y.shape) (178,)

names = load_wine().feature_names



for i in range(13):
    plt.figure(figsize =(30, 20))
    k = 1
    for j in range(i+1,13):
            plt.subplot(2,7,k)
            plt.scatter(X[:,i], X[:, j])
            plt.xlabel(names[i])
            plt.ylabel(names[j])
            k = k + 1
    plt.show()


