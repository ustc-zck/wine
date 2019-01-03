

from sklearn.datasets import load_wine
import numpy as np

X, y = load_wine(return_X_y=True)
#print(X.shape) (178,13)
#print(y.shape) (178,)

print(np.mean(X, axis = 0))
print(np.max(X, axis = 0))
print(np.min(X, axis = 0))
