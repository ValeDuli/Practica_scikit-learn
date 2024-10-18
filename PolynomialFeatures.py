import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
print(X)
poly = PolynomialFeatures(2)
print(poly.fit_transform(X))
poly = PolynomialFeatures(interaction_only=True)
print(poly.fit_transform(X))