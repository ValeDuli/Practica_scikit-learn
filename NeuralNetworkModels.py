from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
print(clf.fit(X, y))

print(clf.predict([[2., 2.], [-1., -2.]]))

print([coef.shape for coef in clf.coefs_])

print(clf.predict_proba([[2., 2.], [1., 2.]]))

X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), random_state=1)
print(clf.fit(X, y))
print(clf.predict([[1., 2.]]))
print(clf.predict([[0., 0.]]))
