import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# Create data and polynomial svm classifier

x, y = make_moons(100, True, noise=0.1)

polynomial_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss='hinge', max_iter=300, tol=0.1))
])
polynomial_svm_clf.fit(x, y)

# Plot the data and some predictions

cdict = {0: 'red', 1: 'blue'}
fig, ax = plt.subplots()
for c in np.unique(y):
    ix = np.where(y == c)
    ax.scatter(x[ix, 0], x[ix, 1], c=cdict[c], label=c)
plt.scatter(-0.5, 0.5, c='black',
            label=f"prediction:{np.int(polynomial_svm_clf.predict([[-0.5, 0.5]]))}")
plt.scatter(0, 0.5, c='black', marker="s",
            label=f"prediction:{np.int(polynomial_svm_clf.predict([[0, 0.5]]))}")
plt.scatter(1.0, 0.5, c='black', marker="*",
            label=f"prediction:{np.int(polynomial_svm_clf.predict([[1.0, 0.5]]))}")
plt.scatter(1.7, 0.5, c='black', marker="^",
            label=f"prediction:{np.int(polynomial_svm_clf.predict([[1.7, 0.5]]))}")
plt.legend()
plt.title("Polynomial SVM Classifier")
plt.show()


# Kernel based svm
# poly kernel svm

poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5)),
])
poly_kernel_svm_clf.fit(x, y)

# Plot the data and make predictions

cdict = {0: 'red', 1: 'blue'}
fig, ax = plt.subplots()
for c in np.unique(y):
    ix = np.where(y == c)
    ax.scatter(x[ix, 0], x[ix, 1], c=cdict[c], label=c)
plt.scatter(-0.5, 0.5, c='black',
            label=f"prediction:{np.int(poly_kernel_svm_clf.predict([[-0.5, 0.5]]))}")
plt.scatter(0, 0.5, c='black', marker="s",
            label=f"prediction:{np.int(poly_kernel_svm_clf.predict([[0, 0.5]]))}")
plt.scatter(1.0, 0.5, c='black', marker="*",
            label=f"prediction:{np.int(poly_kernel_svm_clf.predict([[1.0, 0.5]]))}")
plt.scatter(1.7, 0.5, c='black', marker="^",
            label=f"prediction:{np.int(poly_kernel_svm_clf.predict([[1.7, 0.5]]))}")
plt.legend()
plt.title("Poly Kernel SVM Classifier")
plt.show()


# rbf kernel svm

rbf_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001)),
])
rbf_kernel_svm_clf.fit(x, y)

# Plot the data and make predictions

cdict = {0: 'red', 1: 'blue'}
fig, ax = plt.subplots()
for c in np.unique(y):
    ix = np.where(y == c)
    ax.scatter(x[ix, 0], x[ix, 1], c=cdict[c], label=c)
plt.scatter(-0.5, 0.5, c='black',
            label=f"prediction:{np.int(rbf_kernel_svm_clf.predict([[-0.5, 0.5]]))}")
plt.scatter(0, 0.5, c='black', marker="s",
            label=f"prediction:{np.int(rbf_kernel_svm_clf.predict([[0, 0.5]]))}")
plt.scatter(1.0, 0.5, c='black', marker="*",
            label=f"prediction:{np.int(rbf_kernel_svm_clf.predict([[1.0, 0.5]]))}")
plt.scatter(1.7, 0.5, c='black', marker="^",
            label=f"prediction:{np.int(rbf_kernel_svm_clf.predict([[1.7, 0.5]]))}")
plt.legend()
plt.title("RBF Kernel SVM Classifier")
plt.show()
