
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=10, n_features=6,  n_classes=3, n_informative=3,random_state=0)
print(x[0])
print(y)
print(x.shape)
clf = LinearSVC(random_state=0)
clf.fit(x, y)
print(clf.coef_)
print(clf.coef_.shape)
print(clf.predict([[1,1,1,1,1,1]]))