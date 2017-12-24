from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

iris = datasets.load_iris()
digits = datasets.load_digits()

print (digits.target)
print(digits.images[0])

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])

print(clf.predict(digits.data[-1:]))