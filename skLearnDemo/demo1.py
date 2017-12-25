from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

# minist test
# iris = datasets.load_iris()
digits = datasets.load_digits()

print("target")
print (digits.target)
print("data[:1]")
print (digits.data[:1])
print("images[0]")
print(digits.images[0])

# clf = svm.SVC(gamma=0.001, C=100.)
# clf.fit(digits.data[:-1], digits.target[:-1])
#
# print(clf.predict(digits.data[-1:]))