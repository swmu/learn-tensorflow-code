from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split

digits = load_digits()
data = digits.data

print("data")
print(data)
print(data.shape)
print(digits.images[0])
target = digits.target
print("target")
print(target)
print(target.shape)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)

# model select
model = svm.SVC(kernel='linear')
model.fit(X_train,y_train,)
print("weight")
print(model.coef_)
print(model.coef_.shape)
results = model.predict(X_test)
# print(results)
# print(y_test)
# print(model.score(X_test,y_test))



import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
# plt.show()