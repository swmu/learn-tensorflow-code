from sklearn.neural_network import  MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
data = iris.data
target = iris.target

x_train,x_test,y_train,y_test = train_test_split(data,target, train_size=0.1)
model = MLPClassifier(max_iter=199)
model.fit(x_train,y_train)
results = model.predict(x_test)
print(results)
print(y_test)
print(model.score(x_test,y_test))


# import matplotlib.pyplot as plt
#
# plt.gray()
# plt.show()

