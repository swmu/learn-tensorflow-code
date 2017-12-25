# 导入包
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import train_test_split



# 获取数据集
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)

# 选择模型
clf = svm.SVC()
# 训练
clf.fit(X_train, y_train)

# 预测
pre = clf.predict(X_test)
# 预测结果
print(pre)
# 真实结果
print(y_test)