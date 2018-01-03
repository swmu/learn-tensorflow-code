# 引入包
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection


# 整理数据集
boston = datasets.load_boston()

x = boston.data
y = boston.target

print("x:", x[0])
print(x.shape)

lr = linear_model.LinearRegression()
# lr.fit(x, y)
#
# # 预测
# predicte = lr.predict(boston.data)
# print(predicte[:50])
# print(boston.target[:50])

train_size, train_score, test_score = model_selection.learning_curve(lr, x, y)
print("train_size",train_size)
print("train_score",train_score)
print("test_score",test_score)

# 画图显示
# fig, ax = plt.subplots()
# ax.scatter(y, y, edgecolors=(0, 0, 0))
# ax.plot([predicte.min(), predicte.max()], [predicte.min(), predicte.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()