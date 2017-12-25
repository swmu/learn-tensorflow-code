import numpy as np

#  use array
a = np.array([1,2,3])
print("a")
print(a)
print(a.shape)

# mutiple dimention

b = np.array([[1,  2],  [3,  4]])
print("b")
print(b)
print(b.shape)

#
c = np.array([1,2,3,4,5], ndmin=2).reshape(1,5)
print("c")
print(c)
print(c.shape)


#zhuanzhi

d = np.array([[3,4,5],[1,6,7]])
print("d")
print(d)
print(d.shape)
print(d.T)