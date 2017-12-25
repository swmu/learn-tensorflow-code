import numpy as np

# 进行数组间的运算

# 数组相加
a = np.arange(6).reshape(2,3)
print("a")
print(a)

b = np.array([[2,4,6],[7,1,2]])
print("b",b)

c = np.add(a,b)
print("c",c)

# 数组相减
d = np.subtract(a,b)
print("d",d)

# 乘
e = np.multiply(a,b)
print("e",e)

# 除
f = np.divide(a,b)
print("f",f)


np.power(a,2)
np.mod(a,2)