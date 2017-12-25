import numpy as np
import matplotlib.pyplot as plt


x = np.arange(10)
y = np.array([5,10,20,50,1,5,0,0,30,5])

plt.figure(1)
ax1 = plt.subplot(1,1,1)
rect = ax1.bar(x,y)
for rec in rect:
    x=rec.get_x()
    height=rec.get_height()
    ax1.text(x+0.1,1.02*height,str(height))

# plt.plot(x, y, 'bar')
plt.show()