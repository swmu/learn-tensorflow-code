import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.,5.,0.2)
print(t)
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.ylabel('some numbers')
# plt.axis([0, 6, 0, 20])

line, = plt.plot(t, t, linewidth=5.0)
line.set_antialiased(False)
plt.setp(line, color='r')


plt.show()