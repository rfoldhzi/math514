import math
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return math.e**x

a = [1, 0.995067271766, 0.459837658911, 0.0968909459737]
b = [1, 0.995067271766, 0.459837658911, 0.263376897783]

def s(x):
    if x == 0:
        return 1
    if x<0:
        return a[0]*(x**0) + a[1]*(x**1) + a[2]*(x**2) + a[3]*(x**3)
    else:
        return b[0]*(x**0) + b[1]*(x**1) + b[2]*(x**2) + b[3]*(x**3)

def t(x):
    return (x**0)/math.factorial(0) + (x**1)/math.factorial(1) + (x**2)/math.factorial(2) + (x**3)/math.factorial(3)

X = np.linspace(-1, 1, num=1025)
y1 = np.array([f(x)-s(x) for x in X])
y2 = np.array([f(x)-t(x) for x in X])

norm1 = max(y1)
norm2 = max(y2)

print(norm1, norm2)
print(f(-1), s(-1),t(-1))

plt.plot(X,y1)
plt.plot(X,y2)
plt.show()
