import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom

def ode(xValues, yValues):
    outX = []
    outY = []
    for i in range(len(xValues)-1):
        outX.append(yValues[i])
        h = xValues[i+1]-xValues[i]
        outY.append((yValues[i]-yValues[i+1])/h)
    return outX, outY

def multinomial2(params, n):
    if sum(params) > n:
        return 0
    params.append(n-sum(params))
    return multinomial(params)

def multinomial(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])



missesSides = 3
nearHitSides = 1
hitSides = 4

totalSides = missesSides+nearHitSides+hitSides

diceCount = 8

matrix = np.zeros((diceCount+1,diceCount+1))
for i in range(diceCount+1):
    for j in range(diceCount+1):
        matrix[i,j] = multinomial2([i,j], diceCount) \
        * ((hitSides/totalSides)**i)  * ((nearHitSides/totalSides)**j) * ((missesSides/totalSides)**(diceCount-i-j))
print(matrix)

d = [0] * (diceCount+1)
for i in range(diceCount+1):
    for j in range(diceCount+1):
        if matrix[i,j] > 0:
            hitCount = i + min(1,j)
            d[hitCount] += matrix[i,j]

print(d)

cdf = [0] * (diceCount+2)
for i in range(diceCount+1):
    cdf[i] = sum(d[i:])

x,y = ode(range(len(cdf)),cdf)
plt.plot(x, y)

c = 1

def f(x):
    a = 0.8
    b = 0.3
    #c = 1#0.6
    return c*x**a*(1-x)**b



supposedC = 0
over = 0
for i in range(len(y)):
    if f(x[i]) != 0:
        supposedC += y[i]*y[i]/f(x[i])
        over += y[i]

supposedC /= over

c = supposedC

steps = 100
x2 = np.linspace(0, 1, steps)
y2 = [f(v) for v in x2]
plt.plot(x2, y2)


print(sum(y2)/steps, supposedC)
print(sum(y)) #The sum of y's of this ODE is 1. Could relate this 
#somehow to the integral

plt.show()


#x**a*(1-x)**b
#x*(1-x)*(2*x*x+6*x+5)

