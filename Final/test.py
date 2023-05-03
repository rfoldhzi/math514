import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

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

diceCount = 6

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

print(cdf)

plt.plot(cdf)
plt.plot(d)
plt.show()