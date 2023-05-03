import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom

def PolyCoefficients2(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    print(f'# This is a polynomial of order {o}.')
    y = 0
    for i in range(o):
        y += coeffs[o-i-1][0]*x**i
    return y

def plotSpline(xValues, yValues):
    n = len(xValues)-1

    matrix = np.zeros((4*n, 4*n))
    b = np.zeros((4*n, 1))


    for i in range(n):
        #End points of each cubic must match f(x)
        for j in range(4):
            matrix[2*i, 4*i+j] = xValues[i]**(3-j)
            matrix[2*i + 1, 4*i+j] = xValues[i+1]**(3-j)
        #
        b[i*2,0] = yValues[i]
        b[i*2+1,0] = yValues[i+1]

    for i in range(n-1):
        for j in range(3):
            matrix[i + 2*n, 4*i   +   j] = (3-j)*xValues[i+1]**(2-j)
            matrix[i + 2*n, 4*(i+1) + j] = -(3-j)*xValues[i+1]**(2-j)
        
        matrix[i + 3*n-1, 4*i    ] = 6*xValues[i+1]
        matrix[i + 3*n-1, 4*i + 1] = 2
        matrix[i + 3*n-1, 4*(i+1)] = -6*xValues[i+1]
        matrix[i + 3*n-1, 4*(i+1) + 1] = -2

    matrix[4*n-2, 0] = 6*xValues[0]
    matrix[4*n-2, 1] = 2

    matrix[4*n-1, 4*(n-1)  ] = 6*xValues[n]
    matrix[4*n-1, 4*(n-1)+1] = 2


    X = np.linalg.solve(matrix,b)

    from matplotlib import pyplot as plt

    for i in range(n):
        x = np.linspace(xValues[i], xValues[i+1], 100)
        coeffs = X[4*i:4*(i+1)]
        plt.plot(x, PolyCoefficients2(x, coeffs), "g")

    return X

def xAlongSpline(x, spline, xValues):
    i = 0
    while i < len(xValues)-1 and x > xValues[i]:
        i+=1
    coeffs = spline[4*i:4*(i+1)]
    return PolyCoefficients2(x, coeffs)

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

spline = plotSpline(range(len(cdf)),cdf)
plotSpline(range(len(d)),d)

x = np.linspace(0, len(cdf)-1, 5)
y = [xAlongSpline(X, spline, range(len(cdf))) for X in x]
print(x,y)
plt.plot(x, y, "p")
plotSpline(x,y)

plt.show()