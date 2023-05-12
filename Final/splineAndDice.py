import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom
import math

def PolyCoefficients2(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[o-i-1][0]*x**i
    return y

#Generates a spline for given xValues and yValues, then plots it
def plotSpline(xValues, yValues,format="g"):
    n = len(xValues)-1

    matrix = np.zeros((4*n, 4*n))
    b = np.zeros((4*n, 1))

    #Constraints needed to determine the coefficients for the spline

    for i in range(n):
        #End points of each cubic must match f(x)
        for j in range(4):
            matrix[2*i, 4*i+j] = xValues[i]**(3-j)
            matrix[2*i + 1, 4*i+j] = xValues[i+1]**(3-j)
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

    for i in range(n):
        x = np.linspace(xValues[i], xValues[i+1], 100)
        coeffs = X[4*i:4*(i+1)]
        if i==0:
            plt.plot(x, PolyCoefficients2(x, coeffs), format, label="Exact Spline") #Only wish to label once
        else:
            plt.plot(x, PolyCoefficients2(x, coeffs), format)

    return X

#Similar to the plot spline method before it, but takes the spline throgh
# tanh and tan before outputing, as to bound to 0 and 1
def plotSplineTanH(xValues, yValues,format="g"):
    n = len(xValues)-1

    matrix = np.zeros((4*n, 4*n))
    b = np.zeros((4*n, 1))

    yValues = [math.tanh(y) for y in yValues]

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

    for i in range(n):
        x = np.linspace(xValues[i], xValues[i+1], 100)
        coeffs = X[4*i:4*(i+1)]
        newY = PolyCoefficients2(x, coeffs)
        newY = [math.tan(y) for y in newY]
        if i==0:
            plt.plot(x, newY, format, label="Tangent Spline with 4 Knots") #Only wish to label once
        else:
            plt.plot(x, newY, format)

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


#Input given Parameters

missesSides = 3
nearHitSides = 1
hitSides = 4

totalSides = missesSides+nearHitSides+hitSides

diceCount = 6

#Generate the cdf
matrix = np.zeros((diceCount+1,diceCount+1))
for i in range(diceCount+1):
    for j in range(diceCount+1):
        matrix[i,j] = multinomial2([i,j], diceCount) \
        * ((hitSides/totalSides)**i)  * ((nearHitSides/totalSides)**j) * ((missesSides/totalSides)**(diceCount-i-j))

d = [0] * (diceCount+1)
for i in range(diceCount+1):
    for j in range(diceCount+1):
        if matrix[i,j] > 0:
            hitCount = i + min(1,j)
            d[hitCount] += matrix[i,j]


cdf = [0] * (diceCount+2)
for i in range(diceCount+1):
    cdf[i] = sum(d[i:])

plt.plot(cdf, label="CDF of Y")

spline = plotSpline(range(len(cdf)),cdf)

x = np.linspace(0, len(cdf)-1, 4)
y = [xAlongSpline(X, spline, range(len(cdf))) for X in x]

plotSplineTanH(x,y,"r")

plt.xlabel("Number of Hits")
plt.ylabel("Probability")
plt.legend()
plt.title("Figure 1.1")
plt.show()