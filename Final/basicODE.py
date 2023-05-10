import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom
import matplotlib.colors as colors

def ode(xValues, yValues):
    print("=?=?=? yValues",yValues)
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



missesSides = 6
nearHitSides = 8
hitSides = 8

totalSides = missesSides+nearHitSides+hitSides

diceCount = 10

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

print("CDF",cdf)
cdf[0] = 1 #Floating point may add to not exactly one, but for consistency, we just set it to 1
print("Not CDF",cdf)

x,y = ode(range(len(cdf)),cdf)
plt.plot(x, y)

a = 0.889#0.8
b = 0.6877
c = 0.6

def f(x):
    #a = 0.8163#0.8
    #b = 0.3959
    #c = 1#0.6
    print("c",c)
    print("x", x, "x**a",x**a, "(1-x)**b",(1-x)**b, "subEnd:",x**a*(1-x)**b, c*x**a*(1-x)**b)
    return c*x**a*(1-x)**b



supposedC = 0
over = 0
for i in range(len(y)):
    if f(x[i]) != 0:
        supposedC += y[i]*y[i]/f(x[i])
        over += y[i]

supposedC /= over

def findOptimalScale(a,b, yValues):
    def f(x):
        if x>1: #Weird fix for RuntimeWarning: invalid value encountered in double_scalars
            x =1 
        return x**a*(1-x)**b
    oldX = yValues
    newY = [f(v) for v in x]
    m, c = np.linalg.lstsq(np.vstack([oldX, np.ones(len(oldX))]).T, newY, rcond=None)[0]
    return 1/m

#c = supposedC
c = findOptimalScale(a,b, y)
print("NEW C",c)

steps = 100
x2 = np.linspace(0, 1, steps)
y2 = [f(v) for v in x2]
for v in x2:
    print(v,f(v)) #f(v) is nan for some reason (TODO: Fix)
plt.plot(x2, y2)


print(sum(y2)/steps, supposedC)
print(sum(y)) #The sum of y's of this ODE is 1. Could relate this 
#somehow to the integral

plt.show()


#x**a*(1-x)**b
#x*(1-x)*(2*x*x+6*x+5)

def calculateError(a,b):
    c = 1
    def f(x):
        return c*x**a*(1-x)**b
    
    # supposedC = 0
    # over = 0
    # for i in range(len(y)):
    #     if f(x[i]) != 0:
    #         supposedC += y[i]*y[i]/f(x[i])
    #         over += y[i]

    #supposedC /= over
    #c = supposedC
    c = findOptimalScale(a,b, y)
    
    totalError = 0
    for i in range(len(y)):
        totalError += abs(y[i]-f(x[i]))
    
    return totalError

def scatterPlot(a,b):
    C = 1
    def f(x):
        return C*x**a*(1-x)**b

    #C = 0.6

    oldX = y


    newY = [f(v) for v in x]
    

    plt.plot(oldX,newY,"p")
    X = np.linspace(0, 0.35, 3)
    plt.plot(X,X)

    m, c = np.linalg.lstsq(np.vstack([oldX, np.ones(len(oldX))]).T, newY, rcond=None)[0]
    #c seems to be very close to 0
    newY = [m*v+c for v in oldX]
    plt.plot(oldX, newY, 'r', label='Fitted line')
    print("m,c",m,c)

    newPointsY1 = [f(v) for v in x] #You are here, make sure this works
    newPointsY2 = [(f(v)/m) for v in x]
    print("newPointsY1",newPointsY1,newPointsY2)
    plt.plot(oldX,newPointsY2,"rp")

    plt.show()



#scatterPlot(0.8163,0.4)

print("calculateError11",calculateError(1,1))
print("calculateError01",calculateError(0,1))
print("calculateError10",calculateError(1,0))
print("calculateError00",calculateError(0,0))

print("calculateError0.8,0.3",calculateError(0.8,0.3))

def attempt1():
    x = y = np.linspace(0, 1, 50)
    z = np.array([calculateError(i,j) for j in y for i in x])
    Z = z.reshape(50, 50)
    print(Z)
    plt.imshow(Z, interpolation='bilinear')
    plt.show()

def attempt2():
    xx = np.linspace(0.1, 1, 50)
    yy = np.linspace(0.1, 1, 50)
    z = np.array([calculateError(i,j) for j in yy for i in xx])
    print("min", min(z))
    minValue = min(z)
    Z = z.reshape(50, 50)
    itemindex = np.where(Z == minValue)
    print("itemIndex",itemindex)
    print(xx[itemindex[0]], yy[itemindex[1]])
    
    print(Z)
    plt.imshow(Z, interpolation='bilinear', norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),)
    plt.show()

    a,b = xx[itemindex[1]][0], yy[itemindex[0]][0]
    print("a,b",a,b)
    c = findOptimalScale(a,b,y)

    def f2(x):
        return c*x**a*(1-x)**b

    steps = 100
    x2 = np.linspace(0, 1, steps)
    y2 = [f2(v) for v in x2]
    plt.plot(x2, y2)
    plt.plot(x, y)
    plt.show()

def given(A,B,C,start):
    print("a,b,c",A,B,C)
    xValues = [0,1]
    yValues = [1,start]

    def f2(x):
        return C*x**A*(1-x)**B

    while yValues[-1] > 0:
        xValues.append(len(xValues))
        yValues.append(max(yValues[-1]-f2(yValues[-1]),0))
    
    plt.plot(xValues,yValues)
    print("cdf",cdf)
    plt.plot(range(len(cdf)), cdf)
    plt.show()


    pass #You know, try to reconstruct the cdf from the thing

def attempt3():
    xx = np.linspace(0.1, 1, 50)
    yy = np.linspace(0.1, 1, 50)
    z = np.array([calculateError(i,j) for j in yy for i in xx])
    print("min", min(z))
    minValue = min(z)
    Z = z.reshape(50, 50)
    itemindex = np.where(Z == minValue)
    print("itemIndex",itemindex)
    print(xx[itemindex[0]], yy[itemindex[1]])
    
    print(Z)
    plt.imshow(Z, interpolation='bilinear', norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),)
    plt.show()

    a,b = xx[itemindex[1]][0], yy[itemindex[0]][0]
    print("a,b",a,b)
    c = findOptimalScale(a,b,y)

    given(a,b,c,cdf[1])

attempt3()