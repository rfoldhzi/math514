import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom
import matplotlib.colors as colors

# Generates the multistep ode from the cdf
# Here, the xValues is the number of hits, while yValues is the probability
# of obtaining that number of hits or greater
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


#Generates the cdf for a given set of inputs
def generateCDF(hitSides, nearHitSides, missesSides, diceCount, converts):
    totalSides = missesSides+nearHitSides+hitSides

    matrix = np.zeros((diceCount+1,diceCount+1))
    for i in range(diceCount+1):
        for j in range(diceCount+1):
            matrix[i,j] = multinomial2([i,j], diceCount) \
            * ((hitSides/totalSides)**i)  * ((nearHitSides/totalSides)**j) * ((missesSides/totalSides)**(diceCount-i-j))

    pdf = [0] * (diceCount+1)
    for i in range(diceCount+1):
        for j in range(diceCount+1):
            if matrix[i,j] > 0:
                hitCount = i + min(converts,j) # Here is where we account for the number of conversions
                pdf[hitCount] += matrix[i,j]

    cdf = [0] * (diceCount+2)
    for i in range(diceCount+1):
        cdf[i] = min(1, sum(pdf[i:])) # Floating point may add to slightly more than 1, so we fix that here

    return cdf


#Used for Figure 2.1
def plotODE(show = False): 
    cdf = generateCDF(hitSides = 4, 
                    nearHitSides = 1, 
                    missesSides = 3, 
                    diceCount = 12, 
                    converts = 1)
    x,y = ode(range(len(cdf)),cdf)
    plt.plot(x, y, label="Steps of y_n")
    plt.xlabel("Value of Y")
    plt.ylabel("Value of Y prime")
    if show:
        plt.title("Figure 2.1")
        plt.legend()
        plt.show()

plotODE(True)

def findOptimalScale(a,b, yValues):
    def g(x):
        if x>1:
            x = 1 
        return x**a*(1-x)**b
    oldX = yValues
    newY = [g(v) for v in x]
    m, offset = np.linalg.lstsq(np.vstack([oldX, np.ones(len(oldX))]).T, newY, rcond=None)[0]
    # Here offset is typically very close to 0, so we can generally ignore it
    return 1/m

def FigureX0():

    a = 0.94360902
    b = 0.67744361
    
    def g(x):
        return c*x**a*(1-x)**b

    c = findOptimalScale(a,b, y)

    steps = 100
    x2 = np.linspace(0, 1, steps)
    y2 = [g(v) for v in x2]
    for v in x2:
        print(v,g(v))
    plt.plot(x2, y2)

    plt.show()



def calculateError(a,b):
    c = 1
    def g(x):
        return c*x**a*(1-x)**b
    
    c = findOptimalScale(a,b, y)
    
    totalError = 0
    for i in range(len(y)):
        totalError += abs(y[i]-g(x[i]))
    
    return totalError

def scatterPlot(a,b, xValues, yValues):
    C = 1
    def g(x):
        return C*x**a*(1-x)**b

    oldX = yValues
    newY = [g(v) for v in xValues]

    plt.plot(oldX,newY,"p",label="Original Points")
    X = np.linspace(0, 0.26, 3)
    plt.plot(X,X, label="Target Line x=x")

    m, c = np.linalg.lstsq(np.vstack([oldX, np.ones(len(oldX))]).T, newY, rcond=None)[0]
    #c seems to be very close to 0
    newY = [m*v+c for v in oldX]
    plt.plot(oldX, newY, 'r', label='Fitted line')
    print("m,c",m,c)

    newPointsY1 = [g(v) for v in xValues] 
    newPointsY2 = [(g(v)/m) for v in xValues]
    plt.plot(oldX,newPointsY2,"rp",label="Translated points")
    plt.legend()
    plt.xlabel("Values of Y'(x)")
    plt.ylabel("Values of g(x)")
    plt.title("Figure 2.2")
    plt.show()



cdf = generateCDF(hitSides = 4, 
                nearHitSides = 1, 
                missesSides = 3, 
                diceCount = 12, 
                converts = 1)
x,y = ode(range(len(cdf)),cdf)


def Figure2_2():
    scatterPlot(0.8163,0.4,x,y)

#Figures 2.3 and 2.4
def Figure2_3and4():
    xx = np.linspace(0.1, 1, 50)
    yy = np.linspace(0.1, 1, 50)
    z = np.array([calculateError(i,j) for j in yy for i in xx])

    minValue = min(z)
    Z = z.reshape(50, 50)
    itemindex = np.where(Z == minValue)

    plt.imshow(Z, interpolation='bilinear', norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),)
    plt.xlabel("a Values")
    plt.ylabel("b Values")
    plt.xticks([1,49],["0.1","1.0"])
    plt.yticks([1,49],["0.1","1.0"])
    plt.title("Figure 2.3")
    plt.show()

    a,b = xx[itemindex[1]][0], yy[itemindex[0]][0]
    print("a,b",a,b)
    c = findOptimalScale(a,b,y)

    def f2(x):
        return c*x**a*(1-x)**b

    steps = 100
    x2 = np.linspace(0, 1, steps)
    y2 = [f2(v) for v in x2]
    plt.plot(x, y,label="Steps of y_n")
    plt.plot(x2, y2,label="g(Y)")
    plt.xlabel("Value of Y")
    plt.ylabel("Value of Y prime")
    plt.title("Figure 2.4")
    plt.legend()
    plt.show()

def reconstructCDF(A,B,C,start):
    xValues = [0,1]
    yValues = [1,start]

    def f2(x):
        return C*x**A*(1-x)**B

    while yValues[-1] > 0:
        xValues.append(len(xValues))
        yValues.append(max(yValues[-1]-f2(yValues[-1]),0))

    #Here we offsetX by the difference in expected values
    expectedValue = 0
    for i in range(len(cdf)-1):
        expectedValue += i*(cdf[i]-cdf[i+1])
    
    expectedValueOfG = 0
    for i in range(len(yValues)-1):
        expectedValueOfG += i*(yValues[i]-yValues[i+1])

    xOffset = expectedValue - expectedValueOfG
    xValues = [x+xOffset for x in xValues]
    
    plt.plot(range(len(cdf)), cdf, label="CDF of Y")
    plt.plot(xValues,yValues,'o', label="k_n")
    plt.xlabel("Value of Y")
    plt.ylabel("Value of Y prime")
    plt.title("Figure 2.5")
    plt.legend()
    plt.show()


    pass #You know, try to reconstruct the cdf from the thing

def attempt3():
    xx = np.linspace(0.1, 1, 50)
    yy = np.linspace(0.1, 1, 50)
    z = np.array([calculateError(i,j) for j in yy for i in xx])
    minValue = min(z)
    Z = z.reshape(50, 50)
    itemindex = np.where(Z == minValue)

    a,b = xx[itemindex[1]][0], yy[itemindex[0]][0]
    print("a,b",a,b)
    c = findOptimalScale(a,b,y)

    reconstructCDF(a,b,c,cdf[1])

Figure2_3and4()