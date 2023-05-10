
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom
import matplotlib.colors as colors
import time
np.seterr(all='raise')

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

def findOptimalScale(a,b, xValues, yValues):
    def f(x):
        if x>1: #Weird fix for RuntimeWarning: invalid value encountered in double_scalars
            x =1 
        return x**a*(1-x)**b
    oldX = yValues
    newY = [f(v) for v in xValues]
    m, c = np.linalg.lstsq(np.vstack([oldX, np.ones(len(oldX))]).T, newY, rcond=None)[0]
    return 1/m

def calculateError(a,b, xValues, yValues):
    c = 1
    #print("a,b",a,b)
    def f(x):
        #print("x",x)
        return c*x**a*(1-x)**b

    c = findOptimalScale(a,b, xValues, yValues)
    
    totalError = 0
    for i in range(len(yValues)):
        totalError += abs(yValues[i]-f(xValues[i]))
    
    return totalError

def calcABC(hitSides,nearHitSides,missesSides,diceCount, converts):
    totalSides = missesSides+nearHitSides+hitSides
    matrix = np.zeros((diceCount+1,diceCount+1))
    for i in range(diceCount+1):
        for j in range(diceCount+1):
            matrix[i,j] = multinomial2([i,j], diceCount) \
            * ((hitSides/totalSides)**i)  * ((nearHitSides/totalSides)**j) * ((missesSides/totalSides)**(diceCount-i-j))

    d = [0] * (diceCount+1)
    for i in range(diceCount+1):
        for j in range(diceCount+1):
            if matrix[i,j] > 0:
                hitCount = i + min(converts,j)
                d[hitCount] += matrix[i,j]

    cdf = [0] * (diceCount+2)
    for i in range(diceCount+1):
        cdf[i] = min(sum(d[i:]), 1)

    #cdf[0] = 1 #Floating point may add to not exactly one, but for consistency, we just set it to 1

    x,y = ode(range(len(cdf)),cdf)

    xx = np.linspace(0.1, 1, 400)
    yy = np.linspace(0.1, 1, 400)
    z = np.array([calculateError(i,j,x,y) for j in yy for i in xx])
    minValue = min(z)
    Z = z.reshape(400, 400)
    itemindex = np.where(Z == minValue)

    a,b = xx[itemindex[1]][0], yy[itemindex[0]][0]
    c = findOptimalScale(a,b,x,y)
    return a,b,c

def attempt1(): #Figure 1
    bestAs = []
    bestBs = []
    bestCs = []
    theHitSides = []

    startTime = time.time()
    for i in range(30):
        if i != 0:
            timePerIter = (time.time()-startTime)/i
            iterRemaining = 30-i
            timeRemaining = timePerIter*iterRemaining
            print("Est Time:", timeRemaining)
        print(i)
        hitSides = i*0.03
        nearHitSides = 0.1
        missesSides = 1-hitSides-nearHitSides
        diceCount = 10
        converts = 1
        a,b,c = calcABC(hitSides,nearHitSides,missesSides,diceCount, converts)
        bestAs.append(a)
        bestBs.append(b)
        bestCs.append(c)
        theHitSides.append(hitSides)

    plt.plot(theHitSides, bestAs, label='a value')
    plt.plot(theHitSides, bestBs, label='b value')
    plt.plot(theHitSides, bestCs, label='c value')
    plt.xlabel('Value of Variable')
    plt.ylabel('Percentage of Hit Sides') 
    plt.legend()
    plt.show()



def attempt2(): #Figure 2
    bestAs = []
    bestBs = []
    bestCs = []
    numbersOfDice = []

    startTime = time.time()
    for i in range(30):
    # if True:
    #     i = 26
        if i != 0:
            timePerIter = (time.time()-startTime)/i
            iterRemaining = 30-i
            timeRemaining = timePerIter*iterRemaining
            print("Est Time:", timeRemaining)
        print(i)
        hitSides = 0.7
        nearHitSides = 0.1
        missesSides = 1-hitSides-nearHitSides
        diceCount = i+2 #want 2 as minimum
        converts = 1
        a,b,c = calcABC(hitSides,nearHitSides,missesSides,diceCount, converts)
        bestAs.append(a)
        bestBs.append(b)
        bestCs.append(c)
        numbersOfDice.append(diceCount)

    plt.plot(numbersOfDice, bestAs, label='a value')
    plt.plot(numbersOfDice, bestBs, label='b value')
    plt.plot(numbersOfDice, bestCs, label='c value')
    plt.xlabel('Number of Dice')
    plt.ylabel('Percentage of Hit Sides') 
    plt.legend()
    plt.show()

def attempt3(): #Figure 3
    bestAs = []
    bestBs = []
    bestCs = []
    theHitSides = []

    startTime = time.time()
    for i in range(50):
        if i != 0:
            timePerIter = (time.time()-startTime)/i
            iterRemaining = 30-i
            timeRemaining = timePerIter*iterRemaining
            print("Est Time:", timeRemaining)
        print(i)
        hitSides = i*0.01
        nearHitSides = 0.5-hitSides
        missesSides = 0.5
        diceCount = 10
        converts = 1
        a,b,c = calcABC(hitSides,nearHitSides,missesSides,diceCount, converts)
        bestAs.append(a)
        bestBs.append(b)
        bestCs.append(c)
        theHitSides.append(hitSides)

    plt.plot(theHitSides, bestAs, label='a value')
    plt.plot(theHitSides, bestBs, label='b value')
    plt.plot(theHitSides, bestCs, label='c value')
    plt.xlabel('Number of Dice')
    plt.ylabel('Percentage of Hit Sides') 
    plt.legend()
    plt.show()

def attempt4(): #Figure 4
    bestAs = []
    bestBs = []
    bestCs = []
    convertCounts = []

    startTime = time.time()
    for i in range(15):
        if i != 0:
            timePerIter = (time.time()-startTime)/i
            iterRemaining = 15-i
            timeRemaining = timePerIter*iterRemaining
            print("Est Time:", timeRemaining)
        print(i)
        hitSides = 0.4
        nearHitSides = 0.2
        missesSides = 0.4
        diceCount = 15
        converts = i
        a,b,c = calcABC(hitSides,nearHitSides,missesSides,diceCount, converts)
        bestAs.append(a)
        bestBs.append(b)
        bestCs.append(c)
        convertCounts.append(converts)

    plt.plot(convertCounts, bestAs, label='a value')
    plt.plot(convertCounts, bestBs, label='b value')
    plt.plot(convertCounts, bestCs, label='c value')
    plt.xlabel('Number of Dice')
    plt.ylabel('Percentage of Hit Sides') 
    plt.legend()
    plt.show()

attempt2()