import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom
import random
np.seterr(all='raise')


# Batch Size, Input Neurons, Hidden Neurons, Output Neurons
N, D_in, H, D_out = 16, 5, 1024, 1

model2 = torch.nn.Sequential(
    torch.nn.Linear(D_in, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

model2.load_state_dict(torch.load("Final/model2.pt"))
model2.eval()

modeltanh = torch.nn.Sequential(
    torch.nn.Linear(D_in, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

modeltanh.load_state_dict(torch.load("Final/model2.pt"))
modeltanh.eval()



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


def generateRandomSample():
    hitSides = random.random()
    missesSides = random.random()*(1-hitSides)
    nearHitSides = 1-missesSides-hitSides
    diceCount = random.randint(2,50)
    converts = random.randint(0,diceCount)
    #print(hitSides, missesSides, nearHitSides, diceCount, converts)
    a,b,c = calcABC(hitSides,nearHitSides,missesSides,diceCount, converts)
    return (hitSides, missesSides, nearHitSides, diceCount, converts), (a,b,c)


sample = generateRandomSample()
numpyX = np.asarray(sample[0], dtype=np.float32)
x = torch.from_numpy(numpyX).to(torch.float32)
y = model(x)
print(sample, y)