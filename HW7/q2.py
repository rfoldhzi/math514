from math import *
import numpy as np
import matplotlib.pyplot as plt

def trueF(x):
    return sin(x*x/2)

def ode(x,y):
    return -y + sin(x*x/2) + x * cos(x*x/2)

def rungeKutta2nd(x,y,h):
    k1 = h*ode(x,y)
    k2 = h*ode(x + h, y + h*k1)
    return y + .5*(k1+k2)

def rungeKutta3nd(x,y,h):
    k1 = h*ode(x, y)
    k2 = h*ode(x + h, y + h*k1)
    k3 = h*ode(x + h*.5, y + h/2*(k1+k2)/2)
    return y + (k1+k2+4*k3)/6

L = 6*sqrt(pi)
N = 240
h = L/N

x,y = 0,0

numerical = []
exact = []
Xs = []

for i in range(N):

    Xs.append(x)
    numerical.append(y)
    exact.append(trueF(x))

    y = rungeKutta2nd(x,y,h)
    x += h

# plt.plot(Xs,numerical)
# plt.plot(Xs,exact)

# plt.show()



x = 0
y = 0
h = 10**-4
threshold = 0.1
steps = 0
numericalX = []
numericalY = []

while x != L:
    numericalX.append(x)
    numericalY.append(y)

    h = min(1.1*h, L - x)
    y2nd = rungeKutta2nd(x,y,h)
    y3rd= rungeKutta3nd(x,y,h)
    Test = abs(y2nd-y3rd)/h
    while Test > threshold:
        h *= 0.5
        y2nd = rungeKutta2nd(x,y,h)
        y3rd= rungeKutta3nd(x,y,h)
        Test = abs(y2nd-y3rd)/h
    y = y2nd
    x += h
    steps += 1
    print("steps",steps,"h",h, "x",x)

print("exact", y, sin(x*x/2) )

print(steps)

# plt.plot(numericalX,numericalY)
# plt.plot(Xs,exact)

# plt.show()


def fixedStepIntegrator(N = 2**6):
    x,y = 0,0
    fEvaluations = 0
    h = L/N

    for i in range(N):
        y = rungeKutta2nd(x,y,h)
        fEvaluations += 2 # Two for rungeKutta2nd
        x += h
    
    return abs(y-trueF(x)), fEvaluations

def adaptiveStepIntegrator(threshold = 4**-1):
    x,y = 0,0
    fEvaluations = 0
    h = 10**-4

    while x != L:
        h = min(1.1*h, L - x)
        y2nd = rungeKutta2nd(x,y,h)
        y3rd= rungeKutta3nd(x,y,h)

        fEvaluations += 5

        Test = abs(y2nd-y3rd)/h
        while Test > threshold:
            h *= 0.5
            y2nd = rungeKutta2nd(x,y,h)
            y3rd= rungeKutta3nd(x,y,h)

            fEvaluations += 5

            Test = abs(y2nd-y3rd)/h
        y = y2nd
        x += h
    
    return abs(y-trueF(x)), fEvaluations

NValues = [2**k for k in range(6,20)]
Tvalues = [4**(-k) for k in range(1,15)]
xs1 = []
ys1 = []
xs2 = []
ys2 = []
for N in NValues:
    X,Y = fixedStepIntegrator(N)
    xs1.append(X)
    ys1.append(Y)
for T in Tvalues:
    X,Y = adaptiveStepIntegrator(T)
    xs2.append(X)
    ys2.append(Y)


print(xs2, ys2)

plt.plot(xs1,ys1)
plt.plot(xs2,ys2)
plt.xscale('log')
plt.yscale('log')


plt.show()