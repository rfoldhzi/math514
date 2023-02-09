#!/usr/bin/python3
from math import exp, sin, cos, pi
import sys

a = 0.35
b = 0.25
L = 0.24

def x(t):
    return sin(t) - a*sin(4*t) + b*sin(6*t)

def y(t):
    return cos(t) - a*cos(4*t) + b*cos(6*t)   

def xPrime(t):
    return cos(t) - 4*a*cos(4*t) + 6*b*cos(6*t)

def yPrime(t):
    return -1*sin(t) + 4*a*sin(4*t) + -6*b*sin(6*t)   

# Function to consider
def f(t, prevT):
    return (x(t) - x(prevT))**2 + (y(t) - y(prevT))**2 - L**2

# Derivative of the function
def df(t, prevT):
    return 2*xPrime(t)*(x(t) - x(prevT)) + 2*yPrime(t)*(y(t) - y(prevT))

# Starting guess
tPrev = 0
t = 2*pi*L/13.62

# Perform ten steps of the Newton iteration
n=10
for i in range(n):

    # Take Newton step
    fv=f(t,tPrev)
    #print(i,t,fv)
    change = fv/df(t,tPrev)
    print(i,t,tPrev,fv,df(t,tPrev),change)
    tPrev = t
    t -= change

# Print solution on the final iteration
print(n,t,f(t, tPrev))