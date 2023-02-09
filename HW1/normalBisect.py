#!/usr/bin/python3
from math import exp,cos,sin

# Function to perform root-finding on
def f1(x):
    return (sin(x))**3 - 0.3

def f(x):
    return x**30 - 0.01

# Initial bracket, assuming f(a)<0 and f(b)>0
a=0
b=1
xExact = 0.7334452040040546
xExact2 = 0.8576960563659668

# Perform the bisection search
for i in range(20):
    #print("[",a,",",b,"]")
    c=0.5*(a+b)
    
    print(i, abs(c - xExact2))
    if f(c)<0:
        
        # New interval is [c,b]
        a=c
    else:

        # New interval is [a,c]
        b=c

# Print the approximation to the root, and evaluate the function there
x=0.5*(a+b)
#print("\nRoot at x =",x,"\nf(x) =",f(x))