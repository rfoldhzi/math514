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
    print("[",a,",",b,"]")
    
    d = a-f(a)*(b-a)/(f(b)-f(a))
    print(i, abs(d - xExact))
    if f(d)<0:
        
        # New interval is [d,b]
        a=d
    else:

        # New interval is [a,d]
        b=d

# Print the approximation to the root, and evaluate the function there
x= a-f(a)*(b-a)/(f(b)-f(a))
print("\nRoot at x =",x,"\nf(x) =",f(x))