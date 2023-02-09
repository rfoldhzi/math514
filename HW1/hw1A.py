#!/usr/bin/python3
from math import exp, sin, cos, pi
import sys

c = 1.78

# Function to consider
def f(x):
    return 0.5*(x**2 + c)


# Starting guess
x = 1.78

# Perform ten steps of the Newton iteration
n=20
for i in range(n):

    # Take Newton step
    x = f(x)
    print(i,x)

# Print solution on the final iteration
print(n,x)