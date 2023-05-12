import numpy as np

xValues = [0,0.2,0.5,0.6,1]
#yValues = [1,0.9,0.7,0.5,0.3,0.2,0.1,0]

n = len(xValues)-1

matrix = np.zeros((4*n, 4*n))
b = np.zeros((4*n, 1))

def f(x):
    return -4*(x-1)*x

for i in range(n):
    #End points of each cubic must match f(x)
    for j in range(4):
        matrix[2*i, 4*i+j] = xValues[i]**(3-j)
        matrix[2*i + 1, 4*i+j] = xValues[i+1]**(3-j)
    #
    #b[i*2,0] = yValues[i]
    #b[i*2+1,0] = yValues[i+1]
    b[i*2,0] = f(xValues[i])
    b[i*2+1,0] = f(xValues[i+1])
    

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


from matplotlib import pyplot as plt

def PolyCoefficients2(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    print(f'# This is a polynomial of order {o}.')
    y = 0
    for i in range(o):
        y += coeffs[o-i-1][0]*x**i
    return y

def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    print(f'# This is a polynomial of order {o}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

x = np.linspace(0, 1, 100)
coeffs = [0,4,-4]
plt.plot(x, PolyCoefficients(x, coeffs))

for i in range(n):
    x = np.linspace(xValues[i], xValues[i+1], 100)
    coeffs = X[4*i:4*(i+1)]
    plt.plot(x, PolyCoefficients2(x, coeffs))

plt.xlim(-1, 2)
plt.ylim(-1, 2)






plt.show()