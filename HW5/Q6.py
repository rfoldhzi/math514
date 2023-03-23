# Multiplies two polynomials together
# Inputs are lists of coefficients of each polynomial
def polyMult(A,B):
    result = [0] * (len(A) + len(B) - 1)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if i+j<len(result): result[i+j] += a*b
    return result

def polyScale(A,c):
    result = []
    for a in A:
        result.append(a*c)
    return result

def polyAdd(A,B):
    result = [0] * (max(len(A),len(B)))
    for i in range(len(result)):
        if i<len(A): result[i] += A[i]
        if i<len(B): result[i] += B[i]
    return result

def polyEval(A, x):
    result = 0
    for i, a in enumerate(A):
        result += a * x**i
    return result

# Evaluates the integral of a polynomial from r to s
# Input A is a list of coefficients of the polynomial
def polyIntegrate(A, r, s):
    newPoly = [0] * (len(A)+1)
    for i, a in enumerate(A):
        newPoly[i+1] = a/(i+1)
    return polyEval(newPoly, s) - polyEval(newPoly, r)

def weirdDot(A,B):
    newPoly = polyMult(A,B)
    return \
             polyIntegrate(newPoly, -1.0, -0.5) + \
        10 * polyIntegrate(newPoly, -0.5,  0.0) + \
             polyIntegrate(newPoly,  0.0,  0.5) + \
        10 * polyIntegrate(newPoly,  0.5,  1.0)

# Generate orthogonal polynomials up to degree n with given weight function
def genOrthoPolys(n):
    polys = [[1]]
    for i in range(1,n):
        subPolys = []
        Xn = [0]*i + [1]
        for phi in polys:
            constant = weirdDot(phi, Xn)/weirdDot(phi,phi)
            subPolys.append(polyScale(phi, -1*constant))
        newPoly = Xn
        for poly in subPolys:
            newPoly = polyAdd(newPoly, poly)
        polys.append(newPoly)
    return polys

        
        