import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

#This file is used to generate Figures for the introduction

def multinomial2(params, n):
    if sum(params) > n:
        return 0
    params.append(n-sum(params))
    return multinomial(params)

def multinomial(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])

# Generates and plots the cdf and pdf given an input set of parameters
def generateCDFandPDF(hitSides, nearHitSides, missesSides, diceCount, converts):
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
        cdf[i] = sum(pdf[i:])

    pdf.append(0) # To complete the right side of the graph

    plt.plot(cdf, label="CDF of Y")
    plt.plot(pdf, label="PDF of Y")
    plt.xlabel("Number of Hits")
    plt.ylabel("Probability")
    plt.legend()
    plt.title("Figure 0")
    plt.show()



generateCDFandPDF(hitSides = 4, nearHitSides = 1, missesSides = 3, diceCount = 12, converts = 1)