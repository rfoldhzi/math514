
import matplotlib.pyplot as plt
import numpy as np


def plot(A,q,r=100):
    x0 = []
    y = []
    n = q.shape[0]
    for i in range(n):
        y.append([])
    
    for i in range(r):
        x0.append(i)
        for j in range(n):
            y[j].append(q.item(j))
        q += np.multiply(q, A*q - np.ones([n,1])*np.transpose(q)*A*q)
        print(q)

    for j in range(n):
        plt.plot(x0,y[j])
    plt.show()

# Part 1
plot(
 np.matrix([
     [0, 0, 1],
     [1, 0, 0],
     [0, 1, 0]
 ]),
 np.matrix([
     [0.8],
     [0.1],
     [0.1]
 ])
)

#Part 2
plot(
 np.matrix([
     [0, 0, 1],
     [1, 0, 0],
     [0, 1, 0]
 ]),
 np.matrix([
     [0.3],
     [0.7],
     [0.0]
 ])
)

# Part 3
plot(
 np.matrix([
     [0, 0, 1],
     [2, 0, 0],
     [0, 1, 0]
 ]),
 np.matrix([
     [0.8],
     [0.1],
     [0.1]
 ])
)

#Part 4
plot(
 np.matrix([
     [0, 0, 1, 0, 1],
     [1, 0, 0, 1, 0],
     [0, 1, 0, 0, 1],
     [1, 0, 1, 0, 0],
     [0, 1, 0, 1, 0],
 ]),
 np.matrix([
     [0.6],
     [0.1],
     [0.1],
     [0.1],
     [0.1]
 ])
)

#Part 5
plot(
 np.matrix([
     [0, 0, 1, 0, 1],
     [1, 0, 0, 1, 0],
     [0, 1, 0, 0, 1],
     [2, 0, 1, 0, 0],
     [0, 1, 0, 1, 0],
 ]),
 np.matrix([
     [0.6],
     [0.1],
     [0.1],
     [0.1],
     [0.1]
 ])
)