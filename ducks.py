import matplotlib.pyplot as plt
from skimage import io
import numpy as np

L = []
for i in range(7):
    light = io.imread("duck_photos/single/light%s.jpg" % i)
    light.shape=(light.shape[0]*light.shape[1]*light.shape[2])
    L.append(light)

s = L[0] + L[1] + L[2]

sNorm1 = np.linalg.norm(s,1)
sNorm2 = np.linalg.norm(s,2)
sNormInf = np.linalg.norm(s,np.inf)

print(sNorm1, sNorm2, sNormInf)

S = s/sNormInf

S.shape = (416,624,3)
s.shape = (416,624,3)

plt.imshow(S)
plt.show() # Shows part A

A = np.column_stack(L)


out = ""

for j in range(1):
    p = io.imread("duck_photos/message/msg%s.jpg" % str(j).zfill(3))
    p.shape=(p.shape[0]*p.shape[1]*p.shape[2])

    LeftSide = np.matmul(np.transpose(A),A)
    RightSide = np.matmul(np.transpose(A),p)
    b = np.linalg.lstsq(A,p)[0]
    print(b) # shows part B


    char = 0

    MAX = np.amax(b)

    for i in range(len(b)):
        if b[i] > 0.335*MAX:
            b[i] = 1
            char += 2**i
        else:
            b[i] = 0

    out += chr(char)



print(out) # Shows part C