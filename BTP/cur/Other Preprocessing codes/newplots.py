#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
with open('probcnt.txt','r') as f:
    t = np.genfromtxt(f,dtype=None,delimiter='\t')
a = []
b= [] 
for i in range(len(t)):
    a.append(t[i][0])
    b.append(int(t[i][1]))
x  =range(len(t))
plt.bar(x,b)
plt.xticks(x,a)
plt.show()
