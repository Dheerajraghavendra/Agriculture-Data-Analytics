#!/usr/bin/python
import numpy as np
with open('advices1.txt','r') as f:
    t = np.genfromtxt(f,dtype = None,delimiter = '\t')

with open('prob.txt','r') as f:
    p = np.genfromtxt(f,dtype = None, delimiter = ';')

k = [0 for i in range(len(t))]
flag=0
count=0
for i in range(len(t)):
    flag = 0
    k[i] = t[i][3]
    for j in range(len(p)): 
        for l in range(5):
            if t[i][3].lower().find(p[j][l])>=0:
                if l==0 or not flag:
                    k[i] = p[j][0]
                    count+=1
                    flag=1
np.savetxt('stats2.txt',k, fmt ='%s')
print count
