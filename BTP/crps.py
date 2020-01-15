#!/usr/bin/python
import numpy as np
import re

with open('stats2.txt','r') as f:
    adv = np.genfromtxt(f,dtype=None,delimiter = '\n')
with open('uyears.txt','r') as f:
    y = np.genfromtxt(f,dtype = None)
with open('ucrops.txt','r') as f:
    cp = np.genfromtxt(f,dtype= None)
with open('ucroptypes.txt','r') as f:
    cpt = np.genfromtxt(f,dtype=None)
with open('prob.txt','r')as f:
    pb = np.genfromtxt(f,dtype = None,delimiter = ';')
with open('croptypecount.txt','r') as f:
    ctcnt = np.genfromtxt(f,dtype = None,delimiter = '\t')
d = [{} for i in range(18)]
for i in range(len(cp)):
    for k in range(len(ctcnt)):
        if ctcnt[k][0]==cpt[i]:
            break
    if k<18:
        k1 = re.findall(r'((\w{2,3}_){4}\d{4}_\w{5}_\d+_\w+[a-z]+)',cp[i])
        k1 = k1[0][0]
        if k1 in d[k]:
            d[k][k1]+=1
        else:
            d[k][k1] = 1

for k in range(18):
    p = d[k].keys()
    a = [[0 for i in range(2)] for j in range(len(p))]
    for i in range(len(p)):
        a[i][0] = p[i]
        a[i][1] = d[k][p[i]]
    np.savetxt('new/'+ctcnt[k][0]+'.txt',a,delimiter='\t',fmt='%s')
