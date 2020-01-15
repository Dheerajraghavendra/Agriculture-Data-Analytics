#!/usr/bin/python
import numpy as np
with open('../'+'stats2.txt','r') as f:
    adv = np.genfromtxt(f,dtype = None,delimiter = '\n')
with open('../'+'uyears.txt','r') as f:
    y = np.genfromtxt(f,dtype = None)
with open('../'+'ucrops.txt','r') as f:
    cp = np.genfromtxt(f,dtype= None)
with open('../'+'ucroptypes.txt','r') as f:
    cpt = np.genfromtxt(f,dtype=None)
with open('../'+'prob.txt','r')as f:
    pb = np.genfromtxt(f,dtype = None,delimiter = ';')
with open('../'+'croptypecount.txt','r')as f:
    ctcnt = np.genfromtxt(f,dtype = None,delimiter = '\t')
     
def season(s):
    m = int(s[5:7])
    if m>=7 and m<=10:
        return "kharif"
    elif m>=3 and m<=6:
        return "summer"
    else:
        return "rabi"
d = {}
d["kharif"] = [[0 for i in range(len(pb))] for j in range(len(ctcnt))]
d["rabi"] = [[0 for i in range(len(pb))] for j in range(len(ctcnt))]
d["summer"] = [[0 for i in range(len(pb))] for j in range(len(ctcnt))]

for i in range(len(adv)):
    for k in range(len(ctcnt)):
        if ctcnt[k][0]==cpt[i]:     
            break
    s = season(y[i])
    for j in range(len(pb)):
        if pb[j][0] in adv[i]:
            d[s][k][j]+=1
l = d.keys()
for ssn in l:
    a = d[ssn]
    for i in range(len(a)):
        for j in range(len(a[i])):
            if sum(a[i])>0:
                a[i][j]=a[i][j]/float(sum(a[i]))
    np.savetxt(ssn+".txt",a,delimiter=' ',fmt = '%s')
