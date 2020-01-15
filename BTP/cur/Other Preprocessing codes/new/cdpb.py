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
with open('../'+'probcnt.txt','r') as f:
    pbcnt = np.genfromtxt(f,dtype=None,delimiter='\n')
month = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
def dos(s1,s2):
    a1 = s1[3:]
    b1 = s1[:3]
    a2 = s2[2:4]
    b2 = s2[5:7]
    c2 = s2[8:]
    if(a1==a2):
        k = 4*(int(b2)-month[b1])
        k = k+(int(c2)/7)
    else:
        p = int(a2)-int(a1)
        k = int(b2)+(12*p)
        k-= month[b1]
        k*= 4
        k+= (int(c2)/7)
    return k
age={}
for i in range(len(ctcnt)):
    age[i] = [[1 for i1 in range(346)] for j1 in range(len(pb))]
for i in range(len(adv)):
    wk = dos(cp[i][20:25],y[i])
    for k in range(len(ctcnt)):
        if ctcnt[k][0]==cpt[i]:     
            break
    for j in range(len(pb)):
        if pb[j][0] in adv[i]:
            age[k][j][wk]+=1
for i in range(len(ctcnt)):
    a = age[i]
    for j in range(len(pb)):
        a[j][0] = (a[j][0]+a[j][1])/(float(2))
        a[j][345] = (a[j][344]+a[j][345])/float(2)
        for wk in range(1,345):
            a[j][wk] = (a[j][wk-1]+a[j][wk]+a[j][wk+1])/float(3)
        tmp = float(sum(a[j]))
        for wk in range(0,346):
            a[j][wk] = a[j][wk]/tmp
    np.savetxt('age/'+ctcnt[i][0]+'.txt',a,delimiter=' ',fmt='%s')

