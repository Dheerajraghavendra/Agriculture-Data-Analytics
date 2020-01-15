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
print len(adv)
frmdict = {}
frmcnt = {}
ctcnt = {}
ctdict = {}
d = [list() for i in range(len(pb))]
for i in range(len(adv)):
    k = re.findall(r'((\w{2,3}_){4}\d{4}_\w{5}_\d+_\w+[a-z]+)',cp[i])
    k = k[0][0]
    if k in frmcnt:
        frmcnt[k]+=1
    else:
        ctcnt[k] = cpt[i]
        frmcnt[k] = 1
    if cpt[i] in ctdict:
        ctdict[cpt[i]]+=1
    else:
        ctdict[cpt[i]] = 1
    for j in range(len(pb)):
        if pb[j][0] in adv[i]:
            d[j].append(y[i]+"  "+cp[i]+"   "+cpt[i]+"  "+pb[j][0])

a = frmcnt.keys()
b = []
#frmcnt.values()
for i in range(len(a)):
    b.append(frmcnt[a[i]])

c = [[0 for width in range(3)] for height in range(len(a))]
a1 = []
b1=[]
for i in range(len(a)):
    if frmcnt[a[i]]>10:
        a1.append(a[i])
        b1.append(frmcnt[a[i]])
c1 = [[0 for width in range(3)] for height in range(len(a1))]

for i in range(len(a)):
    c[i][0] = a[i]
    c[i][1] = ctcnt[a[i]]
    c[i][2] = frmcnt[a[i]]
for i in range(len(a1)):
    c1[i][0] = a1[i]
    c1[i][1] = ctcnt[a1[i]]
    c1[i][2] = b1[i]
ctype = ctdict.keys()
ctype2 = []
#ctdict.values()
for i in range(len(ctype)):
    ctype2.append(ctdict[ctype[i]])
crop = [[0 for i in range(2)] for j in range(len(ctype))]
for i in range(len(ctype)):
    crop[i][0] = ctype[i]
    crop[i][1] = ctype2[i]
for i in range(len(pb)):
    np.savetxt('probSpecific/'+pb[i][0]+'.txt',d[i],fmt='%s')

np.savetxt('farmcount.txt',c,delimiter='\t',fmt = '%s')
np.savetxt('farmcount2.txt',c1,delimiter='\t',fmt = '%s')
np.savetxt('croptypecount.txt',crop,delimiter='\t',fmt = '%s')



