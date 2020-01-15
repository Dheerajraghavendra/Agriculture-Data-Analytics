#!/usr/bin/python
import numpy as np
with open('advices1.txt','r') as f:
        t = np.genfromtxt(f,dtype = None,delimiter = '\t')
with open('croptypecount.txt','r') as f:
        cptn = np.genfromtxt(f,dtype=None,delimiter = '\t')

yr = [0 for i in range(len(t))]
cp = [0 for i in range(len(t))]
cpt = [0 for i in range(len(t))]
adv = [0 for i in range(len(t))]

def week(date):
    month = date[5:7]
    day = date[8:]
    wk = (int(month)-1)*4
    wk+= (int(day)/7)+1
    return wk

for i in range(len(t)):
    yr[i] = t[i][0]
    cp[i] = t[i][1]
    cpt[i] = t[i][2]
    adv[i] = t[i][3]

yrs = {}
yrsc = {}
cpd = {}
for i in range(len(cptn)):
    cpd[cptn[i][0]] = i

for i in range(len(t)):
    #r = cp[i]
    #r = r[::-1]
    #r = r[:10]
    #r = r[::-1]
    r = yr[i]
    curyear = r[:4]
    wk = week(r)
    if curyear not in yrs:
        yrs[curyear] = [0 for i1 in range(53)]
    yrs[curyear][wk]+=1
    if curyear not in yrsc:
        yrsc[curyear] = [0 for i1 in range(len(cptn))]
    yrsc[curyear][cpd[cpt[i]]]+=1

k = yrsc.keys()
for y in k:
    ar = [0 for i in range(53)]
    ar2 = [[0 for j in range(2)] for i in range(len(cptn))]
    for i in range(53):
        ar[i] = yrs[y][i]
    for i in range(len(cptn)):
        ar2[i][0] = cptn[i][0]
        ar2[i][1] = yrsc[y][i]
    #np.savetxt('2nd/all'+y+'.txt',ar,delimiter = ' ',fmt = '%s')
    np.savetxt('1st/cpcnt'+y+'.txt',ar2,delimiter = '\t',fmt = '%s')


