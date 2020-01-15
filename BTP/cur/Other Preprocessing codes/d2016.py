#!/usr/bin/python
import numpy as np
import re
import os
with open('stats2.txt','r') as f:
    adv = np.genfromtxt(f,dtype=None,delimiter='\n')
with open('uyears.txt','r') as f:
    y = np.genfromtxt(f,dtype = None)
with open('ucrops.txt','r') as f:
    cp = np.genfromtxt(f,dtype= None)
with open('ucroptypes.txt','r') as f:
    cpt = np.genfromtxt(f,dtype=None)
with open('prob.txt','r')as f:
    pb = np.genfromtxt(f,dtype = None,delimiter = ';')
with open('farmcount2.txt','r') as f:
    frmcnt = np.genfromtxt(f,dtype = None,delimiter = '\t')
with open('croptypecount.txt','r') as f:
    ctcnt = np.genfromtxt(f,dtype = None,delimiter = '\t')
with open('kurnool1.txt','r') as f:
    kn = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('medak1.txt','r') as f:
    mk = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('rajendranagar1.txt','r') as f:
    rj = np.genfromtxt(f,dtype=None,delimiter='\t')
mx = [float(0) for _ in range(8)]
mn = [float(10000) for i in range(8)]
wd2={}
wd2["kur"]={}
wd2["med"]={}
wd2["ran"]={}

for i in range(len(kn)):
    wd2["kur"][kn[i][0]] = [0 for i1 in range(2)]
    wd2["kur"][kn[i][0]][0] = float(kn[i][1])
    wd2["kur"][kn[i][0]][1] = float(kn[i][2])
for i in range(len(mk)):
    wd2["med"][mk[i][0]] = [0 for i1 in range(2)]
    wd2["med"][mk[i][0]][0] = float(mk[i][1])
    wd2["med"][mk[i][0]][1] = float(mk[i][2])
for i in range(len(rj)):
    wd2["ran"][rj[i][0]] = [0 for i1 in range(2)]
    wd2["ran"][rj[i][0]][0] = float(rj[i][1])
    wd2["ran"][rj[i][0]][1] = float(rj[i][2])
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
        p1 = int(a2)-int(a1)
        k = int(b2)+(12*p1)
        k-= month[b1]
        k*= 4
        k+= (int(c2)/7)
    return k

total=[]
onlywthr=[]
clas = [[] for i in range(len(pb))]
clas2 = [[] for i in range(len(pb))]
for i in range(len(adv)):
    date = cp[i][::-1]
    date = date[:10]
    date = date[::-1]
    date2 = y[i]
    age = dos(cp[i][20:25],y[i])
    dist = cp[i][3:6]
    for k in range(len(ctcnt)):
        if ctcnt[k][0]==cpt[i]:
            break
    for j in range(len(pb)):
        if pb[j][0] in adv[i]:
            if (dist in wd2) and (date2 in wd2[dist]):
                p = [0 for _ in range(5)]
                p[0] = float(k)
                p[1] = float(y[i][5:7])
                p[2] = float(age)
                p[3] = wd2[dist][date2][0]
                p[4] = wd2[dist][date2][1]
                clas2[j].append(p)
            if i not in total:
                total.append(i)
ans = []
idx = []
for j in range(len(pb)):
    if(len(clas2[j]))>=5:
	ans.append(clas2[j])
	idx.append(j)
tmp = [[0 for i in range(3)] for _ in range(len(idx))]
for i in range(len(idx)):
    tmp[i][0] = idx[i]
    tmp[i][2] = pb[idx[i]][0]
    tmp[i][1] = i
np.savetxt('tdata2/'+'problemid.txt',tmp,delimiter = '\t',fmt = '%s')
for j in range(len(ans)):
    a = ans[j]
    np.savetxt('tdata2/'+str(j)+'.txt',a,delimiter='\t',fmt='%s')

sm1=0
sm2=0
for j in range(len(pb)):
    sm1+=len(clas[j])
    sm2+=len(clas2[j])
    print len(clas[j]),len(clas2[j]),pb[j][0]
print sm1,sm2

