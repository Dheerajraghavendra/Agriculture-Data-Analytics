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
with open('weather1.txt','r') as f:
    w = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('kurnool1.txt','r') as f:
    kn = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('medak1.txt','r') as f:
    mk = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('rajendranagar1.txt','r') as f:
    rj = np.genfromtxt(f,dtype=None,delimiter='\t')

w = w[9000:]
wd = {}
mx = [float(0) for _ in range(8)]
mn = [float(10000) for i in range(8)]
wd2={}
wd2["kur"]={}
wd2["med"]={}
wd2["ran"]={}
for i in range(len(w)):
    wd[w[i][0]] = [0 for i1 in range(8)]
    for j in range(8):
        print w[i][j+1]
        w[i][j+1] = float(w[i][j+1])
        wd[w[i][0]][j] = float(w[i][j+1])
        if float(w[i][j+1])>mx[j]:
            mx[j] = float(w[i][j+1])
        if float(w[i][j+1])<mn[j]:
            mn[j] = float(w[i][j+1])
'''
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
    for k in range(len(ctcnt)):
        if ctcnt[k][0]==cpt[i]:
            break
    for j in range(len(pb)):
        if pb[j][0] in adv[i]:
            if date2 in wd:
                p = [0 for _ in range(11)]
                p[0] = float(k)
                p[1] = float(y[i][5:7])
                p[2] = float(age)
                p[3] = wd[date2][0]
                p[4] = wd[date2][1]
                p[5] = wd[date2][2]
                p[6] = wd[date2][3]
                p[7] = wd[date2][4]
                p[8] = wd[date2][5]
                p[9] = wd[date2][6]
                p[10] = wd[date2][7]
                clas[j].append(p)
                p = [0 for _ in range(5)]
                p[0] = float(k)
                p[1] = float(y[i][5:7])
                p[2] = float(age)
                p[3] = wd[date2][0]
                p[4] = wd[date2][1]
                clas2[j].append(p)
                break
#                print p,clas[j][-1],cpt[i],y[i]
            elif date2 in wd2["kur"]:
                dist = cp[i][3:6]
                if dist in wd2:
                    p = [0 for _ in range(5)]
                    p[0] = float(k)
                    p[1] = float(y[i][5:7])
                    p[2] = float(age)
                    p[3] = wd2[dist][date2][0]
                    p[4] = wd2[dist][date2][1]
                    clas2[j].append(p)
                    break
            if i not in total:
                total.append(i)
                if date2 in wd:
                    onlywthr.append(i)
'''
a=[[0 for j in range(4)] for i in range(len(total))]
for i in range(len(a)):
    a[i][0] = y[total[i]]
    a[i][1] = cp[total[i]]
    a[i][2] = cpt[total[i]]
    a[i][3] = adv[total[i]]
np.savetxt('total.txt',a,delimiter='\t',fmt='%s')

a=[[0 for j in range(4)] for i in range(len(onlywthr))]
for i in range(len(a)):
    a[i][0] = y[onlywthr[i]]
    a[i][1] = cp[onlywthr[i]]
    a[i][2] = cpt[onlywthr[i]]
    a[i][3] = adv[onlywthr[i]]
np.savetxt('onlywthr.txt',a,delimiter='\t',fmt='%s')
'''

for j in range(len(pb)):
    a = clas[j]
    print len(a)
    path = 'train/'+pb[j][0]
    try: 
            os.makedirs(path)
    except OSError:
            if not os.path.isdir(path):
                        raise
    for l in range(len(a)):
#        b = [[0 for it in range(12)] for _ in range(1)]
#        for u in range(11):
#            b[0][u] = a[l][u]
#        b[0][11] = pb[j][0]
        np.savetxt(path+'/'+str(l+1)+'.txt',a[l],delimiter=' ',fmt='%f')

sm1=0
sm2=0
for j in range(len(pb)):
    sm1+=len(clas[j])
    sm2+=len(clas2[j])
    print len(clas[j]),len(clas2[j]),pb[j][0]
print sm1,sm2
'''
