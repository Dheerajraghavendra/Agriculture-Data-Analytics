#!/usr/bin/python
import numpy as np
import re
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

month = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

#print len(pb),len(y)
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
la = len(adv)
lf = len(frmcnt)
lc = len(ctcnt)
d = {}
for i in range(lc):
    d[ctcnt[i][0]] = i
a = [{} for i in range(lc)]

pbwise      = [[[] for i1 in range(len(ctcnt))] for j1 in range(len(pb))]
pbwisecnt   = [[0 for i1 in range(len(ctcnt))] for j1 in range(len(pb))]
cpwise      = [[[] for i1 in range(len(pb))] for j1 in range(len(ctcnt))]
cpwisecnt   = [[0 for i1 in range(len(pb))] for j1 in range(len(ctcnt))]

#print len(adv)
mw = [-1 for i in range(len(ctcnt))]
for i in range(la):
#    s2 = cp[i][::-1]
#    s2 = s2[:10]
#    s2 = s2[::-1]
    s2 = y[i]
    wk = dos(cp[i][20:25],s2)
    for k in range(len(ctcnt)):
        if ctcnt[k][0]==cpt[i]:
            break  
    if wk>mw[k]:
        mw[k] = wk
'''
mat = [[0 for i in range(2)] for j in range(len(ctcnt))]
for i in range(len(ctcnt)):
    mat[i][0] = ctcnt[i][0]
    mat[i][1] = mw[i]
np.savetxt('maxweek2.txt',mat,delimiter='\t',fmt='%s')
'''
ans=0
tc=0
pm = [[0 for i in range(len(pb))] for j in range(len(ctcnt))]
pmn = [0 for i in range(len(ctcnt))]
for i in range(la):
    for k1 in range(len(ctcnt)):
        if ctcnt[k1][0]==cpt[i]:     
            break
    pmn[k1]+=1
    for j in range(len(pb)):
        if pb[j][0] in adv[i]:
            #print y[i],i
#            s2 = cp[i][::-1]
#            s2 = s2[:10]
#            s2 = s2[::-1]
#            tc+=1
            s2 = y[i]
            curyear = s2[:4]
            for k in range(len(ctcnt)):
                if ctcnt[k][0]==cpt[i]:
                    break
            #if dos(cp[i][20:25],s2)>=0:
            pm[k][j]+=1
            wk = dos(cp[i][20:25],s2)
            if not mw[k]>=18:
                tc+=1
                if wk<=14:
                    ans+=1
            pbwise[j][k].append(wk)
            pbwisecnt[j][k]+=1
            cpwise[k][j].append(wk)
            cpwisecnt[k][j]+=1
#            print s2
            if pb[j][0] not in a[d[cpt[i]]]:
                a[d[cpt[i]]][pb[j][0]] = str(dos(cp[i][20:25],s2))
            else:
                a[d[cpt[i]]][pb[j][0]]+= ','+str(dos(cp[i][20:25],s2))

print tc,ans
for i in range(len(ctcnt)):
    for j in range(len(pb)):
        pm[i][j]=float(pm[i][j])/float(pmn[i])
  #  print sum(pm[i])
np.savetxt('probability.txt',pm,delimiter=' ',fmt='%s')
'''
for i in range(lc):
    fr = a[i].keys()
    sr = a[i].values()
    c = [[0 for i1 in range(2)] for j in range(len(fr))]
    for i1 in range(len(fr)):
        c[i1][0] = fr[i1]
        c[i1][1] = sr[i1]
    np.savetxt('wrtoage/'+ctcnt[i][0]+'.txt',c,delimiter = '\t',fmt = '%s')
'''
for i in range(len(pb)):
    mx=0
    for j in range(len(pbwise[i])):
        if len(pbwise[i][j])>mx:
            mx = len(pbwise[i][j])
    ar = [[-1 for i1 in range(mx)] for j1 in range(len(ctcnt))]
    arn = [[-1 for i1 in range(mx)] for j1 in range(len(ctcnt))]
    for i1 in range(len(ctcnt)):
        for j1 in range(mx):
            if len(pbwise[i][i1])>j1:
                ar[i1][j1] = pbwise[i][i1][j1]
    np.savetxt('age3/pbwise/'+pb[i][0]+'.txt',ar,delimiter=' ',fmt='%s')

for i in range(len(ctcnt)):
    mx=0
    for j in range(len(cpwise[i])):
        if len(cpwise[i][j])>mx:
            mx = len(cpwise[i][j])
    ar = [[-1 for i1 in range(mx)] for j1 in range(len(pb))]
    arn = [[-1 for i1 in range(mx)] for j1 in range(len(ctcnt))] 
    for i1 in range(len(pb)):
        for j1 in range(mx):
            if len(cpwise[i][i1])>j1:
                ar[i1][j1] = cpwise[i][i1][j1]
    np.savetxt('age3/cpwise/'+ctcnt[i][0]+'.txt',ar,delimiter=' ',fmt='%s')

