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

def week(date):
    month = date[5:7]
    day = date[8:]
    wk = (int(month)-1)*4
    wk+= (int(day)/7)+1
    return wk

tmp = [0 for i in range(38)]
fl = [0 for i in range(38)]
for i in range(len(pb)):
    with open('wthr/'+pb[i][0]+'.txt') as f:
        tmp[i] = np.genfromtxt(f,dtype=None,delimiter='\t')
    if (tmp[i].size>1):
        tmp[i] = tmp[i][1:]
#        print tmp[i]
    else:
        fl[i]=1
wt = [[[0 for k in range(8)] for i in range(52)] for j in range(len(pb))]
ct = [[[0 for k in range(8)] for i in range(52)] for j in range(len(pb))]
yrwise = {}
cy = {}
for i in range(len(pb)):
    if not fl[i]:
        for j in range(len(tmp[i])):
            k = tmp[i][j].split()
            #print k[0]
            wk = week(k[0])
            year = k[0][:4]
            if year not in yrwise:
                yrwise[year] = [[[0 for k1 in range(8)] for i1 in range(60)] for j1 in range(len(pb))]
                cy[year] = [[[0 for k1 in range(8)] for i1 in range(60)] for j1 in range(len(pb))]
            #print wk
            for l in range(1,9):
                yrwise[year][i][wk][l-1]+= float(k[l])
                cy[year][i][wk][l-1]+=1
                wt[i][wk][l-1]+=float(k[l])
                ct[i][wk][l-1]+=1
for i in range(len(pb)):
    for j in range(52):
        for k in range(8):
            if ct[i][j][k]:
                wt[i][j][k]/=ct[i][j][k]
                #print wt[i][j][k]
'''
for i in range(len(pb)):
    k = wt[i]
    print k
#    np.savetxt('wthrcorr/'+pb[i][0]+'.txt',k,delimiter='\t',fmt='%s')
'''

#array = [[i0 for i in range(52)] for j in range(len(pb))]
#for i in range(len(pb)):
#    for j in range(52):
#        array[i][j] = wt[i][j][0]

t = yrwise.keys()
print t
for yr in t:
    for i in range(len(pb)):
        for j in range(52):
            for k in range(8):
                if cy[yr][i][j][k]:
                    yrwise[yr][i][j][k]/=cy[yr][i][j][k]
               #     print yrwise[yr][i][j][k]

'''

for yr in t:
    for i in range(len(pb)):
        k = yrwise[yr][i]
#        print k
        np.savetxt('wthrcorr2/'+yr+'/'+pb[i][0]+'.txt',k,delimiter='\t',fmt='%s')
'''

yrs2 = {}
yrsc2 = {}
for i in range(len(adv)):
    for j in range(len(pb)):
        if pb[j][0] in adv[i]:
            r = cp[i]
            r = r[::-1]
            r = r[:10]
            r = r[::-1]
 #           print r
            curyear = r[:4]
            wk = week(r)
            if curyear not in yrs2:
                yrs2[curyear] = [[0 for i1 in range(52)] for j1 in range(len(pb))]
            if curyear not in yrsc2:
                yrsc2[curyear] = [0 for i1 in range(len(pb))]
            yrsc2[curyear][j]+=1    
            yrs2[curyear][j][wk]+=1

t = yrs2.keys()
for yr in t:
    ar = [[0 for i in range(52)] for j in range(len(pb))]
    arn =[[0 for i in range(52)] for j in range(len(pb))]
    for i in range(len(pb)):
        for j in range(52):
            ar[i][j] = yrs2[yr][i][j]
            if yr == '2015' and i is 1:
                print yrsc2[yr][i]
            if ar[i][j] is not 0:
                arn[i][j]= float(yrs2[yr][i][j])/yrsc2[yr][i]
    #np.savetxt(yr+'.txt',ar,delimiter=' ',fmt='%s')
    np.savetxt('normalized/'+yr+'.txt',arn,delimiter=' ',fmt='%s')

p =[0 for i in range(len(pb))]
w =[0 for i in range(52)]

for i in range(len(pb)):
    p[i] = pb[i][0]
for j in range(52):
    w[j] =str(j+1);
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
c1=0
t = yrwise.keys()
for yr in t:
    array = [[0 for i in range(52)] for j in range(len(pb))]
    for i in range(len(pb)):
        for j in range(52):
            array[i][j] = cy[yr][i][j][0]
'''
    nx,ny = len(pb),52
    x = range(ny)
    y = range(nx)
  #  plt.figure()
    hf = plt.figure()
    #c1+=1
    ha = hf.add_subplot(111,projection ='3d')
    X,Y = np.meshgrid(x,y)
    #   print array,len(array[1])
    #ax = hf.gca(projection='3d')
    for i in range(nx):
#    print array[i]
        ha.bar(x,array[i],zs=i,zdir='x')
    ha.set_xticks(y)
    ha.set_xticklabels(p,rotation=90)
    ha.set_yticks(x)
    ha.set_yticklabels(w,rotation=90)
    #ha.set_xlabel('Problems')
    ha.set_title(yr)
    plt.show(block=False)
plt.show()'''
