#!/usr/bin/python
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
with open('weather.txt','r') as f:
    w = np.genfromtxt(f,dtype=None,delimiter='\t')

def rnd(s):
    s = float(s)
    tmp = s-int(s)
    if tmp<=0.2:
        s = float(int(s))
    elif (tmp>=0.3)and(tmp<=0.6):
        s = float(int(s))+0.5
    else:
        s = float(int(s))+1
    return s
h = w[0]
w = w[12000:]
wd = {}
mn = 200
mx = -200
for i in range(len(w)):
    wd[w[i][0]] = w[i][1:]
    if float(w[i][3])>mx:
        mx = float(w[i][3])
        print w[i][0]
    if float(w[i][3])<mn:
        mn = float(w[i][3])
mn = rnd(mn)
mx = rnd(mx)
print mn,mx
rng = mx-mn
rng=rng+1
rng = int(rng)
print rng
ar= [[0 for i in range(rng)] for j in range(len(pb))]
for i in range(len(adv)):
    for j in range(len(pb)):
        if pb[j][0] in adv[i]:
            if y[i] in wd:
                tmp = rnd(wd[y[i]][2])
                tmp = tmp-mn
                #tmp=tmp*2
                tmp = int(tmp)
                ar[j][tmp]+=1
#for i in range(len(pb)):

np.savetxt('3.txt',ar,delimiter='\t',fmt='%s')
p = [0 for i in range(len(pb))]
r = [0 for i in range(rng)]
for i in range(len(pb)):
    p[i] = pb[i][0]
for i in range(rng):
    r[i] = mn+(i*0.5)
'''
nx,ny = len(pb),rng
x = range(ny)
y = range(nx)
#plt.figure()
hf = plt.figure()
ha = hf.add_subplot(111,projection ='3d')
X,Y = np.meshgrid(x,y)
#   print array,len(array[1])
ax = hf.gca(projection='3d')
for i in range(nx):
    #    print array[i]
    ha.bar(x,ar[i],zs=i,zdir='x')
ha.set_xticks(y)
ha.set_xticklabels(p,rotation=90)
ha.set_yticks(x)
ha.set_yticklabels(r)
ha.set_xlabel('Problems')
ha.set_title(h[1])
plt.show()
#plt.show(block=False)'''
