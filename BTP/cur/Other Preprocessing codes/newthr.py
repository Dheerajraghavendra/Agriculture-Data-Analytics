#!/usr/bin/python
import numpy as np
import re
with open('medak.txt','r') as f:
    k = np.genfromtxt(f,dtype=None,delimiter=' ')
print len(k),len(k[0])
a = [[0 for j in range(3)] for i in range(len(k))]
for i in range(len(k)):
    tmp = k[i][1].split('/')
    a[i][0]=tmp[1]
    if int(tmp[1])<10:
        a[i][0] = '0'+a[i][0]
    a[i][0] = tmp[0]+'-'+a[i][0]
    if int(tmp[0])<10:
        a[i][0] = '0'+a[i][0]
    a[i][0] = '2016-'+a[i][0]
    a[i][1] = k[i][4].split('/')
    a[i][1] = a[i][1][0][:2]
    a[i][2] = k[i][5].split('/')
    a[i][2] = a[i][2][0][:2]
np.savetxt('medak1.txt',a,delimiter='\t',fmt='%s')


