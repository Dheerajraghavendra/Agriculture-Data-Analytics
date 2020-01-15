#!/usr/bin/python
import numpy as np
with open('prob.txt','r')as f:
        pb = np.genfromtxt(f,dtype = None,delimiter = ';')
w = [0 for i in range(len(pb))]
for j in range(len(pb)):
    with open('wthr/'+pb[j][0]+'.txt','r') as f:
        w[j] = np.genfromtxt(f,dtype=None,delimiter='\t')
#        print pb[j][0],w[j],w[j].size
avg = [[0 for i in range(8)] for j in range(len(pb))]
for i in range(len(w)):
    if (w[i].size)>1:
        k = w[i][1:]
        for j in range(k.size):
            lst=k[j].split()
            avg[i][0]+=float(lst[1])
            avg[i][1]+=float(lst[2])
            avg[i][2]+=float(lst[3])
            avg[i][3]+=float(lst[4])
            avg[i][4]+=float(lst[5])
            avg[i][5]+=float(lst[6])
            avg[i][6]+=float(lst[7])
            avg[i][7]+=float(lst[8])
        for g in range(8):
            avg[i][g]/=k.size
#            avg[i][g] = int(avg[i][g])
        
a = [[0 for i in range(9)] for j in range(len(pb))]
for i in range(len(pb)):
    a[i][0] = pb[i][0]
    a[i][1] = avg[i][0]
    a[i][2] = avg[i][1]
    a[i][3] = avg[i][2]
    a[i][4] = avg[i][3]
    a[i][5] = avg[i][4]
    a[i][6] = avg[i][5]
    a[i][7] = avg[i][6]
    a[i][8] = avg[i][7]

np.savetxt('wthrcorr.txt',a,delimiter='\t',fmt='%s')
