#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
with open('wthrcorr.txt','r') as f:
    d = np.genfromtxt(f,dtype=None,delimiter='\t')
maxt = []#[0 for j in range(len(d))]
xlbl = []#[0 for j in range(len(d))]
for i in range(len(d)):
    if(float(d[i][8])>0):
        maxt.append(float(d[i][8]))
        xlbl.append(d[i][0])
xa = range(0,len(maxt));
plt.bar(xa,maxt)
plt.xlabel('Status')
plt.ylabel('Evaporation (mm)')
plt.xticks(xa,xlbl,rotation=90)
plt.rc('axes',labelsize=20)
plt.show()
