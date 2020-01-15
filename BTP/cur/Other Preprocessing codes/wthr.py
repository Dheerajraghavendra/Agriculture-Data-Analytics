#!/usr/bin/python
import numpy as np

with open('stats2.txt','r') as f:
    adv = np.genfromtxt(f,dtype=None,delimiter='\n')

with open('weather.txt','r') as f:
    w = np.genfromtxt(f,dtype=None,delimiter='\t')
h=w[0]
w = w[13000:]

with open('ucrops.txt','r') as f:
    y = np.genfromtxt(f,dtype=None)

for i in range(len(y)):
    k=y[i][len(y[i])-10:]
    y[i] = k
#    print y[i]
with open('prob.txt','r') as f:
    pb = np.genfromtxt(f,dtype=None,delimiter=';');

h1 = h[0]+"    "+h[1]+"    "+h[2]+"    "+h[3]+"    "+h[4]+"    "+h[5]+"    "+h[6]+"    "+h[7]+"    "+h[8]

#print h
#print h1
d = [list() for i in range(len(pb))]
years ={}
for i in range(len(w)):
    years[w[i][0]] = w[i][1]+"    "+w[i][2]+"    "+w[i][3]+"    "+w[i][4]+"    "+w[i][5]+"    "+w[i][6]+"    "+w[i][7]+"    "+w[i][8]

for j in range(len(pb)):
    d[j].append(h1)

for i in range(len(adv)):
    for j in range(len(pb)):
        if pb[j][0] in adv[i]:
            if y[i] in years and y[i]+"    "+years[y[i]] not in d[j]:
                d[j].append(y[i]+"    "+years[y[i]])


for i in range(len(pb)):
    print pb[i][0]
    np.savetxt('wthr/'+pb[i][0]+'.txt',d[i],fmt="%s")
