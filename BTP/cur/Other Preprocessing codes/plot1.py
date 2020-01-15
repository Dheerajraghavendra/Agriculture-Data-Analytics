#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
with open('plots2/accuracies.txt','r') as f:
    a1 = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('plots2/alpha.txt','r') as f:
    b1 = np.genfromtxt(f,dtype=None,delimiter='\t')
'''
for i in range(len(a1)):
    a1[i] = a1[i].split('\t')
    b1[i] = b1[i].split('\t')
'''
h = range(5,105,5)
print a1[5][1]
c=[]
a=[]
b=[]
x = range(len(a1))
for i in range(len(a1)):
    c.append(h[i])
    a.append(100*float(a1[i][0]))
    b.append(100*float(a1[i][1]))
print len(a),len(b)

plt.plot(x,a,'b',label = 'training score')
plt.plot(x,b,'r',label = 'testing score')
plt.xlabel('Number of hidden units')
plt.ylabel('Accuracy')
plt.xticks(x,c)
plt.legend(loc= 'upper right')
plt.show()
