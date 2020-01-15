#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
with open('values.txt','r') as f:
    a = np.genfromtxt(f,dtype=None,delimiter='\t')
h = range(20,120,20)
x = range(5)
a1=[]
a2=[]
a3=[]
a4=[]
a5=[]
a6=[]
avg1=[]
avg2=[]
for i in range(5):
    a1.append(float(a[i][1]))
    a2.append(float(a[i][2]))
    a3.append(float(a[i][3]))
    a4.append(float(a[i][4]))
    a5.append(float(a[i][5]))
    a6.append(float(a[i][6]))
    avg1.append((a[i][1]+a[i][2]+a[i][3])/float(3))
    avg2.append((a[i][4]+a[i][5]+a[i][6])/float(3))
plt.plot(x,a4,'b',label='first part')
plt.plot(x,a5,'g',label='second part')
plt.plot(x,a6,'y',label='third part')
plt.plot(x,avg2,'r',label='average 3fold value')
plt.title('Overall 3-fold accuracies')
plt.xlabel('Number of hidden units')
plt.ylabel('Accuracy')
plt.xticks(x,h)
plt.legend(loc='lower right')
plt.show()
