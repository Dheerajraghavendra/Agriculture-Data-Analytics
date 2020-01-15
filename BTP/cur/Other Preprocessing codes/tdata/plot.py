#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
with open('accuracies/6.txt','r') as f:
    a = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('accuracies/7.txt','r') as f:
    b = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('accuracies/8.txt','r') as f:
    c = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('accuracies/10.txt','r') as f:
    d = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('accuracies/12.txt','r') as f:
    e = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('accuracies/15.txt','r') as f:
    f = np.genfromtxt(f,dtype=None,delimiter='\t')

h = range(35,135,10)
a1 =[]
a2 = []
b1 =[]
b2 = []
c1 =[]
c2 = []
d1 =[]
d2 = []
e1 =[]
e2 = []
f1=[]
f2=[]
x = range(len(a))
for i in range(len(a)):
    a1.append(float(a[i][4]))
    a2.append(float(a[i][5]))
    b1.append(float(b[i][4]))
    b2.append(float(b[i][5]))
    c1.append(float(c[i][4]))
    c2.append(float(c[i][5]))
    d1.append(float(d[i][4]))
    d2.append(float(d[i][5]))
    e1.append(float(e[i][4]))
    e2.append(float(e[i][5]))
    f1.append(float(f[i][4]))
    f2.append(float(f[i][5]))

y = range(0,110,10)
#plt.plot(x,a1,'b',label ='6 hidden layers')
plt.plot(x,a2,'b',label ='6 hidden layers')
#plt.plot(x,b1,'g',label ='7 hidden layers')
plt.plot(x,b2,'g',label ='7 hidden layers')
#plt.plot(x,c1,'r',label ='8 hidden layers')
plt.plot(x,c2,'r',label ='8 hidden layers')
#plt.plot(x,d1,'y',label ='10 hidden layers')
plt.plot(x,d2,'y',label ='10 hidden layers')
#plt.plot(x,e1,'c',label ='12 hidden layers')
plt.plot(x,e2,'c',label ='12 hidden layers')
#plt.plot(x,f1,'k',label ='15 hidden layers')
plt.plot(x,f2,'k',label ='15 hidden layers')

plt.title('Testing accuracies')
plt.xlabel("Number of hidden units")
plt.ylabel("Accuracy")
plt.xticks(x,h)
#plt.yticks(y)
plt.legend(loc='lower right')
plt.show()
