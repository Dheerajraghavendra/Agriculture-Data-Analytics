#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
with open('accuracies/12c.txt','r') as f:
    a = np.genfromtxt(f,dtype=None,delimiter=' ')
with open('accuracies/12c.txt','r') as f:
    b = np.genfromtxt(f,dtype=None,delimiter=' ')

'''
with open('accuracies/10.txt','r') as f:
    c = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('accuracies/12.txt','r') as f:
    d = np.genfromtxt(f,dtype=None,delimiter='\t')
with open('accuracies/15.txt','r') as f:
    e = np.genfromtxt(f,dtype=None,delimiter='\t')
'''
h = [5,7,8,10,12,15]
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
x = range(len(a))
for i in range(len(a)):
    a1.append(float(a[i][2]))
    a2.append(float(a[i][4]))
    #h[i]=b[i][1]
    #b2.append(float(b[i][4]))
    #c1.append(float(c[i][2]))
    #c2.append(float(c[i][3]))
    #d1.append(float(d[i][2]))
    #d2.append(float(d[i][3]))
    #e1.append(float(e[i][2]))
    #e2.append(float(e[i][3]))
#y = range(11)
plt.plot(x,a1,'b',label ='Training accuracy')
plt.plot(x,a2,'r',label='Testing accuracy')
#plt.plot(x,b1,'g',label ='tanh')
#plt.plot(x,b2,'g')#,label ='5 hidden layers')
'''
plt.plot(x,c1,'r',label ='10 hidden layers')
#plt.plot(x,c2,'r')#,label ='10 hidden layers')
plt.plot(x,d1,'y',label ='12 hidden layers')
#plt.plot(x,d2,'y')#,label ='12 hidden layers')
plt.plot(x,e1,'k',label ='15 hidden layers')
#plt.plot(x,e2,'k')#,label ='15 hidden layers')
'''
plt.title('Performance plot (for 12 classes)')
plt.xlabel("Number of hidden layers")
plt.ylabel("Accuracy")
plt.xticks(x,h)
#plt.yticks(y)
plt.legend(loc='upper right')
plt.show()
