#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
with open('features.txt','r') as f:
    a = np.genfromtxt(f,dtype=None,delimiter='\t')
a1=[]
a2=[]
a3=[]
for i in range(len(a)):
    a1.append(a[i][0])
    a2.append(a[i][1])
    a3.append(a[i][2])
x = range(len(a))
plt.bar(x,a2)
plt.title("Testing accuracies after removing features")
plt.xlabel("Excluded feature")
plt.ylabel("Accuracy")
plt.xticks(x,a1,rotation=-90)
plt.show()
