#!/usr/bin/python
import numpy as np
f = open('ucroptypes.txt','r')
data = np.genfromtxt(f, dtype = None)
cnt = {}
for word in data:
    if word in cnt:
        cnt[word]+=1
    else:
        cnt[word]=1
a = []
b = []
done = {}
for crop in data:
    if not crop in done:
        a.append(crop)
        b.append(cnt[crop])
        done[crop]=1

c = [[0 for width in range(2)] for height in range(len(a))]
for i in range(len(a)):
    c[i][0] = a[i]
    c[i][1] = b[i]
np.savetxt('croptypecount.txt',c,delimiter="\t",fmt='%s')


