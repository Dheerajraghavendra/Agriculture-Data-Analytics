#!usr/bin/python
import numpy as np
with open('finalformat.txt','r') as f:
	a=np.genfromtxt(f,dtype=None,delimiter='\t')
l=len(a)
m=len(a[0])
print l,m
po=0
out=[[0 for x in range(12)] for y in range(l)]
filename=str(19)
for i in range(l):
	if int(a[i][m-1])==20:
		for j in range(12):
			out[po][j]=a[i][j]
		po=po+1
print po
#with open('tdata'+'/'+filename+'.txt','w') as f:
np.savetxt('tdata'+'/'+filename+'.txt',out,delimiter='\t',fmt='%s') 
