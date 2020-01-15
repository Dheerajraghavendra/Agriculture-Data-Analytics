#!usr/bin/python
import numpy as np
with open('finalformat.txt','r') as f:
	a=np.genfromtxt(f,dtype=None,delimiter='\t')
l=len(a)
m=len(a[0])
po=0
out=[[0 for x in range(11)] for y in range(858)]
filename=0
for i in range(l):
	if int(a[i][m-1])==1:
		for j in range(11):
			out[po][j]=a[i][j]
		po=po+1
#print out
np.savetxt('tdata/'+filename+'.txt',out,delimiter='\t',fmt='%s') 
