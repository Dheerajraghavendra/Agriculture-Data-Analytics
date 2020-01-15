#!/usr/bin/python
import numpy as np
import os
import glob
pointer=0
po=0
po1=0
out=[[0 for x in range(12)] for y in range(3181)]
out1=[[0 for x in range(2)] for y in range(19)]
with open('problems.txt','r') as f:
	pb=np.genfromtxt(f,dtype=None,delimiter='\n')
l=len(pb)
for i in range(0,l):
	path=pb[i]
	count=0
	for filename in os.listdir(path):
		count=count+1
	print count
	if count>=5 :
		pointer=pointer+1
		out1[po1][0]=pointer
		out1[po1][1]=path
		po1=po1+1
		for filename in os.listdir(path):
			with open(path+'/'+filename,'r') as f:
				w=np.genfromtxt(f,dtype=None,delimiter='\n')
			out[po][0]=w[0]
			out[po][1]=w[1]
			out[po][2]=w[2]
			out[po][3]=w[3]
			out[po][4]=w[4]
			out[po][5]=w[5]
			out[po][6]=w[6]
			out[po][7]=w[7]
			out[po][8]=w[8]
			out[po][9]=w[9]
			out[po][10]=w[10]
			out[po][11]=pointer
			po=po+1
print po1
print po	
np.savetxt('finalformat.txt',out,delimiter='\t',fmt='%s')
np.savetxt('Designations.txt',out1,delimiter='\t',fmt='%s')


