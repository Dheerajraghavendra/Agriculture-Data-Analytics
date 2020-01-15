#!/usr/bin/python
import numpy as np
with open('advices1.txt','r') as f:
    t = np.genfromtxt(f,dtype = None,delimiter = '\t')

with open('prob.txt','r') as f:
    p = np.genfromtxt(f,dtype = None, delimiter = ';')
with open('uyears.txt','r') as f:
    year = np.genfromtxt(f,dtype = None)
k = [0 for i in range(len(t))]
yc = [0 for i in range(13)]
cnt = {}
done={}
flag=0
count=0
print len(t)
for i in range(len(t)):
    done[i]=0
    flag = 0
    k[i] = ';'+t[i][3]
    for j in range(len(p)): 
        for l in range(5):
            if p[j][l] in t[i][3].lower():
                if l==0 or not flag:
                    yc[int(year[i][5:7])]+=1
                    if not done[i]:
                        k[i] = p[j][0]
                        done[i]=1
                    else:
                        k[i] = k[i]+";"+p[j][0]
                    if p[j][0] in cnt:
                        cnt[p[j][0]]+=1
                    else:
                        cnt[p[j][0]]=1
                    count+=1
                    flag=1
                    break
#        if flag==1:
#            break

#Problem count
a=[]
b=[]
for j in range(len(p)):
    a.append(p[j][0])
    b.append(cnt[p[j][0]])
c = [[0 for width in range(2)] for height in range(len(p))]
for i in range(len(p)):
    c[i][0]=a[i]
    c[i][1]=b[i]
#for i in range(len(k)):
#    print i,k[i]
np.savetxt('stats2.txt',k,delimiter='\n',fmt ='%s')
print count
np.savetxt('probcnt.txt',c,delimiter = "\t",fmt='%s')
yr = [[0 for i in range(2)] for j in range(12)]
for i in range(12):
    yr[i][0] = i+1
    yr[i][1] = yc[i+1]
#np.savetxt('monthwise.txt',yr,delimiter='\t',fmt = '%s')
