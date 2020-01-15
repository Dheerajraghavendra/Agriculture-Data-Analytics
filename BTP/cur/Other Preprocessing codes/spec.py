#!/usr/bin/python
import numpy as np

with open('stats2.txt','r') as f:
    ad = np.genfromtxt(f,dtype = None,delimiter = '\n')
with open('uyears.txt','r') as f:
    year = np.genfromtxt(f,dtype = None)
with open('prob.txt','r') as f:
    p = np.genfromtxt(f,dtype = None, delimiter =';')

dict =[{} for i in range(12)]
for i in range(12):
    for j in range(len(p)):
        dict[i][p[j][0]]=0
for i in range(len(ad)):
    for j in range(len(p)):
        if p[j][0] in ad[i]:
            if not dict[int(year[i][5:7])-1][p[j][0]]:
                dict[int(year[i][5:7])-1][p[j][0]]=1
            else:
                dict[int(year[i][5:7])-1][p[j][0]]+=1

jan =  [[0 for width in range(2)] for height in range(len(p))]
feb =  [[0 for width in range(2)] for height in range(len(p))]
mar =  [[0 for width in range(2)] for height in range(len(p))]
apr =  [[0 for width in range(2)] for height in range(len(p))]
may =  [[0 for width in range(2)] for height in range(len(p))]
jun =  [[0 for width in range(2)] for height in range(len(p))]
jul =  [[0 for width in range(2)] for height in range(len(p))]
aug =  [[0 for width in range(2)] for height in range(len(p))]
sept = [[0 for width in range(2)] for height in range(len(p))]
octo = [[0 for width in range(2)] for height in range(len(p))]
nov =  [[0 for width in range(2)] for height in range(len(p))]
dec =  [[0 for width in range(2)] for height in range(len(p))]
for i in range(len(p)):
    jan[i][0]=p[i][0]
    jan[i][1]=dict[0][p[i][0]]
    feb[i][0]=p[i][0]
    feb[i][1]=dict[1][p[i][0]]
    mar[i][0]=p[i][0]
    mar[i][1]=dict[2][p[i][0]]
    apr[i][0]=p[i][0]
    apr[i][1]=dict[3][p[i][0]]
    may[i][0]=p[i][0]
    may[i][1]=dict[4][p[i][0]]
    jun[i][0]=p[i][0]
    jun[i][1]=dict[5][p[i][0]]
    jul[i][0]=p[i][0]
    jul[i][1]=dict[6][p[i][0]]
    aug[i][0]=p[i][0]
    aug[i][1]=dict[7][p[i][0]]
    sept[i][0]=p[i][0]
    sept[i][1]=dict[8][p[i][0]]
    octo[i][0]=p[i][0]
    octo[i][1]=dict[9][p[i][0]]
    nov[i][0]=p[i][0]
    nov[i][1]=dict[10][p[i][0]]
    dec[i][0]=p[i][0]
    dec[i][1]=dict[11][p[i][0]]

np.savetxt('January.txt',jan,delimiter='\t',fmt='%s')
np.savetxt('February.txt',feb,delimiter='\t',fmt='%s')
np.savetxt('March.txt',mar,delimiter='\t',fmt='%s')
np.savetxt('April.txt',apr,delimiter='\t',fmt='%s')
np.savetxt('May.txt',may,delimiter='\t',fmt='%s')
np.savetxt('June.txt',jun,delimiter='\t',fmt='%s')
np.savetxt('July.txt',jul,delimiter='\t',fmt='%s')
np.savetxt('August.txt',aug,delimiter='\t',fmt='%s')
np.savetxt('September.txt',sept,delimiter='\t',fmt='%s')
np.savetxt('October.txt',octo,delimiter='\t',fmt='%s')
np.savetxt('November.txt',nov,delimiter='\t',fmt='%s')
np.savetxt('December.txt',dec,delimiter='\t',fmt='%s')



