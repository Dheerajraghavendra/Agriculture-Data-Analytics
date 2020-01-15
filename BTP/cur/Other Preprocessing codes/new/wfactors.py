#!/usr/bin/python
import numpy as np
import re
with open('../'+'stats2.txt','r') as f:
    adv = np.genfromtxt(f,dtype=None,delimiter='\n')
with open('../'+'uyears.txt','r') as f:
    y = np.genfromtxt(f,dtype = None)
with open('../'+'ucrops.txt','r') as f:
    cp = np.genfromtxt(f,dtype= None)
with open('../'+'ucroptypes.txt','r') as f:
    cpt = np.genfromtxt(f,dtype=None)
with open('../'+'prob.txt','r')as f:
    pb = np.genfromtxt(f,dtype = None,delimiter = ';')
with open('../'+'farmcount2.txt','r') as f:
    frmcnt = np.genfromtxt(f,dtype = None,delimiter = '\t')
with open('../'+'croptypecount.txt','r') as f:
    ctcnt = np.genfromtxt(f,dtype = None,delimiter = '\t')
with open('../'+'weather.txt','r') as f:
        w = np.genfromtxt(f,dtype=None,delimiter='\t')


w = w[13000:]
wd = {}
mx = [float(0) for _ in range(8)]
mn = [float(10000) for i in range(8)]

for i in range(len(w)):
    wd[w[i][0]] = [0 for i1 in range(8)]
    for j in range(8):
        w[i][j+1] = float(w[i][j+1])
        wd[w[i][0]][j] = float(w[i][j+1])
        if float(w[i][j+1])>mx[j]:
            mx[j] = float(w[i][j+1])
        if float(w[i][j+1])<mn[j]:
            mn[j] = float(w[i][j+1])

def season(s):
    m = int(s[5:7])
    if m>=7 and m<=10:
        return "kharif"
    elif m>=3 and m<=6:
        return "summer"
    else:
        return "rabi"

def rnd(s):
    s = float(s)
    tmp = s-int(s)
    if tmp<=0.2:
        s = float(int(s))
    elif (tmp>=0.3)and(tmp<=0.6):
        s = float(int(s))+0.5
    else:
        s = float(int(s))+1
    return s

maxt={}
mint={}
rh1={}
rh2={}
ws={}
rf={}
ss={}
evp={}
sd={}
sd={"kharif","summer","rabi"}
w1={}
w2={}
w3={}
w4={}
w5={}
w6={}
w7={}
w8={}
for ssn in sd:
    maxt[ssn]={}
    mint[ssn]={}
    rh1[ssn]={}
    rh2[ssn]={}
    ws[ssn]={}
    rf[ssn]={}
    ss[ssn]={}
    evp[ssn]={}
    w1[ssn]=[[1 for v1 in range(101)] for v2 in range(len(ctcnt))]
    w2[ssn]=[[1 for v1 in range(71)] for v2 in range(len(ctcnt))]
    w3[ssn]=[[1 for v1 in range(101)] for v2 in range(len(ctcnt))]
    w4[ssn]=[[1 for v1 in range(101)] for v2 in range(len(ctcnt))]
    w5[ssn]=[[1 for v1 in range(51)] for v2 in range(len(ctcnt))]
    w6[ssn]=[[1 for v1 in range(301)] for v2 in range(len(ctcnt))]
    w7[ssn]=[[1 for v1 in range(31)] for v2 in range(len(ctcnt))]
    w8[ssn]=[[1 for v1 in range(41)] for v2 in range(len(ctcnt))]
    for i in range(len(ctcnt)):
        maxt[ssn][i] = [[1 for i1 in range(101)] for j1 in range(len(pb))]
        mint[ssn][i] = [[1 for i1 in range(71)] for j1 in range(len(pb))]
        rh1[ssn][i] = [[1 for i1 in range(101)] for j1 in range(len(pb))]
        rh2[ssn][i] = [[1 for i1 in range(101)] for j1 in range(len(pb))]
        ws[ssn][i] = [[1 for i1 in range(51)] for j1 in range(len(pb))]
        rf[ssn][i] = [[1 for i1 in range(301)] for j1 in range(len(pb))]
        ss[ssn][i] = [[1 for i1 in range(31)] for j1 in range(len(pb))]
        evp[ssn][i] = [[1 for i1 in range(41)] for j1 in range(len(pb))]

for i in range(len(adv)):
    date = cp[i][::-1]
    date = date[:10]
    date = date[::-1]
    ssn = season(date)
    for k in range(len(ctcnt)):
        if ctcnt[k][0]==cpt[i]:
            break
    if date in wd:
        idx1 = round(wd[date][0])
        idx1 = int(idx1*2)
        w1[ssn][k][idx1]+=1
        idx1 = round(wd[date][1])
        idx1 = int(idx1*2)
        w2[ssn][k][idx1]+=1
        idx1 = int(wd[date][2])
        w3[ssn][k][idx1]+=1
        idx1 = int(wd[date][3])
        w4[ssn][k][idx1]+=1
        idx1 = round(wd[date][4])
        idx1=int(idx1*2)
        w5[ssn][k][idx1]+=1
        idx1 = round(wd[date][5])
        idx1=int(idx1*2)
        w6[ssn][k][idx1]+=1
        idx1 = round(wd[date][6])
        idx1=int(idx1*2)
        w7[ssn][k][idx1]+=1
        idx1 = round(wd[date][7])
        idx1=int(idx1*2)
        w8[ssn][k][idx1]+=1
        for j in range(len(pb)):
            if pb[j][0] in adv[i]:
                idx = round(wd[date][0])
                idx = int(idx*2)
                maxt[ssn][k][j][idx]+=1
                idx = round(wd[date][1])
                idx = int(idx*2)
                mint[ssn][k][j][idx]+=1
                idx = int(wd[date][2])
                rh1[ssn][k][j][idx]+=1
                idx = int(wd[date][3])
                rh2[ssn][k][j][idx]+=1
                idx = round(wd[date][4])
                idx=int(idx*2)
                ws[ssn][k][j][idx]+=1
                idx = round(wd[date][5])
                idx=int(idx*2)
                rf[ssn][k][j][idx]+=1
                idx = round(wd[date][6])
                idx=int(idx*2)
                ss[ssn][k][j][idx]+=1
                idx = round(wd[date][7])
                idx=int(idx*2)
                evp[ssn][k][j][idx]+=1
for i in range(len(ctcnt)):
    for ssn in sd:
        a = maxt[ssn][i]
        for j in range(len(pb)):
            for val in range(40,89):
                a[j][val] = (a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
            tmp = float(sum(a[j]))
            for val in range(101):
                a[j][val] = a[j][val]/tmp
        np.savetxt('w1/'+ssn+'/'+ctcnt[i][0]+'.txt',a,delimiter=' ',fmt='%s')


for i in range(len(ctcnt)):
    for ssn in sd:
        a = mint[ssn][i]
        for j in range(len(pb)):
            for val in range(10,64):
                a[j][val] = (a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
            tmp = float(sum(a[j]))
            for val in range(71):
                a[j][val] = a[j][val]/tmp
        np.savetxt('w2/'+ssn+'/'+ctcnt[i][0]+'.txt',a,delimiter=' ',fmt='%s')
for i in range(len(ctcnt)):
    for ssn in sd:
        a = rh1[ssn][i]
        for j in range(len(pb)):
            a[j][100] = (a[j][99]+a[j][100])/float(2)
            for val in range(16,100):
                a[j][val] = (a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
            tmp = float(sum(a[j]))
            for val in range(101):
                a[j][val] = a[j][val]/tmp
        np.savetxt('w3/'+ssn+'/'+ctcnt[i][0]+'.txt',a,delimiter=' ',fmt='%s')
for i in range(len(ctcnt)):
    for ssn in sd:
        a = rh2[ssn][i]
        for j in range(len(pb)):
            a[j][100] = (a[j][99]+a[j][100])/float(2)
            for val in range(6,100):
                a[j][val] = (a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
            tmp = float(sum(a[j]))
            for val in range(101):
                a[j][val] = a[j][val]/tmp
        np.savetxt('w4/'+ssn+'/'+ctcnt[i][0]+'.txt',a,delimiter=' ',fmt='%s')

for i in range(len(ctcnt)):
    for ssn in sd:
        a = ws[ssn][i]
        for j in range(len(pb)):
            a[j][0] = (a[j][0]+a[j][1])/float(2)
            for val in range(1,48):
                a[j][val] = (a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
            tmp = float(sum(a[j]))
            for val in range(51):
                a[j][val] = a[j][val]/tmp
        np.savetxt('w5/'+ssn+'/'+ctcnt[i][0]+'.txt',a,delimiter=' ',fmt='%s')
for i in range(len(ctcnt)):
    for ssn in sd:
        a = rf[ssn][i]
        for j in range(len(pb)):
            a[j][0] = (a[j][0]+a[j][1])/float(2)
            a[j][300] = (a[j][299]+a[j][300])/float(2)
            for val in range(1,300):
                a[j][val] = (a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
            tmp = float(sum(a[j]))
            for val in range(301):
                a[j][val] = a[j][val]/tmp
        np.savetxt('w6/'+ssn+'/'+ctcnt[i][0]+'.txt',a,delimiter=' ',fmt='%s')
for i in range(len(ctcnt)):
    for ssn in sd:
        a = ss[ssn][i]
        for j in range(len(pb)):
            a[j][0] = (a[j][0]+a[j][1])/float(2)
            for val in range(1,24):
                a[j][val] = (a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
            tmp = float(sum(a[j]))
            for val in range(31):
                a[j][val] = a[j][val]/tmp
        np.savetxt('w7/'+ssn+'/'+ctcnt[i][0]+'.txt',a,delimiter=' ',fmt='%s')
for i in range(len(ctcnt)):
    for ssn in sd:
        a = evp[ssn][i]
        for j in range(len(pb)):
            a[j][0] = (a[j][0]+a[j][1])/float(2)
            for val in range(1,31):
                a[j][val] = (a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
            tmp = float(sum(a[j]))
            for val in range(41):
                a[j][val] = a[j][val]/tmp
        np.savetxt('w8/'+ssn+'/'+ctcnt[i][0]+'.txt',a,delimiter=' ',fmt='%s')




for ssn in sd:
    a = w1[ssn]
    for j in range(len(ctcnt)):
        for val in range(40,89):
            a[j][val]=(a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
        tmp = float(sum(a[j]))
        for val in range(101):
            a[j][val] = a[j][val]/tmp
    np.savetxt('weather/w1/'+ssn+'.txt',a,delimiter=' ',fmt='%s')
for ssn in sd:
    a = w2[ssn]
    for j in range(len(ctcnt)):
        for val in range(10,64):
            a[j][val]=(a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
        tmp = float(sum(a[j]))
        for val in range(71):
            a[j][val] = a[j][val]/tmp
    np.savetxt('weather/w2/'+ssn+'.txt',a,delimiter=' ',fmt='%s')
for ssn in sd:
    a = w3[ssn]
    for j in range(len(ctcnt)):
        a[j][100] =(a[j][99]+a[j][100])/float(2) 
        for val in range(16,100):
            a[j][val]=(a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
        tmp = float(sum(a[j]))
        for val in range(101):
            a[j][val] = a[j][val]/tmp
    np.savetxt('weather/w3/'+ssn+'.txt',a,delimiter=' ',fmt='%s')
for ssn in sd:
    a = w4[ssn]
    for j in range(len(ctcnt)):
        a[j][100] = (a[j][99]+a[j][100])/float(2)
        for val in range(6,100):
            a[j][val]=(a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
        tmp = float(sum(a[j]))
        for val in range(101):
            a[j][val] = a[j][val]/tmp
    np.savetxt('weather/w4/'+ssn+'.txt',a,delimiter=' ',fmt='%s')
for ssn in sd:
    a = w5[ssn]
    for j in range(len(ctcnt)):
        a[j][0] = (a[j][0]+a[j][1])/float(2)
        for val in range(1,48):
            a[j][val]=(a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
        tmp = float(sum(a[j]))
        for val in range(51):
            a[j][val] = a[j][val]/tmp
    np.savetxt('weather/w5/'+ssn+'.txt',a,delimiter=' ',fmt='%s')
for ssn in sd:
    a = w6[ssn]
    for j in range(len(ctcnt)):
        a[j][0] = (a[j][0]+a[j][1])/float(2)
        a[j][300]=(a[j][299]+a[j][300])/float(2)
        for val in range(1,300):
            a[j][val]=(a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
        tmp = float(sum(a[j]))
        for val in range(301):
            a[j][val] = a[j][val]/tmp
    np.savetxt('weather/w6/'+ssn+'.txt',a,delimiter=' ',fmt='%s')
for ssn in sd:
    a = w7[ssn]
    for j in range(len(ctcnt)):
        a[j][0] = (a[j][0]+a[j][1])/float(2)
        for val in range(1,24):
            a[j][val]=(a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
        tmp = float(sum(a[j]))
        for val in range(31):
            a[j][val] = a[j][val]/tmp
    np.savetxt('weather/w7/'+ssn+'.txt',a,delimiter=' ',fmt='%s')
for ssn in sd:
    a = w8[ssn]
    for j in range(len(ctcnt)):
        a[j][0] = (a[j][0]+a[j][1])/float(2)
        for val in range(1,31):
            a[j][val]=(a[j][val-1]+a[j][val]+a[j][val+1])/float(3)
        tmp = float(sum(a[j]))
        for val in range(41):
            a[j][val] = a[j][val]/tmp
    np.savetxt('weather/w8/'+ssn+'.txt',a,delimiter=' ',fmt='%s')
