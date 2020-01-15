#!/usr/bin/python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
with open('croptypecount.txt','r')as f:
        ctcnt = np.genfromtxt(f,dtype = None,delimiter = '\t')


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

def season(m):
    if m>=7 and m<=10:
        return "kharif"
    elif m>=3 and m<=6:
        return "summer"
    else:
        return "rabi"


#scale = MinMaxScaler()

psx = []
psy=[]
tx1=[]
tx2=[]
tx3=[]
ty1=[]
ty2=[]
ty3=[]
for i in range(18):
    with open('tdata/'+str(i)+'.txt','r') as f:
        tmp = np.genfromtxt(f,dtype=None,delimiter='\t')
    f1 = int(0.33*len(tmp))
    for j in range(f1):
        tx1.append(tmp[j][:11])
        ty1.append(i)
        psx.append(tmp[j][:11])
        psy.append(i)
    f2 = int(0.66*len(tmp))
    for j in range(f1,f2):
        tx2.append(tmp[j][:11])
        ty2.append(i)
        psx.append(tmp[j][:11])
        psy.append(i)
    for j in range(f2,len(tmp)):
        tx3.append(tmp[j][:11])
        ty3.append(i)
        psx.append(tmp[j][:11])
        psy.append(i)
xtrain = [0 for i in range(3)]
xtest = [0 for i in range(3)]
ytrain = [0 for i in range(3)]
ytest = [0 for i in range(3)]
xtrain[0] = np.concatenate((tx1,tx2))
xtest[0] = tx3
xtrain[1] = np.concatenate((tx1,tx3))
xtest[1] = tx2
xtrain[2] = np.concatenate((tx2,tx3))
xtest[2] = tx1
ytrain[0] = np.concatenate((ty1,ty2))
ytest[0] = ty3
ytrain[1] = np.concatenate((ty1,ty3))
ytest[1] = ty2
ytrain[2] = np.concatenate((ty2,ty3))
ytest[2] = ty1

for k in range(3):
    x = xtrain[k]
    y = ytrain[k]

    ##Calculating P(age/problem) for each crop
    age = {}
    month = {}
    p = {}
    p["kharif"] = [[0 for i in range(19)] for j in range(len(ctcnt))]
    p["rabi"] = [[0 for i in range(19)] for j in range(len(ctcnt))]
    p["summer"] = [[0 for i in range(19)] for j in range(len(ctcnt))]
    for i in range(len(ctcnt)):
        age[i] = [[1 for i1 in range(346)] for j1 in range(19)]
        month[i] =[[0 for i1 in range(12)] for j1 in range(19)]
    l = p.keys()
    maxt={}
    mint={}
    rh1={}
    rh2={}
    ws={}
    rf={}
    ss={}
    evp={}
    w1={}
    w2={}
    w3={}
    w4={}
    w5={}
    w6={}
    w7={}
    w8={}
    for ssn in l:
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
            maxt[ssn][i] = [[1 for i1 in range(101)] for j1 in range(19)]
            mint[ssn][i] = [[1 for i1 in range(71)] for j1 in range(19)]
            rh1[ssn][i] = [[1 for i1 in range(101)] for j1 in range(19)]
            rh2[ssn][i] = [[1 for i1 in range(101)] for j1 in range(19)]
            ws[ssn][i] = [[1 for i1 in range(51)] for j1 in range(19)]
            rf[ssn][i] = [[1 for i1 in range(301)] for j1 in range(19)]
            ss[ssn][i] = [[1 for i1 in range(31)] for j1 in range(19)]
            evp[ssn][i] = [[1 for i1 in range(41)] for j1 in range(19)]
    for i in range(len(x)):
        crop = int(x[i][0])
        mth = int(x[i][1])-1
        ssn = season(mth+1)
        wk = int(x[i][2])
        pb = y[i]
        p[ssn][crop][pb]+=1
        age[crop][pb][wk]+=1
        month[crop][pb][mth]+=1

        idx1= rnd(x[i][3])
        idx1 = int(idx1*2)
        w1[ssn][crop][idx1]+=1
        maxt[ssn][crop][pb][idx1]+=1

        idx2= rnd(x[i][4])
        idx2 = int(idx2*2)
        w2[ssn][crop][idx2]+=1
        mint[ssn][crop][pb][idx2]+=1

        idx3= rnd(x[i][5])
        idx3 = int(idx3)
        w3[ssn][crop][idx3]+=1
        rh1[ssn][crop][pb][idx3]+=1

        idx4= rnd(x[i][6])
        idx4 = int(idx4)
        w1[ssn][crop][idx4]+=1
        rh2[ssn][crop][pb][idx4]+=1
         
        idx5= rnd(x[i][7])
        idx5 = int(idx5*2)
        w5[ssn][crop][idx5]+=1
        ws[ssn][crop][pb][idx5]+=1
         
        idx6= rnd(x[i][8])
        idx6 = int(idx6*2)
        w6[ssn][crop][idx6]+=1
        rf[ssn][crop][pb][idx6]+=1
         
        idx7= rnd(x[i][9])
        idx7 = int(idx7*2)
        w7[ssn][crop][idx7]+=1
        ss[ssn][crop][pb][idx7]+=1

        idx8= rnd(x[i][10])
        idx8 = int(idx8*2)
        w8[ssn][crop][idx8]+=1
        evp[ssn][crop][pb][idx8]+=1

    for ssn in l:
        for i in range(len(ctcnt)):
            tmp = float(sum(p[ssn][i]))
            for j in range(19):
                if tmp>0:
                    p[ssn][i][j] = p[ssn][i][j]/tmp
    for i in range(len(ctcnt)):
        for j in range(19):
            age[i][j][0] = (age[i][j][0]+age[i][j][1])/(float(2))
            age[i][j][345] = (age[i][j][344]+age[i][j][345])/float(2)
            for wk in range(1,345):
                age[i][j][wk] = (age[i][j][wk-1]+age[i][j][wk]+age[i][j][wk+1])/float(3)
            tmp = float(sum(age[i][j]))
            for wk in range(0,346):
                age[i][j][wk] = age[i][j][wk]/tmp
            
            ##Calculating P(month/problem) for each crop(not needed)
            tmp = float(sum(month[i][j]))
            for mnth in range(12):
                if tmp>0:
                    month[i][j][mnth]/=tmp
    ##Probabilities - P(factor/problem) for fixed season and crop
    for ssn in l:
        for i in range(len(ctcnt)):
            for j in range(19):
                for val in range(40,89):
                    maxt[ssn][i][j][val] = (maxt[ssn][i][j][val-1]+maxt[ssn][i][j][val]+maxt[ssn][i][j][val+1])/float(3)
                tmp = float(sum(maxt[ssn][i][j]))
                for val in range(101):
                    maxt[ssn][i][j][val] = maxt[ssn][i][j][val]/tmp
    for ssn in l:
        for i in range(len(ctcnt)):
            for j in range(19):
                for val in range(10,64):
                    mint[ssn][i][j][val] = (mint[ssn][i][j][val-1]+mint[ssn][i][j][val]+mint[ssn][i][j][val+1])/float(3)
                tmp = float(sum(mint[ssn][i][j]))
                for val in range(71):
                    mint[ssn][i][j][val] = mint[ssn][i][j][val]/tmp
    for ssn in l:
        for i in range(len(ctcnt)):
            for j in range(19):
                rh1[ssn][i][j][100] = (rh1[ssn][i][j][99]+rh1[ssn][i][j][100])/float(2)
                for val in range(16,100):
                    rh1[ssn][i][j][val] = (rh1[ssn][i][j][val-1]+rh1[ssn][i][j][val]+rh1[ssn][i][j][val+1])/float(3)
                tmp = float(sum(rh1[ssn][i][j]))
                for val in range(101):
                    rh1[ssn][i][j][val] = rh1[ssn][i][j][val]/tmp
    for ssn in l:
        for i in range(len(ctcnt)):
            for j in range(19):
                rh2[ssn][i][j][100] = (rh2[ssn][i][j][99]+rh2[ssn][i][j][100])/float(2)
                for val in range(6,100):
                    rh2[ssn][i][j][val] = (rh2[ssn][i][j][val-1]+rh2[ssn][i][j][val]+rh2[ssn][i][j][val+1])/float(3)
                tmp = float(sum(rh2[ssn][i][j]))
                for val in range(101):
                    rh2[ssn][i][j][val] = rh2[ssn][i][j][val]/tmp
    for ssn in l:
        for i in range(len(ctcnt)):
            for j in range(19):
                ws[ssn][i][j][0] = (ws[ssn][i][j][0]+ws[ssn][i][j][1])/float(2)
                for val in range(1,48):
                    ws[ssn][i][j][val] = (ws[ssn][i][j][val-1]+ws[ssn][i][j][val]+ws[ssn][i][j][val+1])/float(3)
                tmp = float(sum(ws[ssn][i][j]))
                for val in range(51):
                    ws[ssn][i][j][val] = ws[ssn][i][j][val]/tmp
    for ssn in l:
        for i in range(len(ctcnt)):
            for j in range(19):
                rf[ssn][i][j][0] = (rf[ssn][i][j][0]+rf[ssn][i][j][1])/float(2)
		rf[ssn][i][j][300] = (rf[ssn][i][j][299]+rf[ssn][i][j][300])/float(2)
                for val in range(1,300):
                    rf[ssn][i][j][val] = (rf[ssn][i][j][val-1]+rf[ssn][i][j][val]+rf[ssn][i][j][val+1])/float(3)
                tmp = float(sum(rf[ssn][i][j]))
                for val in range(301):
                    rf[ssn][i][j][val] = rf[ssn][i][j][val]/tmp
    for ssn in l:
        for i in range(len(ctcnt)):
            for j in range(19):
                ss[ssn][i][j][0] = (ss[ssn][i][j][0]+ss[ssn][i][j][1])/float(2)
                for val in range(1,24):
                    ss[ssn][i][j][val] = (ss[ssn][i][j][val-1]+ss[ssn][i][j][val]+ss[ssn][i][j][val+1])/float(3)
                tmp = float(sum(ss[ssn][i][j]))
                for val in range(31):
                    ss[ssn][i][j][val] = ss[ssn][i][j][val]/tmp
    for ssn in l:
        for i in range(len(ctcnt)):
            for j in range(19):
                evp[ssn][i][j][0] = (evp[ssn][i][j][0]+evp[ssn][i][j][1])/float(2)
                for val in range(1,31):
                    evp[ssn][i][j][val] = (evp[ssn][i][j][val-1]+evp[ssn][i][j][val]+evp[ssn][i][j][val+1])/float(3)
                tmp = float(sum(evp[ssn][i][j]))
                for val in range(41):
                    evp[ssn][i][j][val] = evp[ssn][i][j][val]/tmp
    #Calculating P(factors) for fixed season and crop
    for ssn in l:
        for j in range(len(ctcnt)):
            for val in range(40,89):
                w1[ssn][j][val]=(w1[ssn][j][val-1]+w1[ssn][j][val]+w1[ssn][j][val+1])/float(3)
            tmp = float(sum(w1[ssn][j]))
            for val in range(101):
                w1[ssn][j][val] = w1[ssn][j][val]/tmp

    for ssn in l:
        for j in range(len(ctcnt)):
            for val in range(10,64):
                w2[ssn][j][val]=(w2[ssn][j][val-1]+w2[ssn][j][val]+w2[ssn][j][val+1])/float(3)
            tmp = float(sum(w2[ssn][j]))
            for val in range(71):
                w2[ssn][j][val] = w2[ssn][j][val]/tmp
    for ssn in l:
        for j in range(len(ctcnt)):
	    w3[ssn][j][100] =(w3[ssn][j][99]+w3[ssn][j][100])/float(2) 
            for val in range(16,100):
                w3[ssn][j][val]=(w3[ssn][j][val-1]+w3[ssn][j][val]+w3[ssn][j][val+1])/float(3)
            tmp = float(sum(w3[ssn][j]))
            for val in range(101):
                w3[ssn][j][val] = w3[ssn][j][val]/tmp
    for ssn in l:
        for j in range(len(ctcnt)):
	    w4[ssn][j][100] =(w4[ssn][j][99]+w4[ssn][j][100])/float(2) 
            for val in range(6,100):
                w4[ssn][j][val]=(w4[ssn][j][val-1]+w4[ssn][j][val]+w4[ssn][j][val+1])/float(3)
            tmp = float(sum(w4[ssn][j]))
            for val in range(101):
                w4[ssn][j][val] = w4[ssn][j][val]/tmp
    for ssn in l:
        for j in range(len(ctcnt)):
	    w5[ssn][j][0] =(w5[ssn][j][0]+w5[ssn][j][1])/float(2) 
            for val in range(1,48):
                w5[ssn][j][val]=(w5[ssn][j][val-1]+w5[ssn][j][val]+w5[ssn][j][val+1])/float(3)
            tmp = float(sum(w5[ssn][j]))
            for val in range(51):
                w5[ssn][j][val] = w5[ssn][j][val]/tmp
    for ssn in l:
        for j in range(len(ctcnt)):
	    w6[ssn][j][0] =(w6[ssn][j][0]+w6[ssn][j][1])/float(2) 
	    w6[ssn][j][300] =(w6[ssn][j][299]+w6[ssn][j][300])/float(2) 
            for val in range(1,300):
                w6[ssn][j][val]=(w6[ssn][j][val-1]+w6[ssn][j][val]+w6[ssn][j][val+1])/float(3)
            tmp = float(sum(w6[ssn][j]))
            for val in range(301):
                w6[ssn][j][val] = w6[ssn][j][val]/tmp
    for ssn in l:
        for j in range(len(ctcnt)):
	    w7[ssn][j][0] =(w7[ssn][j][0]+w7[ssn][j][1])/float(2) 
            for val in range(1,24):
                w7[ssn][j][val]=(w7[ssn][j][val-1]+w7[ssn][j][val]+w7[ssn][j][val+1])/float(3)
            tmp = float(sum(w7[ssn][j]))
            for val in range(31):
                w7[ssn][j][val] = w7[ssn][j][val]/tmp
    for ssn in l:
        for j in range(len(ctcnt)):
	    w8[ssn][j][0] =(w8[ssn][j][0]+w8[ssn][j][1])/float(2) 
            for val in range(1,31):
                w8[ssn][j][val]=(w8[ssn][j][val-1]+w8[ssn][j][val]+w8[ssn][j][val+1])/float(3)
            tmp = float(sum(w8[ssn][j]))
            for val in range(41):
                w8[ssn][j][val] = w8[ssn][j][val]/tmp
    xt= xtest[k]
    yt= ytest[k]
    yp = np.zeros(shape=(len(yt),3))
    for i in range(len(xt)):       
	crop = int(xt[i][0])
        mth = int(xt[i][1])-1
        ssn = season(mth+1)
        wk = int(x[i][2])
        op = [0 for i1 in range(19)]
	for i1 in range(19):
	    op[i1] =  p[ssn][crop][i1]*maxt[ssn][crop][i1][int(xt[i1][3]*2)]
	    op[i1]*= mint[ssn][crop][i1][int(xt[i1][4]*2)]
	    op[i1]*= rh1[ssn][crop][i1][int(xt[i1][5])]
	    op[i1]*= rh2[ssn][crop][i1][int(xt[i1][6])]
	    op[i1]*= ws[ssn][crop][i1][int(xt[i1][7]*2)]
	    op[i1]*= rf[ssn][crop][i1][int(xt[i1][8]*2)]
	    op[i1]*= ss[ssn][crop][i1][int(xt[i1][9]*2)]
	    op[i1]*= evp[ssn][crop][i1][int(xt[i1][10]*2)]
    	    d = w1[ssn][crop][int(xt[i1][3]*2)]
	    d*= w2[ssn][crop][int(xt[i1][4]*2)]
    	    d*= w3[ssn][crop][int(xt[i1][5])]
    	    d*= w4[ssn][crop][int(xt[i1][6])]
    	    d*= w5[ssn][crop][int(xt[i1][7]*2)]
    	    d*= w6[ssn][crop][int(xt[i1][8]*2)]
    	    d*= w7[ssn][crop][int(xt[i1][9]*2)]
    	    d*= w8[ssn][crop][int(xt[i1][10]*2)]
	    op[i1] = op[i1]/float(d)
        sb = sorted(enumerate(op),key = lambda g:g[1])
	yp[i][0] = sb[18][0]
        yp[i][1] = sb[17][0]
        yp[i][2] = sb[16][0]
    ln = np.shape(yt)[0]
    acc = (np.equal(yp[:,0],yt).sum())*100/float(ln)
    print "Testing accuracies"
    print acc,"%"
    yt = np.reshape(yt,[ln,1])
    acc3 = np.equal(yp,yt)
    acc3 = np.amax(acc3,axis=1)
    print (acc3.sum())*100/float(ln),"%"

    xt= xtrain[k]
    yt= ytrain[k]
    yp = np.zeros(shape=(len(yt),3))

    for i in range(len(xt)):       
	crop = int(xt[i][0])
        mth = int(xt[i][1])-1
        ssn = season(mth+1)
        wk = int(x[i][2])
        op = [0 for i1 in range(19)]
	for i1 in range(19):
	    op[i1] =  p[ssn][crop][i1]*maxt[ssn][crop][i1][int(xt[i1][3]*2)]
	    op[i1]*= mint[ssn][crop][i1][int(xt[i1][4]*2)]
	    op[i1]*= rh1[ssn][crop][i1][int(xt[i1][5])]
	    op[i1]*= rh2[ssn][crop][i1][int(xt[i1][6])]
	    op[i1]*= ws[ssn][crop][i1][int(xt[i1][7]*2)]
	    op[i1]*= rf[ssn][crop][i1][int(xt[i1][8]*2)]
	    op[i1]*= ss[ssn][crop][i1][int(xt[i1][9]*2)]
	    op[i1]*= evp[ssn][crop][i1][int(xt[i1][10]*2)]
    	    d = w1[ssn][crop][int(xt[i1][3]*2)]
	    d*= w2[ssn][crop][int(xt[i1][4]*2)]
    	    d*= w3[ssn][crop][int(xt[i1][5])]
    	    d*= w4[ssn][crop][int(xt[i1][6])]
    	    d*= w5[ssn][crop][int(xt[i1][7]*2)]
    	    d*= w6[ssn][crop][int(xt[i1][8]*2)]
    	    d*= w7[ssn][crop][int(xt[i1][9]*2)]
    	    d*= w8[ssn][crop][int(xt[i1][10]*2)]
	    op[i1] = op[i1]/float(d)
        sb = sorted(enumerate(op),key = lambda g:g[1])
	yp[i][0] = sb[18][0]
        yp[i][1] = sb[17][0]
        yp[i][2] = sb[16][0]
    ln = np.shape(yt)[0]
    acc = (np.equal(yp[:,0],yt).sum())*100/float(ln)
    print "Training accuracies"
    print acc,"%"
    yt = np.reshape(yt,[ln,1])
    acc3 = np.equal(yp,yt)
    acc3 = np.amax(acc3,axis=1)
    print (acc3.sum())*100/float(ln),"%"
