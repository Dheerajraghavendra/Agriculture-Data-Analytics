#!/usr/bin/python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def actfn(x,dd = False):
    if dd==True:
        #return np.sign(np.maximum(0,x))
        return x*(1-x)
    else:
        #return np.maximum(0,x)
        return 1/(1+np.exp(-x))
noiter=100
nh = 1
scale = MinMaxScaler()
x1=[]
x2=[]
d= []
for i in range(18):
    with open('tdata/'+str(i)+'.txt','r') as f:
        tmp = np.genfromtxt(f,dtype=None,delimiter='\t')
    f = int(0.8*len(tmp))
    for j in range(f):
        x1.append(tmp[j])
        d.append(tmp[j][:11])
    for j in range(f,len(tmp)):
        x2.append(tmp[j])
        d.append(tmp[j][:11])
for i in range(len(d)):
    d[i]=np.append(d[i],[1])
print(scale.fit(d))
print(scale.data_max_)
print(scale.data_min_)
X = np.ones(shape = (len(x1),12))
Y = np.zeros(shape=(len(x1),18))
for i in range(len(x1)):
    Y[i][int(x1[i][11])-1] = 1
    for j in range(11):
        X[i][j] = float(x1[i][j])
    X[i][11] = 1

Xtest = np.ones(shape = (len(x2),12))
Ytest = np.zeros(shape=(len(x2),18))
for i in range(len(x2)):
    Ytest[i][int(x2[i][11])-1] = 1
    for j in range(11):
        Xtest[i][j] = float(x2[i][j])
    Xtest[i][11] = 1
X = scale.transform(X)
Xtest = scale.transform(Xtest)

wts = [[] for i in range(nh+1)]
nh1,nh2,nh3,nh4,nh5 = 30,50,50,50,30
wts[0] = np.zeros(shape=(12,nh1))#+0.5#np.random.random((11,nh1))
wts[1] = np.zeros(shape=(nh1,18))#+0.5#np.random.random((nh1,nh2))
#wts[2] = np.zeros(shape=(nh2,nh3))#+0.5#np.random.random((nh2,nh3))
#wts[3] = np.zeros(shape=(nh3,nh4))#+0.5#np.random.random((nh3,nh4))
#wts[4] = np.zeros(shape=(nh4,nh5))#+0.5#np.random.random((nh4,nh5))
#wts[5] = np.zeros(shape=(nh5,18))#+0.5#np.random.random((nh5,18))
lr = 0.1

#training
for itr in range(noiter):
    print itr
    for sample in range(len(X)):
        x = X[sample]
        x = x.reshape(1,12)
        y = Y[sample]
        y = y.reshape(1,18)
        tmp = [[] for j in range(nh+2)]
        tmp[0] = x
        #tmp[1] = actfn(np.dot(x,wts[0]))
        #tmp[1]=tmp[1].reshape(1,tmp[1].shape[1])
        for j in range(1,nh+2):
            tmp[j] = actfn(np.dot(tmp[j-1],wts[j-1]))
            tmp[j]=tmp[j].reshape(1,tmp[j].shape[1]) 
        #print y,itr
        #print tmp[nh+1]
        d = [[] for j in range(nh+1)]
        #backpropagation 
        d[nh] = (y-tmp[nh+1])*actfn(tmp[nh+1],dd=True)
        #print actfn(tmp[nh+1],dd=True)
        wud = [[] for j in range(nh+1)]
        for j in range(nh+1):
            wud[j] = np.zeros(shape=wts[j].shape)
        [a,b] = wts[nh].shape
        for k in range(b):
            for j in range(a):
                wud[nh][j][k] = d[nh][0][k]*tmp[nh][0][j]
        for i in range(nh-1,-1,-1):
            [a,b] = wts[i].shape
            d[i] = (d[i+1].dot(wts[i+1].T))*actfn(tmp[i+1],dd=True)
            for k in range(b):
                for j in range(a):
                    wud[i][j][k] = d[i][0][k]*tmp[i][0][j]
        for i in range(nh+1):
        #    print wud[i]
            wts[i]+= lr*wud[i]

print "Training is over"
print "Testing...."
ct = 0
#Testing

print wts

#Train error
for sample in range(len(Xtest)):
    x = X[sample]
    x = x.reshape(1,12)
    y = Y[sample]
    y = y.reshape(1,18)
    tmp = [[] for j in range(nh+1)]
    tmp[0] = actfn(np.dot(x,wts[0]))
    tmp[0]=tmp[0].reshape(1,tmp[0].shape[1])
    for j in range(1,nh+1):
        tmp[j] = actfn(np.dot(tmp[j-1],wts[j]))
        tmp[j]=tmp[j].reshape(1,tmp[j].shape[1]) 
    pred = tmp[nh]
    print y,y.shape
    print pred,pred.shape
    mx=0
    mxi=0
    mxi2=0
    mx2=0
    for j in range(18):
        if pred[0][j]>mx:
            mx= pred[0][j]
            mxi = j
        if y[0][j]>mx2:
            mx2 = y[0][j]
            mxi2 = j
    if mxi!=mxi2:
        ct+=1
print "Train Error: ",ct/float(len(Ytest))
ct=0

#Test error
for sample in range(len(Xtest)):
    x = Xtest[sample]
    x = x.reshape(1,12)
    y = Ytest[sample]
    y = y.reshape(1,18)
    tmp = [[] for j in range(nh+1)]
    tmp[0] = actfn(np.dot(x,wts[0]))
    tmp[0]=tmp[0].reshape(1,tmp[0].shape[1])
    for j in range(1,nh+1):
        tmp[j] = actfn(np.dot(tmp[j-1],wts[j]))
        tmp[j]=tmp[j].reshape(1,tmp[j].shape[1]) 
    pred = tmp[nh]
    print x
    print y[0]
    print pred[0]
    mx=0
    mxi=0
    mxi2=0
    mx2=0
    for j in range(18):
        if pred[0][j]>mx:
            mx= pred[0][j]
            mxi = j
        if y[0][j]>mx2:
            mx2 = y[0][j]
            mxi2 = j
    if mxi!=mxi2:
        ct+=1
print "Test Error: ",ct/float(len(Ytest))
        


