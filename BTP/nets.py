#!/usr/bin/python
import numpy as np
from sklearn import preprocessing
def actfn(x,dd = False):
    if dd==True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))
noiter=10000
nh = 5
with open('finalformat.txt','r') as f:
    tmp = np.genfromtxt(f,dtype=None,delimiter='\t')
X = np.ones(shape = (len(tmp),11))
Y = np.zeros(shape=(len(tmp),18))
#f = int(0.8*len(X))
f = len(X)
for i in range(len(tmp)):
    Y[i][int(tmp[i][11])-1] = 1
    for j in range(11):
        X[i][j] = float(tmp[i][j])
#print X
X = preprocessing.scale(X)
#print X
Xtest = X[f:]
Ytest = Y[f:]
X = X[:f]
Y = Y[:f]
print f
wts = [[] for i in range(nh+1)]
nh1,nh2,nh3,nh4,nh5 = 25,50,75,50,30
wts[0] = np.random.random((11,nh1))
wts[1] = np.random.random((nh1,nh2))
wts[2] = np.random.random((nh2,nh3))
wts[3] = np.random.random((nh3,nh4))
wts[4] = np.random.random((nh4,nh5))
wts[5] = np.random.random((nh5,18))
lr = 1
for itr in range(noiter):
    for sample in range(f):
        x = X[sample]
        x = x.reshape(1,11)
        y = Y[sample]
        y = y.reshape(1,18)
        tmp = [[] for j in range(nh+1)]
        tmp[0] = actfn(np.dot(x,wts[0]))
        tmp[0]=tmp[0].reshape(1,tmp[0].shape[1])
        for j in range(1,nh+1):
            tmp[j] = actfn(np.dot(tmp[j-1],wts[j]))
            tmp[j]=tmp[j].reshape(1,tmp[j].shape[1]) 
        print y,itr
        print tmp[nh]
        
        #backpropagation 
        curerr = y-tmp[nh]
        update = [[] for _ in range(nh+1)]
        #actfn,dd=True - just gives derivatives-1xN array
        for j in range(nh,-1,-1):
            update[j] = curerr*actfn(tmp[j],dd=True);
            curerr = update[j].dot(wts[j].T)

        #wts update
        wts[0]+=  x.T.dot(update[0])
        for j in range(1,nh+1):
            wts[j]+= lr*tmp[j-1].T.dot(update[j])
        
        '''
        pred = actfn(np.dot(Xtest,wts[0]))
        for j in range(1,nh+1):
            pred = actfn(np.dot(pred,wts[j]))
        print pred
        ct=0
        for i in range(len(pred)):
            mx=0
            mxi=0
            mxi2=0
            mx2=0
            for j in range(18):
                if pred[i][j]>mx:
                    mx= pred[i][j]
                    mxi = j
                if Ytest[i][j]>mx2:
                    mx2 = Ytest[i][j]
                    mxi2 = j
            if mxi!=mxi2:
                ct+=1
        print "Error: ",ct/float(len(Ytest))
        '''


