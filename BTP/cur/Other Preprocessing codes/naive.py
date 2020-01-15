#!/usr/bin/python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def lrelu(x):
    return tf.maximum(x,0.2*x)
    
scale = MinMaxScaler()

psx = []
psy=[]
tx1=[]
tx2=[]
tx3=[]
ty1=[]
ty2=[]
ty3=[]
for i in range(19):
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
'''
x1, x2, y1, y2 = train_test_split(psx, psy, test_size=0.33, random_state=2)
st1 = [0 for i in range(18)]
st2 = [0 for i in range(18)]
st3 = [0 for i in range(18)]
for i in range(len(psy)):
    st1[psy[i]]+=1
for i in range(len(y2)):
    st3[y2[i]]+=1
for i in range(len(y1)):
    st2[y1[i]]+=1

for i in range(18):
    print st1[i],st2[i],st3[i]
'''
print len(psx)
print(scale.fit(psx))
print(scale.data_max_)
print(scale.data_min_)

for k in range(3):
    xtrain[k] = scale.transform(xtrain[k])
    xtest[k] = scale.transform(xtest[k])
    print len(ytest[k])
for k in range(3):
    gnb = GaussianNB()
    gnb.fit(xtrain[k],ytrain[k])
    yp = gnb.predict(xtest[k])
    yt = ytest[k]
    ln = np.shape(yt)[0]
    print (np.equal(yp,ytest[k]).sum())*100/float(ln)
    pb = gnb.predict_proba(xtest[k])
    yp3 = np.argsort(pb,axis=1)[:,16:19]
    print yp3,np.argmax(pb,axis=1)
    yt=np.reshape(yt,[ln,1])
    acc3= np.equal(yp3,yt)
    acc3 = np.amax(acc3,axis=1)
    print acc3.sum()*100/float(ln)

