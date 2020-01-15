#!/usr/bin/python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
def lrelu(x):
    return tf.maximum(x,0.1*x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
def relu(x):
    return np.maximum(x,0)
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
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

print len(psx)
print(scale.fit(psx))
print(scale.data_max_)
print(scale.data_min_)

X = np.ones(shape = (len(psx),11))
Y = np.zeros(shape=(len(psx),19))
for i in range(len(psx)):
    Y[i][int(psy[i])] = 1
    for j in range(11):
        X[i][j] = np.float32(psx[i][j])

X = scale.transform(X)

print X,len(X)

with open('mainwts/1.txt','r') as f:
    mw1 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('mainwts/2.txt','r') as f:
    mw2 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('mainwts/3.txt','r') as f:
    mw3 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('mainwts/4.txt','r') as f:
    mw4 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('mainwts/5.txt','r') as f:
    mw5 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('mainwts/6.txt','r') as f:
    mw6 = np.genfromtxt(f,dtype=float,delimiter = '\t')


with open('mainbs/1.txt','r') as f:
    mb1 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('mainbs/2.txt','r') as f:
    mb2 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('mainbs/3.txt','r') as f:
    mb3 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('mainbs/4.txt','r') as f:
    mb4 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('mainbs/5.txt','r') as f:
    mb5 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('mainbs/6.txt','r') as f:
    mb6 = np.genfromtxt(f,dtype=float,delimiter = '\t')
'''
mb1 = np.reshape(mb1,[1,np.shape(mb1)[0]])
mb2 = np.reshape(mb2,[1,np.shape(mb2)[0]])
mb3 = np.reshape(mb3,[1,np.shape(mb3)[0]])
mb4 = np.reshape(mb4,[1,np.shape(mb4)[0]])
mb5 = np.reshape(mb5,[1,np.shape(mb5)[0]])
mb6 = np.reshape(mb6,[1,1])
mw6 = np.reshape(mw6,[105,1])

xi = tf.placeholder(tf.float32)

print mb1.shape
w1 = tf.Variable(np.float32(mw1))
w2 = tf.Variable(np.float32(mw2))
w3 = tf.Variable(np.float32(mw3))
w4 = tf.Variable(np.float32(mw4))
w5 = tf.Variable(np.float32(mw5))
w6 = tf.Variable(np.float32(mw6))
#w7 = tf.Variable(np.float32(mw7))
#w8 = tf.Variable(np.float32(mw8))

b1 = tf.Variable(np.float32(mb1))
b2 = tf.Variable(np.float32(mb2))
b3 = tf.Variable(np.float32(mb3))
b4 = tf.Variable(np.float32(mb4))
b5 = tf.Variable(np.float32(mb5))
b6 = tf.Variable(np.float32(mb6))
#b7 = tf.Variable(np.float32(mb7))
#b8 = tf.Variable(np.float32(mb8))

bl1 = tf.nn.relu(tf.matmul(xi,w1)+b1)
bl2 = tf.nn.relu(tf.matmul(bl1,w2)+b2)
bl3 = tf.nn.relu(tf.matmul(bl2,w3)+b3)
bl4 = tf.nn.relu(tf.matmul(bl3,w4)+b4)
bl5 = tf.nn.relu(tf.matmul(bl4,w5)+b5)
blo = tf.sigmoid(tf.matmul(bl5,w6)+b6)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    c= sess.run(blo,{xi:X})
    print c
'''
bl1 = relu(np.dot(X,mw1)+mb1)
bl2 = relu(np.dot(bl1,mw2)+mb2)
bl3 = relu(np.dot(bl2,mw3)+mb3)
bl4 = relu(np.dot(bl3,mw4)+mb4)
bl5 = relu(np.dot(bl4,mw5)+mb5)
blo = sigmoid(np.dot(bl5,mw6)+mb6)

c = np.floor(blo+0.5)

with open('12wts/2/1.txt','r') as f:
    fw1 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12wts/2/2.txt','r') as f:
    fw2 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12wts/2/3.txt','r') as f:
    fw3 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12wts/2/4.txt','r') as f:
    fw4 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12wts/2/5.txt','r') as f:
    fw5 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12wts/2/6.txt','r') as f:
    fw6 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12wts/2/7.txt','r') as f:
    fw7 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12wts/2/8.txt','r') as f:
    fw8 = np.genfromtxt(f,dtype=float,delimiter = '\t')

with open('12bs/2/1.txt','r') as f:
    fb1 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12bs/2/2.txt','r') as f:
    fb2 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12bs/2/3.txt','r') as f:
    fb3 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12bs/2/4.txt','r') as f:
    fb4 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12bs/2/5.txt','r') as f:
    fb5 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12bs/2/6.txt','r') as f:
    fb6 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12bs/2/7.txt','r') as f:
    fb7 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('12bs/2/8.txt','r') as f:
    fb8 = np.genfromtxt(f,dtype=float,delimiter = '\t')

with open('7wts/2/1.txt','r') as f:
    sw1 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7wts/2/2.txt','r') as f:
    sw2 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7wts/2/3.txt','r') as f:
    sw3 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7wts/2/4.txt','r') as f:
    sw4 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7wts/2/5.txt','r') as f:
    sw5 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7wts/2/6.txt','r') as f:
    sw6 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7wts/2/7.txt','r') as f:
    sw7 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7wts/2/8.txt','r') as f:
    sw8 = np.genfromtxt(f,dtype=float,delimiter = '\t')

with open('7bs/2/1.txt','r') as f:
    sb1 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7bs/2/2.txt','r') as f:
    sb2 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7bs/2/3.txt','r') as f:
    sb3 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7bs/2/4.txt','r') as f:
    sb4 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7bs/2/5.txt','r') as f:
    sb5 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7bs/2/6.txt','r') as f:
    sb6 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7bs/2/7.txt','r') as f:
    sb7 = np.genfromtxt(f,dtype=float,delimiter = '\t')
with open('7bs/2/8.txt','r') as f:
    sb8 = np.genfromtxt(f,dtype=float,delimiter = '\t')

fy = [0 for i in range(len(X))]
fy3 = [[0 for j in range(3)] for i in range(len(X))]
c12 = [0,3,4,5,6,9,10,12,13,15,16,18]
for i in range(len(X)):
    #print c[i], psy[i] in c12
    if c[i]==0:
        fl1 = relu(np.dot(X[i],fw1)+fb1)
	fl2 = relu(np.dot(fl1,fw2)+fb2)
	fl3 = relu(np.dot(fl2,fw3)+fb3)
	fl4 = relu(np.dot(fl3,fw4)+fb4)
	fl5 = relu(np.dot(fl4,fw5)+fb5)
	fl6 = relu(np.dot(fl5,fw6)+fb6)
	fl7 = relu(np.dot(fl6,fw7)+fb7)
	flo = softmax(np.dot(fl7,fw8)+fb8)
    else:
        fl1 = relu(np.dot(X[i],sw1)+sb1)
	fl2 = relu(np.dot(fl1,sw2)+sb2)
	fl3 = relu(np.dot(fl2,sw3)+sb3)
	fl4 = relu(np.dot(fl3,sw4)+sb4)
	fl5 = relu(np.dot(fl4,sw5)+sb5)
	fl6 = relu(np.dot(fl5,sw6)+sb6)
	fl7 = relu(np.dot(fl6,sw7)+sb7)
	flo = softmax(np.dot(fl7,sw8)+sb8)
    fy[i] = np.argmax(flo)
    fy3[i] = np.argsort(flo)[16:19]
    #print fy3[i],psy[i],(psy[i] in fy3[i])
ct = 0
ct3 = 0
total=0
for i in range(len(X)):
    if (psy[i]==fy[i]):# and (psy[i] not in c12):
        ct+=1
    if psy[i] not in c12:
	total+=1
    if (psy[i] in fy3[i]):# and(psy[i] not in c12):
        ct3+=1
print ct*100/float(len(X))
print ct3*100/float(len(X))
cacc=0
for i in range(len(X)):
    if (psy[i] in c12) and (c[i]==0):
	cacc+=1
    if(psy[i] not in c12) and (c[i]==1):
	cacc+=1
print cacc/float(len(X))
