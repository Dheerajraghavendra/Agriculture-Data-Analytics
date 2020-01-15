#!/usr/bin/python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
def lrelu(x):
    return tf.maximum(x,0.2*x)
    
scale = MinMaxScaler()
x1=[]
x2=[]
d= []
y1=[]
y2=[]
strain = [0 for i in range(12)]
stest = [0 for i in range(12	)]
matrix=[0,3,4,5,6,7,8,9,10,11,12,13]
for i in range(12):
    with open('tdata/'+str(matrix[i])+'.txt','r') as f:
        tmp = np.genfromtxt(f,dtype=None,delimiter='\t')
    f = int(0.7*len(tmp))
    for j in range(f):
        x1.append(tmp[j])
        d.append(tmp[j][:11])
        y1.append(tmp[j][11])
        strain[i]+=1
    for j in range(f,len(tmp)):
        x2.append(tmp[j])
        d.append(tmp[j][:11])
        y2.append(tmp[j][11])
        stest[i]+=1
print(scale.fit(d))
print(scale.data_max_)
print(scale.data_min_)
X = np.ones(shape = (len(x1),11))
Y = np.zeros(shape=(len(x1),12))
for i in range(len(x1)):
    if int(y1[i])==1:
    	Y[i][int(y1[i])-1] = 1
    else:
    	Y[i][int(y1[i])-3]=1
    for j in range(11):
        X[i][j] = np.float32(x1[i][j])
Xtest = np.ones(shape = (len(x2),11))
Ytest = np.zeros(shape=(len(x2),12))
for i in range(len(x2)):
    if int(y2[i])==1:
    	Ytest[i][int(y2[i])-1] = 1
    else:
	Ytest[i][int(y2[i])-3] =1
    for j in range(11):
        Xtest[i][j] = np.float32(x2[i][j])
X = scale.transform(X)
Xtest = scale.transform(Xtest)
print len(Ytest)
ni = 11
nh = 5
no =  12
nh1 = 75
nh2 = 75
nh3 = 75
nh4 = 75
nh5 = 75
nh6 = 75
nh7 = 75
nh8 = 75
nh9 = 75
nh10 = 75
nh11 = 75
nh12 = 75
nh13=75
nh14=75
nh15=75
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
np.random.seed(0)
winit = tf.contrib.layers.xavier_initializer(seed=0)
wts = [[] for _ in range(nh+1)]

wts[0] = np.zeros(shape=(ni,nh1))+2*np.random.random((ni,nh1))-1
wts[1] = np.zeros(shape=(nh1,nh2))+2*np.random.random((nh1,nh2))-1
wts[2] = np.zeros(shape=(nh2,nh3))+2*np.random.random((nh2,nh3))-1
wts[3] = np.zeros(shape=(nh3,nh4))+2*np.random.random((nh3,nh4))-1
wts[4] = np.zeros(shape=(nh4,nh5))+2*np.random.random((nh4,nh5))-1
wts[5] = np.zeros(shape=(nh5,no))+2*np.random.random((nh5,no))-1
#wts[6] = np.zeros(shape=(nh6,nh7))+2*np.random.random((nh6,nh7))-1
#wts[7] = np.zeros(shape=(nh7,nh8))+2*np.random.random((nh7,nh8))-1
#wts[8] = np.zeros(shape=(nh8,nh9))+2*np.random.random((nh8,nh9))-1
#wts[9] = np.zeros(shape=(nh9,nh10))+2*np.random.random((nh9,nh10))-1
#wts[10] = np.zeros(shape=(nh10,nh11))+2*np.random.random((nh10,nh11))-1
#wts[11] = np.zeros(shape=(nh11,nh12))+2*np.random.random((nh11,nh12))-1
#wts[12] = np.zeros(shape=(nh12,nh13))+2*np.random.random((nh12,nh13))-1
#wts[13] = np.zeros(shape=(nh13,nh14))+2*np.random.random((nh13,nh14))-1
#wts[14] = np.zeros(shape=(nh14,nh15))+2*np.random.random((nh14,nh15))-1
#wts[15] = np.zeros(shape=(nh15,no))+2*np.random.random((nh15,no))-1


bs = [[] for _ in range(nh+1)]
bs[0] = np.zeros(shape=(1,nh1))#+2*np.random.random((1,nh1))-1+1
bs[1] = np.zeros(shape=(1,nh2))#+2*np.random.random((1,nh2))-1+1
bs[2] = np.zeros(shape=(1,nh3))#+2*np.random.random((1,nh3))-1+1
bs[3] = np.zeros(shape=(1,nh4))#+2*np.random.random((1,nh4))-1+1
bs[4] = np.zeros(shape=(1,nh5))#+2*np.random.random((1,nh5))-1+1
bs[5] = np.zeros(shape=(1,no))#+2*np.random.random((1,nh6))-1+1
#bs[6] = np.zeros(shape=(1,nh7))#+2*np.random.random((1,no))-1+1
#bs[7] = np.zeros(shape=(1,nh8))#+2*np.random.random((1,no))-1+1
#bs[8] = np.zeros(shape=(1,nh9))#+2*np.random.random((1,no))-1+1
#bs[9] = np.zeros(shape=(1,nh10))#-1+2*np.random.random((1,no))-1+1
#bs[10] = np.zeros(shape=(1,nh11))#-1+2*np.random.random((1,no))-1+1
#bs[11] = np.zeros(shape=(1,nh12))#-1+2*np.random.random((1,no))-1+1
#bs[12] = np.zeros(shape=(1,nh13))#-1+2*np.random.random((1,no))-1+1
#bs[13] = np.zeros(shape=(1,nh14))#-1+2*np.random.random((1,no))-1+1
#bs[14] = np.zeros(shape=(1,nh15))#-1+2*np.random.random((1,no))-1+1
#bs[15] = np.zeros(shape=(1,no))#-1+2*np.random.random((1,no))-1+1


'''
w1 = tf.Variable(np.float32(wts[0]))
w2 = tf.Variable(np.float32(wts[1]))
w3 = tf.Variable(np.float32(wts[2]))
w4 = tf.Variable(np.float32(wts[3]))
w5 = tf.Variable(np.float32(wts[4]))
w6 = tf.Variable(np.float32(wts[5]))
#w7 = tf.Variable(np.float32(wts[6]))
#w8 = tf.Variable(np.float32(wts[7]))
#w9 = tf.Variable(np.float32(wts[8]))
'''
w1 = tf.Variable(winit([ni,nh1]))
w2 = tf.Variable(winit([nh1,nh2]))
w3 = tf.Variable(winit([nh2,nh3]))
w4 = tf.Variable(winit([nh3,nh4]))
w5 = tf.Variable(winit([nh4,nh5]))
w6 = tf.Variable(winit([nh5,no]))
#w7 = tf.Variable(winit([nh6,nh7]))
#w8 = tf.Variable(winit([nh7,nh8]))
#w9 = tf.Variable(winit([nh8,nh9]))
#w10 = tf.Variable(winit([nh9,nh10]))
#w11 = tf.Variable(winit([nh10,nh11]))
#w12 = tf.Variable(winit([nh11,nh12]))
#w13 = tf.Variable(winit([nh12,nh13]))
#w14 = tf.Variable(winit([nh13,nh14]))
#w15 = tf.Variable(winit([nh14,nh15]))
#w16 = tf.Variable(winit([nh15,no]))


b1 = tf.Variable(np.float32(bs[0]))
b2 = tf.Variable(np.float32(bs[1]))
b3 = tf.Variable(np.float32(bs[2]))
b4 = tf.Variable(np.float32(bs[3]))
b5 = tf.Variable(np.float32(bs[4]))
b6 = tf.Variable(np.float32(bs[5]))
#b7 = tf.Variable(np.float32(bs[6]))
#b8 = tf.Variable(np.float32(bs[7]))
#b9 = tf.Variable(np.float32(bs[8]))
#b10 = tf.Variable(np.float32(bs[9]))
#b11 = tf.Variable(np.float32(bs[10]))
#b12 = tf.Variable(np.float32(bs[11]))
#b13 = tf.Variable(np.float32(bs[12]))
#b14 = tf.Variable(np.float32(bs[13]))
#b15 = tf.Variable(np.float32(bs[14]))
#b16 = tf.Variable(np.float32(bs[15]))




learning_rate = 1
epochs = 10000
l1 = tf.nn.relu(tf.matmul(x,w1)+b1)
l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)
l3 = tf.nn.relu(tf.matmul(l2,w3)+b3)
l4 = tf.nn.relu(tf.matmul(l3,w4)+b4)
l5 = tf.nn.relu(tf.matmul(l4,w5)+b5)
lo = tf.nn.softmax(tf.matmul(l5,w6)+b6)
#l6 = tf.nn.relu(tf.matmul(l5,w6)+b6))
#l7 = tf.nn.relu(tf.matmul(l6,w7)+b7)
#l8 = tf.nn.relu(tf.matmul(l7,w8)+b8)
#l9 = tf.nn.relu(tf.matmul(l8,w9)+b9)
#l10 = tf.nn.relu(tf.matmul(l9,w10)+b10)
#l11 = tf.nn.relu(tf.matmul(l10,w11)+b11)
#l12 = tf.nn.relu(tf.matmul(l11,w12)+b12)
#l13 = tf.nn.relu(tf.matmul(l12,w13)+b13)
#l14 = tf.nn.relu(tf.matmul(l13,w14)+b14)
#l15 = tf.nn.relu(tf.matmul(l14,w15)+b15)
#lo = tf.nn.softmax(tf.matmul(l15,w16)+b16)
#lo = tf.matmul(l5,w6)+b6;

r1 = tf.nn.l2_loss(w1)+tf.nn.l2_loss(b1)
r2 = tf.nn.l2_loss(w2)+tf.nn.l2_loss(b2)
r3 = tf.nn.l2_loss(w3)+tf.nn.l2_loss(b3)
r4 = tf.nn.l2_loss(w4)+tf.nn.l2_loss(b4)
r5 = tf.nn.l2_loss(w5)+tf.nn.l2_loss(b5)
r6 = tf.nn.l2_loss(w6)+tf.nn.l2_loss(b6)
#r7 = tf.nn.l2_loss(w7)+tf.nn.l2_loss(b7)
#r8 = tf.nn.l2_loss(w8)+tf.nn.l2_loss(b8)
#r9 = tf.nn.l2_loss(w9)+tf.nn.l2_loss(b9)
#r10 = tf.nn.l2_loss(w10)+tf.nn.l2_loss(b10)
#r11 = tf.nn.l2_loss(w11)+tf.nn.l2_loss(b11)
#r12 = tf.nn.l2_loss(w12)+tf.nn.l2_loss(b12)
#r13 = tf.nn.l2_loss(w13)+tf.nn.l2_loss(b13)
#r14 = tf.nn.l2_loss(w14)+tf.nn.l2_loss(b14)
#r15 = tf.nn.l2_loss(w15)+tf.nn.l2_loss(b15)
#r16 = tf.nn.l2_loss(w16)+tf.nn.l2_loss(b16)

r = r1+r2+r3+r4+r5+r6
beta = 0.000001
#cost = tf.losses.hinge_loss(y,lo)
#cost = tf.reduce_sum(tf.square(y-lo))
cost = -tf.reduce_mean(y*tf.log(lo+1e-10))+(beta*r)
#cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=lo)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

batchsize = np.shape(X)[0]
fac = np.shape(X)[0]%batchsize!=0
nbatches = np.shape(X)[0]/batchsize-fac
with tf.Session() as sess:
    sess.run(init)
    #print sess.run([w1,w8])
    for step in range(epochs):
        i_batch = (step%nbatches)*batchsize
        batch = X[i_batch:i_batch+batchsize],Y[i_batch:i_batch+batchsize] 
        sess.run(optimizer,feed_dict = {x:batch[0], y:batch[1]})

        if step%100 == 0:
            print sess.run(cost,feed_dict={x:X, y:Y}),step/100
    ans = tf.equal(tf.argmax(lo,axis=1),tf.argmax(y,axis=1))
    tmp0 = tf.cast(ans,"float")
    #tmp1 = tf.reduce_min(tmp,axis=1)
    #sm = tf.reduce_sum(tmp);
    acc = tf.reduce_mean(tmp0)
    ans2 = tf.nn.top_k(lo,3).indices
    ans2 = tf.to_int64(ans2)
    y1 = tf.argmax(y,axis=1)
    y1 = tf.reshape(y1,[tf.shape(y1)[0],1])
    tmp2 = tf.equal(ans2,y1)
    tmp2 = tf.cast(tmp2,"float")
    sm2 = tf.reduce_max(tmp2,axis=1)
    acc2 = tf.reduce_mean(sm2)
    print sess.run(lo,feed_dict={x:X,y:Y})
    print "this"
    print tmp0.eval({x:X,y:Y})
    print "over"
    print "train accuracy: ",acc.eval({x:X,y:Y})*100, " %" 
    print "train3 accurac: ",acc2.eval({x:X,y:Y})*100," %"
    act = np.argmax(Y,axis=1)+1
    #act= tf.reshape(act,[tf.shape(act)[0],1])
    cl1 = act*(np.int64(tmp0.eval({x:X,y:Y})))
    cl3 = act*(np.int64(sm2.eval({x:X,y:Y})))
    teval = [0 for i in range(12)]
    teval3 = [0 for i in range(12)]
    for i in range(len(cl1)):
        if cl1[i]>0:
            teval[cl1[i]-1]+=1
        if cl3[i]>0:
            teval3[cl3[i]-1]+=1
    print "Classwise train accuraices"
    for i in range(12):
	print teval[i]/float(strain[i]),"  and  ",teval3[i]/float(strain[i])
    print Ytest
    print sess.run(lo,feed_dict={x:Xtest,y:Ytest})
    print "test accuracy: ",acc.eval({x:Xtest,y:Ytest})*100," %"
    print sess.run(tmp2,feed_dict={x:Xtest,y:Ytest})
    print "test3 accurac: ",acc2.eval({x:Xtest,y:Ytest})*100," %"
    act = np.argmax(Ytest,axis=1)+1    
    #print act*sm2.eval({x:Xtest,y:Ytest})
    clt1 = act*(np.int64(tmp0.eval({x:Xtest,y:Ytest})))
    clt3 = act*(np.int64(sm2.eval({x:Xtest,y:Ytest})))
    print act
    print clt3
    tsteval = [0 for i in range(12)]
    tsteval3 = [0 for i in range(12)]
    for i in range(len(clt1)):
        if clt1[i]>0:
            tsteval[clt1[i]-1]+=1
        if clt3[i]>0:
            tsteval3[clt3[i]-1]+=1
    print "Classwise test accuraices"
    for i in range(12):
        print tsteval[i]/float(stest[i]),"  and  ",tsteval3[i]/float(stest[i])
    #print sess.run(w1)
    #print sess.run(w2)
    #print sess.run(w3)
    #print sess.run(w4)
    #print sess.run(w5)
    #print sess.run(w6)
