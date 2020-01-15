#!/usr/bin/python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
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

print(scale.fit(d))
print(scale.data_max_)
print(scale.data_min_)
X = np.ones(shape = (len(x1),11))
Y = np.zeros(shape=(len(x1),18))
for i in range(len(x1)):
    Y[i][int(x1[i][11])-1] = 1
    #Y[i] = int(x1[i][11])-1
    for j in range(11):
        X[i][j] = np.float32(x1[i][j])

Xtest = np.ones(shape = (len(x2),11))
Ytest = np.zeros(shape=(len(x2),18))
for i in range(len(x2)):
    Ytest[i][int(x2[i][11])-1] = 1
    #Ytest[i] = int(x2[i][11]) -1
    for j in range(11):
        Xtest[i][j] = np.float32(x2[i][j])

X = scale.transform(X)
Xtest = scale.transform(Xtest)
print len(Ytest)
ni = 11
nh = 14
no =  18
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
nh13 = 75
nh14 = 75

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
np.random.seed(0)
wts = [[] for _ in range(nh+1)]
wts[0] = np.zeros(shape=(11,nh1))+2*np.random.random((ni,nh1))-1
wts[1] = np.zeros(shape=(nh1,nh2))+2*np.random.random((nh1,nh2))-1
wts[2] = np.zeros(shape=(nh2,nh3))+2*np.random.random((nh2,nh3))-1
wts[3] = np.zeros(shape=(nh3,nh4))+2*np.random.random((nh3,nh4))-1
wts[4] = np.zeros(shape=(nh4,nh5))+2*np.random.random((nh4,nh5))-1
wts[5] = np.zeros(shape=(nh5,nh6))+2*np.random.random((nh5,nh6))-1
wts[6] = np.zeros(shape=(nh6,nh7))+2*np.random.random((nh6,nh7))-1
wts[7] = np.zeros(shape=(nh7,nh8))+2*np.random.random((nh7,nh8))-1
wts[8] = np.zeros(shape=(nh8,nh9))+2*np.random.random((nh8,nh9))-1
wts[9] = np.zeros(shape=(nh9,nh10))+2*np.random.random((nh9,nh10))-1
wts[10] = np.zeros(shape=(nh10,nh11))+2*np.random.random((nh10,nh11))-1
wts[11] = np.zeros(shape=(nh11,nh12))+2*np.random.random((nh11,nh12))-1
wts[12] = np.zeros(shape=(nh12,nh13))+2*np.random.random((nh12,nh13))-1
wts[13] = np.zeros(shape=(nh13,nh14))+2*np.random.random((nh13,nh14))-1
wts[14] = np.zeros(shape=(nh14,no))+2*np.random.random((nh14,no))-1

bs = [[] for _ in range(nh+1)]
bs[0] = np.zeros(shape=(1,nh1))-1#-2*np.random.random((1,nh1))+1
bs[1] = np.zeros(shape=(1,nh2))-1#-2*np.random.random((1,nh2))+1
bs[2] = np.zeros(shape=(1,nh3))-1#-2*np.random.random((1,nh3))+1
bs[3] = np.zeros(shape=(1,nh4))-1#-2*np.random.random((1,nh4))+1
bs[4] = np.zeros(shape=(1,nh5))-1#-2*np.random.random((1,nh5))+1
bs[5] = np.zeros(shape=(1,nh6))-1#-2*np.random.random((1,nh6))+1
bs[6] = np.zeros(shape=(1,nh7))-1#-2*np.random.random((1,no))+1
bs[7] = np.zeros(shape=(1,nh8))-1#-2*np.random.random((1,no))+1
bs[8] = np.zeros(shape=(1,nh9))-1#-2*np.random.random((1,no))+1
bs[9] = np.zeros(shape=(1,nh10))-1#-2*np.random.random((1,no))+1
bs[10] = np.zeros(shape=(1,nh11))-1#-2*np.random.random((1,no))+1
bs[11] = np.zeros(shape=(1,nh12))-1#-2*np.random.random((1,no))+1
bs[12] = np.zeros(shape=(1,nh13))-1#-2*np.random.random((1,no))+1
bs[13] = np.zeros(shape=(1,nh14))-1#-2*np.random.random((1,no))+1
bs[14] = np.zeros(shape=(1,no))-1#-2*np.random.random((1,no))+1



w1 = tf.Variable(np.float32(wts[0]))
w2 = tf.Variable(np.float32(wts[1]))
w3 = tf.Variable(np.float32(wts[2]))
w4 = tf.Variable(np.float32(wts[3]))
w5 = tf.Variable(np.float32(wts[4]))
w6 = tf.Variable(np.float32(wts[5]))
w7 = tf.Variable(np.float32(wts[6]))
w8 = tf.Variable(np.float32(wts[7]))
w9 = tf.Variable(np.float32(wts[8]))
w10 = tf.Variable(np.float32(wts[9]))
w11 = tf.Variable(np.float32(wts[10]))
w12 = tf.Variable(np.float32(wts[11]))
w13 = tf.Variable(np.float32(wts[12]))
w14 = tf.Variable(np.float32(wts[13]))
w15 = tf.Variable(np.float32(wts[14]))

b1 = tf.Variable(np.float32(bs[0]))
b2 = tf.Variable(np.float32(bs[1]))
b3 = tf.Variable(np.float32(bs[2]))
b4 = tf.Variable(np.float32(bs[3]))
b5 = tf.Variable(np.float32(bs[4]))
b6 = tf.Variable(np.float32(bs[5]))
b7 = tf.Variable(np.float32(bs[6]))
b8 = tf.Variable(np.float32(bs[7]))
b9 = tf.Variable(np.float32(bs[8]))
b10 = tf.Variable(np.float32(bs[9]))
b11 = tf.Variable(np.float32(bs[10]))
b12 = tf.Variable(np.float32(bs[11]))
b13 = tf.Variable(np.float32(bs[12]))
b14 = tf.Variable(np.float32(bs[13]))
b15 = tf.Variable(np.float32(bs[14]))


learning_rate = 0.1
epochs = 5000
l1 = tf.nn.softmax(tf.matmul(x,w1)+b1)
l2 = tf.nn.softmax(tf.matmul(l1,w2)+b2)
l3 = tf.nn.softmax(tf.matmul(l2,w3)+b3)
l4 = tf.nn.softmax(tf.matmul(l3,w4)+b4)
l5 = tf.nn.softmax(tf.matmul(l4,w5)+b5)
l6 = tf.nn.softmax(tf.matmul(l5,w6)+b6)
l7 = tf.nn.softmax(tf.matmul(l6,w7)+b7)
l8 = tf.nn.softmax(tf.matmul(l7,w8)+b8)
l9 = tf.nn.softmax(tf.matmul(l8,w9)+b9)
l10 = tf.nn.softmax(tf.matmul(l9,w10)+b10)
l11 = tf.nn.softmax(tf.matmul(l10,w11)+b11)
l12 = tf.nn.softmax(tf.matmul(l11,w12)+b12)
l13 = tf.nn.softmax(tf.matmul(l12,w13)+b13)
l14 = tf.nn.softmax(tf.matmul(l13,w14)+b14)
lo = tf.nn.softmax(tf.matmul(l14,w15)+b15)
#lo = tf.matmul(l5,w6)+b6;

r1 = tf.nn.l2_loss(w1)+tf.nn.l2_loss(b1)
r2 = tf.nn.l2_loss(w2)+tf.nn.l2_loss(b2)
r3 = tf.nn.l2_loss(w3)+tf.nn.l2_loss(b3)
r4 = tf.nn.l2_loss(w4)+tf.nn.l2_loss(b4)
r5 = tf.nn.l2_loss(w5)+tf.nn.l2_loss(b5)
r6 = tf.nn.l2_loss(w6)+tf.nn.l2_loss(b6)
r = r1+r2+r3+r4+r5+r6
beta = 0
#cost = tf.losses.hinge_loss(y,lo)
#cost = tf.reduce_sum(tf.square(y-lo))
cost = -tf.reduce_mean(y*tf.log(lo+1e-10))+(beta*r)
#cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=lo)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(epochs):
        sess.run(optimizer,feed_dict = {x:X, y:Y}),step/100

        if step%1000 == 0:
            print sess.run(cost,feed_dict={x:X, y:Y})
    ans = tf.equal(tf.argmax(lo,axis=1),tf.argmax(y,axis=1))
    tmp = tf.cast(ans,"float")
    #tmp1 = tf.reduce_min(tmp,axis=1)
    acc = tf.reduce_mean(tmp)
    ans2 = tf.nn.top_k(lo,3).indices
    ans2 = tf.to_int64(ans2)
    y1 = tf.argmax(y,axis=1)
    y1 = tf.reshape(y1,[tf.shape(y1)[0],1])
    tmp2 = tf.equal(ans2,y1)
    tmp2 = tf.cast(tmp2,'float')
    sm2 = tf.reduce_max(tmp2,axis=1)
    acc2 = tf.reduce_mean(sm2)
#    testl1 = tf.nn.softmax(tf.matmul(Xtest,w1)+b1)
#    testl2 = tf.nn.softmax(tf.matmul(l1,w2)+b2)
#    testl3 = tf.nn.softmax(tf.matmul(l2,w3)+b3)
#    testl4 = tf.nn.softmax(tf.matmul(l3,w4)+b4)
#    testl5 = tf.nn.softmax(tf.matmul(l4,w5)+b5)
#    testlo = tf.nn.softmax(tf.matmul(l5,w6)+b6)    
    print sess.run(lo,feed_dict={x:X,y:Y})
    print "this"
    print tmp.eval({x:X,y:Y})
    print "over"
    print "train accuracy: ",acc.eval({x:X,y:Y})*100, " %" 
    print "train3 accuracy: ",acc2.eval({x:X,y:Y})*100, " %"

    print Ytest
    print sess.run(lo,feed_dict={x:Xtest,y:Ytest})
    print "test accuracy: ",acc.eval({x:Xtest,y:Ytest})*100," %"
    print sess.run(lo,feed_dict={x:Xtest,y:Ytest})
    print "test3 accuracy: ",acc2.eval({x:Xtest,y:Ytest})*100," %"
