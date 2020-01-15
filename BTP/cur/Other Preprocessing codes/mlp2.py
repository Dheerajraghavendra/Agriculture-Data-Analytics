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
Y = np.zeros(shape=(len(x1)))
for i in range(len(x1)):
    #[i][int(x1[i][11])-1] = 1
    Y[i] = int(x1[i][11])-1
    for j in range(11):
        X[i][j] = np.float32(x1[i][j])

Xtest = np.ones(shape = (len(x2),11))
Ytest = np.zeros(shape=(len(x2)))
for i in range(len(x2)):
    #Ytest[i][int(x2[i][11])-1] = 1
    Ytest[i] = int(x2[i][11]) -1
    for j in range(11):
        Xtest[i][j] = np.float32(x2[i][j])

X = scale.transform(X)
Xtest = scale.transform(Xtest)
print len(Ytest)
ni = 11
nh = 5
no =  18
nh1 = 45
nh2 = 55
nh3 = 45
nh4 = 55
nh5 = 45

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.int32)
np.random.seed(0)
wts = [[] for _ in range(nh+1)]
wts[0] = np.zeros(shape=(11,nh1))+2*np.random.random((ni,nh1))-1
wts[1] = np.zeros(shape=(nh1,nh2))+2*np.random.random((nh1,nh2))-1
wts[2] = np.zeros(shape=(nh2,nh3))+2*np.random.random((nh2,nh3))-1
wts[3] = np.zeros(shape=(nh3,nh4))+2*np.random.random((nh3,nh4))-1
wts[4] = np.zeros(shape=(nh4,nh5))+2*np.random.random((nh4,nh5))-1
wts[5] = np.zeros(shape=(nh5,no))+2*np.random.random((nh5,no))-1

bs = [[] for _ in range(nh+1)]
bs[0] = np.zeros(shape=(1,nh1))-1#-2*np.random.random((1,nh1))+1
bs[1] = np.zeros(shape=(1,nh2))-1#-2*np.random.random((1,nh2))+1
bs[2] = np.zeros(shape=(1,nh3))-1#-2*np.random.random((1,nh3))+1
bs[3] = np.zeros(shape=(1,nh4))-1#-2*np.random.random((1,nh4))+1
bs[4] = np.zeros(shape=(1,nh5))-1#-2*np.random.random((1,nh5))+1
bs[5] = np.zeros(shape=(1,no))-1#-2*np.random.random((1,no))+1



w1 = tf.Variable(np.float32(wts[0]))
w2 = tf.Variable(np.float32(wts[1]))
w3 = tf.Variable(np.float32(wts[2]))
w4 = tf.Variable(np.float32(wts[3]))
w5 = tf.Variable(np.float32(wts[4]))
w6 = tf.Variable(np.float32(wts[5]))

b1 = tf.Variable(np.float32(bs[0]))
b2 = tf.Variable(np.float32(bs[1]))
b3 = tf.Variable(np.float32(bs[2]))
b4 = tf.Variable(np.float32(bs[3]))
b5 = tf.Variable(np.float32(bs[4]))
b6 = tf.Variable(np.float32(bs[5]))

learning_rate = 0.1
epochs = 1000
l1 = tf.nn.relu(tf.matmul(x,w1)+b1)
l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)
l3 = tf.nn.relu(tf.matmul(l2,w3)+b3)
l4 = tf.nn.relu(tf.matmul(l3,w4)+b4)
l5 = tf.nn.relu(tf.matmul(l4,w5)+b5)
#lo = tf.nn.softmax(tf.matmul(l5,w6)+b6)
lo = tf.matmul(l5,w6)+b6;

#cost = tf.losses.hinge_loss(y,lo)
#cost = tf.reduce_sum(tf.square(y-lo))
#cost = -tf.reduce_mean(y*tf.log(lo+1e-10))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=lo))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(epochs):
        sess.run(optimizer,feed_dict = {x:X, y:Y})

        if step%100 == 0:
            print sess.run(cost,feed_dict={x:X, y:Y})
    ans = tf.equal(tf.floor(lo),np.float32(y))
    tmp = tf.cast(ans,"float")
    #tmp1 = tf.reduce_min(tmp,axis=1)
    acc = tf.reduce_mean(tmp)
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
    print acc.eval({x:X,y:Y})

    print Ytest
    print sess.run(lo,feed_dict={x:Xtest,y:Ytest})
    print acc.eval({x:Xtest,y:Ytest})
    #print sess.run(w1)
    #print sess.run(w2)
    #print sess.run(w3)
    #print sess.run(w4)
    #print sess.run(w5)
    #print sess.run(w6)
