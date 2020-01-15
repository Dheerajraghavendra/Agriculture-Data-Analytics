#!/usr/bin/python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
def lrelu(x):
    return tf.maximum(x,0.1*x)
    
scale = MinMaxScaler()

psx = []
psy=[]
z1=[]
z2=[]
z3=[]
tx1=[]
tx2=[]
tx3=[]
ty1=[]
ty2=[]
ty3=[]

c12 = [0,3,4,5,6,9,10,12,13,15,16,18]
c7 = [1,2,7,8,11,14,17]

for i in range(19):
    with open('tdata/'+str(i)+'.txt','r') as f:
        tmp = np.genfromtxt(f,dtype=None,delimiter='\t')
    f1 = int(0.33*len(tmp))
    for j in range(f1):
        tx1.append(tmp[j][:11])
        ty1.append(i)
        psx.append(tmp[j][:11])
        psy.append(i)
	if i in c12:
	    z1.append(0)
	else:
	    z1.append(1)
    f2 = int(0.66*len(tmp))
    for j in range(f1,f2):
        tx2.append(tmp[j][:11])
        ty2.append(i)
        psx.append(tmp[j][:11])
        psy.append(i)
	if i in c12:
	    z2.append(0)
	else:
	    z2.append(1)
    for j in range(f2,len(tmp)):
        tx3.append(tmp[j][:11])
        ty3.append(i)
        psx.append(tmp[j][:11])
        psy.append(i)
	if i in c12:
	    z3.append(0)
	else:
	    z3.append(1)
xtrain = [0 for i in range(3)]
xtest = [0 for i in range(3)]
ytrain = [0 for i in range(3)]
ytest = [0 for i in range(3)]
zt = [0 for i in range(3)]
ztest = [0 for i in range(3)]
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
zt[0] = np.concatenate((z1,z2))
zt[0] = np.reshape(zt[0],[np.shape(zt[0])[0],1])
ztest[0] = z3
ztest[0] = np.reshape(ztest[0],[np.shape(ztest[0])[0],1])
zt[1] = np.concatenate((z1,z3))
zt[1] = np.reshape(zt[1],[np.shape(zt[1])[0],1])
ztest[1] = z2
ztest[1] = np.reshape(ztest[1],[np.shape(ztest[1])[0],1])
zt[2] = np.concatenate((z2,z3))
zt[2] = np.reshape(zt[2],[np.shape(zt[2])[0],1])
ztest[2] = z1
ztest[2] = np.reshape(ztest[2],[np.shape(ztest[2])[0],1])


print len(psx)
print(scale.fit(psx))
print(scale.data_max_)
print(scale.data_min_)
X = [0 for i in range(3)]
Xtest = [0 for i in range(3)]
Y = [0 for i in range(3)]
Ytest = [0 for i in range(3)]
for k in range(3):
    X[k] = np.ones(shape = (len(xtrain[k]),11))
    Y[k] = np.zeros(shape=(len(xtrain[k]),19))
    for i in range(len(xtrain[k])):
        Y[k][i][int(ytrain[k][i])] = 1
        for j in range(11):
            X[k][i][j] = np.float32(xtrain[k][i][j])
    Xtest[k] = np.ones(shape = (len(xtest[k]),11))
    Ytest[k] = np.zeros(shape=(len(xtest[k]),19))
    for i in range(len(xtest[k])):
        Ytest[k][i][int(ytest[k][i])] = 1
        for j in range(11):
            Xtest[k][i][j] = np.float32(xtest[k][i][j])
    X[k] = scale.transform(X[k])
    Xtest[k] = scale.transform(Xtest[k])
    print len(Ytest[k])

for kpart in range(3):
    ni = 11
    nh = 5
    no = 1
    nh1 = 105
    nh2 = 105
    nh3 = 105
    nh4 = 105
    nh5 = 105
    #nh6 = 5
    #nh7 = 5
    x = tf.placeholder(tf.float32)
    z= tf.placeholder(tf.float32)
    np.random.seed(0)
    winit = tf.contrib.layers.xavier_initializer(seed=0)

    bs = [[] for _ in range(nh+1)]

    bs[0] = np.zeros(shape=(1,nh1))
    bs[1] = np.zeros(shape=(1,nh2))
    bs[2] = np.zeros(shape=(1,nh3))
    bs[3] = np.zeros(shape=(1,nh4))
    bs[4] = np.zeros(shape=(1,nh5))
    bs[5] = np.zeros(shape=(1,no))
    #bs[6] = np.zeros(shape=(1,nh7))
    #bs[7] = np.zeros(shape=(1,no))


    w1 = tf.Variable(winit([ni,nh1]))
    w2 = tf.Variable(winit([nh1,nh2]))
    w3 = tf.Variable(winit([nh2,nh3]))
    w4 = tf.Variable(winit([nh3,nh4]))
    w5 = tf.Variable(winit([nh4,nh5]))
    w6 = tf.Variable(winit([nh5,no]))
    #w7 = tf.Variable(winit([nh6,nh7]))
    #w8 = tf.Variable(winit([nh7,no]))

    b1 = tf.Variable(np.float32(bs[0]))
    b2 = tf.Variable(np.float32(bs[1]))
    b3 = tf.Variable(np.float32(bs[2]))
    b4 = tf.Variable(np.float32(bs[3]))
    b5 = tf.Variable(np.float32(bs[4]))
    b6 = tf.Variable(np.float32(bs[5]))
    #b7 = tf.Variable(np.float32(bs[6]))
    #b8 = tf.Variable(np.float32(bs[7]))

    learning_rate = 0.1
    epochs = 10000
    l1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)
    l3 = tf.nn.relu(tf.matmul(l2,w3)+b3)
    l4 = tf.nn.relu(tf.matmul(l3,w4)+b4)
    l5 = tf.nn.relu(tf.matmul(l4,w5)+b5)
    #l6 = tf.nn.relu(tf.matmul(l5,w6)+b6)
    #l7 = tf.nn.relu(tf.matmul(l6,w7)+b7)
    lo = tf.sigmoid(tf.matmul(l5,w6)+b6)

    r1 = tf.nn.l2_loss(w1)+tf.nn.l2_loss(b1)
    r2 = tf.nn.l2_loss(w2)+tf.nn.l2_loss(b2)
    r3 = tf.nn.l2_loss(w3)+tf.nn.l2_loss(b3)
    r4 = tf.nn.l2_loss(w4)+tf.nn.l2_loss(b4)
    r5 = tf.nn.l2_loss(w5)+tf.nn.l2_loss(b5)
    r6 = tf.nn.l2_loss(w6)+tf.nn.l2_loss(b6)
    #r7 = tf.nn.l2_loss(w7)+tf.nn.l2_loss(b7)
    #r8 = tf.nn.l2_loss(w8)+tf.nn.l2_loss(b8)

    r = r1+r2+r3+r4+r5+r6#+r7+r8
    beta = 0#0.000001
    cost2 = -tf.reduce_mean(z*tf.log(lo+1e-10)+(1-z)*tf.log(1-lo+1e-10))
    cost = cost2+(beta*r)
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    

    with tf.Session() as sess:
        sess.run(init)
        for step in range(epochs): 
            sess.run(optimizer,feed_dict = {x:X[kpart], z:zt[kpart]})

            if step%100 == 0:
                print sess.run(cost,feed_dict={x:X[kpart], z:zt[kpart]}),step/100
                print sess.run(cost2,feed_dict={x:X[kpart], z:zt[kpart]})
                print sess.run(lo-z,feed_dict={x:X[kpart], z:zt[kpart]})
                
        ans = tf.equal(tf.floor(lo+0.5),z)
        tmp0 = tf.cast(ans,"float")
        acc = tf.reduce_mean(tmp0)
        print "train accuracy: ",acc.eval({x:X[kpart], z:zt[kpart]})*100, " %" 
 
        print "test accuracy: ",acc.eval({x:Xtest[kpart], z:ztest[kpart]})*100," %"
        print lo.eval({x:Xtest[kpart], z:ztest[kpart]})
        if kpart==0:
            wt1 = sess.run(w1, {x:X[kpart], z:zt[kpart]})
            wt2 = sess.run(w2, {x:X[kpart], z:zt[kpart]})
            wt3 = sess.run(w3, {x:X[kpart], z:zt[kpart]})
            wt4 = sess.run(w4, {x:X[kpart], z:zt[kpart]})
            wt5 = sess.run(w5, {x:X[kpart], z:zt[kpart]})
            wt6 = sess.run(w6, {x:X[kpart], z:zt[kpart]})
            #wt7 = sess.run(w7, {x:X[kpart], z:zt[kpart]})
            #wt8 = sess.run(w8, {x:X[kpart], z:zt[kpart]})
            
	    np.savetxt('mainwts/1.txt',wt1,delimiter='\t',fmt='%f')
	    np.savetxt('mainwts/2.txt',wt2,delimiter='\t',fmt='%f')
	    np.savetxt('mainwts/3.txt',wt3,delimiter='\t',fmt='%f')
	    np.savetxt('mainwts/4.txt',wt4,delimiter='\t',fmt='%f')
	    np.savetxt('mainwts/5.txt',wt5,delimiter='\t',fmt='%f')
	    np.savetxt('mainwts/6.txt',wt6,delimiter='\t',fmt='%f')
	    #np.savetxt('mainwts/7.txt',wt7,delimiter='\t',fmt='%f')
	    #np.savetxt('mainwts/8.txt',wt8,delimiter='\t',fmt='%f')

	    bs1 = sess.run(b1, {x:X[kpart], z:zt[kpart]})
            bs2 = sess.run(b2, {x:X[kpart], z:zt[kpart]})
            bs3 = sess.run(b3, {x:X[kpart], z:zt[kpart]})
            bs4 = sess.run(b4, {x:X[kpart], z:zt[kpart]})
            bs5 = sess.run(b5, {x:X[kpart], z:zt[kpart]})
            bs6 = sess.run(b6, {x:X[kpart], z:zt[kpart]})
            #bs7 = sess.run(b7, {x:X[kpart], z:zt[kpart]})
            #bs8 = sess.run(b8, {x:X[kpart], z:zt[kpart]})

	    np.savetxt('mainbs/1.txt',bs1,delimiter='\t',fmt='%f')
	    np.savetxt('mainbs/2.txt',bs2,delimiter='\t',fmt='%f')
	    np.savetxt('mainbs/3.txt',bs3,delimiter='\t',fmt='%f')
	    np.savetxt('mainbs/4.txt',bs4,delimiter='\t',fmt='%f')
	    np.savetxt('mainbs/5.txt',bs5,delimiter='\t',fmt='%f')
	    np.savetxt('mainbs/6.txt',bs6,delimiter='\t',fmt='%f')
	    #np.savetxt('mainbs/7.txt',bs7,delimiter='\t',fmt='%f')
	    #np.savetxt('mainbs/8.txt',bs8,delimiter='\t',fmt='%f')


