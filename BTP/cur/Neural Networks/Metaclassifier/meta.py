#!/usr/bin/python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
def lrelu(x):
    return tf.maximum(x,0.2*x)
    
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

c12 = [0,3,4,5,6,7,8,9,10,11,12,13]
c7 = [1,2,14,15,16,17,18]

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
    nh = 7
    nhi=2
    ni1 = 35
    ni2 = 35
    nz = 1
    no = 19
    no1 =  19
    no2 = 19
    nh1 = 75
    nh2 = 75
    nh3 = 75
    nh4 = 75
    nh5 = 75
    nh6 = 75
    nh7 = 75
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z= tf.placeholder(tf.float32)
    np.random.seed(0)
    winit = tf.contrib.layers.xavier_initializer(seed=0)
    bis = [[] for _ in range(nhi+1)]
    bis[0] = np.zeros(shape=(1,ni1))
    bis[1] = np.zeros(shape=(1,ni2))
    bis[2] = np.zeros(shape=(1,nz))

    wi1 = tf.Variable(winit([ni,ni1]))
    wi2 = tf.Variable(winit([ni1,ni2]))
    wi3 = tf.Variable(winit([ni2,nz]))

    bi1 = tf.Variable(np.float32(bis[0]))
    bi2 = tf.Variable(np.float32(bis[1]))
    bi3 = tf.Variable(np.float32(bis[2]))

    z1 = tf.nn.relu(tf.matmul(x,wi1)+bi1)
    z2 = tf.nn.relu(tf.matmul(z1,wi2)+bi2)
    zo = tf.sigmoid(tf.matmul(z2,wi3)+bi3)

    bs = [[] for _ in range(nh+1)]
    b7s = [[] for _ in range(nh+1)]
    bs[0] = np.zeros(shape=(1,nh1))
    bs[1] = np.zeros(shape=(1,nh2))
    bs[2] = np.zeros(shape=(1,nh3))
    bs[3] = np.zeros(shape=(1,nh4))
    bs[4] = np.zeros(shape=(1,nh5))
    bs[5] = np.zeros(shape=(1,nh6))
    bs[6] = np.zeros(shape=(1,nh7))
    bs[7] = np.zeros(shape=(1,no1))

    b7s[0] = np.zeros(shape=(1,nh1))
    b7s[1] = np.zeros(shape=(1,nh2))
    b7s[2] = np.zeros(shape=(1,nh3))
    b7s[3] = np.zeros(shape=(1,nh4))
    b7s[4] = np.zeros(shape=(1,nh5))
    b7s[5] = np.zeros(shape=(1,nh6))
    b7s[6] = np.zeros(shape=(1,nh7))
    b7s[7] = np.zeros(shape=(1,no2))

    w1 = tf.Variable(winit([ni,nh1]))
    w2 = tf.Variable(winit([nh1,nh2]))
    w3 = tf.Variable(winit([nh2,nh3]))
    w4 = tf.Variable(winit([nh3,nh4]))
    w5 = tf.Variable(winit([nh4,nh5]))
    w6 = tf.Variable(winit([nh5,nh6]))
    w7 = tf.Variable(winit([nh6,nh7]))
    w8 = tf.Variable(winit([nh7,no1]))

    b1 = tf.Variable(np.float32(bs[0]))
    b2 = tf.Variable(np.float32(bs[1]))
    b3 = tf.Variable(np.float32(bs[2]))
    b4 = tf.Variable(np.float32(bs[3]))
    b5 = tf.Variable(np.float32(bs[4]))
    b6 = tf.Variable(np.float32(bs[5]))
    b7 = tf.Variable(np.float32(bs[6]))
    b8 = tf.Variable(np.float32(bs[7]))

    w71 = tf.Variable(winit([ni,nh1]))
    w72 = tf.Variable(winit([nh1,nh2]))
    w73 = tf.Variable(winit([nh2,nh3]))
    w74 = tf.Variable(winit([nh3,nh4]))
    w75 = tf.Variable(winit([nh4,nh5]))
    w76 = tf.Variable(winit([nh5,nh6]))
    w77 = tf.Variable(winit([nh6,nh7]))
    w78 = tf.Variable(winit([nh7,no2]))

    b71 = tf.Variable(np.float32(b7s[0]))
    b72 = tf.Variable(np.float32(b7s[1]))
    b73 = tf.Variable(np.float32(b7s[2]))
    b74 = tf.Variable(np.float32(b7s[3]))
    b75 = tf.Variable(np.float32(b7s[4]))
    b76 = tf.Variable(np.float32(b7s[5]))
    b77 = tf.Variable(np.float32(b7s[6]))
    b78 = tf.Variable(np.float32(b7s[7]))
    
    x12 = x    
    x7 = x

    learning_rate = 1
    epochs = 10000
    l1 = tf.nn.relu(tf.matmul(x12,w1)+b1)
    l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)
    l3 = tf.nn.relu(tf.matmul(l2,w3)+b3)
    l4 = tf.nn.relu(tf.matmul(l3,w4)+b4)
    l5 = tf.nn.relu(tf.matmul(l4,w5)+b5)
    l6 = tf.nn.relu(tf.matmul(l5,w6)+b6)
    l7 = tf.nn.relu(tf.matmul(l6,w7)+b7)
    l12o = tf.nn.softmax(tf.matmul(l7,w8)+b8)

    l71 = tf.nn.relu(tf.matmul(x7,w71)+b71)
    l72 = tf.nn.relu(tf.matmul(l71,w72)+b72)
    l73 = tf.nn.relu(tf.matmul(l72,w73)+b73)
    l74 = tf.nn.relu(tf.matmul(l73,w74)+b74)
    l75 = tf.nn.relu(tf.matmul(l74,w75)+b75)
    l76 = tf.nn.relu(tf.matmul(l75,w76)+b76)
    l77 = tf.nn.relu(tf.matmul(l76,w77)+b77)
    l7o = tf.nn.softmax(tf.matmul(l77,w78)+b78)

    r1 = tf.nn.l2_loss(w1)+tf.nn.l2_loss(b1)
    r2 = tf.nn.l2_loss(w2)+tf.nn.l2_loss(b2)
    r3 = tf.nn.l2_loss(w3)+tf.nn.l2_loss(b3)
    r4 = tf.nn.l2_loss(w4)+tf.nn.l2_loss(b4)
    r5 = tf.nn.l2_loss(w5)+tf.nn.l2_loss(b5)
    r6 = tf.nn.l2_loss(w6)+tf.nn.l2_loss(b6)
    r7 = tf.nn.l2_loss(w7)+tf.nn.l2_loss(b7)
    r8 = tf.nn.l2_loss(w8)+tf.nn.l2_loss(b8)

    lo = (1-zo)*l12o+zo*l7o
    r = r1+r2+r3+r4+r5+r6+r7+r8
    beta = 0
    cost2 =  -tf.reduce_mean(z*tf.log(zo+1e-10)+(1-z)*tf.log(1-zo+1e-10))#tf.reduce_sum(tf.square(zo - z))
    cost1 = -tf.reduce_mean(y*tf.log(lo+1e-10))
    cost = cost1+cost2+(beta*r)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    strain = [0 for i in range(19)]
    stest = [0 for i in range(19)]
    for i in range(len(X[kpart])):
        idx = np.argmax(Y[kpart][i])
        strain[idx]+=1
    for i in range(len(Ytest[kpart])):
        idx = np.argmax(Ytest[kpart][i])
        stest[idx]+=1
    with tf.Session() as sess:
        sess.run(init)
        for step in range(epochs): 
            sess.run(optimizer,feed_dict = {x:X[kpart], y:Y[kpart], z:zt[kpart]})

            if step%100 == 0:
                print sess.run(cost1,feed_dict={x:X[kpart], y:Y[kpart], z:zt[kpart]}),step/100
                print sess.run(cost2,feed_dict={x:X[kpart], y:Y[kpart], z:zt[kpart]})
                print sess.run(zo-z,feed_dict={x:X[kpart], y:Y[kpart], z:zt[kpart]})
                
        ans = tf.equal(tf.argmax(lo,axis=1),tf.argmax(y,axis=1))
        tmp0 = tf.cast(ans,"float")
        acc = tf.reduce_mean(tmp0)
        ans2 = tf.nn.top_k(lo,3).indices
        ans2 = tf.to_int64(ans2)
        y1 = tf.argmax(y,axis=1)
        y1 = tf.reshape(y1,[tf.shape(y1)[0],1])
        tmp2 = tf.equal(ans2,y1)
        tmp2 = tf.cast(tmp2,"float")
        sm2 = tf.reduce_max(tmp2,axis=1)
        acc2 = tf.reduce_mean(sm2)
        print sess.run(lo,feed_dict={x:X[kpart],y:Y[kpart], z:zt[kpart]})
        print "this"
        print tmp0.eval({x:X[kpart],y:Y[kpart], z:zt[kpart]})
        print "over"
        print "train accuracy: ",acc.eval({x:X[kpart],y:Y[kpart], z:zt[kpart]})*100, " %" 
        print "train3 accurac: ",acc2.eval({x:X[kpart],y:Y[kpart], z:zt[kpart]})*100," %"
        act = np.argmax(Y[kpart],axis=1)+1
        cl1 = act*(np.int64(tmp0.eval({x:X[kpart],y:Y[kpart], z:zt[kpart]})))
        cl3 = act*(np.int64(sm2.eval({x:X[kpart],y:Y[kpart], z:zt[kpart]})))
        teval = [0 for i in range(19)]
        teval3 = [0 for i in range(19)]
        for i in range(len(cl1)):
            if cl1[i]>0:
                teval[cl1[i]-1]+=1
            if cl3[i]>0:
                teval3[cl3[i]-1]+=1
        print "Classwise train accuraices"
        for i in range(18):
	    print teval[i]/float(strain[i]),"  and  ",teval3[i]/float(strain[i])
        print Ytest[kpart]
        print sess.run(lo,feed_dict={x:Xtest[kpart],y:Ytest[kpart], z:ztest[kpart]})
        print "test accuracy: ",acc.eval({x:Xtest[kpart],y:Ytest[kpart], z:ztest[kpart]})*100," %"
        print sess.run(tmp2,feed_dict={x:Xtest[kpart],y:Ytest[kpart], z:ztest[kpart]})
        print "test3 accurac: ",acc2.eval({x:Xtest[kpart],y:Ytest[kpart], z:ztest[kpart]})*100," %"
        act = np.argmax(Ytest[kpart],axis=1)+1    
        clt1 = act*(np.int64(tmp0.eval({x:Xtest[kpart],y:Ytest[kpart], z:ztest[kpart]})))
        clt3 = act*(np.int64(sm2.eval({x:Xtest[kpart],y:Ytest[kpart], z:ztest[kpart]})))
        print act
        print clt3
        tsteval = [0 for i in range(19)]
        tsteval3 = [0 for i in range(19)]
        for i in range(len(clt1)):
            if clt1[i]>0:
                tsteval[clt1[i]-1]+=1
            if clt3[i]>0:
                tsteval3[clt3[i]-1]+=1
        print "Classwise test accuraices"
        for i in range(18):
            print tsteval[i]/float(stest[i]),"  and  ",tsteval3[i]/float(stest[i])


