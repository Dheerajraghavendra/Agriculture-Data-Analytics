#!/usr/bin/python
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
d = load_files('train a')
x= d.data
y = d.target
scale = MinMaxScaler()
#print len(x),len(y)
#print x
for i in range(len(x)):
    x[i] = x[i].split()
    for j in range(len(x[i])):
        x[i][j] = float(x[i][j])
    x[i].append(1)
print(scale.fit(x))
print(scale.data_max_)
print(scale.data_min_)   

#print x
x = scale.transform(x)
for i in range(len(x)):
    x[i][11] = 1
#print x
#x= preprocessing.scale(x)
print len(x)
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.1)
'''
ns=700
x_train = x[ns:]
x_test = x[:ns]
y_train = y[ns:]
y_test = y[:ns]
'''
mx=0
mxa=0
mxh=0
mxh1=0
cval=[]
cval1=[]
itr=0
a=0
trainacc = []
testacc = []
tmp = 0
aopt = 0
mat = []
for nh in [25,45,75,95]:
    for nh1 in [25,45,75,95]:
    #tm = []
        for nh2 in [25,45,75,95]:
            for nh3 in [25,45,75,95]:
                for nh4 in [25,45,75,95]:
                    for a in range(-1,0):
                        tmp2=[]
                        alp = 10**a;
    #alp = a/float(10)
#        for nh1 in range(5,55,5):
                        nn = MLPClassifier(activation='tanh',solver='sgd',learning_rate='adaptive',learning_rate_init=alp,hidden_layer_sizes=(nh,nh1,nh2,nh3,nh4),max_iter=5000)      
                        nn.fit(x_train,y_train)
                        trsc = nn.score(x_train,y_train)
                        tmp2.append(alp)
            #trainacc.append(trsc)
                        tmp2.append(trsc)
                        sc = nn.score(x_test,y_test)
            #testacc.append(sc)
                        tmp2.append(sc)
            #tm.append(str(trsc)+";"+str(sc))
                        if sc>tmp:
                            tmp = sc
                            aopt = alp
#            pred = nn.predict(x_test)
#            a = y_test
        #ct=0
        #for i in range(len(a)):
        #    if pred[i]==a[i]:
        #        ct+=1
        #print ct/float(len(a))
            #scores = cross_val_score(nn,x,y,cv = 5)
                        itr+=1
                        print "regularization parameter: %f  no.of hidden units: %d %d %d %d %d train accuracy: %f test accuracy: %f"%(alp,nh,nh1,nh2,nh3,nh4,100*trsc,100*sc),"%"
#        cval1.append([alp,nh,sc])
                        if sc>mx:
                            mx = sc
                            mxh=nh   
                            mxa = a
                        mat.append(tmp2)
    #nnopt = MLPClassifier(activation='logistic',alpha=aopt,solver='sgd',learning_rate='adaptive',learning_rate_init=1,hidden_layer_sizes=(nh))
#wts = 
print mxh,mxa,mx
#np.savetxt('alpha2.txt',mat,delimiter='\t',fmt='%s')
