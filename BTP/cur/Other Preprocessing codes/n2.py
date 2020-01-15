#!/usr/bin/python
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

d = load_files('train a')
x= d.data
y = d.target
#print len(x),len(y)
#print x
for i in range(len(x)):
	x[i] = x[i].split()
	for j in range(len(x[i])):
		x[i][j] = float(x[i][j])
#        x[i].append(1)
#scale = MinMaxScaler()
#print(scale.fit(x))
#print(scale.data_max_)
#print(scale.data_min_)
#print x
#x = scale.transform(x)
#for i in range(len(x)):
#    x[i][11] = 1

x= preprocessing.scale(x)
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2)
alp=1
#out=[[0 for i in range(2)] for j in range(((100-5)/5))]	
for nh in range(5,100,1):
#    for nh2 in range(5,100,5):
    for a in range(-7,-2):
        alp = 10**a
	nn = MLPClassifier(activation='logistic',solver='sgd',alpha=alp,learning_rate='adaptive',learning_rate_init=1,hidden_layer_sizes=(nh))
	nn.fit(x_train,y_train)
#        print nn.predict(x_test)
#        print y_test
	print nh,nh,nn.score(x_test,y_test)*100,nn.score(x_train,y_train)*100
