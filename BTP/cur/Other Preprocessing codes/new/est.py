#!/usr/bin/python
import numpy as np
sd = {"kharif","rabi","summer"}
with open('../'+'croptypecount.txt','r') as f:
    ctcnt= np.genfromtxt(f,dtype=None,delimiter='\t')
with open('../'+'prob.txt','r') as f:
    pb = np.genfromtxt(f,dtype=None,delimiter=';')

with open('input.txt','r') as f:
    ip = np.genfromtxt(f,dtype=None,delimiter='\t')
#win = [0 for i in range(8)]
count=0
for case in range(len(ip)):
    win=[0 for i in range(8)]
    w = [0 for i in range(8)]
    pw = [0 for i in range(8)]
    crop = ip[case][0]
    for cropid in range(len(ctcnt)):
        if ctcnt[cropid][0]==crop:
            break;
    season = ip[case][1]
    for i in range(8):
        win[i] = float(ip[case][2+i])
    with open('w1/'+season+'/'+crop+'.txt','r') as f:
        w[0] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('w2/'+season+'/'+crop+'.txt','r') as f:
        w[1] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('w3/'+season+'/'+crop+'.txt','r') as f:
        w[2] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('w4/'+season+'/'+crop+'.txt','r') as f:
        w[3] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('w5/'+season+'/'+crop+'.txt','r') as f:
        w[4] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('w6/'+season+'/'+crop+'.txt','r') as f:
        w[5] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('w7/'+season+'/'+crop+'.txt','r') as f:
        w[6] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('w8/'+season+'/'+crop+'.txt','r') as f:
        w[7] = np.genfromtxt(f,dtype=None,delimiter=' ')

    with open('weather/w1/'+season+'.txt','r') as f:
        pw[0] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('weather/w2/'+season+'.txt','r') as f:
        pw[1] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('weather/w3/'+season+'.txt','r') as f:
        pw[2] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('weather/w4/'+season+'.txt','r') as f:
        pw[3] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('weather/w5/'+season+'.txt','r') as f:
        pw[4] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('weather/w6/'+season+'.txt','r') as f:
        pw[5] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('weather/w7/'+season+'.txt','r') as f:
        pw[6] = np.genfromtxt(f,dtype=None,delimiter=' ')
    with open('weather/w8/'+season+'.txt','r') as f:
        pw[7] = np.genfromtxt(f,dtype=None,delimiter=' ')

    with open('season/'+season+'.txt','r') as f:
        prob = np.genfromtxt(f,dtype=None,delimiter=' ')
    op = [0 for i in range(len(pb))]
    for i in range(len(pb)):
        op[i] = prob[cropid][i]
        d = 1
        for j in range(8):
            #print win[j]*2,int(win[j]*2)
            if j==2 or j==3:
                op[i]=op[i]*w[j][i][int(win[j])]
                d = d*pw[j][cropid][int(win[j])]
            else:
                op[i] = op[i]*w[j][i][int(win[j]*2)]
                d=d*pw[j][cropid][int(win[j]*2)]
        d = float(d)
        op[i]/=d
    op = [b[0] for b in sorted(enumerate(op),key=lambda i:i[1])]
    #for i in range(3):
    #print op[i]
    print pb[op[len(op)-1]][0],ip[case][10]
    if pb[op[len(op)-1]][0]!=ip[case][10]:
        count+=1
print "Number of incorrect predictions:%d"%(count)
print "Number of input test vectors:%d"%(len(ip))
print "Accuracy:",((len(ip)-count)/float(len(ip)))*100,"%"
