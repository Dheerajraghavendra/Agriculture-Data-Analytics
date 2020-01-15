#!/usr/bin/python
import numpy as np
import xlrd
w = xlrd.open_workbook('weather.xlsx')
sheet=w.sheet_by_index(0)
r = sheet.nrows
c = sheet.ncols
mat = [[0 for j in range(c)] for i in range(r)]
for i in range(r):
    for j in range(c):
        a = sheet.cell_value(i,j)
        if(i>1 and j==0):
            b = xlrd.xldate_as_tuple(a,w.datemode)
            if(b[1]<10 and b[2]<10):
                a = str(b[0])+"-"+"0"+str(b[1])+"-"+"0"+str(b[2])
            elif(b[1]<10):
                a=str(b[0])+"-"+"0"+str(b[1])+"-"+str(b[2])
            elif(b[2]<10):
                a=str(b[0])+"-"+str(b[1])+"-"+"0"+str(b[2])
            else:
                a = str(b[0])+"-"+str(b[1])+"-"+str(b[2])
        mat[i][j] = a

#np.savetxt('weather.txt',mat,delimiter='\t',fmt='%s')
