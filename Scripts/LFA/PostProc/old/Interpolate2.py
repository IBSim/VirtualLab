import numpy as np

def Interp(data,value):
	k=0
	data2 = np.zeros((len(data)/2,2))
	for i in range(0,len(data)/2):
		for j in range(0,2):
			data2[i,j] = data[k]
			data2[i,j] = data[k]
			k = k+1
	y = np.interp(value,data2[:,0],data2[:,1])
	return y
