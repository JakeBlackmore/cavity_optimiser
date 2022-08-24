import numpy
import os
from matplotlib import pyplot
import pickle

cwd = os.path.dirname(os.path.abspath(__file__))
pyplot.figure()
with open(cwd+"\\V-12.2_core.pkl","rb") as file:
    data = pickle.load(file)

path_data = numpy.genfromtxt(cwd+"\\V-12.2_corepath.csv",delimiter=',')

length = data['L']
roc = data['R']
P = data['P']
pyplot.pcolormesh(length,roc,P.T)

pyplot.plot(path_data[:,0]*1e6,path_data[:,1]*1e6,color='k',marker='o')
pyplot.plot(path_data[0,0]*1e6,path_data[0,1]*1e6,color='r',marker='o')
pyplot.plot(path_data[-1,0]*1e6,path_data[-1,1]*1e6,color='b',marker='o')


pyplot.xlabel("length (µm)")
pyplot.ylabel("roc (µm)")

pyplot.show()
