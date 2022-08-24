import numpy
from cavity import core
import os
import pickle

cwd =os.path.dirname(os.path.abspath(__file__))

D = [50e-6,100e-6,150e-6]
M = [0e-6,10e-6,20e-6]

N = 50

Lengths = numpy.linspace(50e-6,1e-3,N)
Rocs = numpy.linspace(50e-6,1e-3,N-1)

LScat = 1e-9

if not os.path.exists(cwd+"\\data\\clipping"):
    os.makedirs(cwd+"\\data\\clipping")


for p,d in enumerate(D):
    for q,m in enumerate(M):
        clip = numpy.zeros((N,N-1,2))
        for i,l in enumerate(Lengths):
            for j,r in enumerate(Rocs):
                pars = {"length":l,
                        "roc":r,
                        "scatter_loss":LScat,
                        "mis_par":m,
                        "mis_perp":m,
                        "diameter":d}
                cav =core.Cavity(**pars)
                clip[i,j,0] = cav.clip
                clip[i,j,1] = not cav.physical
        fpath = cwd+"\\data\\clipping\\"+"D{:0f}_M{:.0f}.npy".format(d*1e6,m*1e6)
        numpy.save(fpath,clip)

OP = {   "diameter"      : D,
         "misalignment"  : M,
         "lengths": Lengths,
         "ROCS":Rocs}

with open(cwd+"\\data\\clipping\\clipping.pkl","wb") as file:
    pickle.dump(OP,file)
