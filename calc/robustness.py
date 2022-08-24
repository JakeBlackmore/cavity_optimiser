'''

Testing the robustness of an optimised cavity.

'''
import os
import pickle

import numpy

from cavity import core

um = 1e-6
ppm = 1e-6

V = 1e-14
scatter_loss = 100*ppm
m = 5*um

cwd = os.path.dirname(os.path.abspath(__file__))


result = core.optimizer(V,scatter_loss,m,L_lims=(200e-6,1e-3))

print("optimum found")

N=64
#(a)
Trans = numpy.linspace(100,1500,N)*1e-6
lengths = numpy.linspace(150e-6,250e-6,N-1)

#(b)
rocs = numpy.linspace(result['roc']/2,result['roc']*1.5,N-1)
dias = numpy.linspace(result['diameter']/2,result['diameter']*1.5,N)

#(c)
scattering = numpy.linspace(1e-9,250*ppm,N)
misalignments = numpy.linspace(0,12*um,N-1)

P_ext_TL = numpy.zeros((len(Trans),len(lengths)))
Clip_TL = numpy.zeros(P_ext_TL.shape)

P_ext_DR = numpy.zeros((len(dias),len(rocs)))
Clip_DR = numpy.zeros(P_ext_DR.shape)

P_ext_SM = numpy.zeros((len(scattering),len(misalignments)))
Clip_SM = numpy.zeros(P_ext_SM.shape)

#Transmission/length axes:
for i,T in enumerate(Trans):
    for j,L in enumerate(lengths):
        pars = {"roc":result['roc'],
                "length":L,
                "transmission":T,
                "mis_par":m,
                "mis_perp":m,
                "diameter":result['diameter'],
                "scatter_loss":scatter_loss
                }

        Cav = core.Cavity(**pars)
        P_ext_TL[i,j] = Cav.P_ext
        Clip_TL[i,j] = Cav.clip
print("1/3")

OP = { "Optimum":result,"Trans":Trans,
    "lengths":lengths,
    "P":P_ext_TL,
    "clip":Clip_TL}

if not os.path.exists(cwd+"\\data\\robust_TL"):
    os.makedirs(cwd+"\\data\\robust_TL")


with open(cwd+"\\data\\robust_TL\\data.pkl","wb") as file:
    pickle.dump(OP,file)

#Transmission/ROC axes:
for i,D in enumerate(dias):
    for j,R in enumerate(rocs):
        pars = {"roc":R,
                "length":result['length'],
                "transmission":result['transmission'],
                "mis_par":m,
                "mis_perp":m,
                "diameter":D,
                "scatter_loss":scatter_loss
                }
        Cav = core.Cavity(**pars)
        P_ext_DR[i,j] = Cav.P_ext
        Clip_DR[i,j] = Cav.clip
print("2/3")

OP = { "Optimum":result,
    "Trans":Trans,
    "dias":dias,
    "rocs":rocs,
    "P":P_ext_DR,
    "clip":Clip_DR}

if not os.path.exists(cwd+"\\data\\robust_DR"):
    os.makedirs(cwd+"\\data\\robust_DR")

with open(cwd+"\\data\\robust_DR\\data.pkl","wb") as file:
    pickle.dump(OP,file)

#scat/mis axes:
for i,S in enumerate(scattering):
    for j,mis in enumerate(misalignments):
        pars = {"roc":result['roc'],
                "length":result['length'],
                "transmission":result['transmission'],
                "mis_par":mis,
                "mis_perp":mis,
                "diameter":result['diameter'],
                "scatter_loss":S
                }
        Cav = core.Cavity(**pars)
        P_ext_SM[i,j] = Cav.P_ext
        Clip_SM[i,j] = Cav.clip

print("3/3")

OP = { "Optimum":result,
        "scat":scattering,
        "mis":misalignments,
        "P":P_ext_SM,
        "clip":Clip_SM}

if not os.path.exists(cwd+"\\data\\robust_SM"):
    os.makedirs(cwd+"\\data\\robust_SM")

with open(cwd+"\\data\\robust_SM\\data.pkl","wb") as file:
    pickle.dump(OP,file)

output = {  "Optimum":result,
            "P_TL":P_ext_TL,
            "P_DR":P_ext_DR,
            "P_SM":P_ext_SM,
            "C_TL":Clip_TL,
            "C_DR":Clip_DR,
            "C_SM":Clip_SM,
            "lengths":lengths,
            "diameter":dias,
            "rocs":rocs,
            "transmission":Trans,
            "scattering":scattering,
            "misalignments":misalignments}

filepath = cwd +"\\data\\robustness.pkl"
with open(filepath,"wb")as file:
    pickle.dump(output,file)
