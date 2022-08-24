import core
import numpy
import os

cwd = os.path.dirname(os.path.abspath(__file__))

fpath = cwd + "\\data\\"

try:
    os.makedirs(fpath)
except FileExistsError:
    pass

Lscat = 50e-6
m = 0e-6
crit_loss = 1e-6

Vmax = 2e-12 #m^3

Lmin = 300e-6

L_lim = (Lmin,1e-3)

atom = {"wavelength":1e-6,
        "alpha":1/20}

optimum, output = core.optimizer(Vmax,Lscat,m,L_lims=L_lim,
                                local=True)
'''
print("opt",optimum,output)

pars = {"length":optimum[0],
        "roc":optimum[1],
        "diameter":optimum[2],
        "mis_perp":m,
        "mis_par":m,
        "scatter_loss":Lscat,
        "atom":atom}

cav = core.Cavity(**pars)
print(cav.physical)
print(cav.clip)
print(core.spherical_cap(optimum[1],optimum[2]))
'''

data = [[*optimum,output]]
fname = "Optimum_local_Lmin-{:.0f}um_Vmax-{:.0f}um3_Lscat-{:.0f}ppm_m-{:.0f}um.csv".format(Lmin*1e6,Vmax*1e18,Lscat*1e6,m*1e6)
numpy.savetxt(fpath+fname,data,header = "L (m), R (m), D (m), P_ext ")
