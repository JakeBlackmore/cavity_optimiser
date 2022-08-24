import numpy
import core
import os

cwd = os.path.dirname(os.path.abspath(__file__))

fpath = cwd + "\\data\\"
Lscat = 50e-6
m = 0e-6
crit_loss = 1e-6

Vmax = 2e-12 #m^3

Lmin = 300e-6

fname = "Optimum_Lmin-{:.0f}um_Vmax-{:.0f}um3_Lscat-{:.0f}ppm_m-{:.0f}um.csv".format(Lmin*1e6,Vmax*1e18,Lscat*1e6,m*1e6)

data = numpy.genfromtxt(fpath+fname,skip_header=1)
print(core.spherical_cap(data[1],data[2]))
print(data)
