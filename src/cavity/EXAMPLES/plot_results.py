from matplotlib import pyplot,gridspec
import numpy
from jqc import Ox_plot
import os
import core
import pickle


cwd = os.path.dirname(os.path.abspath(__file__))

fpath = cwd + "\\data\\"
Ox_plot.plot_style("Thesis")
V = numpy.logspace(-15,-12,50)
scat = [200e-6]# numpy.linspace(1e-6,500e-6,50)
mis = numpy.linspace(0,25e-6,50)#[0]
pars = [(Vmax,Lscat,m) for Vmax in V for Lscat in scat for m in mis]

Lmin=250e-6

length=numpy.zeros((len(V),len(scat),len(mis)))
roc = length.copy()
diameter = length.copy()
P_ext = length.copy()
Trans = length.copy()
Random = length.copy()
success = length.copy()

if os.path.isfile(fpath+"\\data.pkl"):
    with open(fpath+"\\data.pkl", 'rb') as f:
        data = pickle.load(f)

    length = data['length']
    roc = data['ROC']
    diameter = data['diameter']
    P_ext = data['P ext']
    Trans = data['Trans']
    Random = data['Random']
    success = data['success']
else:
    for i,Vmax in enumerate(V):
        for j,Lscat in enumerate(scat):
            for k,m in enumerate(mis):

                fname = "Optimum_Lmin-{:.0f}um_Vmax-{:.0f}um3_Lscat-{:.0f}ppm_m-{:.0f}um.csv".format(Lmin*1e6,Vmax*1e18,Lscat*1e6,m*1e6)
                L,R,D,P,T,Ra,Su = numpy.genfromtxt(fpath+fname,skip_header=1,delimiter=',')

                print(i,j)
                length[i,j,k]=L
                roc[i,j,k] = R
                diameter[i,j,k] = D
                P_ext[i,j,k] = P
                Trans[i,j,k] = T
                Random[i,j,k] = Ra
                success[i,j,k] = Su

    dict = {"Trans":Trans,
            "length":length,
            "ROC":roc,
            "diameter":diameter,
            "P ext":P_ext,
            "Random":Random,
            "success":success}

    with open(fpath+"data.pkl","wb") as file:
        pickle.dump(dict,file)

fig = pyplot.figure("mis = 0")
grid = gridspec.GridSpec(2,5,height_ratios=[0.1,1])
ax0=fig.add_subplot(grid[1,0])
pcm = ax0.pcolormesh(V*1e12,mis*1e6,1e6*length[:,0,:].T)
ax0.contour(V*1e12,mis*1e6,1e6*length[:,0,:].T,colors='k')

cax0 = fig.add_subplot(grid[0,0])

cb0 =fig.colorbar(pcm,cax=cax0,orientation='horizontal')
cb0.set_label("Length (µm)")

cax0.xaxis.set_ticks_position("top")
cax0.xaxis.set_label_position("top")

ax0.set_xlabel("Volume ($\\times 10^6$ µm$^3$)")
ax0.set_ylabel("Misalignment (µm)")

ax0.set_xscale("log")
ax1=fig.add_subplot(grid[1,1],sharey=ax0)
pcm = ax1.pcolormesh(V*1e12,mis*1e6,1e6*roc[:,0,:].T)
ax1.contour(V*1e12,mis*1e6,1e6*roc[:,0,:].T,colors='k')

cax1 = fig.add_subplot(grid[0,1])

cb1 =fig.colorbar(pcm,cax=cax1,orientation='horizontal')

ax1.set_xlabel("Volume ($\\times 10^6$ µm$^3$)")
ax1.tick_params(labelleft=False)
cb1.set_label("ROC (µm)")
cax1.xaxis.set_ticks_position("top")
cax1.xaxis.set_label_position("top")

ax1.set_xscale("log")

ax2=fig.add_subplot(grid[1,2],sharey=ax1)
pcm = ax2.pcolormesh(V*1e12,mis*1e6,1e6*diameter[:,0,:].T)
ax2.contour(V*1e12,mis*1e6,1e6*diameter[:,0,:].T,colors='k')

cax2 = fig.add_subplot(grid[0,2])
cb2 =fig.colorbar(pcm,cax=cax2,orientation='horizontal')

cb2.set_label("Diameter (µm)")
cax2.xaxis.set_ticks_position("top")
cax2.xaxis.set_label_position("top")

ax2.set_xlabel("Volume ($\\times 10^6$ µm$^3$)")
ax2.tick_params(labelleft=False)

ax2.set_xscale("log")

ax3=fig.add_subplot(grid[1,3],sharey=ax2)
pcm = ax3.pcolormesh(V*1e12,mis*1e6,100*P_ext[:,0,:].T)
ax3.contour(V*1e12,mis*1e6,100*P_ext[:,0,:].T,colors='k')

cax3 = fig.add_subplot(grid[0,3])
cb3 =fig.colorbar(pcm,cax=cax3,orientation='horizontal')

cb3.set_label("P (%)")
cax3.xaxis.set_ticks_position("top")
cax3.xaxis.set_label_position("top")

ax3.set_xlabel("Volume ($\\times 10^6$ µm$^3$)")

ax3.set_xscale("log")
ax3.tick_params(labelleft=False)
print(Trans)
ax4 = fig.add_subplot(grid[1,4],sharey=ax3)
pcm = ax4.pcolormesh(V*1e12,mis*1e6,1e6*Trans[:,0,:].T)
ax4.contour(V*1e12,mis*1e6,1e6*Trans[:,0,:].T,colors='k')

ax4.set_xlabel("Volume ($\\times 10^6$ µm$^3$)")
ax4.set_xscale("log")
ax4.tick_params(labelleft=False)

cax4 = fig.add_subplot(grid[0,4])
cb4 = fig.colorbar(pcm,cax=cax4,orientation="horizontal")

cb4.set_label("Transmission (ppm)")
cax4.xaxis.set_ticks_position("top")
cax4.xaxis.set_label_position("top")


pyplot.tight_layout()

pyplot.figure("test")

pyplot.imshow(P_ext[:,0,:])

pyplot.show()
