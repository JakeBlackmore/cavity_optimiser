import pickle
import os

from matplotlib import pyplot,gridspec

import numpy

cwd = os.path.dirname(os.path.abspath(__file__))

fname = "\\Lmin_200um_data.pkl"

with open(cwd+fname,"rb") as file:
    data = pickle.load(file)

for i,s in enumerate(data['scattering loss']):
    fig = pyplot.figure("cavities_{:.0f}ppm".format(s*1e6))
    grid = gridspec.GridSpec(2,5,height_ratios=[1,1])

    ax0 = fig.add_subplot(grid[0,0])
    pcm = ax0.pcolormesh(data["misalignment"]*1e6,data["volume"]*1e12,1e6*data['length'][:,i,:],vmin=200,vmax=400)
    ax0.set_ylabel("Volume $\\times10^6$ (µm$^3$)")
    ax0.tick_params(labelbottom=False)
    ax0.set_yscale("log")
    cax0 = fig.add_subplot(grid[1,0])
    cb0 = fig.colorbar(pcm,cax=cax0)
    cax0.set_ylabel("Length (µm)")


    ax1 = fig.add_subplot(grid[0,1],sharex = ax0,sharey=ax0)
    ax1.pcolormesh(data["misalignment"]*1e6,data["volume"]*1e12,1e6*data['roc'][:,i,:])
    ax1.set_ylabel("Volume $\\times10^6$ (µm$^3$)")
    ax1.tick_params(labelbottom=False)
    cax1 = fig.add_subplot(grid[1,1])
    cb1 = fig.colorbar(pcm,cax=cax1)
    cax1.set_ylabel("ROC (µm)")

    ax2 = fig.add_subplot(grid[0,2],sharex = ax0,sharey=ax0)
    pcm = ax2.pcolormesh(data["misalignment"]*1e6,data["volume"]*1e12,1e6*data['diameter'][:,i,:])
    ax2.set_ylabel("Volume $\\times10^6$ (µm$^3$)")
    ax2.tick_params(labelbottom=False)
    cax2 = fig.add_subplot(grid[1,2])
    cb2 = fig.colorbar(pcm,cax=cax2)
    cax2.set_ylabel("Diameter (µm)")

    ax3 = fig.add_subplot(grid[0,3],sharex = ax0,sharey=ax0)
    pcm = ax3.pcolormesh(data["misalignment"]*1e6,data["volume"]*1e12,1e6*data['transmission'][:,i,:])
    ax3.set_ylabel("Volume $\\times10^6$ (µm$^3$)")
    ax3.tick_params(labelbottom=False)
    cax3 = fig.add_subplot(grid[1,3])
    cb3 = fig.colorbar(pcm,cax=cax3)
    cax3.set_ylabel("Transmission (ppm)")

    ax4 = fig.add_subplot(grid[0,4],sharex = ax0,sharey=ax0)
    pcm = ax4.pcolormesh(data["misalignment"]*1e6,data["volume"]*1e12,1e2*data['probability'][:,i,:],vmin=0,vmax=100)
    ax4.set_ylabel("Volume $\\times10^6$ (µm$^3$)")
    ax4.set_xlabel("Misalignment (µm)")

    cax4 = fig.add_subplot(grid[1,4])
    cb4 = fig.colorbar(pcm,cax=cax4)
    cax4.set_ylabel("P$_\mathrm{ext}$ (%)")

    fig.subplots_adjust(hspace=0.07,wspace =0.04)
pyplot.show()
