import numpy
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gs
from matplotlib.colors import LogNorm
from jqc import Ox_plot
import pickle
import os
from Figures import draw_brace,jqc_red_blue,Ox_green_blue_r,jqc_sand_red_r
from scipy import constants
from cavity.core import spherical_cap

cwd =os.path.dirname(os.path.abspath(__file__))

Ox_plot.plot_style("wide")

with open(cwd+"\\data\\3D\\Optimise_data.pkl","rb") as file:
    data = pickle.load(file)

'''
output =   {"volume":vol,
            "scattering loss":scatter,
            "misalignment": mis,
            "min_length":mins,
            "length":length,
            "roc":roc,
            "diameter":diameter,
            "probability":probability,
            "transmission":transmission}
'''

vol = data['volume']
min = data['min_length']
mis = data['misalignment']

length = data['length']
roc = data['roc']
diameter = data['diameter']
trans = data['transmission']

''' convert trans to kappa_output '''


prob = data['probability']

locs = numpy.where(prob==0)

roc[locs] = numpy.nan
diameter[locs] = numpy.nan
trans[locs] = numpy.nan
length[locs] = numpy.nan
prob[locs] = numpy.nan

trans = trans *constants.c/(4*2*numpy.pi*length)

mask = numpy.zeros(length.shape)
sag = mask.copy()

for i in range(length.shape[0]):
    for j in range(length.shape[1]):
        for k in range(length.shape[2]):
            for l in range(length.shape[3]):
                mask[i,j,k,l] = (length[i,j,k,l]<=1.01*min[i]) and (length[i,j,k,l]>=0.99*min[i])
                sag[i,j,k,l] = spherical_cap(roc[i,j,k,l],diameter[i,j,k,l])[1]


rmin = 50#numpy.amin(roc)*1e6
rmax = 500# numpy.amax(roc)*1e6

dmin = 15#numpy.amin(diameter)*1e6
dmax = 250#numpy.amax(diameter)*1e6

tmin = 5#numpy.amin(trans)*1e-6
tmax = 350#numpy.amax(trans)*1e-6

pmin = 0.5#numpy.amin(prob)
pmax = 1#numpy.amax(prob)

fig = pyplot.figure()
size = fig.get_size_inches()
size[1]=1.5*size[1]
fig.set_size_inches(size,forward=True)
grid = gs.GridSpec(5,3+1,height_ratios=[.1,.15,1,1,1])

for i in range(3):
    '''
    ROC plots
    '''

    ax = fig.add_subplot(grid[i+2,0])
    pcm = ax.pcolormesh(vol*1e15,min*1e6,roc[:,:,0,i]*1e6,
                                vmin=rmin,vmax=rmax,
                                cmap = Ox_green_blue_r)
    ax.set_xscale("log")

    ct = ax.contour(vol*1e15,min*1e6,roc[:,:,0,i]*1e6,
                    levels = [100,200,400],
                    colors='w',linestyles='--')
    cl = ax.clabel(ct,manual = [(75,91),(55,330),(1,440)])

    if i !=2:
        ax.tick_params(labelbottom=False)

    if i ==1:
        ax.scatter(1e-2,200,137.5,marker='x',color='r')
    if i ==2:
        cax = fig.add_subplot(grid[0,0])
        cb = fig.colorbar(pcm,cax=cax,orientation = 'horizontal',extend = "max")
        cax.xaxis.set_label_position("top")
        cax.set_xlabel("Radius of \n Curvature (µm)",labelpad=5)
        draw_brace(ax,(0,3.1),-0.25,'Minimum Cavity Length (µm)',Radius=150,axis='y')

    '''
    Diameter plots
    '''
    ax = fig.add_subplot(grid[i+2,1],sharex=ax)
    pcm = ax.pcolormesh(vol*1e15,min*1e6,diameter[:,:,0,i]*1e6,
                                vmin=dmin,vmax=dmax,
                                cmap=jqc_sand_red_r)


    ct = ax.contour(vol*1e15,min*1e6,diameter[:,:,0,i]*1e6,
                        levels = [50,100,150],colors='k',linestyles='--')
    cl = ax.clabel(ct,manual = [(2,200),(13,290),(320,420)])
    ax.set_xscale("log")
    ax.tick_params(labelleft=False)

    if i !=2:
        ax.tick_params(labelbottom=False)

    if i ==2:
        cax = fig.add_subplot(grid[0,1])
        cb = fig.colorbar(pcm,cax=cax,orientation = 'horizontal')
        cax.xaxis.set_label_position("top")
        cax.set_xlabel("Diameter (µm)",labelpad=15)

    '''
    Kappa/ Transmission Axes
    '''

    ax = fig.add_subplot(grid[i+2,2],sharex=ax)
    pcm = ax.pcolormesh(vol*1e15,min*1e6,trans[:,:,0,i]*1e-6,
                                norm = LogNorm(vmin=tmin,vmax=tmax),
                                cmap=jqc_red_blue)


    ct = ax.contour(vol*1e15,min*1e6,trans[:,:,0,i]*1e-6,
                    levels = [10,20,40,80],
                    colors='k',linestyles='--')

    cl = ax.clabel(ct,fmt="%d",
                manual=[(27,440),(31,290),(4,140)])
    ax.set_xscale("log")
    ax.tick_params(labelleft=False)
    if i !=2:
        ax.tick_params(labelbottom=False)
    if i ==2:
        cax = fig.add_subplot(grid[0,2])
        cb = fig.colorbar(pcm,cax=cax,orientation = 'horizontal',extend='min')
        cax.xaxis.set_label_position("top")
        cax.set_xticks([10,100])
        cax.set_xticklabels(["10","100"])
        cax.set_xlabel("$\\kappa_\\mathrm{output}/2\\pi$ (MHz)",labelpad=15)

    '''
    Probability plots
    '''

    ax = fig.add_subplot(grid[i+2,3],sharex=ax)
    pcm = ax.pcolormesh(vol*1e15,min*1e6,prob[:,:,0,i],
                                vmin=pmin,vmax=pmax)

    ct = ax.contour(vol*1e15,min*1e6,prob[:,:,0,i],
                    levels = [.50,.6,.7,.8,.9,],
                    colors='k',linestyles='--')


    cl = ax.clabel(ct,manual=[(7,390),(23,280),(100,150)])
    ax.set_xscale("log")
    ax.tick_params(labelleft=False)

    #draw_brace(ax,(0,4.3),-0.5,'Minimum Cavity Length (µm)',Radius=150,axis='y')

    if i ==1:
        ax.scatter(10,200,marker='x',color='r',zorder=5)

    if i !=2:
        ax.tick_params(labelbottom=False)
    if i ==2:
        cax = fig.add_subplot(grid[0,3])
        cb = fig.colorbar(pcm,cax=cax,orientation = 'horizontal',extend='min')
        cax.xaxis.set_label_position("top")
        cax.set_xlabel("Extraction Probability",labelpad=15)

    ax.tick_params(axis='x',which="minor",bottom=True)
    ax.set_xlim(3e-1,250)

    ax.text(1.1,0.5,"$M=${:.0f} µm".format(mis[i]*1e6),rotation=90,
        horizontalalignment="center",verticalalignment="center",transform=ax.transAxes)

draw_brace(ax,(-3.3,1.05),-0.3,'Milled Volume (pL)',Radius=150,axis='x')

pyplot.tight_layout()

pyplot.subplots_adjust(left=0.118,bottom=0.15,right=0.955,top=0.912,wspace=0.1,hspace=0.15)

pyplot.savefig(cwd+"\\figures\\pdf\\optimum.pdf")
pyplot.savefig(cwd+"\\figures\\png\\optimum.png")

fig2 = pyplot.figure("length backup")

ax = fig2.add_subplot(131)

pcm = ax.pcolormesh(vol*1e15,min*1e6,length[:,:,0,0]*1e6,
                                vmin=dmin,vmax=dmax)
ax.set_xscale("log")
ax = fig2.add_subplot(132)

pcm = ax.pcolormesh(vol*1e15,min*1e6,length[:,:,0,1]*1e6,
                                vmin=dmin,vmax=dmax)
ax.set_xscale("log")
ax = fig2.add_subplot(133)

pcm = ax.pcolormesh(vol*1e15,min*1e6,length[:,:,0,2]*1e6,
                                vmin=dmin,vmax=dmax)
ax.set_xscale("log")


fig3 = pyplot.figure("sag backup")

ax = fig3.add_subplot(131)

pcm = ax.pcolormesh(vol*1e15,min*1e6,2*sag[:,:,0,0]*1e6,
                                vmin=dmin,vmax=dmax)
ax.set_xscale("log")

ax = fig3.add_subplot(132)

pcm = ax.pcolormesh(vol*1e15,min*1e6,2*sag[:,:,0,1]*1e6,
                                vmin=dmin,vmax=dmax)
ax.set_xscale("log")
ax = fig3.add_subplot(133)

pcm = ax.pcolormesh(vol*1e15,min*1e6,2*sag[:,:,0,2]*1e6,
                                vmin=dmin,vmax=dmax)
ax.set_xscale("log")

pyplot.show()
