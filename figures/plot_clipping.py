import numpy
from matplotlib import pyplot,gridspec,colors
import os
import pickle
import Figures

from jqc import Ox_plot

Ox_plot.plot_style()

cwd = os.path.dirname(os.path.abspath(__file__))

fpath= cwd+"\\data\\clipping"

with open(fpath+"\\clipping.pkl","rb") as file:
    data = pickle.load(file)

fig = pyplot.figure()
grid = gridspec.GridSpec(5,3,height_ratios = [0.1,0.1,1,1,1])

size = fig.get_size_inches()
size[1]=1.35*size[1]
fig.set_size_inches(size,forward=True)

vmin = 1e-6
vmax = 1e3

for i,d in enumerate(data['diameter']):
    for j,m in enumerate(data['misalignment']):
        IP = numpy.load(fpath+"\\D{:0f}_M{:.0f}.npy".format(d*1e6,m*1e6))
        if i ==0 and j ==0:
            ax = fig.add_subplot(grid[j+2,i])
        else:
            ax = fig.add_subplot(grid[j+2,i],sharex=ax,sharey=ax)

        clip = IP[:,:,0]*1e6

        clip[numpy.where(IP[:,:,1])] = numpy.nan

        pcm = ax.pcolormesh(1e3*data['ROCS'],1e3*data['lengths'],clip,
                norm = colors.LogNorm(vmin,vmax))

        ax.contour(1e3*data['ROCS'],1e3*data['lengths'],clip,
                levels=[1],colors='k',linestyles=':')

        ymin,ymax = ax.get_ylim()
        ax.set_ylim(1.0,0.1)

        if i !=0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_yticks([0,0.5,1])
        if j!=2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticks([0,0.5,1])
        if j ==0:
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("$D$ = {:.0f} µm".format(d*1e6))
        if i ==2:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("$M$ = {:.0f} µm".format(m*1e6))

        '''
        braces for x/y labels
        '''
        if i ==2 and j ==2:
            Figures.draw_brace(ax,(-2.2,1),-0.32,"Radius of Curvature, $R$ (mm)")
        if i ==0 and j ==2:
            Figures.draw_brace(ax,(0,3.3),-0.32,"Cavity Length, $L$ (mm)",axis='y')

'''
colourbar boilerplate here
'''
cax = fig.add_subplot(grid[0,:])
cb = fig.colorbar(pcm,cax=cax,extend="both",orientation="horizontal")
cax.set_xlabel("Clipping Losses, $\\mathcal{L}$$_\\mathrm{clip}$ (ppm)",labelpad=5)
cax.xaxis.set_label_position("top")
cax.tick_params(which="both",labelbottom=False,bottom=False,labeltop=True,top=True)


fig.tight_layout()
fig.subplots_adjust(hspace=0.2,wspace=0.15,left=0.17,bottom=0.13,top=0.85)

pyplot.savefig(cwd+"\\figures\\pdf\\clipping.pdf")
pyplot.savefig(cwd+"\\figures\\png\\clipping.png")


pyplot.show()
