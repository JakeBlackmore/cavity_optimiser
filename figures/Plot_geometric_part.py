# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter,LogFormatter,ScalarFormatter,LogLocator
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import _pickle as cPickle
from Figures import *
from jqc import Ox_plot
import os

cwd=os.path.dirname(os.path.abspath(__file__))

Ox_plot.plot_style()

with open(cwd+r"\Data\divergence.pkl", "rb") as input_file:
    r_data = cPickle.load(input_file)

N_plt = r_data["N_plt"]
Theta = r_data["Theta"]
D_crit = r_data["D_crit"]
tilt_angle = r_data["tilt_angle"]
ROC = r_data['ROC']
cavity_length = r_data['cavity_length']
misalignment = r_data['misalignment']


theta_tilt = tilt_angle/Theta


fig_theta = plt.figure("Fig3")
gs = gridspec.GridSpec(1,2,width_ratios=[1,0.05],figure=fig_theta)
gs2 = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=gs[0])

size = fig_theta.get_size_inches()
size[1]=2.5*size[1]
fig_theta.set_size_inches(size,forward=True)

axs_theta = gs2.subplots(sharex=True,sharey=True)

ratio_levels = [[0.025,0.05,0.1,0.2],[0.125,0.25,0.5,1.,2.],[0.75,1.5,3,6.]]

D_levels = [[50,75,100,125],[50,100,150],[50,100,200]]

man_D =[[(680,350),(700,600),(600,800),(450,1000)],
        [(680,350),(800,800),(400,600)],
        [(600,250),(700,600),(600,800)]]

man_r =[[(900,500),(800,800),(600,800),(750,1400)],
        [(800,350),(500,400),(470,700),(450,800)],
        [(530,220),(470,600),(500,800),(520,950)]]

formats_r = [{0.025:"0.025",0.05:'0.05',0.1:"0.1",0.2:"0.2",0.4:"0.4",0.8:"0.8"},
            {0.125:"0.125",0.25:"0.25",0.5:"0.5",1:"1",2:"2",4:"4"},
            {0.75:"0.75",1.5:"1.5",3:"3",6:"6",1.25:"1.25",2.5:"2.5",5:"5",10:"10"},]

for i in range(N_plt):
        im = axs_theta[i].imshow(Theta[i],cmap='viridis_r',aspect = 'auto',
                            extent=[ROC[0]*1e6,ROC[-1]*1e6,cavity_length[-1]*1e6,
                            cavity_length[0]*1e6],norm=LogNorm(vmin=1.5,vmax=8))

        if i == N_plt-1:
            axs_theta[i].set_xlabel(r'Radius of Curvature, $R$ (µm)')
        else:
            axs_theta[i].tick_params( labelbottom = False)


        ct = axs_theta[i].contour(D_crit[i]*1e6, D_levels[i],
                colors='r', origin='image', extent=[ROC[0]*1e6,ROC[-1]*1e6,
                cavity_length[-1]*1e6,cavity_length[0]*1e6],zorder=2)

        axs_theta[i].clabel(ct,inline=True,fmt='%d',manual=man_D[i])

        ct = axs_theta[i].contour(theta_tilt[i],ratio_levels[i], colors='k',
        linestyles='dashed', origin='image', extent=[ROC[0]*1e6,ROC[-1]*1e6,
        cavity_length[-1]*1e6,cavity_length[0]*1e6],zorder=1.8)

        if len(ct.levels)>1:
            axs_theta[i].clabel(ct,inline=True, fmt=formats_r[i],manual=man_r[i])

        axs_theta[i].text(0.05,0.08,'$M$ = %d'%round(misalignment[i]*1e6)+r' µm',
                    transform=axs_theta[i].transAxes)

cbar_ax = fig_theta.add_subplot(gs[1])

formatter = LogFormatter(10, labelOnlyBase=False)
tb = fig_theta.colorbar(im, cax=cbar_ax,orientation = 'vertical',
                        ticks=[1,2,4,6,8,10],format=formatter)

tb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
tb.ax.yaxis.set_minor_formatter(FormatStrFormatter('%.0f'))

tb.ax.tick_params(which="minor",labelright=False)
tb.ax.minorticks_on()

tb.ax.set_ylabel(r'Divergence Half-Angle, $\theta$ (deg)')
cbar_ax.invert_yaxis()

legend_elements = [Line2D([0], [0], color='k',ls='--', label='$\\phi/\\theta$'),
                   Line2D([0], [0], color='r',label="$D_\mathrm{crit}$ (µm)")]

fig_theta.legend(loc="center",bbox_to_anchor=(0.27,0.9,0.5,0.1),
                handles=legend_elements,ncol =2, fancybox=False,
                bbox_transform=fig_theta.transFigure,mode='expand')


draw_brace(axs_theta[1],(-1.1,2.1),-0.175,'Cavity Length, $L$ (µm)',Radius=150,axis='y')
plt.subplots_adjust(left=0.23,hspace=0.09,right=0.88,bottom=0.07,top=0.912)
fig_theta.savefig(cwd+"\\Figures\\pdf\\geometric.pdf")
fig_theta.savefig(cwd+"\\Figures\\png\\geometric.png")
plt.show()
