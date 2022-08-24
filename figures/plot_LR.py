import numpy
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from jqc import Ox_plot
from cavity.core import spherical_cap

import Figures

Ox_plot.plot_style()
cwd = os.path.dirname(os.path.abspath(__file__))
Lsc = [10e-6,50e-6,250e-6]
fig = plt.figure()
size = fig.get_size_inches()
size[1]=2.5*size[1]
fig.set_size_inches(size,forward=True)

gs =gridspec.GridSpec(3,2,width_ratios = [1,0.05])


with open(cwd+"\\data\\LR_space.pkl","rb") as file:
    data = pickle.load(file)

'''
    data = {'ROC':rocs,
            'cavity_length':lengths,
            'N_plt':len(Lsc),
            'Theta':Thetas,
            'tilt_angle':Phis,
            'D_crit':Diameters,
            'misalignment':M,
            'scatter':Lsc,
            'P':P}


'''

ROC = data['ROC'] *1e6
length = data['cavity_length']*1e6
probability = data['P']
scat = numpy.array(data['scatter'])
D_crit = data['D_crit']



pmin = 0.3
pmax = 1.
lvls = [[0.7,0.75,0.8,0.85,0.9],
        [0.55,0.6,0.65,0.7,0.75],
        [0.25,0.3,0.35,0.4,0.45,0.5,0.55]]
pos =[[(106,105),(240,320),(500,580),(807,877)],
        [(169,100),(288,422),(433,667),(529,750),(711,790)],
        [(150,90),(230,260),(270,380),(380,550),(530,770),(640,850),(860,890)]]


c_levels = [55,60,70,90,150]
ctlvls = [1,10,100,1000,10000]
ct_man =    [[(400,300),(400,450),(450,730)],
            [(450,110),(460,557),(551,900)],
            [(630,294),(550,690),(512,830)]]

ct_strs = ["$1$","$10$","$100$","$10^3$","$10^4$","$10^5$","$10^6$"]



axs=[]
fmt = {}
probability[numpy.where(probability==0)]=numpy.nan
for i,S in enumerate(scat):
    ax = fig.add_subplot(gs[i,0])
    #plot p_ext on R,L
    col = 'k' if i<2 else 'w'
    V_crit = numpy.zeros(D_crit[i].shape)
    for m,R in enumerate(ROC):
        for n,L in enumerate(length):
            V_crit[n,m] = 1e15*spherical_cap(R*1e-6,D_crit[i,n,m])[0]
            #print(V_crit[n,m])
    pcm = ax.pcolormesh(ROC,length,probability[i],vmin=pmin, vmax = pmax)
    ax.invert_yaxis()
    ct = ax.contour(ROC,length,probability[i], levels=lvls[i],colors=col,
                            linestyles='--')

    ax.clabel(ct,inline=True,fmt='%1.2f',colors=col,
                        manual=pos[i])

    ct = ax.contour(ROC,length,V_crit, colors='r',linestyles=':',
                                levels=ctlvls)
    for l,s in zip(ct.levels,ct_strs):
        fmt[l] = s

    ax.clabel(ct,inline=True,fmt = fmt,
                    manual=ct_man[i])

    ax.text(0.01,0.02,
                "$\mathcal{L}_\mathrm{scat}=$"+" {:.0f} ppm".format(scat[i]*2e6),
                transform=ax.transAxes,horizontalalignment="left",color='k')


    if i !=2:
        ax.tick_params(labelbottom=False)
    axs.append(ax)
cbar_ax = fig.add_subplot(gs[:,1])

tb = fig.colorbar(pcm,cax=cbar_ax,orientation = 'vertical',extend='min')

tb.ax.set_ylabel(r'Extraction Probability, $P_\mathrm{ext}$',fontsize=15)


axs[0].text(0.9,0.05,"(a)",fontsize=20,transform=axs[0].transAxes)

axs[1].text(0.9,0.05,"(b)",fontsize=20,transform=axs[1].transAxes)

axs[2].text(0.9,0.05,"(c)",color='w',fontsize=20,transform=axs[2].transAxes)

axs[2].set_xlabel("Radius of Curvature, $R$ (µm)")

Figures.draw_brace(axs[1],(-1.1,2.1),-0.175,"Cavity Length, $L$ (µm)",
            axis='y',Radius=150)


plt.tight_layout()
plt.subplots_adjust(left=0.21,wspace=0.14,
                    hspace=0.1)
plt.savefig(cwd+"\\figures\\pdf\\LR.pdf")
plt.savefig(cwd+"\\figures\\png\\LR.png")
plt.show()
