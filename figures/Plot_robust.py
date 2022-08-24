import numpy
import os
import pickle
from jqc import Ox_plot
from matplotlib import pyplot,gridspec
from cavity.core import spherical_cap
from scipy import optimize


Ox_plot.plot_style("Wide")
cwd = os.path.dirname(os.path.abspath(__file__))
try:
    filepath = cwd +"\\Data\\robustness.pkl"
    with open(filepath,"rb")as file:
        input = pickle.load(file)

except FileNotFoundError and EOFError:
    for i,f in enumerate(["TL","DR","SM"]):
        with open(cwd+"\\data\\robust_"+f+"\\data.pkl","rb") as file:
            data = pickle.load(file)
        if i == 0:
            Trans = data['Trans']
            lengths = data['lengths']
            result = data['Optimum']
            P_ext_TL = data['P']
        elif i ==1:
            dias = data['dias']
            rocs = data['rocs']
            P_ext_DR = data['P']
        elif i ==2:
            misalignments = data['mis']
            scattering = data['scat']
            P_ext_SM = data['P']

    input = {  "Optimum":result,
                "P_TL":P_ext_TL,
                "P_DR":P_ext_DR,
                "P_SM":P_ext_SM,
                "lengths":lengths,
                "rocs":rocs,
                "transmission":Trans,
                "diameter":dias,
                "scattering":scattering,
                "misalignments":misalignments}

    filepath = cwd +"\\robustness.pkl"
    with open(filepath,"wb")as file:
        pickle.dump(input,file)

optimum = input['Optimum']
grid = gridspec.GridSpec(1,6,width_ratios=[1,0.2,1,0.2,1,0.1])
fig = pyplot.figure()
ax = fig.add_subplot(grid[0,0])
print(optimum)
lvls = [optimum['probability']-.10,optimum['probability']-.05,optimum['probability']-.01
    ,optimum['probability']+.01,optimum['probability']+.05,optimum['probability']+.1,]

strs =["-10 %","-5 %","-1 %","+1 %","+5 %","+10 %"]

min = .35
max = .80

ax.pcolormesh(input['lengths']*1e6,input['transmission']*1e6,input['P_TL'],
                    vmin=min,vmax=max)

ax.plot(optimum['length']*1e6,optimum['transmission']*1e6,marker="x",color='k')
cs=ax.contour(input['lengths']*1e6,input['transmission']*1e6,input['P_TL'],
                    colors='k',levels=lvls)

man = [(175,470),(165,340),(170,191)]

fmt = {}
for l,s in zip(cs.levels,strs):
    fmt[l] = s

cs.clabel(inline=True,fmt=fmt,manual=man)

ax.set_xlabel("Length (µm)")
ax.set_ylabel("Transmission (ppm)")

ax.axvline(200,ls='--',color='k')

ax.text(0.87,0.93,"(a)",color='w',fontsize=20,transform=ax.transAxes)

ax = fig.add_subplot(grid[0,2])

ax.pcolormesh(input['rocs']*1e6,input['diameter']*1e6,input['P_DR'],
                    vmin=min,vmax=max)

ax.plot(optimum['roc']*1e6,optimum['diameter']*1e6,marker="x",color='k')
cs=ax.contour(input['rocs']*1e6,input['diameter']*1e6,input['P_DR'],
                                colors='k',levels=lvls)

def find_diameter(R):
    V = 1e-14
    fn = lambda x: spherical_cap(R,x)[0]-V
    root = optimize.root_scalar(fn,bracket = [0,2*R])
    D = root.root
    return D

D = numpy.array([find_diameter(r) for r in input['rocs']])

ax.plot(input['rocs']*1e6,D*1e6,color='k',ls='--')

fmt = {}
for l,s in zip(cs.levels,strs):
    fmt[l] = s

man = [(132,90),(144,90),(170,90),(170,40)]
cs.clabel(inline=True,fmt=fmt,manual=man)

ax.set_xlabel("Radius of Curvature (µm)")
ax.set_ylabel("Diameter (µm)")

ax.text(0.87,0.93,"(b)",color='w',fontsize=20,transform=ax.transAxes)

ax = fig.add_subplot(grid[0,4])
'''
lvls = [optimum['probability']-.10,optimum['probability']-.05,
        optimum['probability']-.01,
        optimum['probability']+.01,optimum['probability']+.05,
        optimum['probability']+.10]

'''

pcm = ax.pcolormesh(input['scattering']*1e6,input['misalignments']*1e6,
                input['P_SM'].T,vmin=min,vmax=max)
cs = ax.contour(input['scattering']*1e6,input['misalignments']*1e6,input['P_SM'].T,
                            colors="k",levels=lvls)

#strs =["-10 %","-5 %","-1 %","+1 %","+5 %","+10 %"]
fmt = {}

for l,s in zip(cs.levels,strs):
    fmt[l] = s

man = [(24,3),(60,3),(91,3),(110,1),(140,1),(180,1)]
cs.clabel(inline=True,fmt=fmt,manual=man)

ax.plot(100,5,marker="x",color='k')

ax.set_xlabel("Scattering Losses (ppm)")
ax.set_ylabel("Misalignment (µm)")

ax.text(0.87,0.93,"(c)",color='w',fontsize=20,transform=ax.transAxes)

cax = fig.add_subplot(grid[0,5])
if max != 1 and min !=0:
    fig.colorbar(pcm,cax=cax,extend="both")
elif max == 1:
    fig.colorbar(pcm,cax=cax,extend="min")
else :
    fig.colorbar(pcm,cax=cax,extend="max")

cax.set_ylabel("Extraction Probability")

fig.tight_layout()
pyplot.subplots_adjust(wspace=0.14,right =0.93)
if not os.path.exists(cwd+"\\figures\\pdf"):
    os.makedirs(cwd+"\\figures\\pdf")

if not os.path.exists(cwd+"\\figures\\png"):
    os.makedirs(cwd+"\\figures\\png")

pyplot.savefig(cwd+"\\figures\\pdf\\robustness.pdf")
pyplot.savefig(cwd+"\\figures\\png\\robustness.png")

pyplot.show()
