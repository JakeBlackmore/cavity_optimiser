from cavity import core

from cavity import core
import pickle
import numpy
import os
from scipy import optimize

cwd = os.path.dirname(os.path.abspath(__file__))
limit = 1e-6
M = 10e-6

def _critical_diam(D,cav,limit):
    clip = cav.clipping_loss(D=D) - limit
    return clip

def _LR_space(par):
    S,L,R = par
    #print(M,L,R)
    fname = cwd+"\\data\\LR_space\\S{:.0f}\\L{:.0f}R{:.0f}.csv".format(S*1e6,L*1e6,R*1e6)
    pars = {"length":L,
            "roc":R,
            "mis_par":M,
            "mis_perp":M,
            "diameter":R,
            "scatter_loss":S, # non-zero to avoid errors
            }

    cav = core.Cavity(**pars)
    try:
        root =  optimize.root_scalar(_critical_diam,args=(cav,limit),
                                            bracket=[0,2*R])
        critical_diam = root.root
    except ValueError:
        critical_diam = numpy.nan

    pars['diameter']=critical_diam
    theta = numpy.rad2deg(cav.thetaprime)
    phi = numpy.rad2deg(cav.phi)
    Prob = cav.P_ext

    numpy.savetxt(fname,[theta,phi,critical_diam,Prob],delimiter=',',
                        header='theta (deg), phi (deg),diameter (m),Probability')

    return

if __name__ == "__main__":
    import multiprocessing
    Lsc = [10e-6,50e-6,250e-6]
    s = 100
    lengths = numpy.linspace(50e-6,1000e-6,s)
    rocs = numpy.linspace(50e-6,1000e-6,s-1)

    pars = [(S,L,R) for S in Lsc for L in lengths for R in rocs]

    for scat in Lsc:
        path =cwd+"\\data\\LR_space\\S{:.0f}\\".format(scat*1e6)
        if not os.path.exists(path):
            os.makedirs(path)


    with multiprocessing.Pool(3) as p:
        p.map(_LR_space,pars)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            +"EXPORTING"+
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #input data is the same for all misalignments so dont need new objects to
    #store it

    #output data is individual for each misalignment
    Thetas = numpy.zeros((len(Lsc),len(lengths),len(rocs)))
    Phis = Thetas.copy()
    Diameters = Thetas.copy()
    P = Thetas.copy()

    for k,S in enumerate(Lsc):
        for i,L in enumerate(lengths):
            for j,R in enumerate(rocs):
                filename = cwd+"\\data\\LR_space\\S{:.0f}\\L{:.0f}R{:.0f}.csv".format(S*1e6,L*1e6,R*1e6)
                data = numpy.genfromtxt(filename,skip_header=1,delimiter=',')
                Thetas[k,i,j] = data[0]
                Phis[k,i,j] = data[1]
                Diameters[k,i,j] = data[2]
                P[k,i,j] = data[3]

    data = {'ROC':rocs,
            'cavity_length':lengths,
            'N_plt':len(Lsc),
            'Theta':Thetas,
            'tilt_angle':Phis,
            'D_crit':Diameters,
            'misalignment':M,
            'scatter':Lsc,
            'P':P}

    with open(cwd+"\\data\\LR_space.pkl", "wb") as output_file:
        pickle.dump(data, output_file)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            +"clean up"+
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #cleanup temporary files
    for S in Lsc:
        for L in lengths:
            for R in rocs:
                os.remove(cwd+"\\data\\LR_space\\S{:.0f}\\L{:.0f}R{:.0f}.csv".format(S*1e6,L*1e6,R*1e6))
        os.rmdir(cwd+"\\data\\LR_space\\S{:.0f}".format(S*1e6))
