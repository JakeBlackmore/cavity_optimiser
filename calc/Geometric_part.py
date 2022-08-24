from cavity import core
import pickle
import numpy
import os
from scipy import optimize

cwd = os.path.dirname(os.path.abspath(__file__))
limit = 1e-6

def _critical_diam(D,cav,limit):
    clip = cav.clipping_loss(D=D) - limit
    return clip

def _divergence(par):
    M,L,R = par
    #print(M,L,R)
    fname = cwd+"\\data\\divergence\\M{:.0f}\\L{:.0f}R{:.0f}.csv".format(M*1e6,L*1e6,R*1e6)
    pars = {"length":L,
            "roc":R,
            "mis_par":M,
            "mis_perp":M,
            "diameter":R,
            "scatter_loss":1e-99, # non-zero to avoid errors
            }

    cav = core.Cavity(**pars)
    theta = numpy.rad2deg(cav.thetaprime)
    phi = numpy.rad2deg(cav.phi)
    try:
        root =  optimize.root_scalar(_critical_diam,args=(cav,limit),
                                            bracket=[0,2*R])
        critical_diam = root.root
    except ValueError:
        critical_diam = numpy.nan
    numpy.savetxt(fname,[theta,phi,critical_diam],delimiter=',',
                        header='theta (deg), phi (deg),diameter (m)')

    return

if __name__ == "__main__":
    import multiprocessing
    Ms = [0,10e-6,20e-6]
    s = 100
    lengths = numpy.linspace(50e-6,1000e-6,s)
    rocs = numpy.linspace(50e-6,1000e-6,s-1)

    pars = [(M,L,R) for M in Ms for L in lengths for R in rocs]

    for M in Ms:
        path =cwd+"\\data\\divergence\\M{:.0f}\\".format(M*1e6)
        if not os.path.exists(path):
            os.makedirs(path)


    with multiprocessing.Pool(3) as p:
        p.map(_divergence,pars)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            +"EXPORTING"+
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #input data is the same for all misalignments so dont need new objects to
    #store it

    #output data is individual for each misalignment
    Thetas = numpy.zeros((len(Ms),len(lengths),len(rocs)))
    Phis = Thetas.copy()
    Diameters = Thetas.copy()

    for k,M in enumerate(Ms):
        for i,L in enumerate(lengths):
            for j,R in enumerate(rocs):
                filename = cwd+"\\data\\divergence\\M{:.0f}\\L{:.0f}R{:.0f}.csv".format(M*1e6,L*1e6,R*1e6)
                data = numpy.genfromtxt(filename,skip_header=1,delimiter=',')
                Thetas[k,i,j] = data[0]
                Phis[k,i,j] = data[1]
                Diameters[k,i,j] = data[2]

    data = {'ROC':rocs,
            'cavity_length':lengths,
            'N_plt':len(Ms),
            'Theta':Thetas,
            'tilt_angle':Phis,
            'D_crit':Diameters,
            'misalignment':Ms}

    with open(cwd+"\\data\\divergence.pkl", "wb") as output_file:
        pickle.dump(data, output_file)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            +"clean up"+
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #cleanup temporary files
    for M in Ms:
        for L in lengths:
            for R in rocs:
                os.remove(cwd+"\\data\\divergence\\M{:.0f}\\L{:.0f}R{:.0f}.csv".format(M*1e6,L*1e6,R*1e6))
        os.rmdir(cwd+"\\data\\divergence\\M{:.0f}".format(M*1e6))
