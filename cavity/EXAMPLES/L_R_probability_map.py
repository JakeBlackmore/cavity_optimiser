import core
import numpy
import os
from multiprocessing import Pool
import scipy.optimize as optimize
import pickle


cwd = os.path.dirname(os.path.abspath(__file__))

def critical_diam(L,R,cav,lim=1e-6):
    loss = lambda x: cav.clipping_loss(L,R,x)-lim
    try:
        root = optimize.root_scalar(loss,bracket=[0,2*R])
        D = root.root
        return D
    except ValueError:
        return numpy.nan

def _pool_fn(pars):
    L,R,S = pars
    cpars = {"length": L,
                "roc":R,
                "scatter_loss":S,
                "mis_par": 5e-6,
                "mis_perp":5e-6,
                "diameter":100e-6
                }
    cav = core.Cavity(**cpars)
    P,T = cav.probability()
    D = critical_diam(L,R,cav)
    V,h = core.spherical_cap(R,D)
    dat = [P,V]
    numpy.savetxt(cwd+"\\L_R_data\\S{:.1f}ppm\\L{:.1f}um_R{:.1f}um.csv".format(S*1e6,L*1e6,R*1e6),dat,delimiter=',')
    return

if __name__ == "__main__":
    X = 25
    Y = 25
    lengths = numpy.linspace(100e-6,1e-3,X)
    rocs = numpy.linspace(50e-6,1e-3,Y)

    scat = [10e-6,50e-6,250e-6]
    Z = len(scat)
    '''
    try:
        os.makedirs(cwd+"\\L_R_data\\")
    except FileExistsError:
        pass

    for s in scat:
        try:
            os.makedirs(cwd+"\\L_R_data\\S{:.1f}ppm\\".format(s*1e6))
        except FileExistsError:
            pass


    pars = [(l,r,s) for l in lengths for r in rocs for s in scat]

    with Pool(5) as p:
        p.map(_pool_fn,pars)
    '''
    Vol_crit = numpy.zeros((X,Y,Z))
    Prob = numpy.zeros((X,Y,Z))

    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                dat = numpy.genfromtxt(cwd+"\\L_R_data\\S{:.1f}ppm\\L{:.1f}um_R{:.1f}um.csv".format(scat[k]*1e6,lengths[i]*1e6,rocs[j]*1e6),delimiter=',')
                Prob[i,j,k] = dat[0]
                Vol_crit[i,j,k] = dat[1]

    output = {"length":lengths,
                "roc":rocs,
                "scatter":scat,
                "critical volume":Vol_crit,
                "probability":Prob}

    with open(cwd+"\\L_R_data\\data.pkl","wb")as file:
        pickle.dump(output,file)
