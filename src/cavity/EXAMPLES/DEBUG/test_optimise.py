import core
import numpy
import os

crit_loss = 1e-6
Lmin = 250e-6

L_lim = (Lmin,5e-3)

cwd = os.path.dirname(os.path.abspath(__file__))

fpath = cwd + "\\data\\"

def func(args):
    Vmax,Lscat,m = args
    fname = "Optimum_Lmin-{:.0f}um_Vmax-{:.0f}um3_Lscat-{:.0f}ppm_m-{:.0f}um.csv".format(Lmin*1e6,Vmax*1e18,Lscat*1e6,m*1e6)

    if os.path.isfile(fpath+fname):
        pass
    else:
        res = core.optimizer(Vmax,Lscat,m,L_lims=L_lim)

        data = [res['length'],
                res['roc'],
                res['diameter'],
                res['probability'],
                res['transmission']]

        numpy.savetxt(fpath+fname,data,
                        header = "L (m), R (m), D (m), P_ext, T (ppm)",
                        delimiter=',')
    return

if __name__ == "__main__":
    import multiprocessing

    atom = {"wavelength":1e-6,
            "alpha":1/20}
    try:
        os.makedirs(fpath)
    except FileExistsError:
        pass


    V = numpy.logspace(-15,-12,50)
    scat = [200e-6]
    mis = numpy.linspace(0,25e-6,50)

    #use multiprocessing to speed up data generation
    pars = [(Vmax,Lscat,m) for Vmax in V for Lscat in scat for m in mis]
    with multiprocessing.Pool(7) as p:
        p.map(func,pars)
    #collate the data into one place
    for i,Vmax in enumerate(V):
        for j,Lscat in enumerate(scat):
            for k,m in enumerate(mis):

                fname = "Optimum_Lmin-{:.0f}um_Vmax-{:.0f}um3_Lscat-{:.0f}ppm_m-{:.0f}um.csv".format(Lmin*1e6,Vmax*1e18,Lscat*1e6,m*1e6)
                L,R,D,P,T,Ra,Su = numpy.genfromtxt(fpath+fname,skip_header=1,delimiter=',')
                length[i,j,k]=L
                roc[i,j,k] = R
                diameter[i,j,k] = D
                P_ext[i,j,k] = P
                Trans[i,j,k] = T

    dict = {"volume":V,
            "scatter_loss":scat,
            "misalignment":mis,
            "Trans":Trans,
            "length":length,
            "ROC":roc,
            "diameter":diameter,
            "P ext":P_ext}

    with open(fpath+"data.pkl","wb") as file:
        pickle.dump(dict,file)
