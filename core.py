import numpy
import scipy.constants as constants
import warnings
import scipy.integrate as integrate
import scipy.optimize as optimize
from matplotlib import pyplot

'''
################################################################################
define universal constants here
################################################################################
'''
pi = numpy.pi
c = constants.c
h = constants.h

'''
################################################################################
Helper functions here
################################################################################
'''
def gaussian(x,y,x0,y0,w):
    ''' area normalised two-dimensional gaussian function

    variables:
    ----------
    x,y = coordinates in two orthogonal axes
    x0,y0 = centre point of beam in x,y
    w = gaussian waist

    returns:
    --------
    G (float): Gaussian normalised to unit area
    '''

    #define the Gaussian in terms of standard deviations for analysis
    #could re-write this to allow for elliptical beams if necessary
    sx = w/2
    sy = w/2

    #prefactor for area normalisation
    prefactor = 1/(2*pi*sx*sy)

    #recentre the x-y axes to x0,y0
    xp = x-x0
    yp = y-y0

    #main Gaussian function - see e.g. https://en.wikipedia.org/wiki/Gaussian_function
    Gaussian = numpy.exp(-xp**2/(2*sx**2)-yp**2/(2*sy**2))

    return Gaussian*prefactor

def rayleigh(wl,w0):
    ''' function to generate the Rayleigh range for a given wavelength and waist

    inputs:
    -------
    wl (float): wavelength of the laser in metres
    w0 (float): waist of the laser beam at the focus in metres

    returns
    -------
    zR (float) : Rayleigh range in metres
    '''
    zR = pi*w0**2/wl
    return zR

def waist(wl,w0,z):
    ''' function to generate the 1/e^2 beam waist at a given point along the
    beam in z.

    inputs:
    -------
    wl (float): wavelength of the laser in metres
    w0 (float): waist of the laser beam at the focus in metres
    zR (float) : Rayleigh range in metres

    returns
    -------
    w (float) : 1/e^2 beam waist in meters
    '''
    zR = rayleigh(wl,w0)
    w = w0*numpy.sqrt(1+(z/zR)**2)
    return w


def Spherical_cap(R,D):
    ''' function to return properties of a given spherical cap

    args:
    -----
    R - radius of curvature of the cap
    d - diameter of the cap
    returns:
    --------
    V - volume of the cap
    h - "height" of the cap from centre to edge. Also called "sag" or "sagitta".
    '''
    a = D/2. #radius of base of cap
    h = R - numpy.sqrt(R**2-a**2) #height of cap
    V = (1/6)*(pi*h)*(3*a**2+h**2)
    #volume of cap  https://en.wikipedia.org/wiki/Spherical_cap
    return V,h

'''
################################################################################
Optimiser Classes
################################################################################
'''
class Cavity:
    ''' class to cover the cavity optimisation problem.

    optimise methods perform a global search of the parameter space to calculate
    the maximum extraction probability given some constraints.

    local flag can be set to force a local search with a specified initial guess
    if performance is required.

    start up parameters are provided as keyword arguments.


    kwargs:
    -----------
    length = cavity length (meters)
    roc = cavity mirror radii of curvature (meters)
    scatter = mirror scattering loss per round trip (parts)
    diameter = mirror diameters (meters)
    atom = atomic properties to carry over

    methods:
    --------
    * clipping_loss(L,ROC,D) - calculates the clipping loss in parts for this geometry.
    * geometry(L,ROC) - calculates the geometric properties of the cavity.
    * calc_C_int(L,R) - Calculates the intrinsic Cooperativity.
    * critical_diam(L,ROC) - calculates the critical diameter where loss hits a
                            critical loss level
    * optimum_trans() - calculates the optimum transmission for the cavity
    * probability(L,ROC) - calculates the extraction probability for the L, ROC
                            provided.
    * optimise_L(Lmin) - optimise the cavity for a fixed ROC,scatter,diameter
    * optimise_R(Rmin) - optimise the cavity for a fixed L,scatter,diameter

    attributes:
    -----------
    length (float)- length of cavity in metres
    diameter (float) - diameter of the mirrors
    roc (float) - radii of curvature of the mirrors in metres
    scatter (float) - scattering loss of cavities in parts
    parallel (float) - misalignment along the cavity axis
    perpendicular (float) -misalignment perpendicular to the cavity axis
    atom (dict):
    * `wavelength` (float) - wavelength of the atomic transition in meters
    * `alpha` (float) - branching ratio of the atomic transition
    '''
    def geometry(self,L = None,roc = None, Mpar = None, Mperp=None, wl = None):
        '''determine geometric properties of the cavity

        if L, roc are not supplied uses the cavity attributes length and roc.

        args:
        -----
        L (float) - cavity length in meters
        roc (float) - cavity radius of curvature in meters
        Mpar (float) - misalignment along cavity axis
        Mperp (float) - misalignment across cavity axis

        returns:
        --------
        phi (float) - tilt angle of the cavity in radians
        Lprime (float) - effective length of the cavity in meters
        thetaprime (float) - divergence half angle of the tilted mode in radians
        w0 (float) - waist of the tilted cavity mode

        updates:
        --------
        phi (float) - tilt angle of the cavity in radians
        Lprime (float) - effective length of the cavity in meters
        thetaprime (float) - divergence half angle of the tilted mode in radians
        w0 (float) - waist of the tilted cavity mode

        physical (bool) - flag for physicality of the cavity
        '''
        # check inputs, if None then use the attributes of the cavity class
        if L == None:
            L = self.length
        if roc == None:
            roc = self.roc
        if Mpar == None:
            Mpar = self.parallel
        if Mperp == None:
            Mperp = self.perpendicular
        if wl == None:
            _lambda = self.atom['wavelength']

        '''apply simple test of cavity stability:
            https://en.wikipedia.org/wiki/Optical_cavity#Stability
        '''
        if (1-L/roc)**2 <= 1 and (1-L/roc)**2 >=0:
            self.physical = True
        else:
            self.physical = False

        if self.physical:
            #find the tilted mode and effective lengths of the cavity

            with warnings.catch_warnings() as w:
                #lets us catch runtime warnings as errors so try/except works without
                #having to include logical tests.
                warnings.simplefilter("error")

                try:
                    # Eq3a
                    phi = numpy.arctan(Mperp/((2*roc-L)-Mpar))
                    # Eq3b
                    Lprime = 2*roc - (2*roc-L-Mpar)/numpy.cos(phi)
                    # Eq3c
                    thetaprime = numpy.sqrt(2*_lambda/(pi*numpy.sqrt(Lprime*(2*roc-Lprime))))

                except RuntimeWarning:
                    #catches when the sqrt in thetaprime is non-physical
                    phi = numpy.nan
                    Lprime = numpy.nan
                    thetaprime = numpy.nan
                    self.physical = False
                '''now calculate the waist of the cavity mode,
                 given these properties
                 '''

                try:
                    w0 = numpy.sqrt(_lambda/(2*pi))*(Lprime*(2*roc-Lprime))**0.25
                except RuntimeWarning:
                    #w0 is probably returning as a NaN so set it to NaN and
                    #raise the nonphysical flag
                    w0 = numpy.nan
                    self.physical = False

        else:
            ''' if cavity is non-physical, it makes no sense to
            speak about the tilted geometry so we return nan.
            '''
            phi = numpy.nan
            Lprime = numpy.nan
            thetaprime = numpy.nan
            w0 = numpy.nan

        ''' save each of the variables as a "prime" as a property of the Cavity
        object except for tilt.
        '''

        self.phi = phi
        self.Lprime = Lprime
        self.thetaprime = thetaprime
        self.waist = w0

        return phi,Lprime,thetaprime,w0

    def clipping_loss(self,L=None,R=None,D=None,recalculate=False):
        ''' calculates the clipping loss of the mode on the mirror assuming a
        hard spherical edge.

        We assume that the misalignment moves the mode only in x and that the
        mirror is centred at x = y = 0.

        args:
        -----
        L (float) - cavity length in meters
        R (float) - radii of curvature of cavity mirrors in meters
        D (float) - diameter of the cavity mirrors in meters
        recalculate (bool) - whether to recalculate the geometric arguments

        returns:
        --------
        loss (float) - clipping loss in parts per round trip

        updates:
        --------
        physical (bool) - flag for physicality of cavity
        clip (float) - clipping loss in parts per round trip

        '''
        # check inputs, if None then use the attributes of the cavity class
        if L == None:
            L = self.length
        if R == None:
            R = self.roc
        if D == None:
            D = self.diameter

        if not recalculate:
            try:
                phi = self.phi
                Lprime = self.Lprime
                thetaprime = self.thetaprime
                w0 = self.w0
            except AttributeError:
                #calculate the tilted geometry
                phi,Lprime,thetaprime,w0 = self.geometry(L,R)
        else:
            #calculate the tilted geometry
            phi,Lprime,thetaprime,w0 = self.geometry(L,R)

        if D > 2*R:
            self.physical = False

        if self.physical:
            #cavity has passed the stability criterion
            radius = D/2.
            shift_x = numpy.tan(self.phi)*(L+self.parallel)

            #propagate the waist to the surface of the mirror
            waist_mirror = waist(self.atom['wavelength'],w0,self.Lprime*0.5)
            #positive and negative y_limits, based on x position
            y_pos = lambda x: +1*numpy.sqrt(radius**2-x**2)
            y_neg = lambda x: -1*numpy.sqrt(radius**2-x**2)

            #wrapper for the gaussian function to do the integration
            f = lambda y,x: gaussian(x,y,shift_x,0,waist_mirror)

            # i represents the remaining power, per bounce, on the mirror
            i,erri = integrate.dblquad(f,-radius,radius,y_neg,y_pos,
                                                epsabs=1e-16,epsrel=1e-10)

            #expect that the approx L=(1-i) is valid to the part-per thousand level when
            # x is at ppm level. percent accuracy is achieved at ~1/10k.

            #NB: not doing this right now: try loss as given siegman's definition of delta notation which should be more accurate...
            try:
                loss = 1-i**2
            except ZeroDivisionError:
                #if remainder is identically zero then we have total loss.
                loss = 1
            #present some warnings to the user to notify of possible issues
            if i>1:
                #this case shouldn't ever occur
                if not self.FixedFlag:
                    self.FixedFlag = True
                    warnings.warn("Value of clipping loss has been fixed to 0",Warning)
                loss = 0
            if loss <= 0:
                if not self.zeroFlag:
                    self.zeroFlag = True
                    warnings.warn("Losses are zero to numerical precision",Warning)
                loss = 1e-16
                #approximate limit of precision from numpy, prevents ghost contours
            self.clip = loss
            return loss

        else:
            #warnings.warn("Cavity is non-physical",Warning)
            #if cavity is non-physical then set losses to 1 (max)
            self.clip = 1.
            return 1.

    def calc_C_int(self,L = None,R = None,alpha = None,Lscat = None,recalculate=False):
        '''calculates the intrinsic cooperativity as given by Eq 15 of gao et al.

        args:
        -----
        L (float) - cavity length in meters
        R (float) - radii of curvature of cavity mirrors in meters
        alpha (float) - atomic branching ratio of the cavity transition
        Lscat (float) - scattering loss of the mirror surfaces in parts per round trip

        returns:
        --------
        C_int (float) - intrinsic cooperativity of the cavity

        updates:
        --------
        C_int (float) - intrinsic cooperativity of the cavity

        '''

        if L == None:
            L = self.length
        if R == None:
            R = self.roc

        if not recalculate:
            try:
                phi = self.phi
                Lprime = self.Lprime
                thetaprime = self.thetaprime
                w0 = self.w0
            except AttributeError:
                #calculate the tilted geometry
                phi,Lprime,thetaprime,w0 = self.geometry(L,R)
        else:
            #calculate the tilted geometry
            phi,Lprime,thetaprime,w0 = self.geometry(L,R)


        if self.physical:
            #only worth doing this if the cavity should actually exist.
            if alpha == None:
                alpha = self.atom['alpha']
            if Lscat == None:
                Lscat = self.scatter
            if Lscat == 0:
                #filter div/0 warnings
                #warnings.simplefilter("ignore")
                Lscat = 1e-9
                warnings.warn("Lscat has been set to 1 ppb",Warning)

            C = 6*alpha*thetaprime**2/Lscat
            self.C_int = C
        else:
            self.C_int = 0
        return self.C_int

    def optimum_trans(self,C=None,Lscat=None):
        ''' optimum transmission for the cavity as given by eq.16 of gao et al.

        args:


        '''

        if C == None:
            try:
                C = self.C_int
            except AttributeError:
                C = calc_C_int()

        if Lscat == None:
            Lscat = self.scatter

        T = numpy.sqrt(C+1)*Lscat
        self.trans = T
        return T

    def probability(self,L=None,R=None):
        ''' calculate the extraction probability for the selected cavity


        args:
        -----
        L (float) - cavity length
        R (float) - radii of curvature of the cavity
        returns:
        --------
        P_ext (float): extraction probability
        updates:
        --------
        P_ext (float): extraction probability
        '''

        #if not supplied, use the cavity attributes.
        if L == None:
            L = self.length
        if R == None:
            R = self.roc
        try:
            C = self.C_int
        except AttributeError:
            C = self.calc_C_int(L,R)

        try:
            T = self.trans
        except AttributeError:
            T = self.optimum_trans()

        P_ext = 1 - 2/(1+numpy.sqrt(1+2*C))

        self.P_ext = P_ext

        return P_ext, T

    def __init__(self,**kwargs):
        ''' initialistion of the class

        kwargs:
        -------
        length:
        roc:
        scatter_loss:
        mis_par:
        mis_perp:
        atom:

        '''
        ''' some flags for calculations.

        These will go True when the criterion is met and tells us about
        the optimisiation process. Later we can use these to filter the
        results
        '''
        self.zeroFlag = False
        self.LowFlag = False
        self.FixedFlag = False

        ''' Extract cavity parameters from input kwargs
        '''

        self.length = kwargs['length']
        self.roc = kwargs['roc']
        self.scatter = kwargs['scatter_loss']
        self.parallel = kwargs['mis_par']
        self.perpendicular = kwargs['mis_perp']
        self.diameter = kwargs['diameter']

        ''' properties of the atomic transitions of interest
        '''
        self.atom = {"wavelength":1e-6,
                    "alpha":1./20.}

        ''' basic check of cavity stability '''
        if (1-self.length/self.roc)**2 <= 1 and (1-self.length/self.roc)**2 >=0:
            # 0 < g1,g2 <1
            if Spherical_cap(self.roc,self.diameter)[1]*2<=self.length:
                # check that the cavity is longer than 2*h
                if self.diameter <= 2*self.roc:
                    #2*ROC > diameter is true for up to hemispherical mirrors
                    self.physical = True
                else:
                    #beyond hemispherical mirrors
                    self.physical = False
            else:
                #not a physical geometry based on clashing mirrors
                self.physical=False
        else:
            #not a physical geometry based on g1,g2
            self.physical = False


def _clipping_loss(varypars,mis,scat):
    ''' ###########INTERNAL FUNCTION########################

    constraint function for the optimiser to use. Do not use.

    returns the cavity having considered the limitations

    args:
    -----
    varypars (tuple) - (L,R,D)
    fixedparts (dict)- (mperp,mpar,Lscat,atom)

    returns:
    --------

    clip: clipping_loss

    '''
    L,R,D = varypars

    fixedpars ={"mis_par":mis,"mis_perp":mis,"scatter_loss":scat}

    pars = fixedpars.copy()

    pars['length'] = L
    pars['roc'] = R
    pars['diameter'] = D

    cav = Cavity(**pars)
    clip = cav.clipping_loss()

    return clip


def _optimise_cavity(varypars,mis,scat,*args):
    ''' ###########INTERNAL FUNCTION########################

    cost function for the optimiser to use. Do not use.

    returns the cavity having considered the limitations

    args:
    -----
    varypars (tuple) - (L,R,D)
    fixedparts (dict)- (mperp,mpar,Lscat,atom)

    returns:
    --------

    -P_ext: returns -1 x the extraction proability to be suitable for minimisation

    '''
    L,R,D = varypars

    fixedpars ={"mis_par":mis,"mis_perp":mis,"scatter_loss":scat}

    pars = fixedpars.copy()

    #print(L,R,D)


    pars['length'] = L
    pars['roc'] = R
    pars['diameter'] = D

    cav = Cavity(**pars)
    P_ext = cav.probability()[0]

    return -1*P_ext

def optimizer(Vmax,scatter_loss,misalignment,crit_loss=1e-6,
                    L_lims = None,R_lims =None,D_lims=None, local = False):
    ''' main optimiser function for the cavity.

    args:
    -----

    Vmax (float) - maximum allowed volume in m^3
    scatter_loss (float) - scattering_loss (in parts per round trip)
    misalignment (float) - misalignment of the mirrors axially and radially (m)
    crit_loss (float) - critical loss level for clipping (in parts per round trip), defaults to 1 ppm

    L_lims (tuple) - pair of minimum/maximum for cavity length
    D_lims (tuple) - pair of minimum/maximum for diameter
    R_lims (tuple) - pair of minimum/maximum for radius of curvature
    local (boolean) - whether to perform a global or local search, defaults to False

    returns:
    --------

    '''
    if L_lims == None:
        Lmin = 1e-6
        Lmax = 3e3
    else:
        Lmin = max([L_lims[0],1e-6])
        Lmax = 3e-3

    if R_lims == None:
        Rmin = 1e-6
        Rmax = 1.5e-3
    else:
        Rmin = max([R_lims[0],1e-6])
        Rmax = 1.5e3

    if D_lims == None:
        Dmin = 2e-6
        Dmax = 3e-3
    else:
        Dmin = max([D_lims[0],1e-6])
        Dmax = 3e3

    bounds = [(Lmin,Lmax),(Rmin,Rmax),(Dmin,Dmax)]

    #currently these are hard-coded on line 504
    #wavelength = 1e-6,
    #alpha = 1/20

    scatter = 1e-6
    mis = 5e-6

    fixed = (misalignment,scatter_loss)

    # x = (L,R,D)
    volume_constraint_fn = lambda x: Vmax - Spherical_cap(x[1],x[2])[0]
    clip_constraint_fn = lambda x: crit_loss - _clipping_loss(x,*fixed)

    #geometric constraints, trying to keep optimizer away from these regions
    roc_diameter_constraint_fn = lambda x: x[1] - 0.5*x[2] #R > D/2
    length_roc_constraint_fn = lambda x: 2*x[1] - x[0] #2*R > L
    length_h_constraint_fn = lambda x: x[0]- 2*Spherical_cap(x[1],x[2])[1] #L > 2*h


    constraints = [{"type":"ineq","fun":clip_constraint_fn},
                    {"type":"ineq","fun":volume_constraint_fn},
                    {"type":"ineq","fun":roc_diameter_constraint_fn},
                    {"type":"ineq","fun":length_roc_constraint_fn},
                    {"type":"ineq","fun":length_h_constraint_fn}]



    ''' MAIN OPTIMISATION STEP OCCURS HERE '''
    warnings.filterwarnings("ignore")
    if not local:

        #options for the optimizer
        opt = {"f_tol":1e-6}

        sol = optimize.shgo(_optimise_cavity,bounds,args=fixed,n =200,iters=5,
                            constraints=constraints,options=opt)
    else:

        x0 = [numpy.random.rand()*(Lmax-Lmin)+Lmin,
                numpy.random.rand()*(Rmax-Rmin)+Rmin,
                numpy.random.rand()*(Dmax-Dmin)+Dmin]
        sol = optimize.minimize(_optimise_cavity,x0,args=fixed,bounds=bounds,
                                constraints=constraints,tol=1e-3)
    warnings.filterwarnings("default")

    if sol.success:
        opt_pars = sol.x
        opt_eff = sol.fun*-1
    else:
        #print("failed")
        opt_pars = [numpy.nan,numpy.nan,numpy.nan]
        opt_eff = 0
    return opt_pars,opt_eff

if __name__ == "__main__":

    Lscat = 50e-6
    m = 5e-6
    crit_loss = 1e-6
    Vmax = 2e-12 #m^3

    L_lim = (300e-6,10e-3)

    atom = {"wavelength":1e-6,
            "alpha":1/20}

    optimum, output = optimizer(Vmax,crit_loss,Lscat,m,L_lims=L_lim,local=True)
    print("opt",optimum)

    pars = {"length":optimum[0],
            "roc":optimum[1],
            "diameter":optimum[2],
            "mis_perp":m,
            "mis_par":m,
            "scatter_loss":Lscat,
            "atom":atom}
    cav = Cavity(**pars)

    print(cav.probability()[0])
