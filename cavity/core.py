import numpy
import scipy.constants as constants
import warnings
import scipy.integrate as integrate
import scipy.optimize as optimize

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
    sx = w/2.
    sy = w/2.

    #prefactor for area normalisation
    prefactor = 1./(2.*pi*sx*sy)

    #recentre the x-y axes to x0,y0
    xp = x-x0
    yp = y-y0

    #main Gaussian function - see e.g. https://en.wikipedia.org/wiki/Gaussian_function
    Gaussian = numpy.exp(-xp**2./(2.*sx**2.)-yp**2./(2.*sy**2.))

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
    zR = pi*w0**2./wl
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
    w = w0*numpy.sqrt(1.+(z/zR)**2.)
    return w


def spherical_cap(R,D):
    ''' function to return properties of a given spherical cap
    https://en.wikipedia.org/wiki/Spherical_cap

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
    V = (1./6.)*(pi*h)*(3.*a**2.+h**2.)
    #volume of cap  https://en.wikipedia.org/wiki/Spherical_cap
    return V,h

def volume_diameter(ROC,V):

    ''' Calculate the diameter for a cap of given volume and ROC.

    Uses a root finding algorithm with spherical_cap to return the diameter of
    a given mirror.

    If the mirror cannot be made using a spherical cap, returns NaN.

    args:
    -----
    ROC (float) - radius of curvature of the mirrors in m
    V (float) - volume of the spherical cap in m^3

    returns:
    --------
    D (float) - diameter of the mirror in m

    '''

    Vhemisphere = (2./3.)*pi*ROC**3

    if Vhemisphere > V:
        # hemisphere is the biggest mirror one can make for a fixed ROC. If we
        # want a mirror smaller than this we can make a cap.
        fn = lambda x: V-spherical_cap(ROC,x)[0]
        root = optimize.root_scalar(fn,bracket=[0,2*ROC],rtol=1e-4)
        D = root.root

    elif Vhemisphere == V:
        # edge case where the hemisphere of ROC R is exactly the volume to be
        # removed for the mirror. Above method won't work here as there is no
        # change of sign for the root finder.
        D = 2*ROC
    else:
        # in this scenario the mirror cannot be made with the desired ROC and
        # volume. i.e. we are trying to make more than a hemisphere.
        D = numpy.nan
    return D
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


        if (1.-L/roc)**2 <= 1. and (1.-L/roc)**2 >=0.:
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
                loss = 1.-i**2
            except ZeroDivisionError:
                #if remainder is identically zero then we have total loss.
                loss = 1.
            #present some warnings to the user to notify of possible issues
            if i>1.:
                #this case shouldn't ever occur
                if not self.FixedFlag:
                    self.FixedFlag = True
                    #warnings.warn("Value of clipping loss has been fixed to 0",Warning)
                loss = 0.
            if loss <= 0.:
                if not self.zeroFlag:
                    self.zeroFlag = True
                    #warnings.warn("Losses are zero to numerical precision",Warning)
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
            try:
                clip = self.clip
            except AttributeError:
                clip = self.clipping_loss()
            if Lscat == 0:
                #filter div/0 warnings
                #warnings.simplefilter("ignore")
                Lscat = 1e-9
                warnings.warn("Lscat has been set to 1 ppb",Warning)

            C = 6.*alpha*thetaprime**2/(Lscat+clip)
            self.C_int = C
        else:
            self.C_int = 0.
        return self.C_int

    def optimum_trans(self,C=None,Lscat=None):
        ''' optimum transmission for the cavity as given by eq.16 of gao et al.

        args:
        -----
        C (float) - Intrinsic cooperativity from the divergence of cavity mode
        Lscat (float) - the scattering loss of the cavity mirrors.

        returns:
        --------
        T (float) - transmission of the ideal cavity mirrors in parts.

        updates:
        --------
        trans (float) - transmission of the ideal cavity mirrors in parts.

        '''

        if C == None:
            try:
                C = self.C_int
            except AttributeError:
                C = self.calc_C_int()

        if Lscat == None:
            Lscat = self.scatter

        try:
            clip = self.clip
        except AttributeError:
            clip = self.clipping_loss()

        T = numpy.sqrt(2*C+1.)*(Lscat+clip)
        self.trans = T
        return T

    def probability(self,L=None,R=None):
        ''' Calculate the extraction probability for the selected cavity from
        the intrinsic cooperativity and the optimum transmission.

        returns the probability and updates the cavity.P_ext attribute.

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

        Lscat = self.scatter
        # if C has not already been calculated, recalculate.
        # should have been covered already by __init__
        try:
            Cint = self.C_int
        except AttributeError:
            Cint = self.calc_C_int(L,R)

        # if T has not already been calculated, recalculate.
        # should have been covered already by __init__
        try:
            T = self.trans
        except AttributeError:
            T = self.optimum_trans(Cint,Lscat)
        # if clipping hasn't been calculated, recalculate.
        # should be covered by __init__
        try:
            clip = self.clip
        except AttributeError:
            clip = self.clipping_loss()


        #C != Cint when T != T_opt

        C = Cint * ((Lscat+clip)/(T+Lscat+clip))

        # eq 10 from gao et al. also eq 1. of goto et al. DOI:10.1106/PhysRevA.99.053843

        #P_ext = 1. - 2./(1.+numpy.sqrt(1.+2.*C))

        # eq 9 from gao et al.
        P_ext = ((2*C)/(2*C+1))*(T/(T+(Lscat+clip)))
        #if we get an invalid answer due to e.g. division by zero, return 0
        if numpy.isnan(P_ext) or not numpy.isfinite(P_ext):
            P_ext = 0.
        else:
            self.P_ext = P_ext
        return P_ext, T

    def __init__(self,**kwargs):
        ''' initialistion of the class

        kwargs:
        -------
        length: cavity length (m)
        roc: cavity mirror radius of curvature (m)
        scatter_loss: scattering loss from mirror surface (parts per round trip)
        mis_par: misalignment along cavity axis (m)
        mis_perp: misalignment perpendicular to cavity axis (m)
        atom: parameters associated with the atomic part of the cavity-atom
                interface.


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

        try:
            self.trans = kwargs['transmission']
        except KeyError:
            self.optimum_trans()



        ''' basic check of cavity stability, also check for spherical cap clash
        and the diameter <= 2* roc '''
        if ((1-self.length/self.roc)**2. <= 1. and #|g|<=1
                (1-self.length/self.roc)**2 >=0. and #|g|>0
                spherical_cap(self.roc,self.diameter)[1]*2.<=self.length and #L>=2h
                self.diameter <= 2.*self.roc): #d>=2*R
            # 0 < g1,g2 <1
            # check that the cavity is longer than 2*h
            #2*ROC > diameter is true for up to hemispherical mirrors
            #print(self.roc,self.diameter/2)
            self.physical = True
        else:
            #not a physical geometry
            self.physical = False

        self.geometry()
        self.clipping_loss()
        self.probability()
'''
###############################################################################
INTERNAL FUNCTIONS
###############################################################################
'''


def _optimise_cavity(varypars,mis,scat,Vmax,crit_loss,trans=False):
    ''' ###########INTERNAL FUNCTION########################

    cost function for the optimiser to use. Do not use.

    returns the cavity having considered the limitations

    args:
    -----
    varypars (tuple) - (L,R)
    fixedparts (dict)- (mperp,mpar,Lscat,atom)

    returns:
    --------

    -P_ext: returns -1 x the extraction proability to be suitable for minimisation

    '''
    L,R = varypars
    fixedpars ={"mis_par":mis,"mis_perp":mis,"scatter_loss":scat}
    pars = fixedpars.copy()

    pars['length'] = L
    pars['roc'] = R

    # find the diameter of the mirror given the constraint on volume
    D = volume_diameter(R,Vmax)

    # if D cannot be met then skip doing any heavy lifting.
    if D == numpy.nan:
        if trans:
            return 0,0
        else:
            return 0.

    pars['diameter'] = D

    cav = Cavity(**pars)
    P_ext = cav.probability()[0]
    clip = cav.clip

    # include constraints into the objective function
    '''
    constraints are:

    clipping loss < critical_loss

    cavity must be physically possible, this includes:
    * 0<|g|<1
    * 2 * h < length
    * 2*R >= diameter

    if constraints aren't met return 0
    '''
    if clip <= crit_loss and cav.physical:
        if trans:
            T = cav.trans
            return -1*P_ext,T
        else:
            return -1*P_ext
    else:
        if trans:
            return 0.,0.
        else:
            return 0.

def _select_initial_point(bounds,args):
    ''' ###########INTERNAL FUNCTION########################

    Algorithm for generating the initial point for optimizer.

    - Selects a random point, away from the bounds in the L/R space
    - verifies that P_ext != 0, if it is returns this random point
    - if P_ext ==0 :
        - repeats up to 10 interations
        - after 10 iterations returns a point with a near-planar cavity.

    args:
    -----
    bounds (iterable) - list of tuple of (Lmin,Lmax),(Rmin,Rmax)
    args (tuple) - tuple of parameters for the cost function
    returns:
    --------
    p0 (tuple) - tuple of initial (L,R) in m

    '''
    Lmin,Lmax = bounds[0]
    Rmin,Rmax = bounds[1]

    p = 0
    i = 0

    while p<= 0. and i < 10:
        L0=numpy.random.rand()*(0.97*Lmax-1.03*Lmin)+(Lmin*1.03)

        R0=numpy.random.rand()*(0.97*Rmax-1.03*Rmin)+(Rmin*1.03)

        p = -1*_optimise_cavity((L0,R0),*args)

        i+=1

    if i ==10:
        # 20 µm move away from the limits to prevent the optimiser crashing into bounds
        L0 = Lmin+20e-6
        R0 = Rmax-20e-6

    p0 = (L0,R0)
    return p0

def _select_initial_point_simple(bounds,L,args):
    ''' ###########INTERNAL FUNCTION########################

    Algorithm for generating the initial point for simple optimizer.

    - Selects a random point, away from the bounds in the L/R space
    - verifies that P_ext != 0, if it is returns this random point
    - if P_ext ==0 :
        - repeats up to 10 interations
        - after 10 iterations returns a point with a near-planar cavity.

    args:
    -----
    bounds (tuple) - tuple of (Rmin,Rmax)
    args (tuple) - tuple of parameters for the cost function
    returns:
    --------
    R0 (float) - initial ROC in m

    '''
    Rmin,Rmax = bounds

    p = 0
    i = 0

    while p<= 0. and i < 10:
        R0=numpy.random.rand()*(0.97*Rmax-1.03*Rmin)+(Rmin*1.03)

        p = -1*_optimise_cavity((L,R0),*args)

        i+=1

    if i ==10:
        # 20 µm move away from the limits to prevent the optimiser crashing into bounds
        R0 = Rmax-20e-6

    return R0

def optimizer(Vmax,scatter_loss,misalignment,crit_loss=1e-6,
                    L_lims = None,R_lims =None,
                        all=False,tol=1e-3):
    ''' main optimiser function for the cavity.

    Optimise the cavity geometry subject to the constraint that volume of the
    mirror is given by Vmax. The optimiser uses the Nelder-Mead local
    optimisation algorithm with a random initial configuration to avoid bias in
    the returned parameters. To approximate a global search the algorithm is
    run twice, if the two extraction probabilities are within tolerance the
    first iteration is returned. If not the higher value is kept and the
    algorithm repeats until a suitable result can be found.

    The optimum cavity results are returned as a dictionary of parameters,
    suitable for building a Cavity object.

    args:
    -----

    required arguments:
    Vmax (float) - maximum allowed volume (m^3)
    scatter_loss (float) - scattering_loss (in parts per round trip)
    misalignment (float) - misalignment of the mirrors axially and radially (m)

    optional arguments:
    crit_loss (float) - critical loss level for clipping (in parts per round trip). Default: 1 ppm
    L_lims (tuple) - pair of minimum/maximum for cavity length (meters). Default: (1 µm,3 mm)
    R_lims (tuple) - pair of minimum/maximum for radius of curvature (meters). Default: (0 m,3 mm)
    all (bool) - return all steps in the Nelder-Mead simplex. Default: False

    returns:
    --------
    results (dict) - dictionary of results keys are:
    * length(float) - optimum length of cavity (m)
    * roc (float) - optimum radius of curvature of cavity mirrors (m)
    * diameter (float) -optimum diameter of the the cavity mirrors (m)
    * transmission (float) - optimum transmission for cavity mirrors (parts)
    * probability (float) - optimum extraction probability
    * path (numpy.ndarray) - path through the L/R optimsation space followed by the
                            Nelder-Mead algorithm (only if all == True). L/R pairs in m
    '''

    # unconstrained optimisation is not helpful as we require that the cavity
    # does not go to zero length, so we set some limits even if the user
    # has not specified limits.
    if L_lims == None:
        Lmin = 1e-6
        Lmax = 3e3
    else:
        Lmin = L_lims[0]
        Lmax = L_lims[1]

    if R_lims == None:
        Rmin = 0.1e-6
        Rmax = 3e-3
    else:
        Rmin = max([R_lims[0],.1e-6])
        Rmax = 1.5e3

    bounds = [(Lmin,Lmax),(Rmin,Rmax)]

    # parameters for the cavity that are not to be changed.
    fixed = (misalignment,scatter_loss,Vmax,crit_loss)


    ''' MAIN OPTIMISATION STEP OCCURS HERE '''

    p0 = _select_initial_point(bounds,fixed)

    #options for the optimiser. In future: open these up to the user
    opt = {"xatol":1e-6,"fatol":1e-6,"return_all":all,
                    'maxiter':4e3,'maxfev':4e3}

    #run the Nelder-Mead algorithm
    sol = optimize.minimize(_optimise_cavity,p0,method="Nelder-Mead",
                                bounds=bounds,args=fixed,options=opt)

    if sol.success:
        opt_pars = sol.x
        opt_eff,opt_T = _optimise_cavity(opt_pars,*fixed,trans=True)
        opt_eff = opt_eff*-1
        opt_pars = [*opt_pars,volume_diameter(opt_pars[1],Vmax)]

    else:
        # in this case the optimiser has failed. We need a set of parameters
        # to compare with so use the initial guess. It is probably bad so assume
        # that the P_ext = 0
        opt_T = 0
        opt_eff = 0
        opt_pars = [*p0,volume_diameter(p0[1],Vmax)]
    diff = True
    counter = 0
    while diff:

        p0 = _select_initial_point(bounds,fixed)

        sol1 = optimize.minimize(_optimise_cavity,p0,method="Nelder-Mead",
                                    bounds=bounds,args=fixed,options=opt)

        warnings.filterwarnings("default")
        if sol1.success:
            opt_pars_1 = sol1.x
            opt_eff_1,opt_T_1 = _optimise_cavity(opt_pars_1,*fixed,trans=True)
            opt_eff_1 = opt_eff_1*-1
            opt_pars_1 = [*opt_pars_1,volume_diameter(opt_pars_1[1],Vmax)]
            # don't need the final case as we should be confident in getting a
            # physical result!

        else:
            # in this case the optimiser has failed for some reason, we don't
            # want to consider these cases.
            opt_T_1 = 0
            opt_eff_1 = 0

        A = max([opt_eff,opt_eff_1])

        B = min([opt_eff,opt_eff_1])
        if A == 0 and B !=0:
            #corrects for div/0 errors by just using B
            opt_eff = opt_eff_1.copy()
            opt_pars = opt_pars_1.copy()
            opt_T = opt_T_1.copy()

        elif A == 0 and B ==0:
            # both are trash, but we may be getting a "true" result so count
            # how many times we end up here in a single calc
            counter += 1
            if counter == 5:
                diff = False
        elif B/A >= 1 - tol:
            # both are equal enough and we have found the same minima,
            # return the first result for simplicity.
            diff = False
        else:
            # possibly have a bad initial starting location so replace the options
            # if the probability is higher == it's a "better" optimum
            if opt_eff < opt_eff_1:
                opt_eff = opt_eff_1.copy()
                opt_pars = opt_pars_1.copy()
                opt_T = opt_T_1.copy()

    results = {'length':opt_pars[0],
                'roc':opt_pars[1],
                'diameter': opt_pars[2],
                'probability':opt_eff,
                'transmission':opt_T
                }

    if all:
        results['path'] = sol.allvecs

    return results

def simple_optimizer(Lmin,Vmax,scatter_loss,misalignment,crit_loss=1e-6,
                    R_lims =None,all=False,tol=1e-3):
    '''simple optimiser function for the cavity. Assumes that you want the
    shortest length cavity to improve the performance.

    Optimise the cavity geometry subject to the constraint that volume of the
    mirror is given by Vmax. The optimiser uses the Nelder-Mead local
    optimisation algorithm with a random initial configuration to avoid bias in
    the returned parameters. To approximate a global search the algorithm is
    run twice, if the two extraction probabilities are within tolerance the
    first iteration is returned. If not the higher value is kept and the
    algorithm repeats until a suitable result can be found.

    The optimum cavity results are returned as a dictionary of parameters,
    suitable for building a Cavity object.

    args:
    -----

    required arguments:
    Lmin (float) - the minimum cavity length (m^3)
    Vmax (float) - maximum allowed volume (m^3)
    scatter_loss (float) - scattering_loss (in parts per round trip)
    misalignment (float) - misalignment of the mirrors axially and radially (m)

    optional arguments:
    crit_loss (float) - critical loss level for clipping (in parts per round trip). Default: 1 ppm
    R_lims (tuple) - pair of minimum/maximum for radius of curvature (meters). Default: (0 m,3 mm)
    all (bool) - return all steps in the Nelder-Mead simplex. Default: False

    returns:
    --------
    results (dict) - dictionary of results keys are:
    * roc (float) - optimum radius of curvature of cavity mirrors (m)
    * diameter (float) -optimum diameter of the the cavity mirrors (m)
    * transmission (float) - optimum transmission for cavity mirrors (parts)
    * probability (float) - optimum extraction probability
    * path (numpy.ndarray) - path through the L/R optimsation space followed by the
                            Nelder-Mead algorithm (only if all == True). L/R pairs in m
    '''

    # unconstrained optimisation is not helpful as we require that the cavity
    # does not go to zero roc, so we set some limits even if the user
    # has not specified limits.

    if R_lims == None:
        Rmin = 0.1e-6
        Rmax = 3e-3
    else:
        Rmin = max([R_lims[0],.1e-6])
        Rmax = 1.5e3

    bounds = [Rmin,Rmax]

    # parameters for the cavity that are not to be changed.
    fixed = (misalignment,scatter_loss,Vmax,crit_loss)

    ''' MAIN OPTIMISATION STEP OCCURS HERE '''

    p0 = _select_initial_point_simple(bounds,Lmin,fixed)

    #options for the optimiser. In future: open these up to the user
    opt = { 'maxiter':4e3}

    simplify = lambda x: _optimise_cavity((Lmin,x),*fixed)

    #run the Nelder-Mead algorithm
    sol = optimize.minimize_scalar(simplify,
                                bounds=bounds,tol=1e-6,options=opt)

    if sol.success:
        opt_pars = sol.x
        opt_pars = (Lmin,opt_pars)
        opt_eff,opt_T = _optimise_cavity(opt_pars,*fixed,trans=True)
        opt_eff = opt_eff*-1
        opt_pars = [*opt_pars,volume_diameter(opt_pars[1],Vmax)]

    else:
        # in this case the optimiser has failed. We need a set of parameters
        # to compare with so use the initial guess. It is probably bad so assume
        # that the P_ext = 0
        opt_T = 0
        opt_eff = 0
        opt_pars = [Lmin,p0,volume_diameter(p0[1],Vmax)]

    results = {'length':Lmin,
                'roc':opt_pars[1],
                'diameter': opt_pars[2],
                'probability':opt_eff,
                'transmission':opt_T
                }

    if all:
        results['path'] = sol.allvecs

    return results
