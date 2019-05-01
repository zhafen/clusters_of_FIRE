import numpy as np
from numba import jit, njit

# Model parameters, calibrated from Grudic et al 2019 simulation suite
Mmin = 100.
n_SFE = 1.
n_CFE_z = [1.4, 2] # 1% solar metallicity and solar metallicity values for index of bound fraction as a function of surface density
sigma_CFE_z = [330, 410] # 1% solar metallicity and solar metallicity values for critical surface density for 50% bound fraction
alpha_M_z = [-1.9, -1.6] # 1% solar metallicity and solar metallicity values for mass function slope from a given cloud
k_M_z = [0.13,0.09] # turnover parameter for cloud-level mass function 
sigma_SFE = 3200. 
SFEmax = 0.55
CFEmax = 1
mass_unit_msun = 1e10
length_unit_pc = 1e3

def time_to_redshift(times):
    z, t = np.loadtxt("snapshot_times.txt")[:,2:4].T
    return np.interp(times, t, z)

def snapnum_to_time(num):
    data = np.loadtxt("snapshot_times.txt")
    n = data[:,0]
    t = data[:,3]
    return np.interp(num, n, t)

@njit
def MassCDF(m, mtot, alpha, k):
    if m==mtot: return 0
    return m**(alpha + 1) * np.exp(-k / (1 - m/mtot))

@jit
def SampleMassFunc(mass_budget, mmin, metallicity,  seed=42):
    """Returns a stochastically-sampled list of masses from a ~M^-2 mass function that goes between mmin and mmax"""
    np.random.seed(seed)
    masses = []
    mtot = 0.

    alpha = np.interp(np.log10(metallicity), [-2,0], alpha_M_z)
    k = np.interp(np.log10(metallicity), [-2,0], k_M_z)

    Nmax = MassCDF(mmin/mass_budget, 1, alpha, k) # mmin**(alpha + 1) * np.exp(k/(1 - mmin/mass_budget))
 #   print(alpha, k, Nmax)
    # do bisection method on the CDF (slow but safe)
    while mtot < 1: #mass_budget:
        x = np.random.rand() * Nmax
        a, b = mmin/mass_budget, 1
        m = (a+b)/2
        f = Nmax #MassCDF(m, mtot, alpha, k)
        while np.abs(f - x) > 1e-2:
            fa = MassCDF(a, 1, alpha, k)
            fb = MassCDF(b, 1, alpha, k)
            f = MassCDF((a+b)/2, 1, alpha, k)
#            print(a,b, f)
            if (f-x)*(fa-x) < 0: # root is between a and (a+b)/2
                b, m = m, (m + a)/2
            else: # root is between (a+b)/2 and b
                a, m = m, (m + b)/2
        mtot += m 
        masses.append(m)

    mtot *= mass_budget
    masses = np.array(masses) * mass_budget
    mass_error = mtot - mass_budget
    if abs(mass_error - masses[-1]) < abs(mass_error): # if rejecting the last one gets us closer to budget, do it
        masses = masses[:-1]
    return np.array(masses)

def SampleSizeFunc(masses, M_GMC, R_GMC, metallicity, seed=42):
    np.random.seed(seed)
#    Rmin = (masses / 1e6 / np.pi)**0.5
    return 3 * (R_GMC/1e6)**.2 * (masses/1e4)**(1./3) * (M_GMC / (np.pi*R_GMC**2))**-1. * metallicity**.1 * 10**(np.random.normal(size=masses.shape)*0.38)

def SampleEFFGamma(N=1, seed=42):
    np.random.seed(seed)
    x = np.random.rand(N)
    X = np.linspace(0,8,1000)
    return np.interp(x, (X/(1.2+X))**0.54, X) + 2  # fit to the distribution of EFF slope parameters from Grudic 2017


class ClusterPopulation:
    """Class that implements the Grudic et al 2019 model for the cluster population properties from a single GMC/OB association, given bulk cloud properties."""
    def __init__(self, snapnum=None, cloud_id=None, cloud_data=None, M_GMC=None, R_GMC=None, tform=None, metallicity=None, tracers=None, seed_offset=0, corrdata=None):
#        np.random.seed(snapnum)
        if cloud_data is not None:
            m = np.array(cloud_data["PartType0"]["Masses"])
            x = np.array(cloud_data["PartType0"]["Coordinates"])
            z = np.array(cloud_data["PartType0"]["Metallicity"][:,0])
            com = np.average(x, axis=0, weights=m)
            r = np.sqrt(np.sum((x-com)*(x-com),axis=1))
            M_GMC = mass_unit_msun * m.sum()
#            self.M_GMC = np.array([M_GMC,])
            R_GMC = np.sqrt(np.average(r**2, weights=m) * 5/3) * length_unit_pc
#            self.R_GMC = np.array([R_GMC,])
            metallicity = np.average(z, weights=m)
            tracers = np.array(cloud_data["Tracers"])
            
        if tform is None and snapnum is not None:
            tform = snapnum_to_time(snapnum)

            
        if snapnum is not None and cloud_id is not None:
            seed = snapnum*1000 + int(cloud_id.replace("Cloud","")) + seed_offset
            corrdata = np.loadtxt(corrdata)
            corrdict = dict(zip(corrdata[:,0], corrdata[:,1]))
            mass_correction_fac = corrdict[snapnum]
        else:
            seed = seed_offset
            mass_correction_fac = 1.
            
        np.random.seed(seed)
        self.Sigma_GMC = M_GMC/(np.pi*R_GMC**2)
        n_CFE = np.interp(np.log10(metallicity), [-2,0], n_CFE_z)
        sigma_CFE = np.interp(np.log10(metallicity), [-2,0], sigma_CFE_z)
        # The following three lines model the scalings obtained in the Grudic et al. 2019 GMC simulations
        self.SFE = (1./SFEmax + (sigma_SFE/self.Sigma_GMC)**n_SFE)**-1.   # star formation efficiency
        # model variance in CFE with a logarithmic variance in the effective surface density
        delta = np.random.normal()*0.6
        self.CFE = (1./CFEmax + (sigma_CFE/self.Sigma_GMC/np.exp(delta))**n_CFE)**-1 # fraction of stars in bound clusters
        self.Mstar = self.SFE * M_GMC * mass_correction_fac   # total stellar mass
#        Mmax = self.Mstar / mass_correction_fac * (1 + (sigma_Mmax/self.Sigma_GMC)**alpha_Mmax)**-1. # maximum cluster mass

        self.Mbound = self.CFE * self.Mstar
#        Mmax = min(self.Mbound, Mmax)  # can't have a cluster more massive than the total bound mass
        if (self.Mbound > Mmin):
            self.ClusterMasses = SampleMassFunc(self.Mbound, Mmin, metallicity, seed = seed) # masses of the star clusters
            self.Mbound = self.ClusterMasses.sum()
        else:
            self.ClusterMasses = np.array([])
            self.Mbound = 0.
            self.FieldMass = np.array([self.Mstar - self.Mbound,])
            self.FieldMetallicity = np.array([metallicity,])                
            
        self.NumClusters = len(self.ClusterMasses)
        if snapnum is not None and cloud_id is not None:
            self.ClusterIDs = ["%d.%s.%d"%(snapnum, cloud_id.replace("Cloud",""), i) for i in range(self.NumClusters)]
        else:
            self.ClusterIDs = np.arange(len(self.ClusterMasses))
        self.ClusterMetallicity = np.repeat(metallicity, self.NumClusters)
        self.R_GMC = np.repeat(R_GMC, self.NumClusters)
        self.ClusterRadii = SampleSizeFunc(self.ClusterMasses, M_GMC, self.R_GMC, self.ClusterMetallicity, seed = seed + 1)   # 3D half-mass radii of the clusters
        self.EFF_Gamma = SampleEFFGamma(self.NumClusters, seed = seed + 2)  # EFF slope parameter gamma

        self.ClusterMetallicity = np.repeat(metallicity, self.NumClusters)

        self.M_GMC = np.repeat(M_GMC, self.NumClusters)

        if snapnum is not None and cloud_id is not None:
            self.ClusterFormationTime = np.repeat(tform, self.NumClusters)
            self.ClusterFormationRedshift = np.repeat(time_to_redshift(tform), self.NumClusters)
            self.FieldFormationTime = np.array([tform,])
            self.ClusterFormationRedshift = np.repeat(time_to_redshift(tform), self.NumClusters)
            self.FieldFormationRedshift = np.array([time_to_redshift(tform),])            
            if np.size(tracers) > 1: # choose from the available galactocentric radii randomly
                self.ClusterTracers = np.random.choice(tracers, self.NumClusters)
            else:
                self.ClusterTracers = np.repeat(tracers[0], self.NumClusters)
            self.FieldTracers = [tracers,]                
        else:
            self.ClusterFormationTime = np.repeat(0, self.NumClusters)
            self.ClusterFormationRedshift = np.repeat(0, self.NumClusters)
            self.FieldFormationTime = np.array([0,])
            self.ClusterFormationRedshift = np.repeat(0, self.NumClusters)
            self.FieldFormationRedshift = np.array([0,])            
            self.ClusterTracers = np.repeat(None, self.NumClusters)
            self.FieldTracers = [None,]
            
        
    def __add__(self, pop2):
        if pop2 == 0: return self
        result = ClusterPopulation()
        result.ClusterMasses = np.concatenate([self.ClusterMasses, pop2.ClusterMasses])
        result.ClusterRadii = np.concatenate([self.ClusterRadii, pop2.ClusterRadii])
        result.M_GMC = np.concatenate([self.M_GMC, pop2.M_GMC])
        result.R_GMC = np.concatenate([self.R_GMC, pop2.R_GMC])        
        result.ClusterFormationTime = np.concatenate([self.ClusterFormationTime, pop2.ClusterFormationTime])
        result.EFF_Gamma = np.concatenate([self.EFF_Gamma, pop2.EFF_Gamma])
        result.ClusterMetallicity = np.concatenate([self.ClusterMetallicity, pop2.ClusterMetallicity])
        result.NumClusters = self.NumClusters + pop2.NumClusters
        result.ClusterFormationTime = np.concatenate([self.ClusterFormationTime, pop2.ClusterFormationTime])
        result.ClusterFormationRedshift = np.concatenate([self.ClusterFormationRedshift, pop2.ClusterFormationRedshift])
        result.ClusterTracers = np.concatenate([self.ClusterTracers, pop2.ClusterTracers])
        result.ClusterIDs = np.concatenate([self.ClusterIDs, pop2.ClusterIDs])
        result.FieldTracers = self.FieldTracers + pop2.FieldTracers
        result.Mstar = self.Mstar + pop2.Mstar
        result.FieldMass = np.concatenate([self.FieldMass, pop2.FieldMass])
        result.FieldMetallicity = np.concatenate([self.FieldMetallicity, pop2.FieldMetallicity])
        result.FieldFormationTime = np.concatenate([self.FieldFormationTime, pop2.FieldFormationTime])
        return result

    def __radd__(self, pop2):
        if pop2 == 0: return self
        result = ClusterPopulation()
        result.ClusterMasses = np.concatenate([self.ClusterMasses, pop2.ClusterMasses])
        result.ClusterRadii = np.concatenate([self.ClusterRadii, pop2.ClusterRadii])
        result.M_GMC = np.concatenate([self.M_GMC, pop2.M_GMC])
        result.R_GMC = np.concatenate([self.R_GMC, pop2.R_GMC])        
        result.ClusterFormationTime = np.concatenate([self.ClusterFormationTime, pop2.ClusterFormationTime])
        result.EFF_Gamma = np.concatenate([self.EFF_Gamma, pop2.EFF_Gamma])
        result.ClusterMetallicity = np.concatenate([self.ClusterMetallicity, pop2.ClusterMetallicity])
        result.NumClusters = self.NumClusters + pop2.NumClusters
        result.ClusterFormationTime = np.concatenate([self.ClusterFormationTime, pop2.ClusterFormationTime])
        result.ClusterFormationRedshift = np.concatenate([self.ClusterFormationRedshift, pop2.ClusterFormationRedshift])
        result.ClusterTracers = np.concatenate([self.ClusterTracers, pop2.ClusterTracers])
        result.ClusterIDs = np.concatenate([self.ClusterIDs, pop2.ClusterIDs])
        result.FieldTracers = self.FieldTracers + pop2.FieldTracers
        result.Mstar = self.Mstar + pop2.Mstar
        result.FieldMass = np.concatenate([self.FieldMass, pop2.FieldMass])
        result.FieldMetallicity = np.concatenate([self.FieldMetallicity, pop2.FieldMetallicity])
        result.FieldFormationTime = np.concatenate([self.FieldFormationTime, pop2.FieldFormationTime])
        return result
