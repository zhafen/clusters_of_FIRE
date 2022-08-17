import numpy as np
from numba import jit, njit, vectorize
import matplotlib.pyplot as plt
from collections.abc import Iterable

# Model parameters, calibrated from Grudic et al 2021 simulation suite
Mmin = 1e2
n_SFE = 1.
n_CFE_z = [1.35, 2.05] # 1% solar metallicity and solar metallicity values for index of bound fraction as a function of surface density
sigma_CFE_z = [320, 390] # 1% solar metallicity and solar metallicity values for critical surface density for 50% bound fraction
alpha_M_z = [-1.9, -1.6] # 1% solar metallicity and solar metallicity values for mass function slope from a given cloud
k_M_z = [0.13,0.09] # turnover parameter for cloud-level mass function 
sigma_SFE = 3200. 
SFEmax = 0.8
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
#    if m==mtot: return 0
    return m**(alpha + 1) * np.exp(-k / (1 - m/mtot))

@jit(forceobj=True)
def SampleMassFunc(mass_budget, mmin, metallicity,  seed=42):
    """Returns a stochastically-sampled list of masses from the special cloud-level mass function described in Grudic et al 2019"""
    np.random.seed(seed)
    masses = []
    mtot = 0.

    alpha = np.interp(np.log10(metallicity/0.02), [-2,0], alpha_M_z)
    k = np.interp(np.log10(metallicity/0.02), [-2,0], k_M_z)

    Nmax = MassCDF(mmin/mass_budget, 1, alpha, k) # mmin**(alpha + 1) * np.exp(k/(1 - mmin/mass_budget))
    mgrid = np.logspace(np.log10(mmin/mass_budget), -1e-7, 1000)
    Nm = MassCDF(mgrid, 1, alpha, k)
    Nm[-1] = 0.
    Nm = Nm.max() - Nm
    while mtot < 1: #mass_budget:
        x = np.random.rand(100) * Nmax
        m = np.interp(x, Nm, mgrid)
        mtot += m.sum()
        masses.append(m)
        
    masses = np.concatenate(masses)
    idx = np.arange(len(masses))[masses.cumsum() > 1][0]
    mtot = masses[:idx+1].sum()
    masses = masses[:idx+1]
    mtot *= mass_budget
    masses = np.array(masses) * mass_budget
    mass_error = mtot - mass_budget
    if abs(mass_error - masses[-1]) < abs(mass_error): # if rejecting the last one gets us closer to budget, do it
        masses = masses[:-1]
    return np.array(masses)

@vectorize
def SampleMassFuncVectorized(mass_budget, mmin, metallicity,  seed=42):
    """Returns a stochastically-sampled list of masses from the special cloud-level mass function described in Grudic et al 2019"""
    np.random.seed(seed)
    masses = []
    mtot = 0.

    alpha = np.interp(np.log10(metallicity/0.02), [-2,0], alpha_M_z)
    k = np.interp(np.log10(metallicity/0.02), [-2,0], k_M_z)

    Nmax = MassCDF(mmin/mass_budget, 1, alpha, k) # mmin**(alpha + 1) * np.exp(k/(1 - mmin/mass_budget))
    mgrid = np.logspace(np.log10(mmin/mass_budget), -1e-7, 1000)
    Nm = MassCDF(mgrid, 1, alpha, k)
    Nm[-1] = 0.
    Nm = Nm.max() - Nm
    while mtot < 1: #mass_budget:
        x = np.random.rand(100) * Nmax
        m = np.interp(x, Nm, mgrid)
        mtot += m.sum()
        masses.append(m)
        
    masses = np.concatenate(masses)
    idx = np.arange(len(masses))[masses.cumsum() > 1][0]
    mtot = masses[:idx+1].sum()
    masses = masses[:idx+1]
    mtot *= mass_budget
    masses = np.array(masses) * mass_budget
    mass_error = mtot - mass_budget
    if abs(mass_error - masses[-1]) < abs(mass_error): # if rejecting the last one gets us closer to budget, do it
        masses = masses[:-1]
    return np.array(masses)

def SampleSizeFunc(masses, M_GMC, R_GMC, metallicity, seed=42):
    np.random.seed(seed)
    return 3 * (M_GMC/1e6)**(1./5) * (masses/1e4)**(1./3) * (M_GMC / (np.pi*R_GMC**2)/100)**-1. * (metallicity/0.02)**.1 * 10**(np.random.normal(size=masses.shape)*0.38)

def SampleEFFGamma(N=1, seed=42):
    np.random.seed(seed)
    x = np.random.rand(N)
    X = np.logspace(-2,2,1000)
    return np.interp(x, (X/(1.2+X))**0.54 * 1.064, X) + 2  # fit to the distribution of EFF slope parameters from Grudic 2017


class ClusterPopulation:
    """Class that implements the Grudic et al 2020 model for the cluster population properties from a single GMC/OB association, given bulk cloud properties."""
    def __init__(self, snapnum=None, cloud_id=None, cloud_data=None, M_GMC=None, R_GMC=None, X_GMC=None, tform=None, metallicity=None, tracers=None, seed_offset=0, corrdata=None, feedback_factor=1, delta_factor = 1, n_CFE_z=None, sigma_CFE_z=None, alpha_M_z=None, k_M_z=None, Rgc=None, cloud_ids=None):
        # model parameters
        if n_CFE_z is None: n_CFE_z = [1.35, 2.05]
        if sigma_CFE_z is None: sigma_CFE_z = [330, 385] # 1% solar metallicity and solar metallicity values for critical surface density for 50% bound fraction
        if alpha_M_z is None: alpha_M_z = [-1.9, -1.6] # 1% solar metallicity and solar metallicity values for mass function slope from a given cloud
        if k_M_z is None: k_M_z = [0.13,0.09] # turnover parameter for cloud-level mass function 
        
# #        np.random.seed(snapnum)
#         if cloud_data is not None and (M_GMC is None or R_GMC is None or metallicity is None or tracers is None):
# #            print("Computing bulk properties...")
#             m = np.array(cloud_data["PartType0"]["Masses"])
#             x = np.array(cloud_data["PartType0"]["Coordinates"])
#             z = np.array(cloud_data["PartType0"]["Metallicity"][:,0])
#             com = np.average(x, axis=0, weights=m)
#             r = np.sqrt(np.sum((x-com)*(x-com),axis=1))#
#             M_GMC = mass_unit_msun * m.sum()
#             R_GMC = np.sqrt(np.average(r**2, weights=m) * 5/3) * length_unit_pc
#             metallicity = np.average(z, weights=m)
#             if "Tracers" in cloud_data.keys():
#                 tracers = np.array(cloud_data["Tracers"])
            
        if tform is None and snapnum is not None:
            tform = snapnum_to_time(snapnum)
            
        if snapnum is not None and cloud_id is not None:
            seed = snapnum*1000 + int(cloud_id.replace("Cloud","")) + seed_offset
 #           corrdata = np.loadtxt(corrdata)
 #           corrdict = dict(zip(corrdata[:,0], corrdata[:,1]))
#            mass_correction_fac = 1. #corrdict[snapnum]
        else:
            seed = seed_offset
 #           mass_correction_fac = 1.
            
        np.random.seed(seed)
        
        self.M_GMC = M_GMC
        self.R_GMC = R_GMC
        self.Sigma_GMC = M_GMC/(np.pi*R_GMC**2)
        n_CFE = np.interp(np.log10(metallicity/0.02), [-2,0], n_CFE_z)
        sigma_CFE = np.interp(np.log10(metallicity/0.02), [-2,0], sigma_CFE_z)
        # The following three lines model the scalings obtained in the Grudic et al. 2019 GMC simulations
        self.GMC_SFE = (1./SFEmax + (sigma_SFE/self.Sigma_GMC *feedback_factor)**n_SFE)**-1.   # star formation efficiency
        # model variance in CFE with a logarithmic variance in the effective surface density
        delta = np.random.normal(size=(len(self.M_GMC),))*0.6 * delta_factor
        self.GMC_CFE = (1./CFEmax + (sigma_CFE/self.Sigma_GMC/np.exp(delta) * feedback_factor)**n_CFE)**-1 # fraction of stars in bound clusters
        self.GMC_Mstar = self.GMC_SFE * M_GMC   # total stellar mass
        self.GMC_Mbound_expected = self.GMC_CFE * self.GMC_Mstar
        if tform is not None:
            self.GMC_StarFormationTime = tform
        self.GMC_Metallicity = metallicity
        if X_GMC is not None:
            self.GMC_Pos = X_GMC
        if cloud_ids is not None:
            self.GMC_ID = cloud_ids
        if Rgc is not None:
            self.GMC_Rgc = Rgc
        if tracers is not None:
            self.GMC_Tracer = np.array([np.random.choice(tracers[i],1)[0] for i in range(len(tracers))])
            self.GMC_Tracers = tracers
        self.ClusterMasses = []
        self.ClusterRadii = []
        self.ClusterMetallicity = []
        self.Cluster_R_GMC = []
        self.Cluster_M_GMC = []
        self.ClusterFormationTime = []
        self.Cluster_Rgc = []
        self.Cluster_Pos = []
        self.Cluster_GMC_ID = []
        self.ClusterTracers = []
#        print(self.GMC_Mbound)
        for i,mb in enumerate(self.GMC_Mbound):
            if mb > Mmin*1.1:
                self.ClusterMasses.append(SampleMassFunc(mb, Mmin, metallicity[i], seed = (seed*i+i)%(2**32)))
                Ncl = len(self.ClusterMasses[-1])
                self.ClusterRadii.append(SampleSizeFunc(self.ClusterMasses[-1], M_GMC[i], R_GMC[i], metallicity[i], seed = (seed*i + i+1)%(2**32)))
                self.ClusterMetallicity.append(np.repeat(metallicity[i], Ncl))
                self.Cluster_R_GMC.append(np.repeat(R_GMC[i], Ncl))
                self.Cluster_M_GMC.append(np.repeat(M_GMC[i], Ncl))
                                    
                if tform is not None:
                    self.ClusterFormationTime.append(np.repeat(tform[i], Ncl))
                if Rgc is not None:
                    self.Cluster_Rgc.append(np.repeat(Rgc[i], Ncl))
                if X_GMC is not None:
#                    print(np.repeat([X_GMC[i]], Ncl,axis=0).shape, X_GMC[i].shape)
                    self.Cluster_Pos.append(np.repeat([X_GMC[i]], Ncl,axis=0))
                if cloud_ids is not None:
                    self.Cluster_GMC_ID.append(np.repeat(cloud_ids[i], Ncl))
                if tracers is not None:
                    if np.size(tracers[i]) > 1: # choose from the available galactocentric radii randomly
                        self.ClusterTracers.append(np.random.choice(tracers[i], Ncl))
                    else:
                        self.ClusterTracers.append(np.repeat(tracers[i][0], Ncl))
            else:
                self.ClusterMasses.append(np.array([]))
                self.ClusterRadii.append(np.array([]))
                if tform is not None: self.ClusterFormationTime.append(np.array([]))
                if Rgc is not None: self.Cluster_Rgc.append(np.array([]))
                if X_GMC is not None: self.Cluster_Pos.append(np.empty((0,3)))
                if cloud_ids is not None: self.Cluster_GMC_ID.append(np.array([]))
                if tracers is not None: self.ClusterTracers.append(np.array([]))
                self.Cluster_R_GMC.append(np.array([]))
                self.Cluster_M_GMC.append(np.array([]))
        self.GMC_Mbound = np.array([a.sum() for a in self.ClusterMasses]) # self.ClusterMasses is still a list of arrays
        self.GMC_NumClusters = np.array([len(a) for a in self.ClusterMasses]) 
        self.GMC_Mfield = self.GMC_Mstar - self.GMC_Mbound    
        if self.GMC_NumClusters.max() > 0:
            self.ClusterMasses = np.concatenate(self.ClusterMasses)
            self.ClusterRadii = np.concatenate(self.ClusterRadii)
            self.ClusterMetallicity = np.concatenate(self.ClusterMetallicity)
            self.Cluster_R_GMC = np.concatenate(self.Cluster_R_GMC)
            self.Cluster_M_GMC = np.concatenate(self.Cluster_M_GMC)
            if tform is not None:
                self.ClusterFormationTime = np.concatenate(self.ClusterFormationTime)
            if Rgc is not None:
                self.Rgc = np.concatenate(self.Rgc)
            if X_GMC is not None:
                #print(self.Cluster_Pos[1].shape)
                self.Cluster_Pos = np.concatenate(self.Cluster_Pos,axis=0)
            if cloud_ids is not None:
                self.Cluster_GMC_ID = np.concatenate(self.Cluster_GMC_ID)
            if tracers is not None:
                self.ClusterTracers = np.concatenate(self.ClusterTracers)
        else:
            self.ClusterMasses = np.array([])
            self.ClusterRadii = np.array([])
            self.ClusterMetallicity = np.array([])
            self.Cluster_R_GMC = np.array([])
            self.Cluster_M_GMC = np.array([])
            self.ClusterFormationTime = np.array([])
            self.Cluster_Rgc = np.array([])
            self.Cluster_Pos = np.empty((0,3))
            self.Cluster_GMC_ID = np.array([])
            self.ClusterTracers = np.array([])

        self.NumClusters = len(self.ClusterMasses) 

        self.Cluster_EFF_Gamma = SampleEFFGamma(self.NumClusters, seed = seed + 2)  # EFF slope parameter gamma            
            
        
    # def __add__(self, pop2):
    #     if pop2 == 0: return self
    #     result = ClusterPopulation() # create an empty population
    #     result.ClusterMasses = np.concatenate([self.ClusterMasses, pop2.ClusterMasses])
    #     result.ClusterRadii = np.concatenate([self.ClusterRadii, pop2.ClusterRadii])
    #     result.M_GMC = np.concatenate([self.M_GMC, pop2.M_GMC])
    #     result.R_GMC = np.concatenate([self.R_GMC, pop2.R_GMC])        
    #     result.ClusterFormationTime = np.concatenate([self.ClusterFormationTime, pop2.ClusterFormationTime])
    #     result.EFF_Gamma = np.concatenate([self.EFF_Gamma, pop2.EFF_Gamma])
    #     result.ClusterMetallicity = np.concatenate([self.ClusterMetallicity, pop2.ClusterMetallicity])
    #     result.NumClusters = self.NumClusters + pop2.NumClusters
    #     result.ClusterFormationTime = np.concatenate([self.ClusterFormationTime, pop2.ClusterFormationTime])
    #     result.ClusterFormationRedshift = np.concatenate([self.ClusterFormationRedshift, pop2.ClusterFormationRedshift])
    #     result.ClusterTracers = np.concatenate([self.ClusterTracers, pop2.ClusterTracers])
    #     result.ClusterIDs = np.concatenate([self.ClusterIDs, pop2.ClusterIDs])
    #     result.FieldTracers = self.FieldTracers + pop2.FieldTracers
    #     result.Mstar = self.Mstar + pop2.Mstar
    #     result.FieldMass = np.concatenate([self.FieldMass, pop2.FieldMass])
    #     result.FieldMetallicity = np.concatenate([self.FieldMetallicity, pop2.FieldMetallicity])
    #     result.FieldFormationTime = np.concatenate([self.FieldFormationTime, pop2.FieldFormationTime])
    #     return result

    # def __radd__(self, pop2):
    #     if pop2 == 0: return self
    #     result = ClusterPopulation()
    #     result.ClusterMasses = np.concatenate([self.ClusterMasses, pop2.ClusterMasses])
    #     result.ClusterRadii = np.concatenate([self.ClusterRadii, pop2.ClusterRadii])
    #     result.M_GMC = np.concatenate([self.M_GMC, pop2.M_GMC])
    #     result.R_GMC = np.concatenate([self.R_GMC, pop2.R_GMC])        
    #     result.ClusterFormationTime = np.concatenate([self.ClusterFormationTime, pop2.ClusterFormationTime])
    #     result.EFF_Gamma = np.concatenate([self.EFF_Gamma, pop2.EFF_Gamma])
    #     result.ClusterMetallicity = np.concatenate([self.ClusterMetallicity, pop2.ClusterMetallicity])
    #     result.NumClusters = self.NumClusters + pop2.NumClusters
    #     result.ClusterFormationTime = np.concatenate([self.ClusterFormationTime, pop2.ClusterFormationTime])
    #     result.ClusterFormationRedshift = np.concatenate([self.ClusterFormationRedshift, pop2.ClusterFormationRedshift])
    #     result.ClusterTracers = np.concatenate([self.ClusterTracers, pop2.ClusterTracers])
    #     result.ClusterIDs = np.concatenate([self.ClusterIDs, pop2.ClusterIDs])
    #     result.FieldTracers = self.FieldTracers + pop2.FieldTracers
    #     result.Mstar = self.Mstar + pop2.Mstar
    #     result.FieldMass = np.concatenate([self.FieldMass, pop2.FieldMass])
    #     result.FieldMetallicity = np.concatenate([self.FieldMetallicity, pop2.FieldMetallicity])
    #     result.FieldFormationTime = np.concatenate([self.FieldFormationTime, pop2.FieldFormationTime])
    #     return result
