#/usr/bin/env python
from sys import argv
from load_from_snapshot import load_from_snapshot
from matplotlib import pyplot as plt
import numpy as np
from natsort import natsorted
from glob import glob

def mass_loss_fac(age_in_Gyr):
    """Fraction of mass from the stellar population remaining at a given age"""
    return 1 - 0.28 * (1 + (0.0144 / age_in_Gyr)**2.18)**-1.

sigma_SFE = 3200.
SFEmax = 0.8
alpha_SFE = 1.

def mstar_from_GMC(MGMC,RGMC):
    sigma = MGMC / (np.pi*RGMC**2)
    SFE = (1./SFEmax + (sigma_SFE/sigma)**alpha_SFE)**-1.
    return MGMC * (1/SFEmax + (sigma_SFE / sigma))**-1.

plt.figure(figsize=(4,4))

def ComputeSFRs(f):
    outputdir = f.split("snapshot_times")[0] + "output"
    print(outputdir)
    n, a, z, t, dt = np.loadtxt(f).T
    t0 = t.max()
    header = "# 601 snapshots\n# times assume cosmology from None\n# i scale-factor redshift time[Gyr] time_width[Myr] Stellar mass formed [Msun] SFR [Msun/yr]"
    af = load_from_snapshot("StellarFormationTime", 4, outputdir, 600)
    m = load_from_snapshot("Masses", 4, outputdir, 600) * 1e10
    tf = np.interp(af, a, t)
    age = t0 - tf
    mf = m / mass_loss_fac(age)
    print(np.mean(mass_loss_fac(age)))
    mbin = np.histogram(tf, weights=mf,bins=t)[0]
    SFR = mbin/np.diff(t*1e9)
    mbin = np.insert(mbin,0,0)
    SFR = np.insert(SFR,0,0)
    name = outputdir.split("/output")[0].split("/")[-1]
    plt.plot(t[:-1], SFR[:-1], label=name.replace("_","\_"),lw=1)
    plt.yscale('log')
    
    # compute total mass formed in GMCs
    gmc_path = "runs/" + name
    Mstar_GMC = []
    
#    for fgmc in natsorted(glob(gmc_path+"/bound_*.dat")):
    for igmc in n:        
        fgmc = glob(gmc_path+"/bound_%d_*.dat"%igmc)
        if len(fgmc) == 0: 
            Mstar_GMC.append(0)
        else:
            fgmc = fgmc[0]
            data = np.loadtxt(fgmc)
            if len(data) == 0: 
                continue
            elif len(data.shape) == 1:
                MGMC, RGMC = np.loadtxt(fgmc)[0::7]
            else:   
                MGMC, RGMC = np.loadtxt(fgmc)[:,0::7].T

            Mstar_GMC.append(mstar_from_GMC(MGMC*1e10,RGMC*1e3).sum())

    np.savetxt("runs/%s/SFR.txt"%(name,), np.c_[n,a,z,t,dt, mbin, SFR, Mstar_GMC])
    
for f in argv[1:]:
    ComputeSFRs(f)

plt.legend()
plt.xlabel("Time (Gyr)")
plt.ylabel("SFR ($M_\odot\,\mathrm{yr}^{-1}$)")
plt.savefig("SFRs.pdf", bbox_inches='tight')
