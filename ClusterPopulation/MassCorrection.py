from sys import argv
import numpy as np
import h5py
from ClusterPopulation import snapnum_to_time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

sfrdata = np.loadtxt("m12i_res7100_mhdcv_sfr.dat")
sfrdict=  dict(zip(sfrdata[:,0], sfrdata[:,1]))
mdict = dict(zip(sfrdata[:,0], sfrdata[:,3]))
SFEmax = 0.8
alpha_SFE = 1
sigma_SFE = 3200.

snapnums = []
corrfacs = []
mstar_actual = []
mstar_model = []
times = []
clustermasses = []
snaps = []
for f in argv[1:]:
    n = int(f.split("bound_")[1].split("_")[0])
    ms = np.loadtxt(f, ndmin=2)[:,0]*1e10
    if len(ms) == 0: continue
    rs = np.loadtxt(f, ndmin=2)[:,7]*1e3
    sigmas = ms/(np.pi*rs*rs)
    SFE = (1./SFEmax + (sigma_SFE/sigmas)**alpha_SFE)**-1.
    mstar = (SFE * ms).sum()

    print(n, mdict[n]/mstar, (SFE*ms).max())
    clustermasses.append(SFE*ms)
#    cloudsigmas.append(sigmas)
    snaps.append(np.repeat(n, len(ms)))
    mstar_actual.append(mdict[n])
    mstar_model.append(mstar)
#    print(n, mstar / mdict[n], m[snapnum==n].sum() / mstar)
    snapnums.append(n)
    corrfacs.append(mstar / mdict[n])
    times.append(snapnum_to_time(n))

#np.savetxt("corrfacs.dat", np.c_[snapnums, 1./np.array(corrfacs)])
print("Mass-weighted correction factor:", np.sum(mstar_actual) / np.sum(mstar_model))
from matplotlib import pyplot as plt

t_order = np.array(times).argsort()
corrfacs = 1/np.array(corrfacs)[t_order]
mstar_actual = np.array(mstar_actual)[t_order]
mstar_model = np.array(mstar_model)[t_order]
#corrfacs = gaussian_filter(corrfacs, 10)
np.savetxt("corrfacs.dat", np.c_[np.array(snapnums)[t_order], corrfacs, mstar_actual, mstar_model])
#plt.plot(np.sort(times), corrfacs)#1/np.array(corrfacs)[np.array(times).argsort()])
#plt.yscale('log')
#plt.ylabel("Correction Factor")
#plt.xlabel("Cosmic Time (Gyr)")
#plt.show()
clustermasses = np.concatenate(clustermasses)
snaps = np.concatenate(snaps)
#cloudsigmas = np.concatenate(cloudsigmas)
np.savetxt("max_cluster_masses.dat", np.c_[snaps,clustermasses])
