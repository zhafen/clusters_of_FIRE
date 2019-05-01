from sys import argv
import numpy as np
import h5py
from ClusterPopulation import snapnum_to_time
from scipy.ndimage.filters import gaussian_filter

sfrdata = np.loadtxt("m12i_res7100_mhdcv_sfr.dat")
sfrdict=  dict(zip(sfrdata[:,0], sfrdata[:,1]))
mdict = dict(zip(sfrdata[:,0], sfrdata[:,3]))
SFEmax = 0.55
alpha_SFE = 1
sigma_SFE = 2300.

clusterdata = np.loadtxt("ClusterPopulation_3.dat")

snapnum = clusterdata[:,1]
m = clusterdata[:,4]

snapnums = []
corrfacs = []
mstar_actual = []
mstar_model = []
times = []
for f in argv[1:]:
    n = int(f.split("bound_")[1].split("_")[0])
    ms = np.loadtxt(f, ndmin=2)[:,0]*1e10
    if len(ms) == 0: continue
    rs = np.loadtxt(f, ndmin=2)[:,7]*1e3
    sigmas = ms/(np.pi*rs*rs)
    SFE = (1./SFEmax + (sigma_SFE/sigmas)**alpha_SFE)**-1.

    mstar = (SFE * ms).sum()
    print(mstar, mdict[n])
    mstar_actual.append(mdict[n])
    mstar_model.append(mstar)
    print(n, mstar / mdict[n], m[snapnum==n].sum() / mstar)
    snapnums.append(n)
    corrfacs.append(mstar / mdict[n])
    times.append(snapnum_to_time(n))

#np.savetxt("corrfacs.dat", np.c_[snapnums, 1./np.array(corrfacs)])
print("Mass-weighted correction factor:", np.sum(mstar_actual) / np.sum(mstar_model))
from matplotlib import pyplot as plt

corrfacs = 1/np.array(corrfacs)[np.array(times).argsort()]
#corrfacs = gaussian_filter(corrfacs, 10)
np.savetxt("corrfacs.dat", np.c_[snapnums, corrfacs])
plt.plot(np.sort(times), corrfacs)#1/np.array(corrfacs)[np.array(times).argsort()])
plt.yscale('log')
plt.ylabel("Correction Factor")
plt.xlabel("Cosmic Time (Gyr)")
plt.show()
