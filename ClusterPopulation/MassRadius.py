import matplotlib.pyplot as plt
import numpy as np
import h5py
from sys import argv

F = h5py.File(argv[1],'r')
m = np.array(F["Mass"])
r = np.array(F["HalfMassRadius"])
z = np.array(F["Metallicity"])
t = np.array(F["FormationTime"])
tmax = np.loadtxt("snapshot_times.txt")[-1,3]
age = tmax - t
#plt.scatter(age[m>1e5], np.log10(m)[m>1e5])

#plt.loglog()
plt.scatter(m[m>1e4], r[m>1e4], s=0.1, facecolor='black', marker='s', label="m12i")

data = np.genfromtxt("ymc_data/Ryon_YMCs.dat")
plt.scatter(data[:,1], data[:,2], label="Local YMCs", s=16, marker='d', color='orange', edgecolor='black', lw=0.1)

plt.xlabel("Mass ($M_\odot$)")
plt.ylabel("Half-Mass Radius ($\mathrm{pc}$)")
plt.loglog()
plt.xlim(1e4,1e7)
plt.ylim(0.01,100)
plt.legend(loc=4)
#plt.xscale('log')
plt.savefig("Cluster_Mass_Radius.png", bbox_inches='tight', dpi=600)
