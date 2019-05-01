from load_from_snapshot import load_from_snapshot
import numpy as np
from matplotlib import pyplot as plt

masses = np.concatenate([load_from_snapshot("Masses", i, '.', 600) for i in (0, 1, 2, 4)])
x = np.concatenate([load_from_snapshot("Coordinates", i, '.', 600) for i in (2, 4)])
center = np.median(x,axis=0)
x = np.concatenate([load_from_snapshot("Coordinates", i, '.', 600) for i in (0, 1, 2, 4)])
sfr = load_from_snapshot("StarFormationRate", 0, '.',600)
n = load_from_snapshot("Density", 0, '.',600)*404

#plt.plot(np.sort(n)[::100], sfr[n.argsort()].cumsum()[::100])
#plt.xscale('log')
#plt.show()
#plt.clf()
#x -= center
#r = np.sqrt((x*x).sum(axis=1))

rgrid = np.logspace(-1,3,1000)
mbin = np.histogram(r, weights=masses, bins=rgrid)[0]
plt.loglog(rgrid[1:], mbin.cumsum())
plt.show()
np.savetxt("r_vs_Menc.dat", np.c_[rgrid[1:], mbin.cumsum() * 1e10], header = "(0) Radius (kpc)\n(1) Enclosed mass (msun)")
#xgas = load_from_snapshot("Coordinates", 0, '.', 600)
#ugas = load_from_snapshot("InternalEnergy", 0, '.', 600)
#print(np.median(xgas[ugas < 1.],0), np.median(xstar,0))

