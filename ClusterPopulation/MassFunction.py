import numpy as np
import h5py
import matplotlib.pyplot as plt
from sys import argv

plt.figure(figsize=(4,4))
mags = []
for line in open("mwgc.dat",'r').readlines()[252:410]:
    if line[46:55].split():
        mags.append(float(line[46:55]))

lum = 10**((4.83 - np.array(mags))/2.5) * 1.5
mwg_masses = lum
plt.loglog(np.sort(mwg_masses), np.arange(1,len(mwg_masses)+1)[::-1], label="MWG GCs (Harris 1996)", color='black')

for f in argv[1:]:
    print(f)
    F = h5py.File(f,'r')
    m = np.array(F["Mass"])

    # parse Harris file
    plt.loglog(np.sort(m), np.arange(1,len(m)+1)[::-1], label="Catalogue %s"%f.split("_")[1].split(".")[0])

    plt.ylabel(r"$N\left(>M\right)$")
    plt.xlabel("Mass ($M_\odot$)")
    plt.xlim(1e2,1e7)
#    plt.plot(np.logspace(2,7,1000), (10**6 * 1e2/np.logspace(2,7,1000)), ls='dashed', color='black', label="$\propto M^{-2}$ Mass Function")
plt.legend()
plt.savefig("MassFunction.pdf")
#print(mass.max())
#plt.hist(np.log10(mass))
#plt.show()
#print(mass)
#print(np.genfromtxt("mwgc.dat"))
