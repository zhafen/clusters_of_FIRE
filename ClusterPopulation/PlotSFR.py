from load_from_snapshot import load_from_snapshot
import numpy as np
from matplotlib import pyplot as plt
import h5py

a = np.loadtxt("snapshot_times.txt")[:,1]
t = np.loadtxt("snapshot_times.txt")[:,3]
n = np.loadtxt("snapshot_times.txt")[:,0]
m = 7100.
af = load_from_snapshot("StellarFormationTime", 4, '.', 600)
tf = np.interp(af, a, t)
tbins = np.concatenate([[0,], t])

#tbins = t[::10]
#mbin = np.histogram(tf, bins=tbins)[0]
#sfr = m * mbin / np.diff(tbins) / 1e9
m_snapshot = np.histogram(tf, bins=tbins, weights=np.repeat(m, len(tf)))[0]
#mtot = 7100 * np.arange(1, len(tf)+1)
sfr = m_snapshot / np.diff(tbins)/1e9
plt.plot(tbins[1:], m_snapshot / np.diff(tbins)/1e9)
#plt.hist(tf, 100, weights=m)
plt.show()



from glob import glob

sfrdict = dict(zip(n, sfr))
cloudsfrdict = {}
cloudsfrdict[0] = 0
for f in glob("../cloud_catalogue/m12i_res7100_mhdcv/Clouds*.hdf5"):
    nn = int(f.split("Clouds_")[1].split("_")[0])
    F = h5py.File(f, 'r')
    cloudsfrdict[nn] = np.sum([np.sum(F[k]["PartType0"]["StarFormationRate"]) for k in F.keys()])
    print(nn, cloudsfrdict[nn], sfrdict[nn])

np.savetxt("m12i_res7100_mhdcv_sfr.dat", np.c_[n, sfr, np.array([cloudsfrdict[int(N)] for N in n]), m_snapshot])
