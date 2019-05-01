from ClusterPopulation import *
from sys import argv
from glob import glob
import h5py
import matplotlib.pyplot as plt
from CloudPhinder import SaveArrayDict
from joblib import delayed, Parallel
#np.random.seed(42)
seed = int(argv[1])
#pops = []
#for f in argv[1:]:
#    print(f)
#    F = h5py.File(f, 'r')
#    n = int(f.split("Clouds_")[1].split("_")[0])

#    for cloud in F.keys():
#        pops.append(ClusterPopulation(cloud_data=F[cloud], snapnum=n, cloud_id=cloud))

rgc, Menc = np.loadtxt("r_vs_Menc.dat").T

def getpop(f):
    print(f)
    pops = []
    F = h5py.File(f, 'r')
    n = int(f.split("Clouds_")[1].split("_")[0])
    for cloud in F.keys():
        if (len(F[cloud]["PartType0"]["Masses"]) >= 10):
            pops.append(ClusterPopulation(cloud_data=F[cloud], snapnum=n, cloud_id=cloud, seed_offset = seed))
    return sum(pops)
    
pops = Parallel(n_jobs=24)(delayed(getpop)(f) for f in glob("cloud_catalogue/Clouds*.hdf5"))

pop = sum(pops)
#print(len(pop.EFF_Gamma), len(pop.ClusterMasses))
star_data = np.load("m12i_res7100_ids_coords_rgc.npy")
rgc_dict = dict(zip(star_data[:,0], star_data[:,-1]))
xgc_dict = dict(zip(star_data[:,0], star_data[:,1:4]))
RGC = np.array([rgc_dict[t] for t in pop.ClusterTracers])
XGC = np.array([xgc_dict[t] for t in pop.ClusterTracers])
outdict = {}
outdict["FormationRedshift"] = pop.ClusterFormationRedshift
outdict["Snapshot"] = np.array([int(s.split(".")[0]) for s in pop.ClusterIDs])
outdict["Cloud"] = np.array([int(s.split(".")[1]) for s in pop.ClusterIDs])
outdict["CloudClusterID"] = np.array([int(s.split(".")[2]) for s in pop.ClusterIDs])
outdict["Mass"] = pop.ClusterMasses
outdict["HalfMassRadius"] = pop.ClusterRadii
outdict["ProgenitorMass"] = pop.M_GMC
outdict["ProgenitorReff"] = pop.R_GMC
outdict["Metallicity"] = pop.ClusterMetallicity
outdict["FormationTime"] = pop.ClusterFormationTime
outdict["EFFGamma"] = pop.EFF_Gamma
outdict["GalactocentricRadius"] = RGC
outdict["GalactocentricMenc"] = np.interp(RGC, rgc, Menc)
outdict["Coordinates"] = XGC
outdict["TracerID"] = pop.ClusterTracers
SaveArrayDict("ClusterPopulation_%d.dat"%(seed,), outdict)

Fout = h5py.File("ClusterPopulation_%d.hdf5"%(seed,), "w")
for k in outdict.keys():
    Fout.create_dataset(k, data=outdict[k])
Fout.close()

print(pop.FieldMass.sum() + pop.ClusterMasses.sum(), pop.Mstar / 1e11)
#plt.plot(np.sort(pop.ClusterMasses[RGC < 30.]), np.arange((RGC<30.).sum())[::-1] + 1)
#plt.loglog()
#plt.show()
