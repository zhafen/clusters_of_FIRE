#!/usr/bin/env python
from ClusterPopulation import *
from sys import argv
from glob import glob
import h5py
from os import path
import matplotlib.pyplot as plt
from CloudPhinder import SaveArrayDict
from numba import jit
#from joblib import delayed, Parallel
from multiprocessing import Pool
#np.random.seed(42)
seed = int(argv[1])


keys_tosave = "FormationRedshift", "Snapshot", "Cloud", "CloudClusterID", "Mass", "HalfMassRadius", "ProgenitorMass", "ProgenitorReff", "Metallicity", "FormationTime", "EFFGamma", #"GalactocentricRadius", "GalactocentricMenc", "Coordinates", "TracerID"

sigma_SFE = 3200.
SFE_max = 0.8

#rgc, Menc = np.loadtxt("r_vs_Menc.dat").T
#corrdict = dict(zip(np.loadtxt("corrfacs.dat")[:,0], np.loadtxt("corrfacs.dat")[:,1]))
#massdict = dict(zip(np.loadtxt("corrfacs.dat")[:,0], np.loadtxt("corrfacs.dat")[:,2]))
sfrdata = argv[2].split("Clouds")[0] + "SFR.txt"
snapshot_lengthdict = dict(zip(np.loadtxt(sfrdata)[:,0], np.loadtxt(sfrdata)[:,-4]))
average_snapshot_length = np.average(np.loadtxt(sfrdata)[:,-4])
overall_corrfac = np.loadtxt(sfrdata)[:,-3].sum() / np.loadtxt(sfrdata)[:,-1].sum() # 15.8
print("avg snapshot length: ", average_snapshot_length, "correction factor:", overall_corrfac)


def ComputeClusterPopulation(f, savetxt=False, overwrite=False):
    print(f)
    pops = []
    F = h5py.File(f, 'r')
    n = int(f.split("Clouds_")[1].split("_")[0])
    outfilename = f.split("Cloud")[0] + "ClusterPopulation_%d_%d.hdf5"%(n, seed)
    if not overwrite:
        if path.isfile(outfilename): return
#    corrfac = 16 #corrdict[n]
#    if not n in massdict.keys(): return ClusterPopulation()
#    m_actual = massdict[n] # actual stellar mass that forms

    # do a first pass to get everybody's bulk properties, because we gotta get the normalization factor for the completness correaction
    M_GMC = {}
    R_GMC = {}
    mstar_model = {}
    tff = {}
    for cloud in F.keys():
#        if len(data["PartType0"]["Masses"]) < 32: continue
        data = F[cloud]
        m = 1e10 * np.array(data["PartType0"]["Masses"])
        x = 1e3 * np.array(data["PartType0"]["Coordinates"])
        com = np.average(x, axis=0, weights=m)
        rSqr = np.sum((x-com)**2,axis=1)
        M_GMC[cloud] = m.sum()
        R_GMC[cloud] = np.sqrt(5./3 * np.average(rSqr, weights=m))
        mstar_model[cloud] = M_GMC[cloud] * (1/SFE_max + (sigma_SFE * np.pi * R_GMC[cloud]**2 / M_GMC[cloud]))**-1.
        tff[cloud] = np.pi/2 * (R_GMC[cloud]**3 / M_GMC[cloud] / 4.301e-3)**0.5

    #resampling_norm_A = massdict[n] / np.sum([mstar_model[cloud] / tff[cloud] for cloud in F.keys()]) # weight according to Mstar / tff
    #resampling_norm_B = massdict[n] / np.sum([mstar_model[cloud] for cloud in F.keys()]) # weight according to Mstar / tff

    expected_resamplings = overall_corrfac*snapshot_lengthdict[n] / average_snapshot_length
    num_resamplings = int(expected_resamplings) #overall_corrfac * snapshot_lengthdict[n] / average_snapshot_length
    if np.random.random() < expected_resamplings - num_resamplings: num_resamplings += 1
#    num_resamplings = np.random.poisson(num_resamplings_avg)
    
    print("Num resamplings of snapshot %d: %d (expected %e)"%(n, num_resamplings, expected_resamplings))
                                                
    for cloud in F.keys():
        data = F[cloud]
        m = 1e10 * np.array(data["PartType0"]["Masses"])
        metallicity = np.average(np.array(data["PartType0"]["Metallicity"])[:,0],weights=m)
#        tracers = np.array(data["Tracers"])
#        if (len(data["PartType0"]["Masses"]) >= 10):
#            print(cloud, len(data["PartType0"]["Masses"]))
        for j in range(num_resamplings): # we re-sample the cloud according to the correction factor
            pops.append(ClusterPopulation(cloud_data=data, snapnum=n, cloud_id=cloud, seed_offset = seed+j, M_GMC=M_GMC[cloud], R_GMC=R_GMC[cloud], metallicity=metallicity))#, tracers=tracers))

    pop = sum(pops)
    if pop==0: return
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
#    outdict["GalactocentricRadius"] = RGC
    #outdict["GalactocentricMenc"] = np.interp(RGC, rgc, Menc)
    #outdict["Coordinates"] = XGC
    #outdict["TracerID"] = pop.ClusterTracers
    if savetxt:
        SaveArrayDict("ClusterPopulation_%d_%d.dat"%(n, seed), outdict)

    Fout = h5py.File(f.split("Cloud")[0] + "ClusterPopulation_%d_%d.hdf5"%(n, seed), "w")
    for k in outdict.keys():
        Fout.create_dataset(k, data=outdict[k])
    Fout.close()
#    return sum(pops)
#print(argv[2:])
#pops = [getpop(f) for f in argv[2:]]
#[ComputeClusterPopulation(f) for f in argv[2:]]
Pool(12).map(ComputeClusterPopulation, argv[2:], chunksize=1)

#pops = Parallel(n_jobs=12)(delayed(getpop)(f) for f in argv[2:])
#exit()
#gamma = []
#redshifts = []
#for p in pops:
#    gamma.append(p.BoundMass / p.FieldMass)

#pop = sum(pops)
#print(len(pop.EFF_Gamma), len(pop.ClusterMasses))
#star_data = np.load("m12i_res7100_ids_coords_rgc.npy")
#rgc_dict = dict(zip(star_data[:,0], star_data[:,-1]))
#xgc_dict = dict(zip(star_data[:,0], star_data[:,1:4]))
#RGC = np.array([rgc_dict[t] for t in pop.ClusterTracers])
#XGC = np.array([xgc_dict[t] for t in pop.ClusterTracers])


#print(pop.FieldMass.sum() + pop.ClusterMasses.sum(), pop.Mstar / 1e11)
#plt.plot(np.sort(pop.ClusterMasses[RGC < 30.]), np.arange((RGC<30.).sum())[::-1] + 1)
#plt.loglog()
#plt.show()
