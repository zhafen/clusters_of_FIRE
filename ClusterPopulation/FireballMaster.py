from sys import argv
from os import system
from os.path import isfile
from glob import glob
from matplotlib import pyplot as plt
from load_from_snapshot import load_from_snapshot
import numpy as np
from multiprocessing import Pool
from ClusterPopulation import ClusterPopulation
from CloudPhinder import SaveArrayDict
import h5py

seed = 42

def GetTracers(f,n):
    print(f)
    n = int(f.split("Clouds_")[1].split("_")[0])
    F = h5py.File(f,'r+') 
#    if F["Header"].attrs["NumPart_Total"][4] == 0: continue
    pos = load_from_snapshot("Coordinates", 4, 'output', n)
    if pos is 0: return
    vel = np.concatenate([load_from_snapshot("Velocities", i, 'output', n) for i in (4,)])
    m = np.concatenate([load_from_snapshot("Masses", i, 'output', n) for i in (4,)])
    ids = np.concatenate([load_from_snapshot("ParticleIDs", i, 'output', n) for i in (4,)])
    tf = np.concatenate([load_from_snapshot("StellarFormationTime", i, 'output', n) for i in (4,)])
    r = np.sqrt(np.sum((pos - np.median(pos,axis=0))**2, axis=1))

    tree = cKDTree(pos)

#    snapids = ids[(formation_snap >= n)*(formation_snap <= n+5)]

    for k in F.keys():
        print(k)
        gas_ids = np.array(F[k]["PartType0"]["ParticleIDs"])
#        print(n, k, np.sum(np.isin(snapids, gas_ids)))
        x = np.array(F[k]["PartType0"]["Coordinates"])
        m = np.array(F[k]["PartType0"]["Masses"])
        v = np.array(F[k]["PartType0"]["Velocities"])
        x_com = np.average(x, weights=m,axis=0)
        v_com  = np.average(v, weights=m,axis=0)
        RGMC = np.sqrt(np.average(5./3 * np.sum((x-x_com)**2,axis=1), weights=m))
        sigmav = 2*np.sqrt(np.average(np.sum((v-v_com)**2,axis=1), weights=m))
        particles = []
        fac = 1
        while len(particles) < 100 and fac < 256:
            particles = tree.query_ball_point(x_com, min(RGMC*fac,1.)) #, x_com)
#            print((np.sum((vel[particles] - v_com)**2,axis=1) < sigmav**2*fac**2))
            particles = np.array(particles)[(np.sum((vel[particles] - v_com)**2,axis=1) < sigmav**2*max(fac/4,1)**2)]
            fac *= 2
        print(k, len(particles))
        if len(particles) == 0: 
            particles = tree.query_ball_point(x_com, 1.)
        if len(particles) == 0:
            particles = tree.query(x_com, 2)[1][:1]
#            print("Couldn't find tracer for cloud " + k)
#            exit()
        if "Tracers" in F[k].keys(): del F[k]["Tracers"]
        F[k].create_dataset("Tracers", data = ids[particles[:100]])
#        print(len(particles), np.log2(fac))



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

def GetCloudProperties(f):
    F = h5py.File(f,'r')
    tff = []
    M_GMC = []
    R_GMC = []
    mstar_model = []
    COM = []
    members = []
    tracers = []
    Z_GMC = []
    for cloud in F.keys():
        data = F[cloud]
        m = 1e10 * np.array(data["PartType0"]["Masses"])
        z = np.array(data["PartType0"]["Metallicity"])[:,0]
        x = 1e3 * np.array(data["PartType0"]["Coordinates"])
        tracers.append(np.array(data["Tracers"]))
        com = np.average(x, axis=0, weights=m)
        rSqr = np.sum((x-com)**2,axis=1)
        M_GMC.append(m.sum())
        Z_GMC.append(np.average(z, weights=m))
        members.append(np.array(data["PartType0"]["ParticleIDs"]))
        COM.append(com)
        R_GMC.append(np.sqrt(5./3 * np.average(rSqr, weights=m)))
#        mstar_model.append(mstar_from_GMC(M_GMC[-1], R_GMC[-1]))
        tff.append(np.pi/2 * (R_GMC[-1]**3 / M_GMC[-1] / 4.301e-3 / 2)**0.5)

    return np.array([k for k in F.keys()]), members, tracers, np.array(M_GMC), np.array(R_GMC), np.array(COM), np.array(tff), np.array(Z_GMC) #, mstar_model.values())

def SaveClusterPop(pop, f, n, savetxt=True):
    outdict = {}
    outdict["Mass"] = pop.ClusterMasses
#    print(pop.Cluster_GMC_ID)
    outdict["GMC_ID"] = np.int_([id.split("Cloud")[1] for id in pop.Cluster_GMC_ID])
    outdict["HalfMassRadius"] = pop.ClusterRadii
    outdict["ProgenitorMass"] = pop.Cluster_M_GMC
    outdict["ProgenitorReff"] = pop.Cluster_R_GMC
    outdict["ClusterPos"] = pop.Cluster_Pos
#    print(outdict["ClusterPos"].shape)
    outdict["Metallicity"] = pop.ClusterMetallicity
    outdict["FormationTime"] = pop.ClusterFormationTime
    outdict["EFFGamma"] = pop.Cluster_EFF_Gamma
    outdict["TracerID"] = pop.ClusterTracers                                                                                                                                                         
#    if savetxt:
 #       SaveArrayDict("ClusterPopulation_%d_%d.dat"%(n, seed), outdict)

    Fout = h5py.File(f.split("Cloud")[0] + "ClusterPopulation_%d_%d.hdf5"%(n, seed), "w")
    for k in outdict.keys():
        print(k)
        Fout.create_dataset(k, data=outdict[k])
    Fout.close()

def SaveGMCPop(pop, f, n, savetxt=True):
    outdict = {}
#    print(pop.Cluster_GMC_ID)
    outdict["GMC_ID"] = np.int_([id.split("Cloud")[1] for id in pop.GMC_ID])
    outdict["Mass"] = pop.M_GMC
    outdict["Reff"] = pop.R_GMC
    outdict["Pos"] = pop.GMC_Pos
    outdict["Metallicity"] = pop.GMC_Metallicity
    outdict["FormationTime"] = pop.GMC_StarFormationTime
    outdict["StellarMass"] = pop.GMC_Mstar
    outdict["BoundStellarMass"] = pop.GMC_Mbound
    outdict["BoundStellarMassExpected"] = pop.GMC_Mbound_expected
#    outdict["TracerID"] = pop.GMC_Tracers                                                                                                                                        
#    if savetxt:
 #       SaveArrayDict("ClusterPopulation_%d_%d.dat"%(n, seed), outdict)

    Fout = h5py.File(f.split("Cloud")[0] + "GMCPopulation_%d_%d.hdf5"%(n, seed), "w")
    for k in outdict.keys():
        print(k)
        Fout.create_dataset(k, data=outdict[k])
    Fout.close()

#plt.figure(figsize=(4,4))

def ComputeSFRs(f):
    outputdir = f.split("snapshot_times")[0] + "/output"
    outputdir = outputdir.replace("//","/")
    print(outputdir)
    n, a, z, t, dt = np.loadtxt(f + "/snapshot_times.txt").T
    t0 = t.max()
    header = "(0) snapshot index\n(1) scale-factor\n(2) redshift\n(3) time[Gyr]\n(4) time_width[Myr]\n(5) Stellar mass formed [Msun]\n(6) SFR [Msun/yr]"
    af = load_from_snapshot("StellarFormationTime", 4, outputdir, 600)
    m = load_from_snapshot("Masses", 4, outputdir, 600) * 1e10
    tf = np.interp(af, a, t)
    age = t0 - tf
    mf = m / mass_loss_fac(age)
    mbin = np.histogram(tf, weights=mf,bins=t)[0]
    SFR = mbin/np.diff(t*1e9)
#    mbin = np.insert(mbin,0,0)
#    SFR = np.insert(SFR,0,0)
    name = outputdir.split("/output")[0].split("/")[-1]
    
    np.savetxt("%s_SFR.txt"%(name,), np.c_[n[:-1],a[:-1],z[:-1],t[:-1],dt[:-1], mbin, SFR],header=header)
    return n, a, z, t, dt, mbin, SFR



def DoStuffForSnapshot(i):
#    for i in range(0,600,1):
    np.random.seed(seed)
    SFR_snap = SFR[i]
    Mstar_snap = mbin[i]
    snapnum = i
    print(snapnum, SFR_snap, Mstar_snap)
    # look for the cloud catalogue for that snapshot
    cloud_file = "clouds/Clouds_%s_n1_alpha2.hdf5"%(str(int(snapnum)))
    if not isfile(cloud_file):
        return

    #INSERT BLOCK THAT GETS TRACERS AND AMBIENT KPC-SCALE PROPERTIES

    # END BLOCK

    cloud_ids, members_GMC, tracers, M_GMC, R_GMC, COM_GMC, tff_GMC, Z_GMC = GetCloudProperties(cloud_file)

    mstar_GMC = mstar_from_GMC(M_GMC, R_GMC)
        
    sample_weights = np.ones_like(tff_GMC) #1/tff_GMC
    sample_weights /= sample_weights.sum()

    if mstar_GMC.sum() == 0:
        return
    N_GMC = len(M_GMC) * int(Mstar_snap/mstar_GMC.sum() + 1)
    Mstar_tot = 0
    while Mstar_tot < Mstar_snap:
        sample = np.random.choice(np.arange(len(M_GMC)),N_GMC, p=sample_weights)
        mstar_GMC = mstar_from_GMC(M_GMC[sample], R_GMC[sample])
        Mstar_tot = mstar_GMC.sum()
        sample = sample[mstar_GMC.cumsum() < Mstar_snap]
        N_GMC = len(sample)

    tform = t[i] + np.random.rand(N_GMC)*dt[i]/1e3

    pop = ClusterPopulation(M_GMC=M_GMC[sample], metallicity=Z_GMC[sample], R_GMC=R_GMC[sample], seed_offset=i, tform=tform, tracers=[tracers[s] for s in sample], X_GMC=COM_GMC[sample], cloud_ids=cloud_ids[sample])

    SaveClusterPop(pop, cloud_file, i)
    SaveGMCPop(pop, cloud_file, i)


for f in argv[1:]:
    n, a, z, t, dt, mbin, SFR = ComputeSFRs(f)
#    system("ComputeSFR.py %s/snapshot_times.txt"%f)

    print("Computing GMC population from SFH and cloud catalogue...")
    Pool(12).map(DoStuffForSnapshot, range(600), chunksize=1)
    

        
#plt.show()
