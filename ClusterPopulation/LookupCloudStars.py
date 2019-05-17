import h5py
from sys import argv
import numpy as np
from numba import jit
from scipy.spatial import cKDTree
data = np.load("m12i_res7100_ids_coords_rgc.npy")
star_ids, coords = np.rint(data[:,0]), data[:,1:]
coords -= np.median(coords,axis=0)

particle_clouds = {}
particle_snaps = {}

@jit
def UpdateDicts(f, particle_clouds, particle_snaps):
    F = h5py.File(f,'r')
    n = f.split("Clouds_")[1].split("_")[0]
    for k in F.keys():
        gas_ids = np.int_(F[k]["PartType0"]["ParticleIDs"])
        for id in gas_ids:
            particle_clouds[id] = "_".join([n,k])
            particle_snaps[id] = n
#    print(particle_clouds.keys())
    return particle_clouds ,particle_snaps


for f in sorted(argv[1:]): # first pass: simply identify the particles that live in clouds, and remember which cloud they live in
    print(f)
    particle_clouds, particle_snaps = UpdateDicts(f, particle_clouds, particle_snaps)

gas_ids, clouds, snapnums = [k for k in particle_clouds.keys()], [v for v in particle_clouds.values()], [v for v in particle_snaps.values()]
#print(np.array(gas_ids).size, np.array(star_ids).size)
# Now, if you are a star particle at z=0, and you were a gas particle in a cloud, congratulations, you are a tracer for that cloud
gas_particle_became_star = dict(zip(gas_ids, np.in1d(gas_ids, star_ids, assume_unique=True)))
cloud_tracers = {}

@jit
def GetCloudTracers():
    for id in gas_ids:
        if gas_particle_became_star[id]:
            cid = particle_clouds[id]
            if cid in cloud_tracers.keys():
                cloud_tracers[cid].append(id)
            else:
                cloud_tracers[cid] = [id,]

GetCloudTracers()
#print(cloud_tracers.keys())
for f in sorted(argv[1:]):
    print(f)
    F = h5py.File(f,'r+')
    n = f.split("Clouds_")[1].split("_")[0]
    centers = []
    clouds_with_tracers = []

    for k in F.keys():
        if "Tracers" in F[k].keys(): del F[k]["Tracers"]
        key = "_".join([n,k])
        if key in  cloud_tracers.keys():
            clouds_with_tracers.append(key)
            m = np.array(F[k]["PartType0"]["Masses"])
            x = np.array(F[k]["PartType0"]["Coordinates"])
            centers.append(np.average(x,axis=0,weights=m))
            F[k].create_dataset("Tracers", data = cloud_tracers[key])
    centers = np.array(centers)
    # now loop over the ones who didn't get any tracers: take a tracer from their nearest neighbor
#    tree = cKDTree(centers)
    for k in F.keys():
        if  "Tracers" in F[k].keys(): continue
        m = np.array(F[k]["PartType0"]["Masses"])
        x = np.array(F[k]["PartType0"]["Coordinates"])
        com = np.average(x, weights=m, axis=0)
        dist = np.sum((com - centers)**2, axis=1)
        F[k].create_dataset("Tracers",data=cloud_tracers[clouds_with_tracers[dist.argmin()]])
   
        
    F.close()
            #GetCloudTracers()
#print("Cloud tracers obtained")
#for id in gas_ids:
    
