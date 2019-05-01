from load_from_snapshot import load_from_snapshot
import numpy as np
from Meshoid import Meshoid
from sys import argv
import os
from time import time

for f in argv[1:]:
    t = time()
    if "snapshot" in f:
        snapnum = int(f.split("snapshot_")[1].split(".")[0])
    #    snappath = f.split("/snap")[0]
    else:
        snapnum = int(f.split("snapdir_")[1].split("/")[0])
    snappath = f.split("/snap")[0]

    pos = load_from_snapshot("Coordinates", 4, snappath, snapnum)
    ids = load_from_snapshot("ParticleIDs", 4, snappath, snapnum)
    v = load_from_snapshot("Velocities", 4, snappath, snapnum)
    Nstar = pos.shape[0]
    pos = np.concatenate([pos, load_from_snapshot("Coordinates", 0, snappath, snapnum)])
    phi = np.concatenate([load_from_snapshot("Potential", 4, snappath, snapnum), load_from_snapshot("Potential", 0, snappath, snapnum)])
    print("Data loaded in at %g seconds"%(time() - t))
    M = Meshoid(pos, particle_mask = np.arange(Nstar), des_ngb=50, n_jobs=24)
    print("Meshoid constructed at %g seconds"%(time() - t))
    tidal_tensor = M.D2(phi, weighted=False)
    hsml = M.hsml
    condition_number = M.d2_condition_number
    del M    
    print("Tidal tensor computed at %g seconds"%(time() - t))
    M_stars = Meshoid(pos[:Nstar], des_ngb=50, n_jobs=24)
    dv = v[:,None,:] - v[M_stars.ngb]
    vdisp = np.average(np.sum(dv*dv, axis=2), axis=1)
    print("Vdisp computed at %g seconds"%(times() - t))
    

    np.savetxt("tidal_tensor_%d.npy"%snapnum, np.c_[ids, tidal_tensor, vdisp, hsml, condition_number])
