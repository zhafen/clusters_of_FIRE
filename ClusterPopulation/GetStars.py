from load_from_snapshot import load_from_snapshot
import numpy as np

x = np.concatenate([load_from_snapshot("Coordinates", i, '.', 600) for i in (2,4)])
m = np.concatenate([load_from_snapshot("Masses", i, '.', 600) for i in (2,4)])
print(m.min(), m.max())
ids = np.concatenate([load_from_snapshot("ParticleIDs", i, '.', 600) for i in (2,4)])
r = np.sqrt(np.sum((x - np.median(x,axis=0))**2, axis=1))
np.save("m12i_res7100_ids_coords_rgc.npy", np.c_[ids, x, r])
