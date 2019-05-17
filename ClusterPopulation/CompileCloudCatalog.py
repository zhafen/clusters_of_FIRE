import numpy as np
from glob import glob
from sys import argv
import h5py

for f in argv[1:]:
    F = h5py.File(f)
    clouds = F.keys()
    print(clouds)
    for c in clouds:
        print(np.average(np.array(F[c]["Metallicity"])[:,0]))
