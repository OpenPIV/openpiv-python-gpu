import numpy as np
import openpiv
from openpiv.tools import imread



if __name__ == "__main__":
    frame_a = np.uint32(imread('exp1_001_a.bmp'))
    frame_b = np.uint32(imread('exp1_001_b.bmp'))
    x,y,u,v, mask = openpiv.process.WiDIM( frame_a.astype(np.int32), 
    frame_b.astype(np.int32),  ones_like(frame_a).astype(np.int32), 
    min_window_size=32, overlap_ratio=0.25, coarse_factor=2, dt=0.02, 
    validation_method='mean_velocity', trust_1st_iter=1, validation_iter=2, 
    tolerance=0.7, nb_iter_max=4, sig2noise_method='peak2peak')