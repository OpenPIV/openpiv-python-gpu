from openpiv import tools, pyprocess, scaling, validation, filters
import numpy as np

import os

# we can run it from any folder
path = os.path.dirname(os.path.abspath(__file__))


frame_a  = tools.imread( os.path.join(path,'../data/test1/exp1_001_a.bmp'))
frame_b  = tools.imread( os.path.join(path,'../data/test1/exp1_001_b.bmp'))

frame_a = (frame_a*1024).astype(np.int32)
frame_b = (frame_b*1024).astype(np.int32)

u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
    window_size=32, overlap=16, dt=0.02, search_area_size=64, sig2noise_method='peak2peak' )

print(u,v,sig2noise)

x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=64, overlap=16 )
u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
u, v, mask = validation.global_val( u, v, (-1000, 2000), (-1000, 1000) )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
tools.save(x, y, u, v, mask, '../data/test1/test_data.vec' )
tools.display_vector_field('../data/test1/test_data.vec', scale=75, width=0.0035)

