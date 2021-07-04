from __future__ import division

"""Run speed tests for GPU vs CPU correlation
"""

import numpy as np
import time

import gpuFFT
import IWarrange

#================================================================================
# PARAMTERS
#================================================================================

im0 = np.random.randn(512, 512).astype(np.float32)
im1 = np.random.randn(1024, 1024).astype(np.float32)
im2 = np.random.randn(2048, 2048).astype(np.float32)
im3 = np.random.randn(2560, 2560).astype(np.float32)

images = [im0, im1, im2, im3]

#window size to do speedtests on
window = np.array([64, 32, 16, 8])

# number of times to iterate
Nt = 10


#================================================================================


def main_speedtest(images, window, Nt):
    """ Main speedtest function
    """

    # store all the time data
    t = np.empty([len(images), len(window)])

    # loop through all the pseudo images and do the speed test
    for i in range(len(images)):
        for j in range(len(window)):

            win_A, win_B = IWarrange(images[i], images[i], window[j], window[j]/2)

            start = time.time()
            for i in range(Nt):
                corr_gpu = gpuFFT(win_A, win_B)
            t[i,j] = (time.time() - start)/Nt

    print(t.shape)
    print_time(t, window)

def print_time(t, w):
    """print all the timing that happened
    """

    print("window  |  512x512  |  1024x1024  |  2048x2048  |  2560x2560")
    print("------------------------------------------------------------")
    for i in range(t.shape[0]):
        print("{:<6}  |  {:2.4f}  |  {:2.4f}  |  {:2.4f}  |  {:2.4f}".format(w[i], t[0,i], t[1,i], t[2,i], t[3,i]))

#
# if __name__ == "__main__":
#     main_speedtest(images, window, Nt)