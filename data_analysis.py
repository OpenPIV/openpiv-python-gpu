from __future__ import division
"""
Script to process all the results from this image dataset. This includes:
- gpu-cpu computation time comparison.
- velocity field comparison between GPU and DaVis results. 
"""

import os
import glob
import sys

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import scipy.interpolate as interp
import scipy.stats
from skimage import io


import openpiv.tools
import math_tools 
import stats_tools

if sys.version_info[0] == 3:
    from progress.bar import Bar

if sys.version_info[0] == 2:
    import openpiv.filters
    from progressbar import *


####################################################
# MUST DEFINE THESE VARIABLES
####################################################

davis_dir = "./davis_velocity_window16_overlap8/"
davis_raw_dir = "./davis_velocity_window16_overlap8/raw_data/"
gpu_raw_dir = "./soscip_gpu_output_data/raw_output_data/"
gpu_dir = "./soscip_gpu_output_data/"

assert os.path.exists(davis_dir), "Davis directory is invalid"
assert os.path.exists(gpu_dir), "gpu directory is invalid"

# dimension information
x_scale = 0.0855384 / 1000 # m/pixel
y_scale = 0.085864 / 1000 # m/pixel
c = 0.3  #chord length
x0 = .115  ## beginning of the image in from the leading edge in mm

# time between image pairs
dt = 90e-6

# for outlier deection
r_thresh = 2

####################################################

def load_raw_data(data_dir):
    """Load all the results
    """

    data_list = sorted(glob.glob(data_dir + "*.txt"))
    x,y,u_tmp,v_tmp,mask = openpiv.tools.load_vectors(data_list[0])

    u = np.empty([len(data_list), x.shape[0], x.shape[1]])
    v = np.empty_like(u)
    mask = np.empty_like(u)
    u[0,:,:] = u_tmp
    v[0,:,:] = v_tmp

    # load the rest of the data
    pbar = Bar("Importing data from {}".format(data_dir), max=len(data_list))
    pbar.start()
    for i in range(1,len(data_list)):
        pbar.next()
        x, y, u[i,:,:], v[i,:,:], mask[i,:,:] = openpiv.tools.load_vectors(data_list[i])

    pbar.finish()

    return(x,y,u,v,mask)


def load_velocity_new_format(u_dir, v_dir, x_file):
    """
    Load all the files from the new format (u_{05d}.npy)
    """
    u_list = sorted(glob.glob(u_dir + "*.npy"))
    v_list = sorted(glob.glob(v_dir + "*.npy"))
    x = np.load(x_file)

    # prep variables to be returned
    u = np.empty([len(u_list), x.shape[0], x.shape[1]])
    v = np.empty_like(u)

    for i in range(0, len(u_list)):
        u[i, :, :] = u_list[i]
        v[i, :, :] = v_list[i]

    return u, v


def load_saved_data(data_list):
    """load data saved in npy format
    """

    print("Loading data")
    if len(data_list) == 5:   
        mask = np.load(data_list[0])
        u = np.load(data_list[1])
        v = np.load(data_list[2])
        x = np.load(data_list[3])
        y = np.load(data_list[4])
        return(x, y, u, v, mask)
    elif len(data_list) == 4:
        u = np.load(data_list[0])
        v = np.load(data_list[1])
        x = np.load(data_list[2])
        y = np.load(data_list[3])
        return(x, y, u, v, mask)   
    else:
        u = np.load(data_list[0])
        v = np.load(data_list[1])
        return(u,v)


def save_data(x, y, u, v, mask, out_dir, descriptor=""):
    """ Save the data to .npy format for quick reading
    """

    print("Saving velocity data...")
    np.save(out_dir + "x{}.npy".format(descriptor), x)
    np.save(out_dir + "y{}.npy".format(descriptor), y)
    np.save(out_dir + "u{}.npy".format(descriptor), u)
    np.save(out_dir + "v{}.npy".format(descriptor), v)
    np.save(out_dir + "mask{}.npy".format(descriptor), mask)
    print("Done.")


def output_vtk(dx, u, v):
    """output a velocity field as a vtk file
    """
    u = np.expand_dims(u,2)
    v = np.expand_dims(v,2)
    w = np.zeros_like(u)

    i = tvtk.ImageData(spacing=(dx, dx ,0), origin=(0,0,0))
    i.point_data.scalars = u.ravel()
    i.point_data.scalars.name = "u"
    i.dimensions = u.shape
    i.point_data.add_array(v.ravel())
    i.point_data.get_array(1).name = "v"
    i.point_data.update()
    #i.point_data.add_array(w.ravel())
    #i.point_data.get_array(2).name = "w"
    #i.point_data.update()

    write_data(i, "vtk_test.vtk")


def to_csv(x, y, u, v):
    """
    """

    f = open("vel_data.csv", "w")
    f.write("x, y, z, u, v, w\n")

    x = x.ravel()
    y = y.ravel()
    u = u.ravel()
    v = v.ravel()

    for i in range(u.size):
        f.write("{:3.5f}, {:3.5f}, {:3.5f}, {:3.5f}, {:3.5f}, {:3.5f}\n".format(x[i], y[i], 0,  u[i], v[i], 0))

    f.close()


def mask_data(u_gpu, v_gpu, u_dv, v_dv, mask_dv):
    """Mask the data using the davis mask
    """

    mask_tmp = np.array(mask_dv > 4)
    u_mask1 = np.array(u_gpu > 30)
    u_mask2 = np.array(u_gpu < -30)
    v_mask1 = np.array(v_gpu > 30)
    v_mask2 = np.array(v_gpu < -30)

    mask = ~(~mask_tmp * ~u_mask1 * ~u_mask2 * ~v_mask1 * ~v_mask2) 

    u_gpu_ma = ma.masked_array(u_gpu, mask)
    v_gpu_ma = ma.masked_array(v_gpu, mask)


    u_dv_ma = ma.masked_array(u_dv, mask)
    v_dv_ma = ma.masked_array(v_dv, mask)

    return(u_gpu_ma, v_gpu_ma, u_dv_ma, v_dv_ma)


def compare_data(u_gpu_ma, v_gpu_ma, u_dv_ma, v_dv_ma, bins=500):
    """Compare the two data sets
    """

    # get bins size
    # see https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    iqr = scipy.stats.iqr(u_gpu_ma.flatten())
    bin_size = 2*iqr/(u_gpu_ma.size**(1/3))
    nbins = 4/bin_size

    # get total velocity
    U_gpu = ma.sqrt(u_gpu_ma**2 + v_gpu_ma**2)
    U_dv = ma.sqrt(u_dv_ma**2 + v_dv_ma**2)
    U_diff = U_gpu - U_dv

    u_diff = u_gpu_ma - u_dv_ma
    v_diff = v_gpu_ma - v_dv_ma

    u_diff_norm = u_diff / x_scale * dt
    v_diff_norm = v_diff / y_scale * dt

    bins = np.linspace(-2, 2, nbins)

    # u difference
    plt.hist(u_diff.flatten(), bins = bins, histtype = "bar", ec = "k")
    plt.title("U velocity difference", fontsize=20)
    plt.xlabel("Difference [pixel/dt]", fontsize = 18)
    plt.ylabel("Frequncy", fontsize = 18)
    plt.xlim([-2, 2])
    plt.show()

    # v differece
    plt.hist(v_diff.flatten(), bins = bins, histtype = "bar", ec = "k")
    plt.title("V velocity difference", fontsize=20)
    plt.xlabel("Difference [pixel/dt]", fontsize = 18)
    plt.ylabel("Frequncy", fontsize = 18)
    plt.show()

    # total velocity differece
    plt.hist(U_diff.flatten(), bins = bins, histtype = "bar", ec = "k")
    plt.title("Total velocity difference", fontsize=20)
    plt.xlabel("Difference [pixel/dt]", fontsize = 18)
    plt.ylabel("Frequncy", fontsize = 18)
    plt.show()


def sinlge_error_hist(u1, u1_real, xlim=None, title=None):
    """Plot a single error histogram
    """

    diff = u1 - u1_real

    if xlim is None:
        xlim = [diff.min(), diff.max()]

    # get binsize and bin arra
    bin_size = get_bin_size(diff)
    if bin_size < 0.01:
        bin_size = 0.01
    bins = np.arange(xlim[0], xlim[1] + bin_size, bin_size)

    # get bins and bar size
    hist, tmp = np.histogram(diff.flatten(), bins=bins)
    hist = hist/float(diff.size)*100.
    plt.bar(bins[:-1], hist, align="edge", width = bin_size, color='b', ec='k')
    plt.xlim(xlim[0], xlim[1])
    plt.xlabel(r"Difference [$m/s$]", fontsize = 18)
    plt.ylabel(r"Percent of Data [$\%$]", fontsize=18)
    if title is not None:
        plt.title(title, fontsize = 20)
    plt.tight_layout()
    plt.show()


def total_error_hist(u, v, u_dv, v_dv, xlim=None, title=None, vline=False):
    """Plot a single error histogram
    """

    u = u.flatten()
    v = v.flatten()
    u_dv = u_dv.flatten()
    v_dv = v_dv.flatten()

    full_mask = np.isfinite(u)*np.isfinite(v)*np.isfinite(u_dv)*np.isfinite(v_dv)
    u = u[full_mask]
    v = v[full_mask]
    u_dv = u_dv[full_mask]
    v_dv = v_dv[full_mask]

    U = np.sqrt(u**2 + v**2)
    U_dv = np.sqrt(u_dv**2 + v_dv**2)
    diff = U - U_dv

    # get binsize and bin arra
    bin_size = get_bin_size(diff)
    if bin_size < 0.01:
        bin_size = 0.01
    bins = np.arange(diff.min(), diff.max() + bin_size, bin_size)

    # get bins and bar size
    hist, tmp = np.histogram(diff.flatten(), bins=bins)
    hist = hist/float(diff.size)*100.

    # get where 95% of the data falls within
    hsum = np.sum(hist)
    h = 0
    i=0
    l = np.where(hist == hist.max())[0][0]
    while( h/hsum < 0.95):
        i+=1
        h = np.sum(hist[l-i:l+i+1])   

    ind_max = bins[l+i]
    ind_min = bins[l-i]

    if xlim is None:
        d_tmp = (ind_max - ind_min)*0.2
        xlim = [ind_min - d_tmp, ind_max + d_tmp]

    plt.bar(bins[:-1], hist, align="edge", width = bin_size, color='b', ec='k')
    plt.xlim(xlim)
    plt.xlabel(r"Difference [m/s]", fontsize = 18)
    plt.ylabel(r"Percent of Data [$\%$]", fontsize=18)
    plt.tick_params(axis = "both", labelsize=14)
    if title is not None:
        plt.title(title, fontsize = 20)
    if vline:
        plt.vlines(ind_max, 0, hist.max(), color='r', linestyle='-.', label="95% of data")
        plt.vlines(ind_min, 0, hist.max(), color='r', linestyle='-.')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def get_bin_size(data):
    """ get bin size for a particular dataset
    see https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    """

    iqr = scipy.stats.iqr(data.flatten())
    bin_size = 2*iqr/(data.size**(1/3.))

    return(bin_size)


def outlier_detection(u, v, r_thresh, mask, max_iter=2):
    """Outlier detection
    """

    u_out = np.copy(u)
    v_out = np.copy(v)

    widgets = ["Detecting Outliers", Percentage(), ' ', Bar(marker='-',left='[',right=']'), ' ']
    N = (u.shape[0]-2)*(u.shape[1] - 2)
    maxval = max_iter*N
    pbar = ProgressBar(widgets=widgets, maxval=100)
    pbar.start()

    for n in range(max_iter):
        for i in range(1, u.shape[0] - 2):
            for j in range(1, u.shape[1] - 2):
                pbar.update(100*( n*N + (i*(u.shape[1] -2) + j))/maxval)#progress update

                if mask[i,j] == False:
                    if np.isfinite(u_out[i,j]):
                        Ui = np.delete(u_out[i-1:i+2, j-1:j+2].flatten(), 4)
                        Um = np.nanmedian(Ui)
                        rm = np.nanmedian(np.abs(Ui - Um))
                        ru0 = np.abs(u_out[i,j] - Um)/(rm + 0.1)
                        if ru0 > r_thresh:
                            u_out[i,j] = np.nan
                        if not np.isfinite(Um):
                            u_out[i,j] = np.nan

                    if np.isfinite(v_out[i,j]):
                        Vi = np.delete(v_out[i-1:i+2, j-1:j+2].flatten(), 4)
                        Vm = np.nanmedian(Vi)
                        rm = np.nanmedian(np.abs(Vi - Vm))
                        rv0 = np.abs(v_out[i,j] - Vm)/(rm + 0.1)
                        if rv0 > r_thresh:
                            v_out[i,j] = np.nan
                        if not np.isfinite(Vm):
                            v_out[i,j] = np.nan


    pbar.finish()

    print("Number of u outliers: {}".format( np.sum(np.isnan(u_out)) - np.sum(mask)) )
    print("Percentage: {}".format( (np.sum(np.isnan(u_out))- np.sum(mask)) /u.size*100))
    print("Number of v outliers: {}".format( np.sum(np.isnan(v_out)) - np.sum(mask) ))
    print("Percentage: {}".format( (np.sum(np.isnan(v_out)) - np.sum(mask))/v.size*100) )
        
    print("Replacing Outliers")
    u_out, v_out = openpiv.filters.replace_outliers(u_out,  v_out)

    return(u_out,v_out)


def replace_all_outliers(u, v, r_thresh, mask):
    """This is gonna take a while
    """

    print("Replacing all outliers.")
    for i in range(u.shape[0]):
        print("Filtering field {} of {}".format(i, u.shape[0]))
        u[i,:,:], v[i,:,:] = outlier_detection(u[i,:,:], v[i,:,:], r_thresh, mask)

    return(u, v)


def smooth_fluctuations(x, y, u, v, N):
    """ average function in windows of size N
    """

    assert u.ndim == 2, "u must be 2D"
    assert v.ndim == 2, "v must be 2D"

    u_avg = np.empty(np.array(u.shape)//N)
    v_avg = np.empty(np.array(v.shape)//N)
    x_avg = np.empty(u.shape[1]//N)
    y_avg = np.empty(u.shape[0]//N)

    for i in range(u_avg.shape[0]):
        for j in range(u_avg.shape[1]):
            u_avg[i,j] = np.mean(u[i*N:(i+1)*N, j*N:(j+1)*N])
            v_avg[i,j] = np.mean(v[i*N:(i+1)*N, j*N:(j+1)*N])
            y_avg[i] = np.mean(y[i*N:(i+1)*N])
            x_avg[j] = np.mean(x[j*N:(j+1)*N])

    return(x_avg, y_avg, u_avg, v_avg)


def pcolormesh(x, y, f, f_dv, title='', cbar_label = ""):
    """
    """

    global c, x0

    max_val = np.nanmax(np.array([f_dv, f]))
    min_val = np.nanmin(np.array([f_dv, f]))
    nbins = 25

    xt = np.argmax(~np.isfinite(f[0,20:]))
    xlim = np.array([0, x[0,xt-1]])/c + x0/c
    ylim = np.array([0.04, y[0,-1]])/c

    # get coormap info
    cmap = plt.get_cmap('jet')
    levels = MaxNLocator(nbins=nbins).tick_values(min_val, max_val)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots(2,1, figsize=(7.3,6), sharex=True)

    ax[0].set_title(r"$OpenPIV$", fontsize=16)
    plt.xlabel(r"$x/c$", fontsize=18)
    im0 = ax[0].pcolormesh(x/c + x0/c, y/c, f , cmap=cmap, norm=norm, vmin = min_val, vmax = max_val)
    ax[0].set_ylim(ylim)
    ax[0].set_xlim(xlim)
    ax[0].tick_params(axis = "both", labelsize=14)

    im1 = ax[1].pcolormesh(x/c + x0/c, y/c, f_dv, cmap=cmap, norm=norm, vmin = min_val, vmax = max_val)
    ax[1].set_title(r"$DaVis$", fontsize=16)
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)
    ax[1].tick_params(axis = "both", labelsize=14)

    fig.text(0.01, 0.5, r"$y/c$", va = "center", rotation="vertical", fontsize=18)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.65])
    cbar0 = fig.colorbar(im0, cax=cbar_ax)
    cbar0.set_label(cbar_label, fontsize = 18)

    plt.tight_layout(rect=[0.035, 0.0, 0.85, 0.95])
    plt.show()


def pcolormesh_single(x, y, u, title='', cbar_label=""):
    """
    """

    max_val = np.nanmax(u)
    min_val = np.nanmin(u)
    nbins=15

    # get coormap info
    cmap = plt.get_cmap('jet')
    levels = MaxNLocator(nbins=nbins).tick_values(min_val, max_val)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots(1,1, figsize=(12,4))
    im0 = ax.pcolormesh(x, y, u , cmap=cmap, norm=norm, vmin = min_val, vmax = max_val)
    cbar0 = fig.colorbar(im0, ax=ax)
    cbar0.set_label(cbar_label, fontsize = 18)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def q_criterion(x, y, u, v, mask):
    """Return Q-criterion
    """

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Q = math_tools.q_criterion_2D(u, v, dx, dy)

    # normalize Q criterion
    Q = Q/np.max(np.abs(Q))

    Q[mask] = np.nan

    title = "Q-Criterion"
    cbar_label = r"$Q$"
    pcolormesh_single(x, y, Q, title=title, cbar_label=cbar_label)

    return(Q)


def q_criterion_compare(x, y, u, v, u_dv, v_dv, n, mask, boolean=False):
    """ Compare Q criterion fields
    """

    u_mean = np.nanmean(u, axis=0)
    v_mean = np.nanmean(v, axis=0)
    u_dv_mean = np.nanmean(u_dv, axis=0)
    v_dv_mean = np.nanmean(v_dv, axis=0)

    u_fluc = u[n,:,:] - u_mean
    v_fluc = v[n,:,:] - v_mean
    u_dv_fluc = u_dv[n,:,:] - u_dv_mean
    v_dv_fluc = v_dv[n,:,:] - v_dv_mean

    Q = math_tools.q_criterion_2D(u_fluc, v_fluc)
    Q = Q/np.max(np.abs(Q))
    Q[mask] = np.nan

    Q_dv = math_tools.q_criterion_2D(u_dv_fluc, v_dv_fluc)
    Q_dv = Q_dv/np.max(np.abs(Q_dv))
    Q_dv[mask] = np.nan

    if boolean:
        Q[Q >= 0] = 1
        Q[Q < 0] = 0
        Q_dv[Q_dv >= 0] = 1
        Q_dv[Q_dv < 0] = 0

    title = "Q-Criterion Comparison"
    cbar_label = r"$Q$"
    pcolormesh(x, y, Q, Q_dv, title=title, cbar_label=cbar_label)


def apply_mask(x, y, mask):
    """
    """

    im = openpiv.tools.imread("images/C0000_a.tif").astype(np.int32)
    im_new = np.copy(im)
    x_new = np.linspace(x.min(), x.max(), im.shape[1])
    y_new = np.linspace(y.min(), y.max(), im.shape[0])

    f = interp.RectBivariateSpline(y, x, mask.astype(int))
    mask_new = f(y_new, x_new)
    mask_new[mask_new > 0.8] = 1
    mask_new[mask_new <= 0.8] = 0
    mask_new = mask_new.astype(bool)

    im_new[mask_new] = 0.0
    plt.imshow(im, cmap="gray")
    plt.show()
    plt.imshow(im_new, cmap="gray")
    plt.show()


def peak_locking(u, v, u_dv, v_dv):
    """ look at peak locking in the data
    """

    u = u.flatten()
    v = v.flatten()
    u_dv = u_dv.flatten()
    v_dv = v_dv.flatten()

    full_mask = np.isfinite(u)*np.isfinite(v)*np.isfinite(u_dv)*np.isfinite(v_dv)
    u = u[full_mask]
    v = v[full_mask]
    u_dv = u_dv[full_mask]
    v_dv = v_dv[full_mask]

    bs0 = stats_tools.get_bin_size(u)
    bs1 = stats_tools.get_bin_size(v)
    bs2 = stats_tools.get_bin_size(u_dv)
    bs3 = stats_tools.get_bin_size(v_dv)

    bin_size = np.min([bs0, bs1, bs2, bs3])
    if bin_size < 0.1:
        bin_size = 0.1
    minbin_u = np.min([np.min(u), np.min(u_dv)])
    maxbin_u = np.max([np.max(u), np.max(u_dv)])
    bins_u = int((maxbin_u - minbin_u)/bin_size + 1)
    minbin_v = np.min([np.min(v), np.min(v_dv)])
    maxbin_v = np.max([np.max(v), np.max(v_dv)])
    bins_v = int((maxbin_v - minbin_v)/bin_size + 1)

    #weights = np.ones(u.size)/u.size*100
    weights = np.ones(u.size)
    u_hist = plt.hist(u, color = "r", bins=bins_u, histtype="step", weights=weights, label="OpenPIV")
    u_dv_hist = plt.hist(u_dv, color="b", bins=bins_u, histtype="step", weights=weights, label="DaVis")
    plt.ylabel("% of data", fontsize=18)
    plt.xlabel(r"u velocity $[m/s]$", fontsize=18)
    plt.xlim([-10, 10])
    plt.legend()
    plt.tight_layout()
    plt.show()

    v_hist = plt.hist(v, color = "r", bins=bins_v, histtype="step", weights=weights, label="OpenPIV")
    v_dv_hist = plt.hist(v_dv, color="b", bins=bins_v, histtype="step", weights=weights, label="DaVis")
    plt.ylabel("% of data", fontsize=18)
    plt.xlabel(r"v velocity $[m/s]$", fontsize=18)
    plt.xlim([-10, 10])
    plt.legend()
    plt.tight_layout()
    plt.show()

    return(u_hist[0], u_dv_hist[0], v_hist[0], v_dv_hist[0])


def quiver_plot(x, y, u, v, u_dv, v_dv, title="", row_step=1, col_step=1):
    """ Quiver plot of the data
    """

    N = 5
    x_tmp, y_tmp, u, v = smooth_fluctuations(x,y,u,v, N)
    x, y, u_dv, v_dv = smooth_fluctuations(x,y,u_dv,v_dv, N)

    fig, (ax0, ax1) = plt.subplots(2,1, figsize=(12,6), sharex = True )
    ax0.quiver(x,y,u,v, headaxislength=2.0)
    ax0.set_title("OpenPIV Data", fontsize=16)
    plt.xlabel(r"Streamwise distance [$m$]", fontsize=18 )
    ax1.quiver(x,y,u_dv,v_dv, headaxislength=2.0)
    ax1.set_title("DaVis Data", fontsize=16)
    fig.text(0.02, 0.5, "Transverse Distance", va = "center", rotation="vertical", fontsize=18)
    plt.tight_layout(rect=[0.04, 0.0, 1, 1])
    plt.show() 


def image_mask(mask, expand_mask=None):
    """get the mask to go over the whole image
    """

    im_a = io.imread("B00020_frame1a.tiff")
    im_a = io.imread("B00020_frame1b.tiff")

    mask_im = np.zeros_like(im_a).astype(bool)
    row = np.linspace(0, mask.shape[0]-1, im_a.shape[0])
    col = np.linspace(0, mask.shape[1]-1, im_a.shape[1])

    try:
        mask_im = np.load("mask_im.npy")
    except FileNotFoundError:
        pbar = Bar("Calculating new mask", max = im_a.size)
        for i in range(im_a.shape[0]):
            for j in range(im_a.shape[1]):
                pbar.next()

                m_row = int(np.round(row[i]))
                m_col = int(np.round(col[j]))

                mask_im[i,j] = mask[m_row, m_col]
        pbar.finish()
        np.save("mask_im.npy", mask_im)

    im_tmp = np.copy(im_a).astype(float)
    im_tmp[mask_im] = np.nan
    plt.imshow(im_tmp, cmap="gray")
    plt.show()

    # expand the mask to cover the boundary layer
    if expand_mask is not None:
        try:
            exp_mask_im = np.load("expanded_mask_im.npy")
        except FileNotFoundError:
            exp_mask_im = np.copy(mask_im)
            mask_im[0:100,:] = 0
            for i in range(exp_mask_im.shape[1]):
                r = np.argmax(mask_im[:,i])
                exp_mask_im[r-expand_mask:r,i] = 1
            np.save("expanded_mask_im.npy", new_mask)

        # mask everything greater than x > 2000
        exp_mask_im[:,2000:] = 1
        exp_mask_im[:,0:5] = 1
        im_tmp = np.copy(im_a).astype(float)
        im_tmp[exp_mask_im] = np.nan
        plt.imshow(im_tmp, cmap="gray")
        plt.show()

        try:
            exp_mask = np.load("expanded_mask.npy")
        except FileNotFoundError:

            fm = interp.RectBivariateSpline(row, col, exp_mask_im.astype(int))
            exp_mask = fm(np.arange(mask.shape[0]), np.arange(mask.shape[1]))

            exp_mask = np.array(exp_mask > 0.5)
            plt.imshow(exp_mask, cmap="gray")
            plt.show()
            np.save("expanded_mask.npy", exp_mask)


def line_plots(x, y, u, v, u_dv, v_dv, mask, loc):
    """ plot any field f and see what happens
    """

    global c

    assert len(loc) == 4, "Need to have 4 streamwise locations"

    # plot the fields

    f = (u, v)
    f_dv = (u_dv, v_dv)
    field=["u", "v"]

    for i in range(2):

        fig, ax = plt.subplots(2,2, figsize = (10,7), sharex=True, sharey=True)

        ax[0,0].plot(y/c, f[i][:,loc[0]], 's', color="deepskyblue", fillstyle="none", label="OpenPIV")
        ax[0,0].plot(y/c, f_dv[i][:,loc[0]], 'o', color="darkblue", fillstyle="none", label="DaVis")
        ax[0,0].set_ylabel("{} (x/c = {})".format(field[i], loc[0]/c), fontsize=18)
        ax[0,0].legend(loc=0)

        ax[0,1].plot(y/c, f[i][:,loc[1]], 's', color="red", fillstyle="none", label="OpenPIV")
        ax[0,1].plot(y/c, f_dv[i][:,loc[1]], 'o', color="darkred", fillstyle="none", label="DaVis")
        ax[0,1].set_ylabel("{} (x/c = {})".format(field[i], loc[1]/c), fontsize=18)
        ax[0,1].legend(loc=0)

        ax[1,0].plot(y/c, f[i][:,loc[2]], 's', color="limegreen", fillstyle="none", label="OpenPIV")
        ax[1,0].plot(y/c, f_dv[i][:,loc[2]], 'o', color="darkgreen", fillstyle="none", label="DaVis")
        ax[1,0].set_ylabel("{} (x/c = {})".format(field[i], loc[2]/c), fontsize=18)
        ax[1,0].legend(loc=0)


        ax[1,1].plot(y/c, f[i][:,loc[3]], 's', color="magenta", fillstyle="none", label="OpenPIV")
        ax[1,1].plot(y/c, f_dv[i][:,loc[3]], 'o', color="purple", fillstyle="none", label="DaVis")
        ax[1,1].set_ylabel("{} (x/c = {})".format(field[i], loc[3]/c), fontsize=18)
        ax[1,1].legend(loc=0)

        fig.text(0.5, 0.035, 'x/c', fontsize=18, ha='center')
        plt.tight_layout(rect = [0, 0.05, 1, 1])
        plt.show()


def clever_line_plots(x, y, u, u_dv, ustd, ustd_dv, mask, loc):
    """ plot any field f and see what happens
    """

    global c, x0

    # bias = std/sqrt(N)
    N = u.size
    u_bias = ustd/np.sqrt(N)
    u_dv_bias = ustd_dv/np.sqrt(N)

    # plot the fields
    u1 = np.copy(u[:,loc[0]])
    u2 = np.copy(u[:,loc[1]])
    u3 = np.copy(u[:,loc[2]])

    u1_dv = np.copy(u_dv[:,loc[0]])
    u2_dv = np.copy(u_dv[:,loc[1]])
    u3_dv = np.copy(u_dv[:,loc[2]])

    Uc_loc = int(u.shape[0]/2)
    Uc = np.nanmax([u_dv, u])
    #Uc = 5

    u1 = u1/Uc
    u2 = u2/Uc
    u3 = u3/Uc
    u1_dv = u1_dv/Uc
    u2_dv = u2_dv/Uc
    u3_dv = u3_dv/Uc

    # get plot limits
    ylim = np.array([0.04, y[0,-1]])/c
    xmin = np.nanmin([u1, u2, u3, u1_dv, u2_dv, u3_dv])
    xmax = np.nanmax([u1, u2, u3, u1_dv, u2_dv, u3_dv])
    xdiff = xmax-xmin
    xlim = [xmin - 0.05*xdiff, xmax+0.05*xdiff]

    fig = plt.figure()
    grid = gridspec.GridSpec(1,3, hspace = 0.0, wspace=0.0)

    ax0 = plt.Subplot(fig, grid[0])
    ax0.plot(u1, y[:,0]/c, color='b', label = "OpenPIV")
    ax0.plot(u1_dv, y[:,0]/c, linestyle='--', color='r', label="DaVis")
    ax0.fill_between(u1 - u_bias, u1 + u_bias, y[:,0]/c)
    ax0.fill_between(u1_dv - u_dv_bias, u1_dv + u_dv_bias, y[:, 0] / c)
    ax0.set_ylabel(r"$y/c$", fontsize=18)
    ax0.set_title(r"$x/c = {:1.2f}$".format(x[0, loc[0]]/c + x0/c), fontsize=18)
    ax0.set_ylim(ylim)
    ax0.set_xlim(xlim)
    ax0.legend(fontsize=14)
    ax0.tick_params(axis='both', labelsize=14)
    fig.add_subplot(ax0)

    ax1 = plt.Subplot(fig, grid[1])
    ax1.plot(u2, y[:,0]/c, color='b')
    ax1.plot(u2_dv, y[:,0]/c, linestyle='--', color='r')
    ax1.fill_between(u2 - u_bias, u2 + u_bias, y[:, 0] / c)
    ax1.fill_between(u2_dv - u_dv_bias, u2_dv + u_dv_bias, y[:, 0] / c)
    ax1.tick_params(labelleft=False)
    ax1.set_xlabel(r"$\overline{u}/\overline{u}_{max}$", fontsize=18)
    ax1.set_title(r"$x/c = {:1.2f}$".format(x[0, loc[1]]/c + x0/c), fontsize=18)
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)
    ax1.tick_params(axis='both', labelsize=14)
    fig.add_subplot(ax1)

    ax2 = plt.Subplot(fig, grid[2])
    ax2.plot(u3, y[:,0]/c, color='b', label=r"$OpenPIV$")
    ax2.plot(u3_dv, y[:,0]/c, linestyle='--', color='r', label=r"$DaVis$")
    ax2.fill_between(u3 - u_bias, u3 + u_bias, y[:, 0] / c)
    ax2.fill_between(u3_dv - u_dv_bias, u3_dv + u_dv_bias, y[:, 0] / c)
    ax2.tick_params(labelleft=False)
    ax2.set_title(r"$x/c = {:1.2f}$".format(x[0, loc[2]]/c + x0/c), fontsize=18)
    ax2.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.tick_params(axis='both', labelsize=14)
    fig.add_subplot(ax2) 

    plt.tight_layout()
    plt.show()


def shitty_line_plot(x, y, u, u_dv, loc):
    """do a regular line plot
    """

    umax = np.max([np.nanmax(u), np.nanmax(u_dv)])

    u1 = np.copy(u[:,loc[0]])/umax
    u2 = np.copy(u[:,loc[1]])/umax
    u3 = np.copy(u[:,loc[2]])/umax

    u1_dv = np.copy(u_dv[:,loc[0]])/umax
    u2_dv = np.copy(u_dv[:,loc[1]])/umax
    u3_dv = np.copy(u_dv[:,loc[2]])/umax

    plt.plot(u1, y[:,0],  's', color="deepskyblue", fillstyle="none", label="OpenPIV: x={}".format(x[0,loc[0]]))
    plt.plot(u1_dv, y[:,0],'o', color="darkblue", fillstyle="none", label="Davis: x={}".format(x[0,loc[0]]))

    plt.plot(u2,y[:,0],  's', color="red", fillstyle="none", label="OpenPIV: x={}".format(x[0,loc[1]]))
    plt.plot(u2_dv, y[:,0], 'o', color="darkred", fillstyle="none", label="DaVis: x={}".format(x[0,loc[1]]))

    plt.plot(u3, y[:,0], 's', color="magenta", fillstyle="none", label="OpenPIV: x={}".format(x[0,loc[2]]))
    plt.plot(u3_dv, y[:,0], 'o', color="purple", fillstyle="none", label="DaVis: x={}".format(x[0,loc[2]]))

    plt.xlabel(r"$u/u_{max}$", fontsize=18)
    plt.ylabel(r"Transverse Distance [$m$]", fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()





#===============================================================================
# FUNCTION CALLS
#===============================================================================


if __name__ == "__main__":

    # load davis data
    if True:
        if "x_dv" not in locals():
            data_list = glob.glob(davis_dir + "*.npy")
            if len(data_list) == 5:
                data_list.sort()
                x_dv, y_dv, u_dv, v_dv, mask_dv = load_saved_data(data_list)
            else:
                x_dv, y_dv, u_dv, v_dv, mask_dv = load_raw_data(davis_raw_dir)
                # davis v data is backwards
                v_dv = -v_dv            
                save_data(x_dv, y_dv, u_dv, v_dv, mask_dv, davis_dir, descriptor="_dv")

            v_dv = -v_dv
            mask_new = np.array(mask_dv.mean(axis=0) > 4.5)

    # get gpu data
    if False:
        if "u_gpu" not in locals():
            data_list = glob.glob(gpu_dir + "*gpu.npy")
            if len(data_list) == 5:
                data_list.sort()
                x_gpu, y_gpu, u_gpu, v_gpu, mask_gpu,= load_saved_data(data_list)
            else:
                x_gpu, y_gpu, u_gpu, v_gpu, mask_gpu = load_raw_data(gpu_raw_dir)
                save_data(x_gpu, y_gpu, u_gpu, v_gpu, mask_dv, gpu_dir, descriptor="_gpu")  

        mask = np.array(np.mean(mask_gpu, axis=0) > 4.5).astype(bool)  
        u_gpu[:,mask_gpu] = np.nan
        v_gpu[:,mask_gpu] = np.nan

    # load the locations only
    if True:
        if "x_gpu" not in locals():
            x_gpu = np.load("./soscip_gpu_output_data/x_gpu.npy")
            y_gpu = np.load("./soscip_gpu_output_data/y_gpu.npy")
            mask_gpu = np.load("./soscip_gpu_output_data/mask_gpu.npy")
            mask = np.array(np.mean(mask_gpu, axis=0) > 4.5).astype(bool)

    # get outlier replaced gpu data
    if True:
        if "u_out" not in locals():
            data_list = glob.glob(gpu_dir + "*out.npy")
            if len(data_list) == 2:
                data_list.sort()
                u_out, v_out = load_saved_data(data_list)
            else:
                u_out, v_out = replace_all_outliers(u_gpu, v_gpu, r_thresh, mask)
                np.save(gpu_dir + "u_out.npy", u_out)
                np.save(gpu_dir + "v_out.npy", v_out)

            if False:
                min_val_u = np.nanmean(u_out) - 3*np.nanstd(u_out)
                u_out[u_out < min_val_u] = np.nan
                min_val_v = np.nanmean(v_out) - 3*np.nanstd(v_out)
                u_out[u_out < min_val_v] = np.nan

            #u_out_mean = np.nanmean(u_out, axis=0)
            #v_out_mean = np.nanmean(v_out, axis=0)

    # mask data
    if True:
        if "exp_mask" not in locals():
            try:
                exp_mask = np.load("expanded_mask.npy")
            except:
                full_mask = np.logical_or(mask, mask_new)
                image_mask(full_mask, expand_mask=147)
                exp_mask = np.load("expanded_mask.npy")
            
            u_out[:,exp_mask] = np.nan
            v_out[:,exp_mask] = np.nan
            u_dv[:,exp_mask] = np.nan
            v_dv[:,exp_mask] = np.nan       

    if False:
        if "U" not in locals():
            U = np.sqrt(u_out**2 + v_out**2)
            U_dv = np.sqrt(u_dv**2 + v_dv**2)

    

    #===============================================================================
    # PLOTS
    #===============================================================================

    #compare_data(u_gpu_ma, v_gpu_ma, u_dv_ma, v_dv_ma)
    #compare_data(u_gpu_ma, v_gpu_ma, u_dv_ma, v_dv_ma, bins=500)
    #sinlge_error_hist(u_gpu_ma, u_dv_ma, xlim=[-1,1], title="U-Velocity Difference")
    #sinlge_error_hist(v_gpu_ma, v_dv_ma, xlim=[-1,1], title="V-Velocity Difference")

    total_error_hist(u_out, v_out, u_dv, v_dv, vline=True)

    # q criterion
    #Q = q_criterion(x_gpu[0,:], y_gpu[:,0], u_out.mean(axis=0), v_out.mean(axis=0), mask)
    #for i in range(0,500, 50):
    #    q_criterion_compare(x_gpu[0,:], y_gpu[:,0], u_out, v_out, u_dv, v_dv, i, mask)

    #quiver_plot(x_gpu[0,:], y_gpu[:,0], np.nanmean(u_out, axis=0), np.nanmean(v_out, axis=0), np.nanmean(u_dv, axis=0), np.nanmean(v_dv,axis=0))

    # vector field
    title_pc = "Streamwise Velocity Comparison"
    cbar_pc = r"$u \;[m/s]$"
    #pcolormesh(x_gpu, y_gpu, np.nanmean(u_out, axis=0), np.nanmean(u_dv, axis=0), title=title_pc, cbar_label=cbar_pc)

    #u_hist, u_dv_hist, v_hist, v_dv_hist = peak_locking(u_out, v_out, u_dv, v_dv)

    #line_plots(x_gpu[0,:], y_gpu[:,0], np.nanmean(u_out, axis=0), np.nanmean(v_out, axis=0), np.nanmean(u_dv, axis=0), np.nanmean(v_dv,axis=0), exp_mask, [75, 150, 225, 300])
    clever_line_plots(x_gpu, y_gpu, np.nanmean(u_out, axis=0), np.nanmean(u_dv, axis=0), np.nanstd(u_out, axis=0), np.nanstd(u_dv, axis=0), exp_mask, [20, 120, 230])
    #shitty_line_plot(x_gpu, y_gpu, np.nanmean(u_out, axis=0), np.nanmean(u_dv, axis=0), [50, 100, 200])
    
