"""
PIV / IMAGE PROCESSING TOOLS

Tools for PIV image processing

Cameron Dallas
University of Toronto
Department of Mechanical Engineering
Turbulence Research Laboratory

"""

import sys
import os
import glob

import numpy as np
from skimage import io, color, exposure
import matplotlib.pyplot as plt

#if sys.version_info[0] == 2:
#    import ReadIM


def contrast(frame,  r_low = 60, r_high = 99):
    """
    Adjust constrast of images

    Inputs:
        frame: 2d array
            image you want to adjust
        r_low, r_high: int
            pecentage range of brightness to enhance

    Outputs:
        frame_a, frame_b: 2d array
            contrast adjusted image
    """

    #adjust contrast
    #by adjusting the numbers in np.percentile, you choose which range of brightness you enhance
    pLow, pHigh = np.percentile(frame, (r_low, r_high))
    frame = exposure.rescale_intensity(frame, in_range = (pLow, pHigh))

    return(frame)


def mask(frame_a, frame_b, mask):

    """
    Sets the pixel intensity at the mask positions to zero.

    Inputs:
        frame_a, frame_b: 2d array
            first and second image pair
        mask: 2d boolean array
            true where the mask image_size

    Outputs:
        frame_a, frame_b: 2d array
            masked images
    """

    #apply the mask
    frame_a[mask] = 0.
    frame_b[mask] = 0.

    return(frame_a, frame_b)


def final_velocity(data_dir, file_ext, skip = 0, std = False):

    """
    Outputs the final velocity vector from an ensemble of PIV results

    Inputs:
        data_dir: string
            path to the directory of the velocity results
        file_ext: string
            file extension of the data
        I, J: int
            number of rows and columns of each vector
        skip: int
            number of header lines to skip in the data file
        std: boolean
            If TRUE, then the standard deviation of the velocity is also returned

    Outputs:
        x,y: 1d array
            velocity vector locations
        u,v: 1d arrays
            velocity components
        uStd, vStd: 2d array
            if std=True, this also returns the standard deviation of the velocity vectors
    """

    os.chdir(data_dir)
    #put a dot at the beginning of the file extension
    if file_ext[0] != '.':
        file_ext = '.' + file_ext

    #get list of image file names
    dataList = glob.glob('*'+ file_ext)
    dataList.sort()

    N = len(dataList)

    #prealloacte space for data
    dataTmp = np.loadtxt(dataList[0], skiprows = skip)
    M = dataTmp.shape[0]
    x = np.zeros([M,N])
    y = np.zeros([M,N])
    u = np.zeros([M,N])
    v = np.zeros([M,N])
    w = np.zeros([M,N])
    flag = np.zeros([M,N])

    for i in range(N):
        x[:,i], y[:,i],u[:,i],v[:,i],w[:,i],flag[:,i] = np.squeeze(np.split(np.loadtxt(dataList[i], skiprows = skip), 6, axis = 1 ))

    uMean = np.mean(u, axis = 1)
    vMean = np.mean(v, axis = 1)

    if std:
        return x[:,0], y[:,0], uMean, vMean, np.std(u, axis = 1), np.std(v, axis = 1)

    else:
        return x[:,0], y[:,0], uMean, vMean


def display_vectors(filePath, skip=0, flag=None):
    """
    Quiver plot of the velocity field

    Inputs:
        filePath: string
            absolute path to the file with the velocity data
        skip: int
            how many rows of the data file to skip
        flag: int
            which column of variables in a the mask is
    """

    #load data
    a = np.loadtxt(filePath, skiprows = skip)

    if flag is not None:
        mask = a[:,flag].astype(bool)
        plt.quiver(a[mask,0], a[mask,1], a[mask,2], a[mask,3])
        plt.show()
    else:
        plt.quiver(a[:,0], a[:,1], a[:,2], a[:,3])
        plt.show()


def load_vectors(filename, skip = 2,  uncrt = None):
    """
    Loads the velocity fields / uncertainty from an OpenPIV data output file
    returns the data from each file in a 3D matrix

    Parameters
    ----------
    filename: string
        The absolute path to the PIV data

    skip: int
        Number of header lines to skip in the data files. Defaults to 2 as that is what the save function defaults to.

    uncrt: tuple
        Column (zero indexed) of the u and v velocity uncertainty.
        Default is None, which returns no uncertainty.

    Returns
    -------
    x,y : 2D numpy array
        The x and y locations of the velocity vector. Since each frame is the same,
        only one 2D array for each is returned

    u,v : 3D numpy array
        The u amd v velocity fields for each piv file

    mask : 3D array
        Values that were masked during the validation process.
    """

    if uncrt is None:
        # data is in 1D arrays. Reshape to 2D arrays.
        x, y, u, v, mask = data_unravel(filename, 4, skip=skip)

        return(x,y,u,v,mask)
    else:
        x,y,u,v, mask, Ux, Uy = data_unravel(filename, skip=skip, uncrt=uncrt)


def data_unravel(filename, maskCol=4, skip=2, uncrt=None):
    """
    Takes data stored as flattened arrays and turns it back into 2D arrays.
    This is necessary for the fast bilinear spline in the image_dewarp function to work.

    Parameters
    ----------
    filename: string
        name of the file piv data results are stored in

    maskCol: int
        column where the mask data is stored

    uncrt: 2 component tuple
        contains the column index of the u and v uncertainties

    Returns
    -------
    x,y: 2d array, float
        coordinates of the velocity vectors

    u,v: 2d array, float
        velocity vectors

    mask: 2d array, bool
        location of the masked elements

    Ux, Uy: 2d array,
        u and v uncertainties for each vector
    """

    data = np.loadtxt(filename, skiprows = skip)
    x_tmp = data[:,0]
    y_tmp = data[:,1]
    u_tmp = data[:,2]
    v_tmp = data[:,3]
    mask_tmp = data[:,maskCol]

    if uncrt is not None:
        Ux_tmp = data[:, uncrt[0]]
        Uy_tmp = data[:, uncrt[1]]

    # find size of the 2d array assuming y location stored in blocks
    col = 0
    while y_tmp[0] == y_tmp[col]:
        col += 1

    row = len(x_tmp)/col

    # If the x variable is stored in blocks
    if col <= 1:
        row = 0
        while x_tmp[0] == x_tmp[row]:
            row += 1
        col = len(y_tmp)/row

    # check vector dimensions
    if col <= 1:
        raise ValueError('Unable to determine size of array. If data is unstructured, must use interp2d in image dewarp function')
    if row <= 1:
        raise ValueError('Unable to determine size of array. If data is unstructured, must use interp2d in image dewarp function')

    # Reshape arrays
    if uncrt is not None:
        x = x_tmp.reshape(row, col)
        y = y_tmp.reshape(row, col)
        u = u_tmp.reshape(row, col)
        v = v_tmp.reshape(row, col)
        mask = mask_tmp.reshape(row, col)
        Ux = Ux_tmp.reshape(row, col)
        Uy = Uy_tmp.reshape(row, col)

        return x, y, u, v, mask, Ux, Uy
    else:
        x = x_tmp.reshape(row, col)
        y = y_tmp.reshape(row, col)
        u = u_tmp.reshape(row, col)
        v = v_tmp.reshape(row, col)
        mask = mask_tmp.reshape(row, col)
        return x, y, u, v, mask


def save(x, y, u, v, mask, filename, uncrt = None, fmt='%8.4f', delimiter='\t'):
    """
    Save flow field to an ascii file.

    Inputs:
        x: 2d array
            x location of the vectors
        y: 2d array
            y location of the vectors
        u: 2d array
            u velocity vectors
        v: 2d array
            v velocity vectors
        mask: 2d array
            a two dimensional boolen array where elements corresponding to
            invalid vectors are True.
        filename: string
            the path of the file where to save the flow field
        uncrt: tuple containing 2 2d arrays
            uncertainty in x and y direction at each vector location
            ie. uncrt = (Ux, Uy)
        fmt: string
            a format string. See documentation of numpy.savetxt
            for more details.
        delimiter: string
            character separating columns

    """

    if uncrt is not None:
        Ux, Uy = uncrt
        header = 'Variables: x, y, u, v, mask, Ux, Uy'
        out = np.vstack( [m.ravel()] for m in [x,y,u,v,mask, Ux, Uy ] )
        #safe the file
        np.savetxt(filename, out.T, fmt = fmt, delimiter = delimiter, header = header)
    else:
        header = 'Variables: x, y, u, v, mask'
        out = np.vstack( [m.ravel()] for m in [x,y,u,v,mask ])
        #safe the file
        np.savetxt(filename, out.T, fmt = fmt, delimiter = delimiter, header = header)


def load_davis(filename, skip = 2, uncrt = None):
    """
    Loads the velocity fields / uncertainty from an OpenPIV  data output file
    returns the data from each file in a 3D matrix

    Parameters
    ----------
    filename: string
        The absolute path to the PIV data

    skip: int
        Number of header lines to skip in the data files. Defaults to 2 as that is what the save function defaults to.

    uncrt: tuple
        Column (zero indexed) of the u and v velocity uncertainty.
        Default is None, which returns no uncertainty.

    Returns
    -------

    u,v : 3D numpy array
        The u amd v velocity fields for each piv file

    mask : 3D array
        Values that were masked during the validation process.
    """

    vbuff, vatts   =  ReadIM.extra.get_Buffer_andAttributeList(dataList[0])
    velocity, vbuff = ReadIM.extra.buffer_as_array(vbuff)
    del(vbuff, vatts)

    u = velocity[1,:,:]
    v = velocity[2,:,:]
    mask = velocity[0,:,:]

    return(u,v,mask)


def shift(x,y,img):
    """
    Shifts an image by a certain number of pixels in the x and y direction

    Parameters
    ----------
    x, y : int
        number of pixels to shift in the x and y directions respectively. must be positive ints

    img : 2D array
        image you want to shift

    Returns
    -------
    img_shift : 2D array
        shifted image
    """

    assert np.all([x>0, y>0]), "x and y must be positive integers"

    # allocate space for return
    img_shift = np.zeros_like(img, dtype = np.int32)

    # flip image as they are stored like this
    img = np.flipud(img)
    img_shift[y:, x:] = img[:-y, :-x]
    img_shift = np.flipud(img_shift)

    return(img_shift)


def batch_histogram_adjust(data_dir, output_dir, pattern_a, pattern_b, r_low=60, r_high=99, out_file='npy', show_image=False):
    """Adjust the contast of each PIV image in a pair for a batch of PIV images
    """

    if data_dir[-1] != '/':
        data_dir = data_dir + '/'

    # check inputs
    assert out_file == 'tif' or out_file == 'npy', "Unknown save format. Use either tif or npy"

    # get list of images
    im_list_A = sorted(glob.glob(data_dir + pattern_a))
    im_list_B = sorted(glob.glob(data_dir + pattern_b))

    assert len(im_list_A) == len(im_list_B), 'Different number of images in pattern A and B. Each image needs a pair.'

    if len(im_list_A) == 0:
        raise ValueError('No images found, check pattern or directory.')

    # adjust the contrast of each image pair
    for i in range(len(im_list_A)):
        # load image
        im_A = io.imread(im_list_A[i])
        im_B = io.imread(im_list_B[i])

        # adjust contrast of each image
        im_A = contrast(im_A, r_low = r_low, r_high=r_high)
        im_B = contrast(im_B, r_low = r_low, r_high=r_high)

        if show_image == True:
            fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 4.5))
            ax0.imshow(im_A, cmap="gray")
            ax1.imshow(im_B, cmap="gray")
            plt.show()

        # save image based on input name
        outname_A = os.path.splitext(os.path.basename(im_list_A[i]))[0]
        outname_B = os.path.splitext(os.path.basename(im_list_B[i]))[0]

        if out_file == 'tif':
            io.imsave(output_dir + outname_A + "_adjusted.tif", im_A)
            io.imsave(output_dir + outname_B + "_adjusted.tif", im_B)
        elif out_file == 'npy':
            np.save(output_dir + outname_A + "_adjusted.npy", im_A)
            np.save(output_dir + outname_B + "_adjusted.npy", im_B)
        else:
            raise ValueError("Unknown save format. Use either tif or npy")


def test_histogram_adjust(data_dir, file_A, file_B, r_low=60, r_high=99):
    """ test the values for adjusting PIV images
    """
    # Don't need this since I have to navigate to the directory anyways
    # Uncomment this if you are not already in the directory where the raw files are stored
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'

    # load images
    im_A = io.imread(data_dir + file_A).astype(np.int8)
    im_B = io.imread(data_dir + file_B).astype(np.int8)

    # adjust contrast of each image
    im_A = contrast(im_A, r_low = r_low, r_high=r_high)
    im_B = contrast(im_B, r_low = r_low, r_high=r_high)

    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 4.5))
    ax0.imshow(im_A, cmap="gray")
    ax1.imshow(im_B, cmap="gray")
    plt.show()

    # create histograms
    bins = np.arange(0, np.max([im_A, im_B])+1, 10)
    counts, nbins, patches = plt.hist([im_A.flatten(), im_B.flatten()], bins = bins, rwidth = 1, color = ['steelblue', (0.85, 0.75, 0.85)], label = ["Image A", "Image B"])

    # label all bins and show legend
    plt.xticks(bins)
    plt.legend()
    plt.show()

    # return the difference in counts between image A and image B
    diff = abs(counts[0]-counts[1])

    return(diff)

