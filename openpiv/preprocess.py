"""This module contains image processing routines that improve
images prior to PIV processing."""

__licence_ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


from scipy.ndimage import median_filter, gaussian_filter, binary_fill_holes
from skimage import io, img_as_float, exposure, data, img_as_uint
from skimage.filters import sobel, rank, threshold_otsu
import numpy as np


def dynamic_masking(image,method='edges',filter_size=7,threshold=0.005):
    """ Dynamically masks out the objects in the PIV images
    
    Parameters
    ----------
    image: image
        a two dimensional array of uint16, uint8 or similar type
        
    method: string
        'edges' or 'intensity':
        'edges' method is used for relatively dark and sharp objects, with visible edges, on 
        dark backgrounds, i.e. low contrast
        'intensity' method is useful for smooth bright objects or dark objects or vice versa, 
        i.e. images with high contrast between the object and the background
    
    filter_size: integer
        a scalar that defines the size of the Gaussian filter
    
    threshold: float
        a value of the threshold to segment the background from the object
        default value: None, replaced by sckimage.filter.threshold_otsu value
            
    Returns
    -------
    image : array of the same datatype as the incoming image with the object masked out
        as a completely black region(s) of zeros (integers or floats).
    
    
    Example
    --------
    frame_a  = openpiv.tools.imread( 'Camera1-001.tif' )
    imshow(frame_a) # original
    
    frame_a = dynamic_masking(frame_a,method='edges',filter_size=7,threshold=0.005)
    imshow(frame_a) # masked 
        
    """
    imcopy = np.copy(image)
    # stretch the histogram
    image = exposure.rescale_intensity(img_as_float(image), in_range=(0, 1))
    # blur the image, low-pass
    blurback = gaussian_filter(image,filter_size)
    if method is 'edges':
        # identify edges
        edges = sobel(blurback)
        blur_edges = gaussian_filter(edges,21)
        # create the boolean mask 
        bw = (blur_edges > threshold)
        bw = binary_fill_holes(bw)
        imcopy -= blurback
        imcopy[bw] = 0.0
    elif method is 'intensity':
        background = gaussian_filter(median_filter(image,filter_size),filter_size)
        imcopy[background > threshold_otsu(background)] = 0

        
    return imcopy #image
  
    
def subtract_min(image):
    """
    Subtract the minimum pixel value from each pixel in the image.
    Readjusting the contrast should be done after this.

    Parameters
    ----------
    image: 2d array - int
        PIV image

    Returns
    -------
    image: 2d array - int
        image with minimum value subtracted
    """

    # make sure image is correct data type
    assert image.dtype == 'int32', 'Input image is not in correct format for openpiv. Change to int32.'
    assert image.min() >= 0, 'Image has negative values.'

    image -= image.min().astype(np.int32)

    return(image)


def subtract_background(image):
    """
    Subtract the mean intensity from each pixel in the image. 
    After subtraction, set all negative values to zero.
    Readjusting the contrast should be done after this.

    Parameters
    ----------
    image: 2d array - int
        PIV image

    Returns
    -------
    image_sub: 2d array - int
        image with background intensity subtracted
    """

    # make sure image is correct data type
    assert image.dtype == 'int32', 'Input image is not in correct format for openpiv. Change to int32.'

    image_sub = image - image.mean()
    image_sub[image_sub < 0] = 0

    return(image_sub.astype(np.int32))


def rescale_intensity(image,  p_range = None):
    """
    Adjust constrast of an image

    Parameters
    ----------
    image: 2d array
        image you want to adjust
    p_range: tuple of ints
        pecentage range of brightness to enhance (low, high)

    Returns
    -------
    image_rescale: 2d array
        contrast adjusted image 
    """

    # make sure image is correct data type
    assert image.dtype == 'int32', 'Input image is not in correct format for openpiv. Change to int32.'

    #adjust contrast
    #by adjusting the numbers in np.percentile, you choose which range of brightness you enhance

    if p_range is not None:
        #rescale only part of the intensity range of the image
        pLow, pHigh = np.percentile(image, p_range)
        image_rescale = exposure.rescale_intensity(image, in_range = (pLow, pHigh))
        return(image_rescale)
    else:
        #rescale the entire intensity range of the image
        image_rescale = exposure.rescale_intensity(image)
        return(image_rescale)


def gauss_smooth(image):
    """
    3x3 Gaussian smoother to an image

    Parameters
    ----------
    image: 2d array - int32
        PIV image to be smoothed

    Returns
    -------
    im_smooth: 2d array - int
        Smoothed image
    """

    im_smooth = gaussian_filter(image, 1, truncate = 1)

    return(im_smooth.astype(np.int32))
