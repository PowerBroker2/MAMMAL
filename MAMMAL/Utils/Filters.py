from copy import deepcopy

import numpy as np
from numpy import fft
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfilt_zi


def compute_wavevectors(dx: float,
                        dy: float,
                        nx: int,
                        ny: int) -> list:
    '''
    Compute the wave vectors for a spatial fourier transfrom in 
    Cartesian coordinates

    Parameters
    ----------
    dx
        x-pixel size (m)
    dy
        y-pixel size (m)
    nx
        Number of pixels in x
    ny
        Number of pixesl in y

    Returns
    -------
    list
        Spatial wavevectors kx, ky as 2D grids --> [kx, ky]
    '''

    kx = 2 * np.pi * fft.fftfreq(n=nx, d=dx)
    ky = 2 * np.pi * fft.fftfreq(n=ny, d=dy)

    kx2d, ky2d = np.meshgrid(kx, ky)

    return [kx2d, ky2d]

def filt(data:   np.ndarray,
         cutoff: float,
         fs:     float,
         btype:  str,
         order:  int=6,
         axis:   int=-1) -> np.ndarray:
    '''
    Filter an MxN array of data in a single direction
    without phase shift
    
    Parameters
    ----------
    data
        MxN array of original data
    cutoff
        Filter's cutoff frequency
    fs
        Sample frequency of dataset
    btype
        Filter type (i.e. 'low' for LPF)
    order
        Filter order
    axis
        Axis along which to filter
        
    Returns
    -------
    np.ndarray
        MxN array of filtered data
    '''
    
    sos = butter(order,
                 cutoff,
                 fs=fs,
                 btype=btype,
                 analog=False,
                 output='sos')
    
    return sosfiltfilt(sos,
                       data,
                       axis=axis)

def bpf(data:   np.ndarray,
        fstart: float,
        fstop:  float,
        fs:     float,
        order:  int=6,
        axis:   int=-1) -> np.ndarray:
    '''
    Band pass filter data array
    
    Parameters
    ----------
    data
        NxM array of data to be filtered
    fstart
        Start frequency for the band (Hz)
    fstop
        Stop frequency for the band (Hz)
    fs
        Data sample frequency (Hz)
    order
        Order of the filter

    Returns
    -------
    x_filt
        Band pass filtered data
    '''
    
    return filt(data,
                [fstart, fstop],
                fs,
                'band',
                order,
                axis)

def lpf(data:   np.ndarray,
        cutoff: float,
        fs:     float,
        order:  int=6,
        axis:   int=-1) -> np.ndarray:
    '''
    Low pass filter an MxN array of data in a single direction
    
    Parameters
    ----------
    data
        MxN array of original data
    cutoff
        Filter's cutoff frequency
    fs
        Sample frequency of dataset
    order
        Filter order
    axis
        Axis along which to filter
        
    Returns
    -------
    np.ndarray
        MxN array of filtered data
    '''
    
    return filt(data,
                cutoff,
                fs,
                'low',
                order,
                axis)

def lpf2(data:   np.ndarray,
         cutoff: float,
         dx:     float,
         dy:     float,
         order:  float=6) -> np.ndarray:
    '''
    Low pass filter a 2D array of data
    
    Parameters
    ----------
    data
        MxN array of original data
    cutoff
        Filter's max spacial wavelength
    dx
        Difference in x coordinates (i.e. meters
        per pixel in the x direction)
    dy
        Difference in y coordinates (i.e. meters
        per pixel in the y direction)
    
    Returns
    -------
    data_lpf
        MxN array of low pass filtered data
    '''
    
    data_copy = deepcopy(data)
    
    # Filter in Y direction
    data_lpf_y = lpf(data   = data_copy,
                     cutoff = 1/cutoff,
                     fs     = 1/dy,
                     order  = order,
                     axis   = 0)
    
    # Filter in X direction
    data_lpf = lpf(data   = data_lpf_y,
                   cutoff = 1/cutoff,
                   fs     = 1/dx,
                   order  = order,
                   axis   = 1)
    
    return data_lpf

def hpf(data:   np.ndarray,
        cutoff: float,
        fs:     float,
        order:  int=6,
        axis:   int=-1) -> np.ndarray:
    '''
    High pass filter an MxN array of data in a single direction
    
    Parameters
    ----------
    data
        MxN array of original data
    cutoff
        Filter's cutoff frequency
    fs
        Sample frequency of dataset
    order
        Filter order
    axis
        Axis along which to filter
        
    Returns
    -------
    np.ndarray
        MxN array of filtered data
    '''
    
    return filt(data,
                cutoff,
                fs,
                'high',
                order,
                axis)