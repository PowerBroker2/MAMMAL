import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from ppigrf import igrf


def b_earth_ned_igrf(lats:    np.ndarray,
                     lons:    np.ndarray,
                     heights: np.ndarray,
                     dates:   list) -> np.ndarray:
    '''
    Find the IGRF Earth field in the NED frame
    
    Parameters
    ----------
    lats
        Latitude (dd) or 1xN array of latitudes (dd)
    lons
        Longitude (dd) or 1xN array of longitudes (dd)
    heights
        Height MSL (m) or 1xN array of heights MSL (m)
    dates
        DateTime (UTC) object or 1xN List of DateTime (UTC) objects

    Returns
    -------
    np.ndarray
        Nx3 array of IGRF Earth field (nT) -> [Bn, Be, Ba] <- 1st time sample
                                              [.   .   . ]
                                              [.   .   . ]
                                              [.   .   . ] <- Nth time sample
    '''
    
    Be, Bn, Bu = igrf(lons, lats, heights / 1000, dates)
    
    if type(lats) is not np.ndarray:
        return np.hstack((Bn.squeeze(), Be.squeeze(), -Bu.squeeze()))
    elif (len(lats) > 1) or (len(lons) > 1):
        return np.hstack((Bn.squeeze()[:, np.newaxis], Be.squeeze()[:, np.newaxis], -Bu.squeeze()[:, np.newaxis]))
    else:
        return np.hstack((Bn.squeeze(), Be.squeeze(), -Bu.squeeze()))

def gen_b_truth_dcm(lat:    float,
                    lon:    float,
                    height: float,
                    date:   dt.datetime,
                    dcms:   np.ndarray) -> np.ndarray:
    '''
    Rotate the IGRF Earth field into the senor frame across
    entire calibration using a tensor of direction cosine
    matrices
    
    Parameters
    ----------
    lat
        Latitude (dd)
    lon
        Longitude (dd)
    height
        Height MSL (m)
    date
        datetime.datetime (UTC) object
    dcms
        Nx3x3 Tensor array of direction cosine matrices
        where N is the number of samples taken in the
        calibration

    Returns
    -------
    np.ndarray
        Nx3 array of IGRF Earth field (nT) rotated
        into sensor frame -> [Bx, By, Bz] <- 1st time sample
                             [.   .   . ]
                             [.   .   . ]
                             [.   .   . ] <- Nth time sample
        where N is the number of samples taken in the
        calibration
    '''
    
    b_earth = b_earth_ned_igrf(lat, lon, height, date)
    
    return dcms @ b_earth

def gen_b_truth_euler(lat:     float,
                      lon:     float,
                      height:  float,
                      date:    dt.datetime,
                      eulers:  np.ndarray,
                      rot_seq: str='zyx',
                      degrees: bool=True) -> np.ndarray:
    '''
    Rotate the IGRF Earth field into the senor frame across
    entire calibration using an array of euler angles
    
    Parameters
    ----------
    lat
        Latitude (dd)
    lon
        Longitude (dd)
    height
        Height MSL (m)
    date
        datetime.datetime (UTC) object
    eulers
        Nx3 array of sensor pitch, roll, yaw euler angles
        (rotates sensor frame - _not measurement vector_ - from NED to body)
        where N is the number of samples taken in the
        calibration -> [roll, pitch, yaw] <- 1st time sample
                       [.     .      .  ]
                       [.     .      .  ]
                       [.     .      .  ] <- Nth time sample
    rot_seq
        Rotation sequence for the euler angles
    degrees
        Whether the euler angles are in degrees or not

    Returns
    -------
    np.ndarray
        Nx3 array of IGRF Earth field (nT) rotated
        into sensor frame -> [Bx, By, Bz] <- 1st time sample
                             [.   .   . ]
                             [.   .   . ]
                             [.   .   . ] <- Nth time sample
        where N is the number of samples taken in the
        calibration
    '''
    
    flpd_angs = np.flip(-eulers, axis=1) # Negate angles because we want to rotate NED vectors into body frame and flip angle order cause scipy rotations are stupid
    dcms      = R.from_euler(rot_seq, flpd_angs, degrees=degrees).as_matrix()
    
    return gen_b_truth_dcm(lat, lon, height, date, dcms)

def ab_to_x(a: np.ndarray,
            b: np.ndarray) -> np.ndarray:
    '''
    Combine distortion matrix and bias vector into a single calibration vector
    
    Parameters
    ----------
    a
        3x3 distortion matrix
    b
        1x3 bias vector

    Returns
    -------
    np.ndarray
        1x12 array of elements where the first 9 are from `a` and the last 3 are from `b`
    '''
    
    return np.concatenate([a.flatten(), b.flatten()])

def x_to_ab(x: np.ndarray) -> np.ndarray:
    ''' 
    Split calibration vector into distortion matrix and bias vector
    
    Parameters
    ----------
    x
        1x12 array of elements where the first 9 are from `a` and the last 3 are from `b`

    Returns
    -------
    a
        3x3 distortion matrix
    b
        1x3 bias vector
    '''
    
    a = x[0:9].reshape(3,3)
    b = x[9:12]
    
    return a, b

def apply_dist_to_vec(vec: np.ndarray,
                      a:   np.ndarray,
                      b:   np.ndarray) -> np.ndarray:
    '''
    Apply distortion to vector data
    
    Parameters
    ----------
    vec
        Nx3 array of vector magnetometer measurements
        where N is the number of samples taken in the
        calibration
    a
        3x3 distortion matrix
    b
        1x3 bias vector
    
    Returns
    -------
    np.ndarray
        Nx3 array of distorted vector measurements (nT)
        where N is the number of samples taken in the
        calibration
    '''
    
    return (a @ (vec + b).T).T

def apply_distx_to_vec(vec: np.ndarray,
                       x:   np.ndarray) -> np.ndarray:
    '''
    Apply distortion vector to vector data
    
    Parameters
    ----------
    vec
        Nx3 array of vector magnetometer measurements
        where N is the number of samples taken in the
        calibration
    x
        1x12 calibration vector
    
    Returns
    -------
    np.ndarray
        Nx3 array of distorted vector measurements (nT)
        where N is the number of samples taken in the
        calibration
    '''
    
    return apply_dist_to_vec(vec, *x_to_ab(x))

def apply_cal_to_vec(vec: np.ndarray,
                     a:   np.ndarray,
                     b:   np.ndarray) -> np.ndarray:
    '''
    Apply distortion and bias calibration to vector data
    
    Parameters
    ----------
    vec
        Nx3 array of vector magnetometer measurements
        where N is the number of samples taken in the
        calibration
    a
        3x3 distortion matrix
    b
        1x3 bias vector
    
    Returns
    -------
    np.ndarray
        Nx3 array of calibrated vector measurements (nT)
        where N is the number of samples taken in the
        calibration
    '''
    
    return la.solve(a, vec.T).T - b

def apply_calx_to_vec(vec: np.ndarray,
                      x:   np.ndarray) -> np.ndarray:
    '''
    Apply calibration vector to vector data
    
    Parameters
    ----------
    vec
        Nx3 array of vector magnetometer measurements
        where N is the number of samples taken in the
        calibration
    x
        1x12 calibration vector
    
    Returns
    -------
    np.ndarray
        Nx3 array of calibrated vector measurements (nT)
        where N is the number of samples taken in the
        calibration
    '''
    
    return apply_cal_to_vec(vec, *x_to_ab(x))

def min_func_vec(x: np.ndarray,
                 *args,
                 **kwargs) -> float:
    '''
    Cost function to minimize when calibrating vector magnetometer
    
    Parameters
    ----------
    x
        1x12 calibration vector
    *args
        list of distorted and "true" measurements -> [b_distorted (nT), b_true (nT)]
    
    Returns
    -------
    float
        Calibration cost
    '''

    b_distorted = args[0]
    b_true      = args[1]

    guess    = apply_calx_to_vec(b_distorted, x)
    vec_diff = np.linalg.norm((b_true - guess)**2, axis=1)
    
    return vec_diff.sum()

def calx(x0:          np.ndarray,
         b_distorted: np.ndarray,
         b_true:      np.ndarray,
         min_func) -> np.ndarray:
    '''
    Find the calibration vector that minimizes the given cost function
    
    Parameters
    ----------
    x
        1x12 calibration vector
    *args
        list of distorted and "true" measurements -> [b_distorted (nT), b_true (nT)]
    
    Returns
    -------
    np.ndarray
        1x12 calibration vector
    '''
    
    return minimize(min_func, x0, args=(b_distorted, b_true), method='Nelder-Mead').x

def calibrate_vec(x0:           np.ndarray,
                  b_distorted:  np.ndarray,
                  b_true:       np.ndarray,
                  max_iters:    int=100,
                  converge_lim: float=0.1) -> np.ndarray:
    '''
    Iterative approach to calibrate vector magnetometer data
    
    Parameters
    ----------
    x0
        1x12 calibration vector (initial guess)
    b_distorted
        Nx3 array of uncalibrated vector measurements (nT)
        where N is the number of samples taken in the
        calibration
    b_true
        Nx3 array of the IGRF Earth field (nT) rotated into the sensor frame
        where N is the number of samples taken in the
        calibration
    max_iters
        Maximum number of iterations the calibration is allowed to run
    converge_lim
        The minimum magnitude difference between the previous and current
        calibration iterations that denote convergence
    
    Returns
    -------
    calibrated
        Nx3 array of calibrated vector measurements
        where N is the number of samples taken in the
        calibration
    x
        1x12 calibration vector (final estimate)
    '''
    
    distorted  = b_distorted
    x          = calx(x0, b_distorted, b_true, min_func_vec)
    calibrated = apply_calx_to_vec(b_distorted, x)
    
    i  = 0
    dx = la.norm(distorted - calibrated)
    
    while (dx > converge_lim) and (i < max_iters):
        distorted  = b_distorted
        x          = calx(x, b_distorted, b_true, min_func_vec)
        calibrated = apply_calx_to_vec(b_distorted, x)
        
        i += 1
        dx = la.norm(distorted - calibrated)
        
    if i > max_iters:
        print('calibrate_vec took too many iterations to converge')
    
    return calibrated, x

def plot_sphere(ax,
                radius: float,
                color:  str='r'):
    '''
    Add a wireframe sphere to an axis ax
    
    Parameters
    ----------
    ax
        Axis object to draw in, must support 3d projections
    radius
        Radius of sphere to draw
    color
        Color of sphere to draw, defaults to red
    '''
    
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    
    x = np.cos(u) * np.sin(v) * radius
    y = np.sin(u) * np.sin(v) * radius
    z = np.cos(v) * radius
    
    ax.plot_wireframe(x, y, z, color=color) 

def plot_spin_data(b_vec_true: np.ndarray,
                   b_vec_dist: np.ndarray,
                   b_vec_cal:  np.ndarray=None):
    '''
    Plot vector data from spin test (before or after calibration)
    
    Parameters
    ----------
    b_vec_true
        Nx3 array of ideal vector measurements (Earth's
        field rotated in the sensor's frame with no
        distortion)
    b_vec_dist
        Nx3 array of distorted vector measurements (Earth's
        field rotated in the sensor's frame with distortion)
    b_vec_cal
        Nx3 array of calibrated vector measurements
    '''
    
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    
    b_earth_mag = la.norm(b_vec_true, axis=1).mean()
    
    plot_sphere(ax, b_earth_mag)
    
    b_vec_true_xs = b_vec_true[:, 0]
    b_vec_true_ys = b_vec_true[:, 1]
    b_vec_true_zs = b_vec_true[:, 2]
    
    ax.scatter(b_vec_true_xs,
               b_vec_true_ys,
               b_vec_true_zs,
               label="True Earth Field")
    
    b_vec_dist_xs = b_vec_dist[:, 0]
    b_vec_dist_ys = b_vec_dist[:, 1]
    b_vec_dist_zs = b_vec_dist[:, 2]
    
    ax.scatter(b_vec_dist_xs,
               b_vec_dist_ys,
               b_vec_dist_zs,
               label="Distorted Measurements")
    
    if b_vec_cal is not None:
        b_vec_cal_xs = b_vec_cal[:, 0]
        b_vec_cal_ys = b_vec_cal[:, 1]
        b_vec_cal_zs = b_vec_cal[:, 2]
        
        ax.scatter(b_vec_cal_xs,
                   b_vec_cal_ys,
                   b_vec_cal_zs,
                   marker='x',
                   label="Calibrated Measurements")
    
    plt.title('Spin Test Data')
    ax.legend()


if __name__ == '__main__':
    lat    = 39.775784
    lon    = -84.109811
    height = 160
    date   = dt.datetime.utcnow()
    
    print('B Earth NED IGRF:\n', b_earth_ned_igrf(lat, lon, height, date))
    
    eulers = np.array([[ 0,  0,  0],  # No rotation
                       [90,  0,  0],  # 90 deg roll right
                       [ 0, 90,  0],  # 90 deg pitch up
                       [ 0,  0, 90],  # 90 deg yaw right
                       [90, 90,  0],  # 90 deg pitch up then 90 deg roll right
                       [ 0, 90, 90],  # 90 deg yaw right then 90 deg pitch up
                       [90,  0, 90],  # 90 deg yaw right then 90 deg roll right
                       [90, 90, 90]]) # 90 deg yaw right then 90 deg pitch up then 90 deg roll right
    
    b_true = gen_b_truth_euler(lat, lon, height, date, eulers)
    
    print('True:\n', b_true)
    print('')
    
    a = np.array([[1.1, 0, 0],
                  [0, 1.2, 0],
                  [0, 0, 1.3]])
    b = np.array([3, 4, 2])
    
    b_distorted = apply_dist_to_vec(b_true, a, b)
    
    print('Distorted:\n', b_distorted)
    print('Distortion Vector:\n', ab_to_x(a, b))
    print('')
    
    a  = np.eye(3)
    b  = np.zeros(3)
    x0 = ab_to_x(a, b)
    
    calibrated, x = calibrate_vec(x0, b_distorted, b_true)
    
    print('Estimated Distortion Vector:\n', x)
    print('Calibrated:\n', calibrated)
    print('True vs Calibrated Difference:\n', b_true - calibrated)
    print('')
    
    plot_spin_data(b_true,
                   b_distorted,
                   calibrated)
    plt.show()