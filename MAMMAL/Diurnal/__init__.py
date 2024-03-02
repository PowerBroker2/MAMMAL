import sys
from os.path import dirname, realpath

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize

sys.path.append(dirname(realpath(__file__)))
sys.path.append(dirname(dirname(realpath(__file__))))

from Utils import Filters


E_ROT_DEG_S = 360.9856 / 24 / 60 / 60 # Earth's rotation rate in degrees/s


def interp_reference_df(df:            pd.DataFrame,
                        timestamps:    np.ndarray,
                        survey_lon:    float=None,
                        ref_scale:     float=1,
                        ref_offset:    float=0,
                        subtract_core: bool=False) -> list:
    '''
    Generate interpolated reference station data from a given magnetic DataFrame
    
    Parameters
    ----------
    df
        DataFrame of reference station data. DataFram must include the
        following columns:
        
        - LONG:      Longitude (dd)
        - epoch_sec: UNIX epoch timestamp (s)
        - X:         Magnetic field measurement in the North direction (nT)
        - Y:         Magnetic field measurement in the East direction (nT)
        - Z:         Magnetic field measurement in the Down direction (nT)
        - IGRF_X:    IGRF magnetic field in the North direction (nT)
        - IGRF_y:    IGRF magnetic field in the East direction (nT)
        - IGRF_z:    IGRF magnetic field in the Down direction (nT)
        
    survey_lon
        Approximate longitude of the survey area (dd)
    timestamps
        1xN array of UNIX epoch timestamps to interpolate the INTERMAGNET
        data at (s)
    ref_scale
        Amount to scale the reference station data by
    ref_offset
        Amount to bias the reference station data by (nT)
    subtract_core
        Whether or not to subtract IGRF Earth core field from returned reference data
    
    Returns
    -------
    list
        Interpolated vector and scalar reference station
        measurements --> [Nx3 vector array (nT), 1xN scalar array (nT)]
    '''
    
    if survey_lon is not None:
        t, f, x, y, z = longitude_norm(df, survey_lon)
    else:
        t = np.array(df.epoch_sec)
        
        x = np.array(df.X)
        y = np.array(df.Y)
        z = np.array(df.Z)
        f = np.array(df.F)
    
    IGRF_x = np.array(df.IGRF_X)
    IGRF_y = np.array(df.IGRF_Y)
    IGRF_z = np.array(df.IGRF_Z)
    IGRF_f = np.array(df.IGRF_F)
    
    min_timestamp = timestamps.min()
    max_timestamp = timestamps.max()
    min_t = t.min()
    max_t = t.max()
    
    assert min_t <= min_timestamp, 'Reference data must start before the start of the survey, is currently off by {}s'.format(min_t - min_timestamp)
    assert max_t >= max_timestamp, 'Reference data must extend beyond the end of the survey, is currently off by {}s'.format(max_timestamp - max_t)
    
    interp_x  = interpolate.interp1d(t, x, 'linear')
    interp_y  = interpolate.interp1d(t, y, 'linear')
    interp_z  = interpolate.interp1d(t, z, 'linear')
    interp_f  = interpolate.interp1d(t, f, 'linear')
    
    interp_IGRF_x  = interpolate.interp1d(df.epoch_sec, IGRF_x, 'linear')
    interp_IGRF_y  = interpolate.interp1d(df.epoch_sec, IGRF_y, 'linear')
    interp_IGRF_z  = interpolate.interp1d(df.epoch_sec, IGRF_z, 'linear')
    interp_IGRF_f  = interpolate.interp1d(df.epoch_sec, IGRF_f, 'linear')
    
    IGRF_x_interp = interp_IGRF_x(timestamps)
    IGRF_y_interp = interp_IGRF_y(timestamps)
    IGRF_z_interp = interp_IGRF_z(timestamps)
    IGRF_f_interp = interp_IGRF_f(timestamps)
    
    x_interp_no_core = interp_x(timestamps) - IGRF_x_interp
    y_interp_no_core = interp_y(timestamps) - IGRF_y_interp
    z_interp_no_core = interp_z(timestamps) - IGRF_z_interp
    f_interp_no_core = interp_f(timestamps) - IGRF_f_interp
    
    x_interp_cal_no_core = apply_cal([ref_offset, ref_scale], x_interp_no_core)
    y_interp_cal_no_core = apply_cal([ref_offset, ref_scale], y_interp_no_core)
    z_interp_cal_no_core = apply_cal([ref_offset, ref_scale], z_interp_no_core)
    f_interp_cal_no_core = apply_cal([ref_offset, ref_scale], f_interp_no_core)
    
    x_interp_cal = x_interp_cal_no_core + IGRF_x_interp
    y_interp_cal = y_interp_cal_no_core + IGRF_y_interp
    z_interp_cal = z_interp_cal_no_core + IGRF_z_interp
    f_interp_cal = f_interp_cal_no_core + IGRF_f_interp
    
    if subtract_core:
        ref_vec = np.hstack([x_interp_cal_no_core[:, np.newaxis],
                             y_interp_cal_no_core[:, np.newaxis],
                             z_interp_cal_no_core[:, np.newaxis]])
        ref_mag = f_interp_cal_no_core
    
    else:
        ref_vec = np.hstack([x_interp_cal[:, np.newaxis],
                             y_interp_cal[:, np.newaxis],
                             z_interp_cal[:, np.newaxis]])
        ref_mag = f_interp_cal
    
    return [ref_vec, ref_mag]

def longitude_norm(ref_df:       pd.DataFrame,
                   survey_lon:   float,
                   corner_frq:   float=1/(3600*3)) -> list:
    '''
    When using data from a reference station with sufficient
    difference in longitude from the survey site, the
    difference in longitude can be accounted for. This is
    done by time shifting the low frequency content of the
    reference station data
    
    Parameters
    ----------
    ref_df
        DataFrame of reference station data
    survey_lon
        Average longitude of the survey site
    corner_frq
        Cutoff frequency that separates which frequencies
        will be time shifted and which frequencies will
        retain their original timestamps
    
    Returns
    -------
    list
        List of new timestamps and scalar measurements
        based on the required time shift -> [timestamps (s), scalar (nT), x/North vector (nT), y/East vector (nT), z/Down vector (nT)]
    '''
    
    t = np.array(ref_df.epoch_sec)
    f = np.array(ref_df.F)
    x = np.array(ref_df.X)
    y = np.array(ref_df.Y)
    z = np.array(ref_df.Z)
    
    from2toLonDiff = ref_df.LONG.mean() - survey_lon
    lon_t_offset   = pd.Timedelta(seconds=from2toLonDiff / E_ROT_DEG_S)
    
    lpf_f = Filters.lpf(f, corner_frq, 1, 1)
    hpf_f = Filters.hpf(f, corner_frq, 1, 1)
    
    lpf_x = Filters.lpf(x, corner_frq, 1, 1)
    hpf_x = Filters.hpf(x, corner_frq, 1, 1)
    
    lpf_y = Filters.lpf(y, corner_frq, 1, 1)
    hpf_y = Filters.hpf(y, corner_frq, 1, 1)
    
    lpf_z = Filters.lpf(z, corner_frq, 1, 1)
    hpf_z = Filters.hpf(z, corner_frq, 1, 1)
    
    lpf_t = t + lon_t_offset.total_seconds()
    
    interp_lpf_f = interpolate.interp1d(lpf_t, lpf_f, 'cubic')
    interp_hpf_f = interpolate.interp1d(t,     hpf_f, 'cubic')
    
    interp_lpf_x = interpolate.interp1d(lpf_t, lpf_x, 'cubic')
    interp_hpf_x = interpolate.interp1d(t,     hpf_x, 'cubic')
    
    interp_lpf_y = interpolate.interp1d(lpf_t, lpf_y, 'cubic')
    interp_hpf_y = interpolate.interp1d(t,     hpf_y, 'cubic')
    
    interp_lpf_z = interpolate.interp1d(lpf_t, lpf_z, 'cubic')
    interp_hpf_z = interpolate.interp1d(t,     hpf_z, 'cubic')
    
    combined_t = lpf_t[np.logical_and(lpf_t >= t.min(), lpf_t <= t.max())] # Clip interpolation times
    
    lpf_interp_f = interp_lpf_f(combined_t)
    hpf_interp_f = interp_hpf_f(combined_t)
    
    lpf_interp_x = interp_lpf_x(combined_t)
    hpf_interp_x = interp_hpf_x(combined_t)
    
    lpf_interp_y = interp_lpf_y(combined_t)
    hpf_interp_y = interp_hpf_y(combined_t)
    
    lpf_interp_z = interp_lpf_z(combined_t)
    hpf_interp_z = interp_hpf_z(combined_t)
    
    combined_f = lpf_interp_f + hpf_interp_f
    combined_x = lpf_interp_x + hpf_interp_x
    combined_y = lpf_interp_y + hpf_interp_y
    combined_z = lpf_interp_z + hpf_interp_z
    
    return [combined_t, combined_f, combined_x, combined_y, combined_z]

def apply_dist(x:    np.ndarray,
               data: np.ndarray) -> np.ndarray:
    '''
    Apply distortion to diurnal data
    
    Parameters
    ----------
    x
        1xN array of diurnal measurements
    data
        1x2 distortion vector --> [offset (nT), scale]
    
    Returns
    -------
    np.ndarray
        1xN array of distorted diurnal measurements
    '''
    
    if type(x) == tuple: # Idk why, but the code sometimes makes x a tuple of one element...
        x = x[0]
    
    return (data - x[0]) / x[1]

def apply_cal(x:    np.ndarray,
              data: np.ndarray) -> np.ndarray:
    '''
    Apply calibration to distorted diurnal data
    
    Parameters
    ----------
    x
        1xN array of distorted diurnal measurements
    data
        1x2 calibration vector --> [offset (nT), scale]
    
    Returns
    -------
    np.ndarray
        1xN array of calibrated diurnal measurements
    '''
    
    return (x[1] * data) + x[0]

def min_func(x: np.ndarray,
             *args,
             **kwargs) -> float:
    '''
    Cost function to minimize when calibrating diurnal measurements
    
    Parameters
    ----------
    x
        1x2 calibration vector --> [offset (nT), scale]
    *args
        list of distorted and "true" diurnal measurements -> [b_distorted (nT), b_true (nT)]
    
    Returns
    -------
    float
        Calibration cost
    '''
    
    data   = args[0]
    target = args[1]
    
    guess = apply_cal(x, data)
    diff  = (target - guess)**2
    
    return diff.sum()

def calibrate(x0:       np.ndarray,
              localRef: np.ndarray,
              intMag:   np.ndarray) -> np.ndarray:
    '''
    Find the calibration vector that minimizes the given cost function
    
    Parameters
    ----------
    x
        1x2 calibration vector "initial guess" --> [offset (nT), scale]
    *args
        list of distorted and "true" diurnal measurements -> [b_distorted (nT), b_true (nT)]
    
    Returns
    -------
    np.ndarray
        1x2 calibration vectorr --> [offset (nT), scale]
    '''
    
    return minimize(min_func, x0, args=(localRef, intMag), method='Nelder-Mead').x