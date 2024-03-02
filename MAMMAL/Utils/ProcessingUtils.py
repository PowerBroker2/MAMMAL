import sys
from copy import deepcopy
from os.path import dirname, realpath

import numpy as np
import pandas as pd
import rioxarray as rxr
import scipy.linalg as la
from numpy import sin, cos
from scipy import interpolate
from scipy.spatial import distance
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as WNK
from tqdm import tqdm
from ppigrf import igrf

sys.path.append(dirname(realpath(__file__)))
sys.path.append(dirname(dirname(realpath(__file__))))

import Diurnal
from MapLvl import pcaLvl
from MapLvl import tieLvl
from SensorCal import spinCal as sc
from VehicleCal import TL as tl
from Utils import coordinateUtils as cu
from Utils import Filters as filt
from Utils import mapUtils as mu


# Enumerate line directions
HORIZ = 0 # Horizontal
VERT  = 1 # Vertical


def rmse(arr_1, arr_2, axis=None):
    mask = np.logical_and(~np.isnan(arr_1), ~np.isnan(arr_2))
    return np.sqrt(np.mean(np.square(arr_1[mask] - arr_2[mask]), axis=axis))

def clip_data(unclipped, high_clip, low_clip):
    '''
    Clip unclipped between high_clip and low_clip. 
    unclipped contains a single column of unclipped data.
    
    Credit:
    https://stackoverflow.com/a/71230493/9860973
    '''
    
    # convert to np.array to access the np.where method
    np_unclipped = np.array(unclipped)
    
    # clip data above HIGH_CLIP or below LOW_CLIP
    cond_high_clip = (np_unclipped > high_clip) | (np_unclipped < low_clip)
    np_clipped     = np.where(cond_high_clip, np.nan, np_unclipped)
    
    return np_clipped.tolist()


def ewma_fb(df_column, span):
    '''
    Apply forwards, backwards exponential weighted moving average (EWMA) to df_column.
    
    Credit:
    https://stackoverflow.com/a/71230493/9860973
    '''
    
    # Forwards EWMA.
    fwd = pd.Series.ewm(df_column, span=span).mean()
    
    # Backwards EWMA.
    bwd = pd.Series.ewm(df_column[::-1], span=span).mean()
    
    # Add and take the mean of the forwards and backwards EWMA.
    stacked_ewma = np.vstack((fwd, bwd[::-1]))
    fb_ewma      = np.mean(stacked_ewma, axis=0)
    
    return fb_ewma
    
    
def remove_outliers(spikey, fbewma, delta):
    '''
    Remove data from df_spikey that is > delta from fbewma.
    
    Credit:
    https://stackoverflow.com/a/71230493/9860973
    '''
    
    np_spikey = np.array(spikey)
    np_fbewma = np.array(fbewma)
    
    cond_delta         = (np.abs(np_spikey-np_fbewma) > delta)
    np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
    
    return np_remove_outliers

def reject_outliers(df:          pd.DataFrame,
                    window_size: int=100,
                    std_lim:     float=3,
                    col:         str='F') -> pd.DataFrame:
    '''
    Reject ouliers within the scalar data
    
    **NOTE**: This assumes 'F' column is already
    included in the dataframe
    
    Parameters
    ----------
    df
        Dataframe of INTERMAGNET data
    window_size
        Number of scalar samples within the window used for outlier
        rejection. Set to `None` to prevent outlier rejection
    std_lim
        Any values outside this number of standard deviations within
        the given window of scalar values will be discarded
    col
        Name of column of data to process
    
    Returns
    -------
    pd.DataFrame
        Dataframe without outlier datapoints
    '''
    
    new_df = deepcopy(df)
    
    for i in tqdm(range(0, len(new_df[col]), window_size)):
        window_df = deepcopy(new_df.iloc[i:i+window_size])
        window_df[(window_df[col] - window_df[col].mean()).abs() >= (std_lim * window_df[col].std())] = np.nan
        new_df.iloc[i:i+window_size] = window_df

    return new_df.dropna()

def add_igrf_cols(df:        pd.DataFrame,
                  fast_mode: bool=True,
                  chunk:     int=1000) -> pd.DataFrame:
    '''
    Add IGRF vector and magnitude columns for all samples
    in the dataframe
    
    **NOTE**: This assumes 'LONG', 'LAT', 'ALT', and
    'datetime' columns are already included in the
    dataframe
    
    Parameters
    ----------
    df
        Dataframe of INTERMAGNET data
    fast_mode
        Calculate IGRF values based only on the first corrdinate
        in the .sec file (speeds up parsing considerably)
    chunk
        If not parsing in `fast_mode`, set the number of samples
        to simultaneously calculate IGRF values for
    
    Returns
    -------
    pd.DataFrame
        Dataframe of INTERMAGNET data with new IGRF columns
    '''
    
    if fast_mode:
        Be, Bn, Bu = igrf(df.LONG[~np.isnan(df.LONG)].mean(),
                          df.LAT[~np.isnan(df.LAT)].mean(),
                          df.ALT[~np.isnan(df.ALT)].mean() / 1000,
                          df.datetime.mean())
        
        Bn =  Bn.squeeze()
        Be =  Be.squeeze()
        Bd = -Bu.squeeze()
        
        df['IGRF_X'] = Bn
        df['IGRF_Y'] = Be
        df['IGRF_Z'] = Bd
        df['IGRF_F'] = la.norm([Bn, Be, Bd], axis=0)
    
    else:
        df['IGRF_X'] = ''
        df['IGRF_Y'] = ''
        df['IGRF_Z'] = ''
        df['IGRF_F'] = ''
        
        for i in range(0, len(df['datetime']), chunk):
            start = i
            stop  = i + chunk
            
            Be, Bn, Bu = igrf(df.LONG.iloc[start:stop],
                              df.LAT.iloc[start:stop],
                              df.ALT.iloc[start:stop] / 1000,
                              df.datetime.iloc[start:stop])

            Bn = np.diagonal( Bn) # Must use np.diagonal because passing multiple locations to ppigrf is for calculating IGRF values over a grid (provides more values than we want)
            Be = np.diagonal( Be)
            Bd = np.diagonal(-Bu)
            
            df.IGRF_X.iloc[start:stop] = Bn
            df.IGRF_Y.iloc[start:stop] = Be
            df.IGRF_Z.iloc[start:stop] = Bd
            df.IGRF_F.iloc[start:stop] = la.norm([Bn, Be, Bd], axis=0)
    
    return df

def angle2dcm(angles:            np.ndarray,
              angle_unit:        str='degrees',
              NED_to_body:       bool=True,
              rotation_sequence: int=321) -> np.ndarray:
    '''
    Convert euler angles to direction cosine matrix (DCM)
    
    Parameters
    ----------
    angles
        Nx2 array of euler angles (can be in rad or deg) -> [roll, pitch, yaw]
    angle_unit
        'degrees' if angles are in degrees, 'radians' if angles are in radians
    NED_to_body
        Set to True if rotation is North-East-Down (NED) to body frame
    rotation_sequence
        321 for standard ZYX rotation sequence
    
    Returns
    -------
    dcm
        Direction cosine matrix that corresponds to the given euler angles
    '''
    
    if len(angles.shape) == 1:
        angles = angles.reshape(1, 3)
    
    num_angles = angles.shape[0]
    
    if angle_unit.lower() == 'degrees':
        roll  = np.radians(angles[:, 0])
        pitch = np.radians(angles[:, 1])
        yaw   = np.radians(angles[:, 2])
    else:
        roll  = angles[:, 0]
        pitch = angles[:, 1]
        yaw   = angles[:, 2]
    
    # For a single angle, DCM R1 would be:
    # R1 = np.array([[1,          0,         0],
    #                [0,  cos(roll), sin(roll)],
    #                [0, -sin(roll), cos(roll)]])
    
    R1 = np.zeros((num_angles, 3, 3))
    R1[:, 0, 0] =  1
    R1[:, 1, 1] =  cos(roll)
    R1[:, 1, 2] =  sin(roll)
    R1[:, 2, 1] = -sin(roll)
    R1[:, 2, 2] =  cos(roll)

    # For a single angle, DCM R2 would be:
    # R2 = np.array([[cos(pitch), 0, -sin(pitch)],
    #                [0,          1,           0],
    #                [sin(pitch), 0,  cos(pitch)]])
    
    R2 = np.zeros((num_angles, 3, 3))
    R2[:, 0, 0] =  cos(pitch)
    R2[:, 0, 2] = -sin(pitch)
    R2[:, 1, 1] =  1
    R2[:, 2, 0] =  sin(pitch)
    R2[:, 2, 2] =  cos(pitch)

    # For a single angle, DCM R3 would be:
    # R3 = np.array([[ cos(yaw), sin(yaw), 0],
    #                [-sin(yaw), cos(yaw), 0],
    #                [ 0,        0,        1]])
    
    R3 = np.zeros((num_angles, 3, 3))
    R3[:, 0, 0] =  cos(yaw)
    R3[:, 0, 1] =  sin(yaw)
    R3[:, 1, 0] = -sin(yaw)
    R3[:, 1, 1] =  cos(yaw)
    R3[:, 2, 2] =  1

    if rotation_sequence == 321:
        dcms = R1 @ R2 @ R3
    elif rotation_sequence == 312:
        dcms = R2 @ R1 @ R3
    elif rotation_sequence == 231:
        dcms = R1 @ R3 @ R2
    elif rotation_sequence == 213:
        dcms = R3 @ R1 @ R2
    elif rotation_sequence == 132:
        dcms = R2 @ R3 @ R1
    elif rotation_sequence == 123:
        dcms = R3 @ R2 @ R1
    else:
        dcms = R1 @ R2 @ R3

    if not NED_to_body:
        return np.transpose(dcms, axes=(0, 2, 1))

    return dcms

def cal_spin_df(df: pd.DataFrame) -> list:
    '''
    Determine distortion matrix and bias vector used
    to calibrate the vector magnetometer
    
    Parameters
    ----------
    df
        Dataframe of spin test calibration data
    
    Returns
    -------
    a
        3x3 distortion matrix
    b
        1x3 bias vector
    '''
    
    vec    = np.hstack([np.array(df.X)[:, np.newaxis], 
                        np.array(df.Y)[:, np.newaxis],
                        np.array(df.Z)[:, np.newaxis]])
    lat    = df.LAT.mean()
    lon    = df.LONG.mean()
    height = df.ALT.mean()
    date   = df.datetime[0]
    eulers = np.hstack([np.array(df.ROLL)[:, np.newaxis], 
                        np.array(df.PITCH)[:, np.newaxis],
                        np.array(df.AZIMUTH)[:, np.newaxis]])
    
    a  = np.eye(3)
    b  = np.zeros(3)
    x0 = sc.ab_to_x(a, b)
    
    vec_true = sc.gen_b_truth_euler(lat    = lat,
                                    lon    = lon,
                                    height = height,
                                    date   = date,
                                    eulers = eulers)
    
    _, xf = sc.calibrate_vec(x0          = x0,
                             b_distorted = vec,
                             b_true      = vec_true)
    
    return sc.x_to_ab(xf)

def cal_tl_df(df:         pd.DataFrame,
              use_filter: bool,
              fstart:     float=0,
              fstop:      float=1,
              a:          np.ndarray=np.eye(3),
              b:          np.ndarray=np.zeros(3),
              terms:      int=tl.ALL_TERMS) -> np.ndarray:
    '''
    Determine Tolles-Lawson calibration coefficients used
    to calibrate the scalar magnetometer
    
    Parameters
    ----------
    df
        a
    use_filter
        Use band pass filter if true
    fstart
        Start frequency for the band (Hz)
    fstop
        Stop frequency for the band (Hz)
    a
        3x3 distortion matrix
    b
        1x3 bias vector
    terms
        Terms to include in A-matrix. Options include:
        
        - ALL_TERMS
        - PERMANENT
        - INDUCED
        - EDDY
    
    Returns
    -------
    np.ndarray
        1xK T-L calibration coefficients where K is the
        number of A matrix terms to use for calibration
    '''
    
    b_scalar = df.F
    b_vector = np.hstack([np.array(df.X)[:, np.newaxis], 
                          np.array(df.Y)[:, np.newaxis],
                          np.array(df.Z)[:, np.newaxis]])
    b_vector = sc.apply_cal_to_vec(vec = b_vector,
                                   a   = a,
                                   b   = b)
    
    delta_t = np.diff(df.epoch_sec).mean()
    
    return tl.tlc(b_scalar  = b_scalar,
                  b_vector  = b_vector,
                  delta_t    = delta_t,
                  use_filter = use_filter,
                  fstart     = fstart,
                  fstop      = fstop,
                  terms      = terms)

def calibrate(survey_df: pd.DataFrame,
              a:         np.ndarray=np.eye(3),
              b:         np.ndarray=np.zeros(3),
              c:         np.ndarray=np.zeros(18),
              terms:     int=tl.ALL_TERMS,
              debug:     bool=True):
    '''
    Applies given calibration terms to the scalar sensor data
    
    Parameters
    ----------
    survey_df
        Dataframe containing flight data from the survey
        Minimum required columns include:
        
        - datetime
        - epoch_sec
        - LAT
        - LONG
        - ALT
        - F
        - X
        - Y
        - Z
        - LINE
        - LINE_TYPE
        
    a
        3x3 distortion matrix
    b
        1x3 bias vector
    c
        1xK T-L calibration coefficients where K is the
        number of A matrix terms to use for calibration
    terms
        Terms to include in A-matrix. Options include:
        
        - ALL_TERMS
        - PERMANENT
        - INDUCED
        - EDDY

    debug
        Whether or not debug prints/plot should be generated
    
    Returns
    -------
    calibrated_df
        Copy of the survey DataFrame, but with calibrated total
        field values in the 'F' column
    '''
    
    timestamps = np.array(survey_df.epoch_sec)
    b_scalar   = np.array(survey_df.F)
    b_vector   = np.hstack([np.array(survey_df.X)[:, np.newaxis], 
                            np.array(survey_df.Y)[:, np.newaxis],
                            np.array(survey_df.Z)[:, np.newaxis]])
    
    if debug:
        print('Applying spin test calibration to vector magnetometer measurements')
    
    b_vector = sc.apply_cal_to_vec(vec = b_vector,
                                   a   = a,
                                   b   = b)
    delta_t  = np.diff(timestamps).mean()
    
    if debug:
        print('Applying Tolles-Lawson calibration to scalar magnetometer measurements')
    
    b_body = tl.apply_tlc(c        = c,
                          b_scalar = b_scalar,
                          b_vector = b_vector,
                          delta_t  = delta_t,
                          terms    = terms)
    
    b_scalar -= b_body
    
    calibrated_df      = deepcopy(survey_df)
    calibrated_df['F'] = b_scalar
    
    return calibrated_df

def find_anomaly(survey_df:       pd.DataFrame,
                 ref_df:          pd.DataFrame,
                 ref_scale:       float=1,
                 ref_offset:      float=0,
                 enable_lon_norm: bool=False,
                 filt_cutoff:     float=None,
                 debug:           bool=True):
    '''
    Finds magnetic anomaly values by subtracting out the
    core field and temporal effects from the *calibrated*
    scalar magnetometer values
    
    Parameters
    ----------
    survey_df
        Dataframe containing flight data from the survey
        Minimum required columns include:
        
        - epoch_sec
        - LAT
        - LONG
        - ALT
        - F
        
    ref_df
        DataFrame containing reference station data coinciding
        with the flight
    ref_scale
        Amount to scale the reference station data by
    ref_offset
        Amount to bias the reference station data by (nT)
    enable_lon_norm
        Whether or not to conduct frequency based longitudinal
        normalization to provided reference station data
    filt_cutoff
        Low pass filter cutoff frequency for scalar data. Set to None
        to prevent filtering
    debug
        Whether or not debug prints/plot should be generated
    
    Returns
    -------
    anomaly_df
        Copy of the survey DataFrame, but with anomaly values in
        the 'F' column
    '''
    
    timestamps = np.array(survey_df.epoch_sec)
    lats       = np.array(survey_df.LAT)
    lons       = np.array(survey_df.LONG)
    heights    = np.array(survey_df.ALT)
    b_scalar   = np.array(survey_df.F)
    
    if debug:
        print('Removing core field from measurements')
        
    try:
        b_scalar -= survey_df.IGRF_F
        
    except:
        Be, Bn, Bu = igrf(lons,
                          lats,
                          heights / 1000,
                          survey_df.datetime[0])

        IGRF = np.hstack((Bn.squeeze()[:, np.newaxis],
                          Be.squeeze()[:, np.newaxis],
                         -Bu.squeeze()[:, np.newaxis]))
        IGRF_F = la.norm(IGRF, axis=1)
        
        b_scalar -= IGRF_F
    
    if ref_df is not None:
        if debug:
            print('Removing diurnal/space weather effects with reference station data')
        
        if enable_lon_norm:
            survey_lon = lons.mean()
        else:
            survey_lon = None
        
        _, ref_mag = Diurnal.interp_reference_df(df            = ref_df,
                                                 timestamps    = timestamps,
                                                 survey_lon    = survey_lon,
                                                 ref_scale     = ref_scale,
                                                 ref_offset    = ref_offset,
                                                 subtract_core = True)
        b_scalar -= ref_mag
    
    if filt_cutoff is not None:
        if debug:
            print('Low pass filtering scalar anomaly measurements w/ {}Hz cutoff'.format(filt_cutoff))
        
        filt_fs  = 1.0 / np.diff(survey_df.epoch_sec).mean() # Find avg sample frequency
        b_scalar = filt.lpf(data   = b_scalar,
                            cutoff = filt_cutoff,
                            fs     = filt_fs)
    
    anomaly_df      = deepcopy(survey_df)
    anomaly_df['F'] = b_scalar
    
    return anomaly_df

def meshgrid_from_lines(survey_df: pd.DataFrame,
                        dx:        float,
                        dy:        float,
                        buffer:    float,
                        line_type: int=1) -> list:
    '''
    Creates a latitude/longitude meshgrid of coordinates based
    on the max/min coordinates of the given line data (either
    flight or tie lines)
    
    Parameters
    ----------
    survey_df
        Dataframe containing flight data from the survey
        Minimum required columns include:
        
        - LAT
        - LONG
        - LINE_TYPE
        
    dx
        Pixel distance in the E/W direction (m)
    dy
        Pixel distance in the N/S direction (m)
    buffer
        The amount of extra distance to include around the edges of
        the min/max line coordinates (m)
    line_type
        Set to 1 to make a meshgrid of flight line data and set
        to 2 for a meshgrid of tie line data
    
    Returns
    -------
    list of np.ndarrays
        Two NxM meshgridded latitude and longitudearrays -> [latitude (dd) grid, longitud (dd) grid]
    '''
    
    mask = (survey_df.LINE_TYPE == line_type)

    lats = np.array(survey_df.LAT[mask],  dtype=np.float64)
    lons = np.array(survey_df.LONG[mask], dtype=np.float64)

    min_lat    = lats.min()
    min_lon    = lons.min()
    max_lat    = lats.max()
    max_lon    = lons.max()
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Create a meshgrid of pixel coordinates to interpolate data at plus add min(h AGL)/2 buffer on edges
    _, min_lon = cu.coord_coord(lat     = center_lat,
                                lon     = min_lon,
                                dist    = buffer / 1000,
                                bearing = 270)
    _, max_lon = cu.coord_coord(lat     = center_lat,
                                lon     = max_lon,
                                dist    = buffer / 1000,
                                bearing = 90)
    min_lat, _ = cu.coord_coord(lat     = min_lat,
                                lon     = center_lon,
                                dist    = buffer / 1000,
                                bearing = 180)
    max_lat, _ = cu.coord_coord(lat     = max_lat,
                                lon     = center_lon,
                                dist    = buffer / 1000,
                                bearing = 0)
    
    dist_x  = cu.coord_dist(lat_1 = center_lat,
                            lon_1 = min_lon,
                            lat_2 = center_lat,
                            lon_2 = max_lon) * 1000
    dist_y  = cu.coord_dist(lat_1 = min_lat,
                            lon_1 = center_lon,
                            lat_2 = max_lat,
                            lon_2 = center_lon) * 1000
    num_lon = int(dist_x / dx)
    num_lat = int(dist_y / dy)

    interp_lons = np.linspace(min_lon, max_lon, num_lon)
    interp_lats = np.linspace(min_lat, max_lat, num_lat)
    
    return np.meshgrid(interp_lons, interp_lats)

def normalize_alts(anomaly_df: pd.DataFrame,
                   dx:         float,
                   dy:         float,
                   buffer:     float=0) -> pd.DataFrame:
    '''
    Normalizes the flight and tie line samples (if given)
    to a constant altitude. The altitude of the highest
    sample within the flight/tie lines is the new constant
    altitude for all samples
    
    Parameters
    ----------
    anomaly_df
        Dataframe containing flight data from the survey
        Minimum required columns include:
        
        - LAT
        - LONG
        - ALT
        - F (anomaly values)
        - LINE_TYPE
        
    dx
        Pixel distance in the E/W direction (m)
    dy
        Pixel distance in the N/S direction (m)
    buffer
        The amount of extra distance to include around the edges of
        the min/max line coordinates (m)
    line_type
        Set to 1 to make a meshgrid of flight line data and set
        to 2 for a meshgrid of tie line data
    
    Returns
    -------
    norm_alt_anom_df
        Copy of the anomaly DataFrame, but with altitude-normalized
        anomaly values in the 'F' column, plus a new constant
        altitude in the 'ALT' column
    '''
    
    # Create a DataFrame copy to be returned at the end
    norm_alt_anom_df = deepcopy(anomaly_df)
    
    ########################################################################################################
    # Mask out flight line data
    ########################################################################################################
    fl_mask = (anomaly_df.LINE_TYPE == 1)

    fl_scalar  = np.array(anomaly_df.F[fl_mask],    dtype=np.float64)
    fl_lats    = np.array(anomaly_df.LAT[fl_mask],  dtype=np.float64)
    fl_lons    = np.array(anomaly_df.LONG[fl_mask], dtype=np.float64)
    fl_heights = np.array(anomaly_df.ALT[fl_mask],  dtype=np.float64)

    fl_fit_points = np.hstack([fl_lons[:, np.newaxis],
                               fl_lats[:, np.newaxis]])

    max_fl_height    = fl_heights.max()
    cartesian_height = max_fl_height
    
    grid_fl_lon, grid_fl_lat = meshgrid_from_lines(survey_df  = anomaly_df,
                                                   dx         = dx,
                                                   dy         = dy,
                                                   buffer     = buffer,
                                                   line_type  = 1)
    
    # Interpolate pixel altitudes
    interp_fl_heights = interpolate.griddata(fl_fit_points,
                                             fl_heights,
                                             (grid_fl_lon, grid_fl_lat),
                                             method='linear')
    
    # Clamp pixel heights
    interp_fl_heights[interp_fl_heights < fl_heights.min()] = fl_heights.min()
    interp_fl_heights[interp_fl_heights > fl_heights.max()] = fl_heights.max()
    
    ########################################################################################################
    # Mask out tie line data (if available)
    ########################################################################################################
    tl_mask = (anomaly_df.LINE_TYPE == 2)
    
    if len(anomaly_df[tl_mask]) > 0:
        tl_scalar  = np.array(anomaly_df.F[tl_mask],    dtype=np.float64)
        tl_lats    = np.array(anomaly_df.LAT[tl_mask],  dtype=np.float64)
        tl_lons    = np.array(anomaly_df.LONG[tl_mask], dtype=np.float64)
        tl_heights = np.array(anomaly_df.ALT[tl_mask],  dtype=np.float64)

        tl_fit_points = np.hstack([tl_lons[:, np.newaxis],
                                   tl_lats[:, np.newaxis]])

        max_tl_height = tl_heights.max()
        
        if max_tl_height > cartesian_height:
            cartesian_height = max_tl_height
        
        grid_tl_lon, grid_tl_lat = meshgrid_from_lines(survey_df  = anomaly_df,
                                                       dx         = dx,
                                                       dy         = dy,
                                                       buffer     = buffer,
                                                       line_type  = 1)

        # Interpolate pixel altitudes
        interp_tl_heights = interpolate.griddata(tl_fit_points,
                                                 tl_heights,
                                                 (grid_tl_lon, grid_tl_lat),
                                                 method='linear')
    
        # Clamp pixel heights
        interp_tl_heights[interp_tl_heights < tl_heights.min()] = tl_heights.min()
        interp_tl_heights[interp_tl_heights > tl_heights.max()] = tl_heights.max()
    
    ########################################################################################################
    # Altitude-normalize flight line samples
    ########################################################################################################
    # Interpolate pixel values
    interp_fl_scalar = interpolate.griddata(fl_fit_points,
                                            fl_scalar,
                                            (grid_fl_lon, grid_fl_lat),
                                            method='cubic')

    # Altitude-normalize interpolated flight line data
    cartesian_fl_scalar = mu.drape2lvl(drape_map        = interp_fl_scalar,
                                       drape_heights    = interp_fl_heights,
                                       delta_x          = dx,
                                       delta_y          = dy,
                                       cartesian_height = cartesian_height)

    # Remove NaNs in input data so that interpolate.griddata doesn't return all NaNs
    flat_cartesian_fl_scalar        = cartesian_fl_scalar.flatten()
    no_nan_fl_mask                  = ~np.isnan(flat_cartesian_fl_scalar)
    no_nan_flat_cartesian_fl_scalar = flat_cartesian_fl_scalar[no_nan_fl_mask]
    
    flat_cartesian_fl_points        = np.hstack([grid_fl_lon.flatten()[:, np.newaxis],
                                                 grid_fl_lat.flatten()[:, np.newaxis]])
    no_nan_flat_cartesian_fl_points = flat_cartesian_fl_points[no_nan_fl_mask, :]
    
    
    # Resample altitude-normalized flight line data where the flight line samples are located
    norm_alt_fl_scalar = interpolate.griddata(no_nan_flat_cartesian_fl_points,
                                              no_nan_flat_cartesian_fl_scalar,
                                              fl_fit_points,
                                              method='nearest')

    # Save the altitude-normalized flight line samples
    norm_alt_anom_df['F'].loc[fl_mask] = norm_alt_fl_scalar
    
    ########################################################################################################
    # Altitude-normalize tie line samples (if available)
    ########################################################################################################
    if len(anomaly_df[tl_mask]) > 0:
        # Interpolate pixel altitudes
        interp_tl_heights = interpolate.griddata(tl_fit_points,
                                                 tl_heights,
                                                 (grid_tl_lon, grid_tl_lat),
                                                 method='linear')
    
        # Interpolate pixel values
        interp_tl_scalar = interpolate.griddata(tl_fit_points,
                                                tl_scalar,
                                                (grid_tl_lon, grid_tl_lat),
                                                method='cubic')
        
        # Altitude-normalize interpolated tie line data
        cartesian_tl_scalar = mu.drape2lvl(drape_map        = interp_tl_scalar,
                                           drape_heights    = interp_tl_heights,
                                           delta_x          = dx,
                                           delta_y          = dy,
                                           cartesian_height = cartesian_height)

        # Remove NaNs in input data so that interpolate.griddata doesn't return all NaNs
        flat_cartesian_tl_scalar        = cartesian_tl_scalar.flatten()
        no_nan_tl_mask                  = ~np.isnan(flat_cartesian_tl_scalar)
        no_nan_flat_cartesian_tl_scalar = flat_cartesian_tl_scalar[no_nan_tl_mask]
        
        flat_cartesian_tl_points        = np.hstack([grid_tl_lon.flatten()[:, np.newaxis],
                                                     grid_tl_lat.flatten()[:, np.newaxis]])
        no_nan_flat_cartesian_tl_points = flat_cartesian_tl_points[no_nan_tl_mask, :]
        
        # Resample altitude-normalized tie line data where the tie line samples are located
        norm_alt_tl_scalar = interpolate.griddata(no_nan_flat_cartesian_tl_points,
                                                  no_nan_flat_cartesian_tl_scalar,
                                                  tl_fit_points,
                                                  method='nearest')
        
        # Save the altitude-normalized tie line samples
        norm_alt_anom_df['F'].loc[tl_mask] = norm_alt_tl_scalar
    
    # Save the normalized altitude values for all samples
    norm_alt_anom_df['ALT'] = cartesian_height
    
    return norm_alt_anom_df

def lvl_flight_lines(anomaly_df:     pd.DataFrame,
                     lvl_type:       str=None,
                     num_ptls:       int=5,
                     ptl_locs:       np.ndarray=None,
                     percent_thresh: float=0.85) -> pd.DataFrame:
    '''
    Applies the given flight line leveling technique and
    returns a DataFrame of the leveled data
    
    Parameters
    ----------
    anomaly_df
        Dataframe containing flight data from the survey
        Minimum required columns include:
        
        - LAT
        - LONG
        - F (anomaly values)
        - LINE
        - LINE_TYPE
    
    lvl_type
        Specify map leveling approach, set to None to disable. Options
        include:
        
        - 'pca'
        - 'tie'
    
    num_ptls
        Only used if lvl_type is set to 'pca'.
        Number of pseudo tie lines to use for leveling
    ptl_locs
        Only used if lvl_type is set to 'pca'.
        Kx1 array of relative locations of the pseudo tie lines.
        Each relative location is a percent distance from the
        edge of the survey area where the first sample of the
        first flight line was taken. For example, in order to
        set two pseudo tie lines at opposite ends of the dataset,
        set ptl_locs = [0.0, 1.0]. Values must be between 0 and 1
    percent_thresh
        Only used if lvl_type is set to 'pca'.
        Value ranging from 0 to 1 (not inclusive) that
        specifies the minimum cumulative contribution
        rate of the components to use for the PCA
        reconstruction
    
    Returns
    -------
    pd.DataFrame
        Copy of anomaly DataFrame, but with leveled flight line
        anomaly values in the 'F' column
    '''
    
    assert lvl_type.lower() in ['pca', 'tie'], 'Uknown map leveling type specified'
    
    if lvl_type.lower() == 'pca':
        return pcaLvl.pca_lvl(survey_df      = anomaly_df,
                              num_ptls       = num_ptls,
                              ptl_locs       = ptl_locs,
                              percent_thresh = percent_thresh)
    
    elif lvl_type.lower() == 'tie':
        return tieLvl.tie_lvl(survey_df = anomaly_df,
                              approach  = 'lobf') # Hard code Line Of Best Fit (lobf) per individual flight line approach

def interp_flight_lines(anomaly_df:      pd.DataFrame,
                        dx:              float,
                        dy:              float,
                        max_terrain_msl: float,
                        buffer:          float=0,
                        sensor_sigma:    float=0,
                        interp_type:     str='bicubic',
                        neighbors:       int=None,
                        skip_na_mask:    bool=False,
                        debug:           bool=True) -> dict:
    '''
    Interpolate flight line data into a grid array for easy
    map creation
    
    Parameters
    ----------
    anomaly_df
        Dataframe containing flight data from the survey
        Minimum required columns include:
        
        - LAT  (dd)
        - LONG (dd)
        - F    (anomaly values in nT)
        - ALT  (m above MSL)
        - LINE_TYPE
    
    dx
        Pixel distance in the E/W direction (m)
    dy
        Pixel distance in the N/S direction (m)
    max_terrain_msl
        Maximum altitude above MSL of the survey terrain
        across all survey points. This parameter is
        mainly used to determine the distance at which
        map pixels are too far away from all survey
        samples and therefore need to be converted
        to NaN values
    buffer
        The amount of extra distance to include around the edges of
        the min/max line coordinates (m)
    sensor_sigma
        Scalar magnetometer noise variance (nT)
    interp_type
        Specify interpolator to use for final map value interpolation.
        Options include:
        
        - 'bicubic': Bicubic spline interpolator
        - 'rbf':     Radial basis function interpolator
        - 'gpr':     Gaussian process regression
    
    neighbors
        Only used for RBF interpolation: If specified, the value of the
        interpolant at each evaluation point will be computed using only
        this many nearest data points. All the data points are used by default
    skip_na_mask
        Prevent the NaN masking of pixels that are considered too far
        from the survey data (saves computation time for large datasets)
    debug
        Whether or not debug prints/plot should be generated
    
    Returns
    -------
    interp_dict
        Dictionary of meshgridded/interpolated coordinate, scalar anomaly,
        altitude, and standard deviation values. Columns include:
        
        - LAT
        - LONG
        - F
        - ALT
        - STD
    '''
    
    fl_mask = (anomaly_df.LINE_TYPE == 1)

    fl_scalar  = np.array(anomaly_df.F[fl_mask],    dtype=np.float64)
    fl_lats    = np.array(anomaly_df.LAT[fl_mask],  dtype=np.float64)
    fl_lons    = np.array(anomaly_df.LONG[fl_mask], dtype=np.float64)
    fl_heights = np.array(anomaly_df.ALT[fl_mask],  dtype=np.float64)

    fl_fit_points = np.hstack([fl_lons[:, np.newaxis],
                               fl_lats[:, np.newaxis]])
    
    grid_lon, grid_lat = meshgrid_from_lines(survey_df  = anomaly_df,
                                             dx         = dx,
                                             dy         = dy,
                                             buffer     = buffer,
                                             line_type  = 1)
    grid_coords = np.hstack([grid_lon.flatten()[:, np.newaxis],
                             grid_lat.flatten()[:, np.newaxis]])
    
    interp_shape = grid_lon.shape
    
    dist_thresh_m = fl_heights.min() - max_terrain_msl
    
    if debug:
        print('Interpolating survey anomaly data to map coordinates')
    
    if interp_type.lower() == 'bicubic':
        if debug:
            print('Running bicubic spline interpolation for all map pixels')
        
        interp_scalar = interpolate.griddata(fl_fit_points,
                                             fl_scalar,
                                             (grid_lon, grid_lat),
                                             method='cubic')
        interp_std = np.ones(interp_shape) * np.nan
    
    elif interp_type.lower() == 'rbf':
        if debug:
            print('Running radial basis function (RBF) interpolation for all map pixels')
        
        rbfi_scalar = interpolate.RBFInterpolator(fl_fit_points,
                                                  fl_scalar,
                                                  neighbors=neighbors,
                                                  kernel='linear',
                                                  smoothing=0)
        interp_scalar = rbfi_scalar(grid_coords).reshape(interp_shape)
        interp_std    = np.ones(interp_shape) * np.nan
    
    elif interp_type.lower() == 'gpr':
        if debug:
            print('Running Gaussian Process Regression (GPR) interpolation for all map pixels')
        
        # Find average distance between samples (m)
        unique_fl_nums = np.unique(anomaly_df.LINE[anomaly_df.LINE_TYPE == 1])
        avg_samp_dists = np.zeros(len(unique_fl_nums))

        for i, line in enumerate(unique_fl_nums):
            mask = np.where(anomaly_df.LINE == line)[0]
            
            lats = np.array(anomaly_df.LAT[mask])
            lons = np.array(anomaly_df.LONG[mask])
            
            line_len = cu.coord_dist(lat_1 = lats[0],
                                     lon_1 = lons[0],
                                     lat_2 = lats[-1],
                                     lon_2 = lons[-1]) * 1000
            avg_samp_dists[i] = line_len / len(lats)

        avg_samp_dist = avg_samp_dists.mean()
        
        # GPR interpolate
        kernel = WNK(sensor_sigma) + RBF([dist_thresh_m, avg_samp_dist])
        gpr    = GPR(kernel=kernel)
        
        gpr.fit(fl_fit_points, fl_scalar)
        
        interp_points = np.hstack([grid_lon.flatten()[:, np.newaxis],
                                   grid_lat.flatten()[:, np.newaxis]])
        
        interp_scalar, interp_std = gpr.predict(interp_points, return_std=True)
        
        interp_scalar = interp_scalar.reshape(interp_shape)
        interp_std    = interp_std.reshape(interp_shape)
    
    if not skip_na_mask:
        if debug:
            print('NaN\'ing-out pixes that are at least min(h AGL)/2 dist away from all sample locations')
        
        angle_thresh = cu.arc_angle(dist_thresh_m / 1000)
        
        fl_grid_lon    = fl_lons.flatten()[:, np.newaxis]
        fl_grid_lat    = fl_lats.flatten()[:, np.newaxis]
        fl_grid_coords = np.hstack([fl_grid_lon, fl_grid_lat])
        
        flat_grid_lon    = grid_lon.flatten()[:, np.newaxis]
        flat_grid_lat    = grid_lat.flatten()[:, np.newaxis]
        flat_grid_coords = np.hstack([flat_grid_lon, flat_grid_lat])
        
        for grid_coord in tqdm(flat_grid_coords):
            min_angle = distance.cdist(fl_grid_coords,
                                       [grid_coord],
                                       'euclidean').min()
            
            if min_angle > angle_thresh:
                nan_mask = np.logical_and(grid_lon == grid_coord[0],
                                          grid_lat == grid_coord[1])
                
                interp_scalar[nan_mask] = np.nan
                
                if interp_std is not None:
                    interp_std[nan_mask] = np.nan
    
    interp_heights = np.ones(interp_shape) * fl_heights.mean() # Interpolate pixel heights?
    
    interp_dict = {'LAT':  grid_lat,
                   'LONG': grid_lon,
                   'F':    interp_scalar,
                   'ALT':  interp_heights,
                   'STD':  interp_std}
    
    return interp_dict

def gen_map(out_dir:         str,
            map_name:        str,
            survey_df:       pd.DataFrame,
            ref_df:          pd.DataFrame,
            dx:              float,
            dy:              float,
            max_terrain_msl: float=0,
            ref_scale:       float=1,
            ref_offset:      float=0,
            enable_lon_norm: bool=False,
            a:               np.ndarray=np.eye(3),
            b:               np.ndarray=np.zeros(3),
            c:               np.ndarray=np.zeros(18),
            terms:           int=tl.ALL_TERMS,
            enable_alt_norm: bool=False,
            lvl_type:        str=None,
            num_ptls:        int=5,
            ptl_locs:        np.ndarray=None,
            percent_thresh:  float=0.85,
            sensor_sigma:    float=0.0,
            interp_type:     str='bicubic',
            filt_cutoff:     float=None,
            debug:           bool=True) -> rxr.rioxarray.raster_dataset.xarray.DataArray:
    '''
    Generate a map based on the provided sensor calibration
    parameters, reference station data (if given), and flight survey
    data
    
    Parameters
    ----------
    out_dir
        Path to directory where the map (as a GeoTIFF) will
        be exported to
    map_name
        Description of survey area or other identifier
    survey_df
        Dataframe containing flight data from the survey
        Minimum required columns include:
        
        - datetime
        - epoch_sec
        - LAT
        - LONG
        - ALT
        - F (total field)
        - X
        - Y
        - Z
        - LINE
        - LINE_TYPE
        
    ref_df
        DataFrame containing reference station data coinciding
        with the flight
    dx
        Pixel distance in the E/W direction (m)
    dy
        Pixel distance in the N/S direction (m)
    max_terrain_msl
        Maximum altitude above MSL of the survey terrain
        across all survey points. This parameter is
        mainly used to determine the distance at which
        map pixels are too far away from all survey
        samples and therefore need to be converted
        to NaN values
    ref_scale
        Amount to scale the reference station data by
    ref_offset
        Amount to bias the reference station data by (nT)
    enable_lon_norm
        Whether or not to conduct frequency based longitudinal
        normalization to provided reference station data
    a
        3x3 distortion matrix
    b
        1x3 bias vector
    c
        1xK T-L calibration coefficients where K is the
        number of A matrix terms to use for calibration
    terms
        Terms to include in A-matrix. Options include:
        
        - ALL_TERMS
        - PERMANENT
        - INDUCED
        - EDDY

    enable_alt_norm
        Whether or not to normalize survey data to a constant
        altitude. Useful for creating a level map from a drape
        survey or a survey with high altitude variance
    lvl_type
        Specify map leveling approach, set to None to disable. Options
        include:
        
        - 'pca'
        - 'tie' (Currently unsupported)
    
    num_ptls
        Only used if lvl_type is set to 'pca'.
        Number of pseudo tie lines to use for leveling
    ptl_locs
        Only used if lvl_type is set to 'pca'.
        Kx1 array of relative locations of the pseudo tie lines.
        Each relative location is a percent distance from the
        edge of the survey area where the first sample of the
        first flight line was taken. For example, in order to
        set two pseudo tie lines at opposite ends of the dataset,
        set ptl_locs = [0.0, 1.0]. Values must be between 0 and 1
    percent_thresh
        Only used if lvl_type is set to 'pca'.
        Value ranging from 0 to 1 (not inclusive) that
        specifies the minimum cumulative contribution
        rate of the components to use for the PCA
        reconstruction
    sensor_sigma
        Scalar magnetometer noise variance (nT)
    interp_type
        Specify interpolator to use for final map value interpolation.
        Options include:
        
        - 'bicubic': Bicubic spline interpolator
        - 'rbf':     Radial basis function interpolator
        - 'gpr':     Gaussian process regression
        
    filt_cutoff
        Low pass filter cutoff frequency for scalar data. Set to None
        to prevent filtering
    debug
        Whether or not debug prints/plot should be generated
    
    Returns
    -------
    rxr.rioxarray.raster_dataset.xarray.DataArray
        Generated map as a rioxarray
    '''
    
    interp_type = interp_type.lower()
    
    if debug:
        print('Generating map from survey data')
    
    # Calibrate only if the calibration terms are not all 0
    if ((~(a == 0).all()) or (~(b == 0).all()) or (~(c == 0).all())):
        if debug:
            print('Calibrating survey data')
        
        calibrated_df = calibrate(survey_df = survey_df,
                                  a         = a,
                                  b         = b,
                                  c         = c,
                                  terms     = terms,
                                  debug     = debug)
    else:
        calibrated_df = survey_df
    
    
    if debug:
        print('Finding survey anomaly scalar values')
    
    anomaly_df = find_anomaly(survey_df       = calibrated_df,
                              ref_df          = ref_df,
                              ref_scale       = ref_scale,
                              ref_offset      = ref_offset,
                              enable_lon_norm = enable_lon_norm,
                              filt_cutoff     = filt_cutoff,
                              debug           = debug)
    
    if enable_alt_norm:
        if debug:
            print('Altitude-normalizing survey anomaly scalar values')
        
        alt_norm_df = normalize_alts(anomaly_df = anomaly_df,
                                     dx         = dx,
                                     dy         = dy)
    else:
        alt_norm_df = anomaly_df
    
    if lvl_type is not None:
        if debug:
            print('Leveling flight line anomaly scalar values using {} method'.format(lvl_type))
        
        lvl_df = lvl_flight_lines(anomaly_df     = alt_norm_df,
                                  lvl_type       = lvl_type,
                                  num_ptls       = num_ptls,
                                  ptl_locs       = ptl_locs,
                                  percent_thresh = percent_thresh)
    else:
        lvl_df = alt_norm_df
    
    if debug:
        print('Interpolating anomaly scalar values')
    
    interp_df = interp_flight_lines(anomaly_df      = lvl_df,
                                    dx              = dx,
                                    dy              = dy,
                                    max_terrain_msl = max_terrain_msl,
                                    buffer          = (lvl_df.ALT.min() - max_terrain_msl) / 2.0,
                                    sensor_sigma    = sensor_sigma,
                                    interp_type     = interp_type,
                                    debug           = debug)
    
    interp_lats    = interp_df['LAT']
    interp_lons    = interp_df['LONG']
    interp_scalar  = interp_df['F']
    interp_heights = interp_df['ALT']
    interp_std     = interp_df['STD']
    
    if debug:
        print('Applying a final low pass filter to the interpolated anomaly scalar values')
    
    filt_interp_scalar = filt.lpf2(data   = interp_scalar,
                                   cutoff = max_terrain_msl - interp_heights.min(),
                                   dx     = dx,
                                   dy     = dy)
    
    if debug:
        print('Exporting map as a GeoTIFF')
    
    return mu.export_map(out_dir  = out_dir,
                         location = map_name,
                         date     = survey_df.datetime[0],
                         lats     = interp_lats,
                         lons     = interp_lons,
                         scalar   = filt_interp_scalar,
                         heights  = interp_heights,
                         stds     = interp_std,
                         vector   = None)