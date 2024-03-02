import datetime as dt
import sys
from os.path import dirname, join, realpath, exists
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rxr
import scipy.linalg as la
import scipy.stats as stat
import xarray as xr
from matplotlib import cm
from tqdm import tqdm
from ppigrf import igrf

sys.path.append(dirname(realpath(__file__)))
sys.path.append(dirname(dirname(realpath(__file__))))

import Diurnal
from SensorCal import spinCal as sc
from Utils import coordinateUtils as cu
from Utils import mapUtils as mu
from VehicleCal import TL as tl


# Enumerate line directions
HORIZ = 0 # Horizontal
VERT  = 1 # Vertical

# Enumerate noise types
ZERO = 0
BIAS = 1 << 0
POLY = 1 << 1
AWGN = 1 << 2
FILE = 1 << 3

M2FT = 3.280839895 # m to ft conversion
FT2M = 1 / M2FT    # ft to m conversion
KM2M = 1000        # km to m conversion
M2KM = 1 / KM2M    # m to km conversion


def save_dataset(type:    str,
                 out_dir: str,
                 date:    Union[dt.date, dt.datetime],
                 data:    dict,
                 debug:   bool=True) -> pd.DataFrame:
    '''
    Save dictionary of data into a .csv file and return
    said data as a Pandas DataFrame
    
    Parameters
    ----------
    type
        Custom field that will be prepended to the .csv's file name
    out_dir
        Path to the directory to save the .csv file to
    date
        Date/datetime object that corresponds to the dataset
    data
        Dictionary of data to save
    debug
        Whether or not debug prints/plot should be generated
    
    Returns
    -------
    df
        Pandas DataFrame of the given data
    '''
    
    i = 0
    fname = '{type}_{year}_{month}_{day}_{num}.csv'.format(type  = type,
                                                           year  = date.year,
                                                           month = date.month,
                                                           day   = date.day,
                                                           num   = i)
    full_path = join(out_dir, fname)
    
    while(exists(full_path)):
        i += 1
        fname = '{type}_{year}_{month}_{day}_{num}.csv'.format(type  = type,
                                                               year  = date.year,
                                                               month = date.month,
                                                               day   = date.day,
                                                               num   = i)
        full_path = join(out_dir, fname) 
    
    df = pd.DataFrame(data)
    df.to_csv(full_path)
    
    if debug:
        print('Saved data to', full_path)
    
    return df

def gen_parallel_search(lon_min:          float,
                        lon_max:          float,
                        lat_min:          float,
                        lat_max:          float,
                        lon_total_dist_m: float,
                        lat_total_dist_m: float,
                        lon_sub_dist_m:   float,
                        lat_sub_dist_m:   float,
                        line_dir:         int,
                        last_line_num:    int=0) -> list:
    '''
    Generate an array of lat/lon coordinates of a parallel
    search pattern lines within a given box boundary
    
    Parameters
    ----------
    lon_min
        Minimum longitude of the generated lines
    lon_max
        Maximum longitude of the generated lines
    lat_min
        Minimum latitude of the generated lines
    lat_max
        Maximum latitude of the generated lines
    lon_total_dist_m
        Total longitudinal distance of the spread
        of the generated lines (m)
    lat_total_dist_m
        Total latitudinal distance of the spread
        of the generated lines (m)
    lon_sub_dist_m
        Distance between samples in the
        longitudinal direction of the lines
    lat_sub_dist_m
        Distance between samples in the
        latitudinal direction of the lines
    line_dir
        Direction the generated lines run. Options include:
        
        - HORIZ: Horizontal - East/West
        - VERT:  Vertical   - North/South
    
    last_line_num
        Number of the last line "surveyed"
    
    Returns
    -------
    list
        2xN array of lat/lon coordinates of a parallel
        search pattern within a given box boundary and
        1xN array of line numbers (flight or tie) --> [[ coord 0 lat (dd), coord 1 lat (dd), ... ]                                                 ]
                                                      [[ coord 0 lon (dd), coord 1 lon (dd), ... ], [coord 0 line number, coord 1 line number, ...]]
    '''
    
    x_num = int(lon_total_dist_m / lon_sub_dist_m) + 1
    y_num = int(lat_total_dist_m / lat_sub_dist_m) + 1
    
    x_coords = np.linspace(lon_min, lon_max, x_num)
    y_coords = np.linspace(lat_min, lat_max, y_num)
    
    if line_dir == HORIZ:
        x_coord_mesh = np.zeros((y_num, x_num))
        x_coord_mesh[::2, :]  = x_coords
        x_coord_mesh[1::2, :] = x_coords[::-1]
        x_coord_mesh = x_coord_mesh.flatten()

        y_coord_mesh = np.tile(y_coords, (x_num, 1)).T.flatten()
        
        line_nums = np.sort(np.tile(np.arange(y_num), x_num)) + last_line_num + 1
    
    else:
        y_coord_mesh = np.zeros((x_num, y_num))
        y_coord_mesh[::2, :]  = y_coords
        y_coord_mesh[1::2, :] = y_coords[::-1]
        y_coord_mesh = y_coord_mesh.flatten()

        x_coord_mesh = np.tile(x_coords, (y_num, 1)).T.flatten()
        
        line_nums = np.sort(np.tile(np.arange(x_num), y_num)) + last_line_num + 1
    
    return [np.vstack([y_coord_mesh,
                       x_coord_mesh]), line_nums]

def gen_timestamps(survey_coords:   np.ndarray,
                   survey_start_dt: dt.datetime,
                   survey_vel_mps:  float) -> np.ndarray:
    '''
    Generate Unix epoch timestamps for each survey sample based on
    the survey's start time, path coordinates, and vehicle velocity
    
    Parameters
    ----------
    survey_coords
        2xN array of survey coordinates in lat/lon (dd) --> [ coord 0 lat (dd), coord 1 lat (dd), ... ]
                                                            [ coord 0 lon (dd), coord 1 lon (dd), ... ]
    survey_start_dt
        Start date/time of survey (UTC)
    survey_vel_mps
        Velocity of survey vehicle (m/s)
    
    Returns
    -------
    timestamps
        Unix epoch timestamps of survey data points
    '''
    
    timestamps = np.zeros(survey_coords.shape[1])
    
    timestamps[0] = survey_start_dt.timestamp() # Unix epoch timestamp
    
    for i in tqdm(range(1, survey_coords.shape[1])):
        prev_lat = survey_coords[0, i-1]
        prev_lon = survey_coords[1, i-1]
        cur_lat  = survey_coords[0, i]
        cur_lon  = survey_coords[1, i]
        
        timestamps[i] = timestamps[i-1] + cu.coord_dist(prev_lat,
                                                        prev_lon,
                                                        cur_lat,
                                                        cur_lon,) * KM2M / survey_vel_mps
    
    return timestamps

def gen_spin_data(out_dir:    str,
                  lat:        float,
                  lon:        float,
                  height:     float,
                  date:       dt.datetime,
                  headings:   np.ndarray=np.linspace(0, 720, 1000),
                  elevations: np.ndarray=np.linspace(0, 7200, 1000),
                  a:          np.ndarray=np.eye(3),
                  b:          np.ndarray=np.zeros(3),
                  debug:      bool=True) -> pd.DataFrame:
    '''
    Generate simulated spin test data, save it to a .csv
    file, and return the data as a Pandas DataFrame
    
    Parameters
    ----------
    out_dir
        Path to directory where the simulated spin test DataFrame will be exported to
        (set to None to prevent writing out the data to disk)
    lat
        Latitude of the simulated spin test (dd)
    lon
        Longitude of the simulated spin test (dd)
    height
        Height of the simulated spin test above MSL (m)
    date
        Date and time of when the simulated spin test began (UTC)
    headings
        1xN array of heading angles (Degrees)
    elevations
        1xN array of elevation angles (Degrees)
    a
        3x3 vector magnetometer distortion matrix to be applied to the data
    b
        1x3 vector magnetometer bias vector to be applied to the data (nT)
    debug
        Whether or not debug prints/plot should be generated
    
    Returns
    -------
    pd.DataFrame
        DataFrame of the simulated spin test data
    '''
    
    if debug:
        print('Generating simulated spin test data')
    
    rolls  = np.zeros(headings.shape)
    eulers = np.hstack([rolls[:, np.newaxis],
                        elevations[:, np.newaxis],
                        headings[:, np.newaxis]])
    
    if debug:
        print('Generating perfect simulated spin test measurements')
    
    b_true = sc.gen_b_truth_euler(lat, lon, height, date, eulers)
    
    if debug:
        print('Applying vector distortion to simulated spin test data')
    
    b_dist = sc.apply_dist_to_vec(b_true, a, b)
    
    b_dist_x = b_dist[:, 0]
    b_dist_y = b_dist[:, 1]
    b_dist_z = b_dist[:, 2]
    b_dist_f = la.norm(b_dist, axis=1)
    
    if debug:
        print('Calculating IGRF values for simulated spin test')
    
    IGRF = sc.b_earth_ned_igrf(lat, lon, height, date)
    
    IGRF_x = IGRF[0]
    IGRF_y = IGRF[1]
    IGRF_z = IGRF[2]
    IGRF_f = la.norm(IGRF)
    
    if debug:
        sc.plot_spin_data(b_true,
                          b_dist)
        
        print('Exporting simulated spin test data as a CSV')
    
    data = {'datetime':  date,
            'epoch_sec': date.timestamp(),
            'LAT':       lat,
            'LONG':      lon,
            'ALT':       height,
            'PITCH':     elevations,
            'ROLL':      rolls,
            'AZIMUTH':   headings,
            'X':         b_dist_x,
            'Y':         b_dist_y,
            'Z':         b_dist_z,
            'F':         b_dist_f,
            'IGRF_X':    IGRF_x,
            'IGRF_Y':    IGRF_y,
            'IGRF_Z':    IGRF_z,
            'IGRF_F':    IGRF_f}
    
    if out_dir is not None:
        return save_dataset(type    = 'spin',
                            out_dir = out_dir,
                            date    = date,
                            data    = data)
    return pd.DataFrame(data)

def gen_TL_data(out_dir:    str,
                center_lat: float,
                center_lon: float,
                height:     float,
                start_dt:   dt.datetime,
                box_xlen_m: float,
                box_ylen_m: float,
                c:          np.ndarray,
                vel_mps:    float=20,
                sample_hz:  float=50,
                dither_hz:  float=1,
                dither_amp: float=10,
                terms:      int=tl.ALL_TERMS,
                a:          np.ndarray=np.eye(3),
                b:          np.ndarray=np.zeros(3),
                debug:      bool=True) -> pd.DataFrame:
    '''
    Generate simulated Tolles-Lawson flight data, save it to a .csv
    file, and return the data as a Pandas DataFrame
    
    Parameters
    ----------
    out_dir
        Path to directory where the simulated Tolles-Lawson flight DataFrame will be exported to
        (set to None to prevent writing out the data to disk)
    center_lat
        Latitude of the center of the simulated Tolles-Lawson box flight (dd)
    center_lon
        Longitude of the center of the simulated Tolles-Lawson box flight (dd)
    height
        Height above MSL of the simulated Tolles-Lawson box flight (m)
    start_dt
        Date and time of when the simulated Tolles-Lawson flight began "collecting" data (UTC)
    box_xlen_m
        Total distance simulated Tolles-Lawson box flight spans in the x/East direction (m)
    box_ylen_m
        Total distance simulated Tolles-Lawson box flight spans in the y/North direction (m)
    c
        Tolles-Lawson coefficients used to distort the scalar magnetometer data. The number
        of coefficients must correspond to the terms specified by `terms`
    vel_mps
        Velocity of the plane in the simulated Tolles-Lawson flight (m/s)
    sample_hz
        Sample rate of the sensors for the simulated Tolles-Lawson flight (Hz)
    dither_hz
        Frequency at which the "plane" oscilates pitch, roll, and azimuth for the simulated
        Tolles-Lawson flight (Hz)
    dither_amp
        Amplitude at which the "plane" oscilates pitch, roll, and azimuth for the simulated
        Tolles-Lawson flight (Degrees)
    terms
        Specify which terms to use to distort the data. Options include:
        
        - ALL_TERMS
        - PERMANENT
        - INDUCED
        - EDDY
    
    a
        3x3 distortion matrix
    b
        1x3 bias vector
    debug
        Whether or not debug prints/plot should be generated

    Returns
    -------
    pd.DataFrame
        DataFrame of the simulated Tolles-Lawson flight data
    '''
    
    if debug:
        print('Generating simulated TL calibration flight data')
    
    # General setup
    delta_t       = 1 / sample_hz
    sample_dist_m = vel_mps / sample_hz
    num_xsamples  = int(box_xlen_m / sample_dist_m)
    num_ysamples  = int(box_ylen_m / sample_dist_m)
    num_samples   = (2 * num_xsamples) + (2 * num_ysamples)
    datetimes     = [start_dt + dt.timedelta(seconds=(delta_t * i)) for i in range(num_samples)]
    timestamps    = [datetime.timestamp() for datetime in datetimes]
    
    # Calculate lat/lon coordinates for box flight
    _, x_min = cu.coord_coord(center_lat, center_lon, box_xlen_m * M2KM / 2, 270)
    _, x_max = cu.coord_coord(center_lat, center_lon, box_xlen_m * M2KM / 2, 90)
    y_min, _ = cu.coord_coord(center_lat, center_lon, box_ylen_m * M2KM / 2, 180)
    y_max, _ = cu.coord_coord(center_lat, center_lon, box_ylen_m * M2KM / 2, 0)
    
    x_coords = np.linspace(x_min, x_max, num_xsamples)
    y_coords = np.linspace(y_min, y_max, num_ysamples)
    
    #               Fly LLHC to ULHC           Fly ULHC to URHC           Fly URHC to LRHC           Fly LRHC to LLHC         --> LLHC = Lower Left Hand Corner, etc...
    lats = np.array(list(y_coords)           + ([y_max] * num_xsamples) + list(y_coords[::-1])     + ([y_min] * num_xsamples))
    lons = np.array(([x_min] * num_ysamples) + list(x_coords)           + ([x_max] * num_ysamples) + list(x_coords[::-1]))
    
    if debug:
        print('Calculating IGRF values for simulated TL calibration flight')
    
    # Calculate IGRF values for flight
    IGRF = sc.b_earth_ned_igrf(lats, lons, height, start_dt)
    
    IGRF_x = IGRF[:, 0]
    IGRF_y = IGRF[:, 1]
    IGRF_z = IGRF[:, 2]
    IGRF_f = la.norm(IGRF, axis=1)
    
    # Calculate euler angles for flight
    dither_xlen_m = box_xlen_m / 3 # Divide each side of the box into 3 sections since we need to independently dither pitch, roll, and azimuth
    dither_ylen_m = box_ylen_m / 3
    
    xdither_num_samples = int(dither_xlen_m / sample_dist_m) - 1
    ydither_num_samples = int(dither_ylen_m / sample_dist_m) - 1
    
    xdither_t = np.linspace(0,
                            xdither_num_samples * delta_t,
                            xdither_num_samples)
    xdither = dither_amp * np.sin(2 * np.pi * dither_hz * xdither_t)
    
    ydither_t = np.linspace(0,
                            ydither_num_samples * delta_t,
                            ydither_num_samples)
    ydither = dither_amp * np.sin(2 * np.pi * dither_hz * ydither_t)
    
    if debug:
        print('Dithering orientation angles ({} Hz, ±{}°)'.format(dither_hz, dither_amp))
        print('Dithering pitch angles')
    
    xpitch = np.zeros(num_xsamples)
    ypitch = np.zeros(num_ysamples)
    
    xpitch[0:len(xdither)] = xdither
    ypitch[0:len(ydither)] = ydither
    
    pitch = np.hstack([xpitch, ypitch, xpitch, ypitch])
    
    if debug:
        print('Dithering roll angles')
    
    xroll = np.zeros(num_xsamples)
    yroll = np.zeros(num_ysamples)
    
    xroll[len(xdither):len(xdither)*2] = xdither
    yroll[len(ydither):len(ydither)*2] = ydither
    
    roll = np.hstack([xroll, yroll, xroll, yroll])
    
    if debug:
        print('Dithering azimuth angles')
        
    xazimuth = np.zeros(num_xsamples)
    yazimuth = np.zeros(num_ysamples)
    
    xazimuth[len(xdither)*2:len(xdither)*3] = xdither
    yazimuth[len(ydither)*2:len(ydither)*3] = ydither
    
    azimuth = np.hstack([xazimuth + 0, # Add biases for the fact that each side of the box has the plane fly different average azimuths
                         yazimuth + 90,
                         xazimuth + 180,
                         yazimuth + 270])

    eulers = np.vstack([roll, pitch, azimuth]).T
    
    if debug:
        print('Generating true TL readings (assuming no anomaly - only IGRF is used)')
    
    b_scalar_true = IGRF_f
    b_vector      = mu.ned2body(IGRF, eulers)
    
    if debug:
        print('Applying TL distortion to simulated calibration flight data')
    
    b_scalar_dist = tl.apply_tl_dist(c,
                                     b_scalar_true,
                                     b_vector,
                                     delta_t,
                                     None,
                                     terms=terms)
    
    if debug:
        print('Applying spin test distortion to simulated simulated TL readings')
    
    IGRF = sc.apply_dist_to_vec(IGRF, a, b)
    
    b_dist_x = b_vector[:, 0]
    b_dist_y = b_vector[:, 1]
    b_dist_z = b_vector[:, 2]
    
    if debug:
        _, (ax1, ax2) = plt.subplots(1, 2)
        
        ax1.set_title('TL Box Flight Angles')
        ax1.plot(datetimes, pitch,   label='Pitch')
        ax1.plot(datetimes, roll,    label='Roll')
        ax1.plot(datetimes, azimuth, label='Azimuth')
        ax1.set_ylabel('°')
        ax1.set_xlabel('Date')
        ax1.legend()
        ax1.grid()
        
        ax2.set_title('TL Magnetometer Data')
        ax2.plot(datetimes, b_scalar_true, label='Original Scalar Data')
        ax2.plot(datetimes, b_scalar_dist, linestyle='dashed', label='Distorted Scalar Data')
        ax2.set_ylabel('nT')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid()
        
        print('Exporting simulated TL flight data as a CSV')
    
    # Save results
    data = {'datetime':  datetimes,
            'epoch_sec': timestamps,
            'LAT':       lats,
            'LONG':      lons,
            'ALT':       height,
            'PITCH':     pitch,
            'ROLL':      roll,
            'AZIMUTH':   azimuth,
            'X':         b_dist_x,
            'Y':         b_dist_y,
            'Z':         b_dist_z,
            'F':         b_scalar_dist,
            'IGRF_X':    IGRF_x,
            'IGRF_Y':    IGRF_y,
            'IGRF_Z':    IGRF_z,
            'IGRF_F':    IGRF_f}
    
    if out_dir is not None:
        return save_dataset(type    = 'tl',
                            out_dir = out_dir,
                            date    = start_dt,
                            data    = data)
    return pd.DataFrame(data)

def gen_ref_station_data(out_dir:   str,
                         lat:       float,
                         lon:       float,
                         height:    float,
                         start_dt:  dt.datetime,
                         dur_s:     float,
                         scale:     float=1,
                         offset:    float=0,
                         awgn_std:  float=0.1,
                         sample_hz: float=1,
                         file_df:   pd.DataFrame=None,
                         debug:     bool=True) -> pd.DataFrame:
    '''
    Generate simulated reference station data, save it to a .csv file, and return
    the data as a Pandas DataFrame
    
    Parameters
    ----------
    out_dir
        Path to directory where the simulated reference station DataFrame will be exported to
        (set to None to prevent writing out the data to disk)
    lat
        Latitude of the simulated reference station (dd)
    lon
        Longitude of the simulated reference station (dd)
    height
        Height of the simulated reference station above MSL (m)
    start_dt
        Date of when the simulated reference station began "collecting" data (UTC)
    dur_s
        Number of seconds the simulated reference station "collected" data (s)
    scale
        Inversely scale the simulated reference data by this factor
    offset
        Negatively bias the simulated reference data by this amount (mT)
    awgn_std
        If 'type & AWGN', add random noise with this standard deviation to the data
    sample_hz
        Sample rate of simulated reference station (Hz)
    file_df
        If 'type & FILE', add data from this DataFrame to the diurnal data
    debug
        Whether or not debug prints/plot should be generated
    
    Returns
    -------
    pd.DataFrame
        DataFrame of the simulated reference station data
    '''
    
    if debug:
        print('Generating simulated reference station data')
    
    t_start        = start_dt.timestamp()
    t_stop         = t_start + dur_s
    timestamps     = np.linspace(t_start, t_stop, (dur_s * sample_hz) + 1)
    num_timestamps = len(timestamps)
    datetimes      = np.array([dt.datetime.fromtimestamp(t) for t in timestamps])
    
    ref_vec = np.zeros((num_timestamps, 3))
    ref_mag = np.zeros(num_timestamps)
    
    if file_df is not None:
        _, file_ref_mag = Diurnal.interp_reference_df(df            = file_df,
                                                      timestamps    = timestamps,
                                                      survey_lon    = None,
                                                      subtract_core = True)
        
        if debug:
            print('Incorporating file data into simulated reference data')
        
        ref_mag += file_ref_mag
    
    if debug:
        print('Incorporating scale and offset into simulated reference data')
    
    ref_mag = Diurnal.apply_dist(x    = [offset, scale],
                                 data = ref_mag)
    
    if debug:
        print('Incorporating AWGN into simulated reference data')
    
    awgn_ref_vec = np.random.normal(0, awgn_std, ref_vec.shape)
    awgn_ref_mag = la.norm(awgn_ref_vec, axis=1)
    
    ref_mag += awgn_ref_mag
    
    if debug:
        print('Calculating IGRF values at simulated reference station')
    
    IGRF = sc.b_earth_ned_igrf(lat, lon, height, start_dt)
    
    IGRF_x = IGRF[0]
    IGRF_y = IGRF[1]
    IGRF_z = IGRF[2]
    IGRF_f = la.norm(IGRF)
    
    IGRF_dcs = IGRF / IGRF_f
    
    if debug:
        print('Adding IGRF core field to simulated reference station scalar values')
    
    ref_mag += IGRF_f
    
    if debug:
        print('Projecting simulated reference station scalar values into vector values using IGRF direction cosines')
    
    x = ref_mag * IGRF_dcs[0]
    y = ref_mag * IGRF_dcs[1]
    z = ref_mag * IGRF_dcs[2]
    f = ref_mag
    
    if debug:
        plt.plot(datetimes, f)
        plt.title('Total Simulated Reference Station Scalar Values')
        plt.ylabel('nT')
        plt.xlabel('Date')
        plt.grid()
        
        print('Exporting simulated reference station data as a CSV')
    
    data = {'datetime':  datetimes,
            'epoch_sec': timestamps,
            'LAT':       lat,
            'LONG':      lon,
            'ALT':       height,
            'PITCH':     0,
            'ROLL':      0,
            'AZIMUTH':   0,
            'X':         x,
            'Y':         y,
            'Z':         z,
            'F':         f,
            'IGRF_X':    IGRF_x,
            'IGRF_Y':    IGRF_y,
            'IGRF_Z':    IGRF_z,
            'IGRF_F':    IGRF_f}
    
    if out_dir is not None:
        return save_dataset(type    = 'ref',
                            out_dir = out_dir,
                            date    = start_dt,
                            data    = data)
    return pd.DataFrame(data)

def gen_sim_map(out_dir:        str,
                location:       str,
                center_lat:     float,
                center_lon:     float,
                dx_m:           float,
                dy_m:           float,
                x_dist_m:       float,
                y_dist_m:       float,
                height:         float,
                date:           dt.datetime,
                anomaly_locs:   np.ndarray,
                anomaly_scales: np.ndarray,
                anomaly_covs:   np.ndarray,
                upcontinue:     bool=False,
                debug:          bool=True) -> rxr.rioxarray.raster_dataset.xarray.DataArray:
    '''
    Generate a simulated anomaly map as a multi-band GeoTIFF file in
    WGS-84 coordinates. Bands include:
    
    - Band 0: Scalar anomaly values (nT)
    - Band 1: x/North vector anomaly values (nT)
    - Band 2: y/East vector anomaly values (nT)
    - Band 3: z/Down vector anomaly values (nT)
    - Band 3: Pixel height values (m)
    
    Parameters
    ----------
    out_dir
        Path to directory where the GeoTIFF will be exported to
        (set to None to prevent writing out the data to disk)
    location
        Description of survey area
    center_lat
        Latitude of the center of the map (dd)
    center_lon
        Longitude of the center of the map (dd)
    dx_m
        X pixel size (m)
    dy_m
        Y pixel size (m)
    x_dist_m
        Total distance map spans in the x/East direction (m)
    y_dist_m
        Total distance map spans in the y/North direction (m)
    height
        Height above MSL of the level map (m)
    date
        Date of when the survey was "collected" (UTC)
    anomaly_locs
        2xN array of lat/lon coordinates of all magnetic anomalies
        in the map area --> [anomaly 0 lat (dd), anomaly 1 lat (dd), ...]
                            [anomaly 0 lon (dd), anomaly 1 lon (dd), ...]
    anomaly_scales
        1xN array of scale values to control the amplitudes of
        all magnetic anomalies in the map area at MSL (NOT at `height`)
    anomaly_covs
        Nx2x2 array of covariance matricies that control the spread
        direction and spread amount of all magnetic anomalies in the
        map area
    upcontinue
        Whether or not to create the initial map at MSL and upcontinue
        to the specified height (otherwise simiply leave the initial
        map as the final map)
    debug
        Whether or not debug prints/plot should be generated
    
    Returns
    -------
    rxr.rioxarray.raster_dataset.xarray.DataArray
        Map as a rioxarray
    '''
    
    if debug:
        print('Generating simulated anomaly map')
    
    # Calculate number of samples and lat/lon coordinates for map
    num_xsamples = int(x_dist_m / dx_m)
    num_ysamples = int(y_dist_m / dy_m)
    
    _, min_lon = cu.coord_coord(center_lat, center_lon, x_dist_m * M2KM / 2, 270)
    _, max_lon = cu.coord_coord(center_lat, center_lon, x_dist_m * M2KM / 2, 90)
    min_lat, _ = cu.coord_coord(center_lat, center_lon, y_dist_m * M2KM / 2, 180)
    max_lat, _ = cu.coord_coord(center_lat, center_lon, y_dist_m * M2KM / 2, 0)
    
    lons = np.linspace(min_lon, max_lon, num_xsamples)
    lats = np.linspace(min_lat, max_lat, num_ysamples)
    
    num_anomalies = anomaly_locs.shape[1]
    anomaly_lats = anomaly_locs[0]
    anomaly_lons = anomaly_locs[1]
    
    x, y = np.meshgrid(lons, lats)
    
    scalar = np.zeros((len(lats), len(lons)))
    
    if debug:
        print('Processing simulated anomaly sctructures at MSL')
    
    # Add each anomaly field to the map individually by spacially
    # sampling a 2D Gaussian distribution that represents the
    # given anomaly
    for i in range(num_anomalies):
        pos          = np.zeros((*scalar.shape, 2))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        
        rv      = stat.multivariate_normal([anomaly_lons[i], anomaly_lats[i]], anomaly_covs[i])
        rv_norm = rv.pdf(pos) / rv.pdf(pos).max()
        
        new_anomaly_vals = anomaly_scales[i] * rv_norm
        
        scalar += new_anomaly_vals
    
    if upcontinue:
        if debug:
            plt.figure()
            plt.title('Simulated Total Anomaly Map at MSL')
            plt.pcolormesh(x, y, scalar, shading='nearest', cmap=cm.coolwarm)
            plt.ylabel('Latitude (dd)')
            plt.xlabel('Longitude (dd)')
            plt.colorbar()
            
        print('Upward continuing map to {}m'.format(height))
            
        scalar = mu.upcontinue(scalar, dx_m, dy_m, height)
        
        if debug:
            plt.figure()
            plt.title('Simulated Total Anomaly Map at {}m MSL'.format(height))
            plt.pcolormesh(x, y, scalar, shading='nearest', cmap=cm.coolwarm)
            plt.ylabel('Latitude (dd)')
            plt.xlabel('Longitude (dd)')
            plt.colorbar()
    
    else:
        plt.figure()
        plt.title('Simulated Total Anomaly Map at {}m MSL'.format(height))
        plt.pcolormesh(x, y, scalar, shading='nearest', cmap=cm.coolwarm)
        plt.ylabel('Latitude (dd)')
        plt.xlabel('Longitude (dd)')
        plt.colorbar()
    
    if debug:
        print('Calculating IGRF values for simulated map')
    
    # Calculate IGRF values for map
    mesh_lons, mesh_lats = np.meshgrid(lons, lats)
    Be, Bn, Bu = igrf(mesh_lons, mesh_lats, height * M2KM, date)
    
    IGRF = np.zeros((3, *scalar.shape))
    IGRF[0, :, :] =  Bn.squeeze()
    IGRF[1, :, :] =  Be.squeeze()
    IGRF[2, :, :] = -Bu.squeeze()
    
    IGRF_x = IGRF[0, :, :]
    IGRF_y = IGRF[1, :, :]
    IGRF_z = IGRF[2, :, :]
    IGRF_f = la.norm(IGRF, axis=0)
    
    if debug:
        print('Projecting simulated map scalar measurements into vector measurements using IGRF direction cosines')
    
    IGRF_dcs = np.zeros(IGRF.shape)
    IGRF_dcs[0, :, :] = IGRF_x / IGRF_f
    IGRF_dcs[1, :, :] = IGRF_y / IGRF_f
    IGRF_dcs[2, :, :] = IGRF_z / IGRF_f
    
    vector = np.zeros(IGRF.shape)
    vector[0, :, :] = scalar * IGRF_dcs[0, :, :]
    vector[1, :, :] = scalar * IGRF_dcs[1, :, :]
    vector[2, :, :] = scalar * IGRF_dcs[2, :, :]
    
    if debug:
        print('Exporting simulated map as a GeoTIFF')
    
    map = mu.export_map(out_dir  = out_dir,
                        location = location,
                        date     = date,
                        lats     = lats,
                        lons     = lons,
                        scalar   = scalar,
                        heights  = height,
                        stds     = None,
                        vector   = vector)
    
    if debug:
        plt.figure()
        map[0].plot(cmap=cm.coolwarm)
        plt.title('Final Simulated Scalar Anomaly Map From GeoTIFF')
        
        plt.figure()
        map[6].plot(cmap=cm.coolwarm)
        plt.title('Final Simulated X-Gradient Map From GeoTIFF')
        
        plt.figure()
        map[7].plot(cmap=cm.coolwarm)
        plt.title('Final Simulated Y-Gradient Map From GeoTIFF')
        
        plt.figure()
        map[4].plot(cmap=cm.coolwarm)
        plt.title('Final Simulated Height Map From GeoTIFF')
    
    return map

def gen_survey_data(out_dir:         str,
                    map:             rxr.rioxarray.raster_dataset.xarray.DataArray,
                    survey_height_m: float,
                    survey_start_dt: dt.datetime,
                    survey_vel_mps:  float,
                    survey_e_buff_m: float,
                    survey_w_buff_m: float,
                    survey_n_buff_m: float,
                    survey_s_buff_m: float,
                    sample_hz:       float,
                    ft_line_dist_m:  float,
                    ft_line_dir:     int=HORIZ,
                    a:               np.ndarray=np.eye(3),
                    b:               np.ndarray=np.zeros(3),
                    c:               np.ndarray=np.zeros(18),
                    terms:           int=tl.ALL_TERMS,
                    scalar_awgn_std: float=0,
                    diurnal_df:      pd.DataFrame=None,
                    diurnal_dist:    np.ndarray=None,
                    use_tie_lines:   bool=False,
                    tie_dist_m:      float=None,
                    debug:           bool=True) -> pd.DataFrame:
    '''
    Generate simulated flight survey data based on several
    parameters
    
    Parameters
    ----------
    out_dir
        Path to directory where the simulated survey data
        will be exported to
        (set to None to prevent writing out the data to disk)
    map
        Rioxarray of the "truth map" used to generate the
        simulated anomaly samples
    survey_height_m
        Height of the simulated survey MSL (m)
    survey_start_dt
        Start datetime of the simulated survey (UTC)
    survey_vel_mps
        Simulated survey vehicle velocity (m/s)
    survey_e_buff_m
        Buffer distance between the East extreme of the map
        and the East extreme of the survey samples (m)
    survey_w_buff_m
        Buffer distance between the West extreme of the map
        and the West extreme of the survey samples (m)
    survey_n_buff_m
        Buffer distance between the North extreme of the map
        and the North extreme of the survey samples (m)
    survey_s_buff_m
        Buffer distance between the South extreme of the map
        and the South extreme of the survey samples (m)
    sample_hz
        Sample rate of the simulated magnetometers (hz)
    ft_line_dist_m
        Distance between flight lines (m)
    ft_line_dir
        Direction the flight lines run. Tie lines (if uesd)
        will be oriented in the orthogonal direction.
        Options include:
        
        - HORIZ: Horizontal - East/West
        - VERT:  Vertical   - North/South
        
    a
        3x3 distortion matrix
    b
        1x3 bias vector
    c
        Tolles-Lawson coefficients used to distort the scalar magnetometer data. The number
        of coefficients must correspond to the terms specified by `terms`
    terms
        Terms to include in A-matrix. Options include:
        
        - ALL_TERMS
        - PERMANENT
        - INDUCED
        - EDDY
    
    scalar_awgn_std
        Standard deviation of the AWGN to add to the survey
        magnetometer measurements
    diurnal_df
        DataFrame of reference station data
    diurnal_dist
        1x2 array of distortion parameters to apply to
        diurnal/reference station data --> [offset (nT), scale]
    use_tie_lines
        Whether or not to generate tie line samples
    tie_dist_m
        Distance between tie lines (m)
    debug
        Whether or not debug prints/plot should be generated
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the simulated survey data
    '''
    
    if debug:
        print('Generating simulated survey data')
    
    # General setup
    delta_t       = 1 / sample_hz
    sample_dist_m = survey_vel_mps / sample_hz
    
    if debug:
        print('Generating simulated survey flight path')
    
    map_min_lat = map.y.min().item()
    map_max_lat = map.y.max().item()
    map_min_lon = map.x.min().item()
    map_max_lon = map.x.max().item()
    
    _, suvey_min_lon = cu.coord_coord(lat     = map_min_lat,
                                      lon     = map_min_lon,
                                      dist    = survey_w_buff_m * M2KM,
                                      bearing = 90)
    _, suvey_max_lon = cu.coord_coord(lat     = map_min_lat,
                                      lon     = map_max_lon,
                                      dist    = survey_e_buff_m * M2KM,
                                      bearing = 270)
    suvey_min_lat, _ = cu.coord_coord(lat     = map_min_lat,
                                      lon     = map_min_lon,
                                      dist    = survey_s_buff_m * M2KM,
                                      bearing = 0)
    suvey_max_lat, _ = cu.coord_coord(lat     = map_max_lat,
                                      lon     = map_min_lon,
                                      dist    = survey_n_buff_m * M2KM,
                                      bearing = 180)
    
    lon_total_dist_m = cu.coord_dist(lat_1 = suvey_min_lat,
                                     lon_1 = suvey_min_lon,
                                     lat_2 = suvey_min_lat,
                                     lon_2 = suvey_max_lon) * KM2M
    lat_total_dist_m = cu.coord_dist(lat_1 = suvey_min_lat,
                                     lon_1 = suvey_min_lon,
                                     lat_2 = suvey_max_lat,
                                     lon_2 = suvey_min_lon) * KM2M
    
    # Create survey "flight path"
    if ft_line_dir == HORIZ: # Flight lines run horizontal
        ft_line_pts, ft_line_nums = gen_parallel_search(lon_min          = suvey_min_lon,
                                                        lon_max          = suvey_max_lon,
                                                        lat_min          = suvey_min_lat,
                                                        lat_max          = suvey_max_lat,
                                                        lon_total_dist_m = lon_total_dist_m,
                                                        lat_total_dist_m = lat_total_dist_m,
                                                        lon_sub_dist_m   = sample_dist_m,
                                                        lat_sub_dist_m   = ft_line_dist_m,
                                                        line_dir         = HORIZ)
        
        if use_tie_lines:
            tie_line_pts, tie_line_nums = gen_parallel_search(lon_min          = suvey_min_lon,
                                                              lon_max          = suvey_max_lon,
                                                              lat_min          = suvey_min_lat,
                                                              lat_max          = suvey_max_lat,
                                                              lon_total_dist_m = lon_total_dist_m,
                                                              lat_total_dist_m = lat_total_dist_m,
                                                              lon_sub_dist_m   = tie_dist_m,
                                                              lat_sub_dist_m   = sample_dist_m,
                                                              line_dir         = VERT,
                                                              last_line_num    = ft_line_nums[-1])
        
    elif ft_line_dir == VERT: # Flight lines run vertical
        ft_line_pts, ft_line_nums = gen_parallel_search(lon_min          = suvey_min_lon,
                                                        lon_max          = suvey_max_lon,
                                                        lat_min          = suvey_min_lat,
                                                        lat_max          = suvey_max_lat,
                                                        lon_total_dist_m = lon_total_dist_m,
                                                        lat_total_dist_m = lat_total_dist_m,
                                                        lon_sub_dist_m   = ft_line_dist_m,
                                                        lat_sub_dist_m   = sample_dist_m,
                                                        line_dir         = VERT)
        
        if use_tie_lines:
            tie_line_pts, tie_line_nums = gen_parallel_search(lon_min          = suvey_min_lon,
                                                              lon_max          = suvey_max_lon,
                                                              lat_min          = suvey_min_lat,
                                                              lat_max          = suvey_max_lat,
                                                              lon_total_dist_m = lon_total_dist_m,
                                                              lat_total_dist_m = lat_total_dist_m,
                                                              lon_sub_dist_m   = sample_dist_m,
                                                              lat_sub_dist_m   = tie_dist_m,
                                                              line_dir         = HORIZ,
                                                              last_line_num    = ft_line_nums[-1])

    if use_tie_lines:
        survey_coords    = np.hstack([ft_line_pts, tie_line_pts])
        survey_line_nums = np.hstack([ft_line_nums, tie_line_nums])
        line_types       = np.hstack([np.ones(len(ft_line_nums)), np.zeros(len(tie_line_nums))])
        
    else:
        survey_coords    = ft_line_pts
        survey_line_nums = ft_line_nums
        line_types       = np.ones(len(ft_line_nums))
    
    lats = survey_coords[0, :]
    lons = survey_coords[1, :]
    
    if debug:
        plt.figure()
        map[0].plot(cmap=cm.coolwarm)
        plt.title('Survey Flight Path')
        plt.plot(lons, lats, label='Flight Path')
        plt.plot(lons, lats, 'x', label='Sample Points')
        plt.legend()
    
    # More setup stuff
    num_samples = survey_coords.shape[1]
    datetimes   = [survey_start_dt + dt.timedelta(seconds=(delta_t * i)) for i in range(num_samples)]
    
    survey_mag = np.zeros(num_samples)
    pitches    = np.zeros(num_samples)
    rolls      = np.zeros(num_samples)
    azimuths   = np.zeros(num_samples)
    
    # Generate a raster map of the IGRF field in the survey area
    IGRF_map = mu.igrf_WGS84(map_WGS84   = map,
                             map_alt_m   = survey_height_m,
                             survey_date = survey_start_dt)
    IGRF_vec = np.zeros((survey_coords.shape[1], 3))
    
    if debug:
        print('Calculating simulated survey scalar anomaly, azimuth, and IGRF values')
    
    # This loop samples both the anomaly and IGRF field maps and also
    # calculates the azimuth in degrees between the current and next
    # flight coordinate
    for i in tqdm(range(num_samples)):
        x = lons[i]
        y = lats[i]
        
        survey_mag[i] = mu.sample_map(map  = map,
                                      x    = x,
                                      y    = y,
                                      band = mu.SCALAR)
        
        try:
            next_x = lons[i + 1]
            next_y = lats[i + 1]
            
            azimuths[i] = cu.coord_bearing(lat_1 = y,
                                           lon_1 = x,
                                           lat_2 = next_y,
                                           lon_2 = next_x)
        except IndexError:
            azimuths[i] = azimuths[i - 1]
        
        IGRF_vec[i, 0] = mu.sample_map(map  = IGRF_map,
                                       x    = x,
                                       y    = y,
                                       band = mu.VEC_X) # x/North IGRF component (nT)
        IGRF_vec[i, 1] = mu.sample_map(map  = IGRF_map,
                                       x    = x,
                                       y    = y,
                                       band = mu.VEC_Y) # y/East IGRF component (nT)
        IGRF_vec[i, 2] = mu.sample_map(map  = IGRF_map,
                                       x    = x,
                                       y    = y,
                                       band = mu.VEC_Z) # z/Down IGRF component (nT)
    
    if debug:
        plt.figure()
        plt.title('Perfect Scalar Anomaly Survey Measurements')
        plt.plot(datetimes, survey_mag)
        plt.ylabel('nT')
        plt.xlabel('Date')
        plt.grid()
    
    IGRF_x = IGRF_vec[:, 0]
    IGRF_y = IGRF_vec[:, 1]
    IGRF_z = IGRF_vec[:, 2]
    IGRF_f = la.norm(IGRF_vec, axis=1)
    
    IGRF_dcs = np.zeros(IGRF_vec.shape)
    IGRF_dcs[:, 0] = IGRF_x / IGRF_f
    IGRF_dcs[:, 1] = IGRF_y / IGRF_f
    IGRF_dcs[:, 2] = IGRF_z / IGRF_f
    
    if debug:
        print('Adding core field to simulated survey scalar measurements')
    
    survey_mag += IGRF_f
    
    if debug:
        plt.figure()
        plt.title('Perfect Scalar Total Field Survey Measurements')
        plt.plot(datetimes, survey_mag)
        plt.ylabel('nT')
        plt.xlabel('Date')
        plt.grid()
    
    timestamps = gen_timestamps(survey_coords   = survey_coords,
                                survey_start_dt = survey_start_dt,
                                survey_vel_mps  = survey_vel_mps)
    
    # Apply diurnal noise if diurnal data is provided
    if diurnal_df is not None:
        _, diurnal_mag = Diurnal.interp_reference_df(df            = diurnal_df,
                                                     timestamps    = timestamps,
                                                     survey_lon    = None,
                                                     subtract_core = True)
        
        # Apply scale and offset to diurnal data if given
        if diurnal_dist is not None:
            diurnal_mag = Diurnal.apply_dist(x    = diurnal_dist,
                                             data = diurnal_mag)
        
        if debug:
            print('Adding diurnal to simulated survey scalar measurements')
    
        survey_mag += diurnal_mag
    
    if debug:
        plt.figure()
        plt.title('Scalar Total Field Survey Measurements + Durnal')
        plt.plot(datetimes, survey_mag)
        plt.ylabel('nT')
        plt.xlabel('Date')
        plt.grid()
    
    if scalar_awgn_std != 0:
        if debug:
            print('Adding AWGN to simulated survey scalar measurements')
    
        scalar_awgn = np.random.normal(0, scalar_awgn_std, survey_mag.shape)
        survey_mag += scalar_awgn
    
    if debug:
        plt.figure()
        plt.title('Scalar Total Field Survey Measurements + Durnal + AWGN')
        plt.plot(datetimes, survey_mag)
        plt.ylabel('nT')
        plt.xlabel('Date')
        plt.grid()
        
        print('Applying TL distortion to simulated survey scalar measurements')
    
    survey_mag = tl.apply_tl_dist(c          = c,
                                  b_scalar   = survey_mag,
                                  b_vector   = IGRF_vec,
                                  delta_t    = delta_t,
                                  b_external = None,
                                  terms      = terms)
    
    if debug:
        plt.figure()
        plt.title('Scalar Total Field Survey Measurements + Durnal + AWGN + TL Distortion')
        plt.plot(datetimes, survey_mag)
        plt.ylabel('nT')
        plt.xlabel('Date')
        plt.grid()
    
    if debug:
        print('Projecting simulated survey scalar measurements into NED vector measurements using IGRF direction cosines')
    
    # Compute vector readings by rotating the scalar "measurements" to
    # point in the same direction as the IGRF field at that date/time
    # and coordinate
    survey_vec = np.zeros(IGRF_dcs.shape)
    survey_vec[:, 0] = survey_mag * IGRF_dcs[:, 0]
    survey_vec[:, 1] = survey_mag * IGRF_dcs[:, 1]
    survey_vec[:, 2] = survey_mag * IGRF_dcs[:, 2]
    
    if debug:
        print('Rotating NED vector measurements into sensor\'s body frame')
    
    eulers = np.hstack([rolls[:, np.newaxis],
                        pitches[:, np.newaxis],
                        azimuths[:, np.newaxis]])
    
    survey_vec = mu.ned2body(ned_vecs = survey_vec,
                             eulers   = eulers)
    
    if debug:
        print('Applying spin test distortion to simulated survey vector measurements')
    
    survey_vec = sc.apply_dist_to_vec(vec = survey_vec,
                                      a   = a,
                                      b   = b)
    survey_vec_x = survey_vec[:, 0]
    survey_vec_y = survey_vec[:, 1]
    survey_vec_z = survey_vec[:, 2]
    
    if debug:
        print('Exporting simulated survey data as a CSV')
    
    if debug:
        print('Survey start datetime/timestamp: {}/{}s'.format(datetimes[0], timestamps[0]))
        print('Survey end datetime/timestamp: {}/{}s'.format(datetimes[-1], timestamps[-1]))
        print('Flight line samples end at timestamp: {}s'.format(timestamps[len(ft_line_pts)]))
        print('Survey Duration: {}s'.format(timestamps[-1] - timestamps[0]))
    
    # Save results
    data = {'datetime':  datetimes,
            'epoch_sec': timestamps,
            'LAT':       lats,
            'LONG':      lons,
            'ALT':       survey_height_m,
            'LINE':      survey_line_nums,
            'LINE_TYPE': line_types,
            'PITCH':     pitches,
            'ROLL':      rolls,
            'AZIMUTH':   azimuths,
            'X':         survey_vec_x,
            'Y':         survey_vec_y,
            'Z':         survey_vec_z,
            'F':         survey_mag,
            'IGRF_X':    IGRF_x,
            'IGRF_Y':    IGRF_y,
            'IGRF_Z':    IGRF_z,
            'IGRF_F':    IGRF_f}
    
    if out_dir is not None:
        return save_dataset(type    = 'survey',
                            out_dir = out_dir,
                            date    = survey_start_dt,
                            data    = data)
    return pd.DataFrame(data)