import sys
import datetime as dt
from os import listdir
from os.path import dirname, join, realpath

import numpy as np
import pandas as pd
import re
from tqdm import tqdm

sys.path.append(dirname(realpath(__file__)))
sys.path.append(dirname(dirname(realpath(__file__))))

from Utils.ProcessingUtils import add_igrf_cols


def natural_sort(l):
    '''
    Credits:
        - https://stackoverflow.com/a/4836734/9860973
    '''
    
    convert      = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key:  [convert(c) for c in re.split('([0-9]+)', key)]
    
    return sorted(l, key=alphanum_key)

def parse_devLog(fname:      str,
                 start_dt:   dt.datetime=dt.datetime.utcnow(),
                 lat:        float=0,
                 lon:        float=0,
                 alt:        float=0,
                 rej_thresh: float=500,
                 fast_mode:  bool=True,
                 chunk:      int=1000) -> pd.DataFrame:
    '''
    Parse the MFAM dev kit SD log file and return a
    pandas data frame with the resulting parsed data
    
    Parameters
    ----------
    fname
        File path/name to the dev kit log file to parse
    start_dt
        Start date/time of the log in UTC
    lat
        Approximate geodetic latitude of the collect (dd)
    lon
        Approximate geodetic longitude of the collect (dd)
    alt
        Approximate altitude of the collect above MSL (m)
    rej_thresh
        Core field rejection threshold (nT). Scalar samples outside the range
        determined by the calculated IGRF field +- this threshold will
        be removed from the dataset. For instance, if this threshold is
        500nT and the IGRF field is 50000nT, a scalar sample of 7000nT
        would be rejected, where a sample of 50300nT would not.
    fast_mode
        Calculate IGRF values based only on the first corrdinate
        in the dev kit log file (speeds up parsing considerably)
    chunk
        If not parsing in `fast_mode`, set the number of samples
        to simultaneously calculate IGRF values for
    
    Returns
    -------
    pd.DataFrame
        Dataframe of data parsed from the given dev kit log
        file - includes the following columns/fields at
        a minimum:
        
        - F:               Magnetic field measurement magnitude (nT)
        - datetime:        Datetime object (UTC)
        - epoch_sec:       UNIX epoch timestamp (s)
        - ID:              Global millisecond counter
        - FIDUCIAL:        Number of samples since last PPS pulse
        - FRAME_ID:        Subset of the FrameID value
        - SYS_STATUS:      See manual
        - SCALAR_1:        Sensor head 1 value (nT)
        - SCALAR_1_STATUS: Sensor head 1 status
        - SCALAR_2:        Sensor head 2 value (nT)
        - SCALAR_2_STATUS: Sensor head 1 status
        - AUX_0            See manual
        - AUX_1            See manual
        - AUX_2            See manual
        - AUX_3            See manual
        - LAT:             Latitude (dd)
        - LONG:            Longitude (dd)
        - ALT:             Altitude MSL (km)
        - IGRF_X:          IGRF magnetic field in the North direction (nT)
        - IGRF_Y:          IGRF magnetic field in the East direction (nT)
        - IGRF_Z:          IGRF magnetic field in the Down direction (nT)
        - IGRF_F:          IGRF magnetic field magnitude (nT)
    '''
    
    pps_id = None
    pps_dt = None
    
    true_lat = None
    true_lon = None
    true_alt = None
    
    ID              = []
    FIDUCIAL        = []
    FRAME_ID        = []
    SYS_STATUS      = []
    SCALAR_1        = []
    SCALAR_1_STATUS = []
    SCALAR_2        = []
    SCALAR_2_STATUS = []
    AUX_0           = []
    AUX_1           = []
    AUX_2           = []
    AUX_3           = []
    
    with open(fname, 'r') as log:
        contents = log.read()
    
    for _, line in enumerate(contents.split('\n')):
        line_pieces = line.split(',')
        
        if len(line_pieces) == 12:
            try:
                int(line_pieces[0])
                int(line_pieces[1])
                int(line_pieces[2])
                int(line_pieces[3])
                float(line_pieces[4])
                int(line_pieces[5])
                float(line_pieces[6])
                int(line_pieces[7])
                int(line_pieces[8])
                int(line_pieces[9])
                int(line_pieces[10])
                int(line_pieces[11])
            
                ID.append(int(line_pieces[0]))
                FIDUCIAL.append(int(line_pieces[1]))
                FRAME_ID.append(int(line_pieces[2]))
                SYS_STATUS.append(int(line_pieces[3]))
                SCALAR_1.append(float(line_pieces[4]))
                SCALAR_1_STATUS.append(int(line_pieces[5]))
                SCALAR_2.append(float(line_pieces[6]))
                SCALAR_2_STATUS.append(int(line_pieces[7]))
                AUX_0.append(int(line_pieces[8]))
                AUX_1.append(int(line_pieces[9]))
                AUX_2.append(int(line_pieces[10]))
                AUX_3.append(int(line_pieces[11]))
            except ValueError:
                pass
        
        elif line.startswith('PPS:'):
            subline = line[line.index('PPS:'):]
            pieces  = subline.split(', ')
            
            pps_id  = int(pieces[1])
            pps_str = pieces[2]
            
            pps_str = ' '.join(pps_str.split())
            
            try:
                pps_dt = dt.datetime.strptime(pps_str, r'%a %b %d %H:%M:%S %Y').replace(tzinfo=dt.timezone.utc)
            except ValueError:
                pass
        
        elif '$GPGGA' in line:
            subline = line[line.index('$GPGGA'):]
            pieces  = subline.split(',')
            
            if int(pieces[6]) > 0: # Process if GNSS data is valid
                lat_dm   = float(pieces[2])
                lat_dir  = pieces[3]
                lat_d    = int(lat_dm / 100)
                lat_m    = lat_dm - (lat_d * 100)
                true_lat = lat_d + (lat_m / 60.0)
                
                if lat_dir == 'S':
                    true_lat *= -1
                
                lon_dm   = float(pieces[4])
                lon_dir  = pieces[5]
                lon_d    = int(lon_dm / 100)
                lon_m    = lon_dm - (lon_d * 100)
                true_lon = lon_d + (lon_m / 60.0)
                
                if lon_dir == 'W':
                    true_lon *= -1
                
                true_alt = float(pieces[9])
    
    # Create/condition the DataFrame
    df = pd.DataFrame({'ID':              ID[1:],
                       'FIDUCIAL':        FIDUCIAL[1:],
                       'FRAME_ID':        FRAME_ID[1:],
                       'SYS_STATUS':      SYS_STATUS[1:],
                       'SCALAR_1':        SCALAR_1[1:],
                       'SCALAR_1_STATUS': SCALAR_1_STATUS[1:],
                       'SCALAR_2':        SCALAR_2[1:],
                       'SCALAR_2_STATUS': SCALAR_2_STATUS[1:],
                       'AUX_0':           AUX_0[1:],
                       'AUX_1':           AUX_1[1:],
                       'AUX_2':           AUX_2[1:],
                       'AUX_3':           AUX_3[1:]})
    df.SCALAR_1 = df.SCALAR_1.astype(float)
    df.SCALAR_2 = df.SCALAR_2.astype(float)
    df.ID       = df.ID.astype(int)
    
    # Add columns for compatibility
    if pps_id is None:
        df['epoch_sec'] = ((df.ID - df.ID.iloc[0]) / 1000.0) + start_dt.timestamp()
    else:
        df['epoch_sec'] = ((df.ID - pps_id) / 1000.0) + pps_dt.timestamp()
    
    if true_lat is None:
        df['LAT'] = np.ones(len(df.ID)) * lat
    else:
        df['LAT'] = np.ones(len(df.ID)) * true_lat
    
    if true_lon is None:
        df['LONG'] = np.ones(len(df.ID)) * lon
    else:
        df['LONG'] = np.ones(len(df.ID)) * true_lon
    
    if true_alt is None:
        df['ALT'] = np.ones(len(df.ID)) * alt
    else:
        df['ALT'] = np.ones(len(df.ID)) * true_alt
    
    df['datetime']       = pd.to_datetime(df.epoch_sec, unit='s')
    df['SCALAR_1_VALID'] = np.ones(len(df.ID)) * np.nan
    df['SCALAR_2_VALID'] = np.ones(len(df.ID)) * np.nan
    df['X']              = np.ones(len(df.ID)) * np.nan
    df['Y']              = np.ones(len(df.ID)) * np.nan
    df['Z']              = np.ones(len(df.ID)) * np.nan
    df['F']              = df.SCALAR_1
    
    df = add_igrf_cols(df, fast_mode, chunk)
    
    # Find valid min/max values
    min_F = df.IGRF_F - rej_thresh
    max_F = df.IGRF_F + rej_thresh
    
    # Find rows that have valid samples for each sensor head
    scalar_1_valid_mask = (df.SCALAR_1 >= min_F) & (df.SCALAR_1 <= max_F)
    scalar_2_valid_mask = (df.SCALAR_2 >= min_F) & (df.SCALAR_2 <= max_F)
    
    df['SCALAR_1_VALID'] = scalar_1_valid_mask
    df['SCALAR_2_VALID'] = scalar_2_valid_mask
    
    # Find when both and neither sensor head values are valid
    both_valid_mask    = scalar_1_valid_mask & scalar_2_valid_mask
    neither_valid_mask = ~(scalar_1_valid_mask | scalar_2_valid_mask)
    
    # Test if all data is "bad"
    if neither_valid_mask.all():
        both_valid_mask    = ~both_valid_mask
        neither_valid_mask = ~neither_valid_mask
        
        print('WARNING: All scalar data was found to be outside the acceptable range from the expected IGRF magnitude, no data will be clipped!')
    
    # Find when only sensor head 2 is valid
    only_scalar_2_valid_mask = scalar_2_valid_mask & (~scalar_1_valid_mask)
    
    # Use sensor head 2 when sensor head 1 is invalid
    df.F.iloc[only_scalar_2_valid_mask] = df.SCALAR_2[only_scalar_2_valid_mask]
    
    # Average sensor values when both are valid
    df.F.iloc[both_valid_mask] = (df.SCALAR_1[both_valid_mask] + df.SCALAR_2[both_valid_mask]) / 2.0
    
    # Drop when both heads are invalid
    df.F.iloc[neither_valid_mask] = 0.0
    
    return df

def parse_devACQU(dir:        str,
                  start_dt:   dt.datetime=dt.datetime.utcnow(),
                  lat:        float=0,
                  lon:        float=0,
                  alt:        float=0,
                  rej_thresh: float=500,
                  fast_mode:  bool=True,
                  chunk:      int=1000) -> pd.DataFrame:
    '''
    Parse all logs from a single acquisition on a MFAM
    dev kit SD log and return a pandas data frame with
    the resulting parsed data
    
    Parameters
    ----------
    dir
        File path to the dev kit survey acquisition directory
    start_dt
        Start date/time of the log in UTC
    lat
        Approximate geodetic latitude of the collect (dd)
    lon
        Approximate geodetic longitude of the collect (dd)
    alt
        Approximate altitude of the collect above MSL (m)
    rej_thresh
        Core field rejection threshold (nT). Scalar samples outside the range
        determined by the calculated IGRF field +- this threshold will
        be removed from the dataset. For instance, if this threshold is
        500nT and the IGRF field is 50000nT, a scalar sample of 7000nT
        would be rejected, where a sample of 50300nT would not.
    fast_mode
        Calculate IGRF values based only on the first corrdinate
        in the dev kit log file (speeds up parsing considerably)
    chunk
        If not parsing in `fast_mode`, set the number of samples
        to simultaneously calculate IGRF values for
    
    Returns
    -------
    pd.DataFrame
        Dataframe of data parsed from the given dev kit log
        file - includes the following columns/fields at
        a minimum:
        
        - F:               Magnetic field measurement magnitude (nT)
        - datetime:        Datetime object (UTC)
        - epoch_sec:       UNIX epoch timestamp (s)
        - ID:              Global millisecond counter
        - FIDUCIAL:        Number of samples since last PPS pulse
        - FRAME_ID:        Subset of the FrameID value
        - SYS_STATUS:      See manual
        - SCALAR_1:        Sensor head 1 value (nT)
        - SCALAR_1_STATUS: Sensor head 1 status
        - SCALAR_2:        Sensor head 2 value (nT)
        - SCALAR_2_STATUS: Sensor head 1 status
        - AUX_0            See manual
        - AUX_1            See manual
        - AUX_2            See manual
        - AUX_3            See manual
        - LAT:             Latitude (dd)
        - LONG:            Longitude (dd)
        - ALT:             Altitude MSL (km)
        - IGRF_X:          IGRF magnetic field in the North direction (nT)
        - IGRF_Y:          IGRF magnetic field in the East direction (nT)
        - IGRF_Z:          IGRF magnetic field in the Down direction (nT)
        - IGRF_F:          IGRF magnetic field magnitude (nT)
    '''
    
    df_list = []
    
    file_list = natural_sort(listdir(dir))
    
    for log in tqdm(file_list):
        if not log.lower().endswith('.txt'):
            continue
        
        try:
            int(log.split('.')[0])
        except:
            continue
        
        if len(df_list) > 0:
            start_dt = df_list[-1].datetime.iloc[-1]
        
        df_list.append(parse_devLog(join(dir, log),
                                    start_dt,
                                    lat,
                                    lon,
                                    alt,
                                    rej_thresh,
                                    fast_mode,
                                    chunk))
    
    return pd.concat(df_list)