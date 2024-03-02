import sys
import datetime as dt
from os.path import dirname, realpath

import numpy as np
import pandas as pd

sys.path.append(dirname(realpath(__file__)))
sys.path.append(dirname(dirname(realpath(__file__))))

from Utils.ProcessingUtils import add_igrf_cols


def parse_date(fname: str) -> dt.datetime:
    '''
    Find collection start date in GSMP log file
    
    Parameters
    ----------
    fname
        File path/name to the GSMP log file
    
    Returns
    -------
    dt.datetime
        Collection start date in GSMP log file
    '''
    
    with open(fname, 'r') as inFile:
        contents = inFile.readlines()
    
    for i, line in enumerate(contents):
        if line.strip() == 'G':
            return dt.datetime.strptime(contents[i + 1][:-1], '%d %m %Y')
    raise Exception('Could not find start date')

def skip_lines(fname: str) -> int:
    '''
    Find number of header lines to skip when reading GSMP log file
    as a dataframe
    
    Parameters
    ----------
    fname
        File path/name to the GSMP log file
    
    Returns
    -------
    int
        Number of header lines to skip when reading GSMP log file
        as a dataframe
    '''
    
    with open(fname, 'r') as inFile:
        contents = inFile.readlines()

    for i, line in enumerate(contents):
        if 'hhmmss.s' in line:
            return i
    return 0

def fix_datetime(df: dt.datetime) -> pd.DataFrame:
    '''
    Corrected TIME and DATE columns in dataframe of
    GSMP log file data, plus add datetime and
    epoch_sec columns
    
    Parameters
    ----------
    df
        Dataframe of GSMP log file data
    
    Returns
    -------
    pd.DataFrame
        Dataframe of GSMP log file data with corrected
        TIME and DATE columns, plus additional datetime
        and epoch_sec columns
    '''
    
    # Fix TIME column to a pure seconds counter
    hrs  = np.floor(df.TIME / 10000)
    mins = np.floor(df.TIME / 100) % 100
    secs = df.TIME % 100
    
    df.TIME   = (hrs * 3600) + (mins * 60) + secs
    time_diff = np.diff(df.TIME, prepend=df.TIME[0])
    time_diff[time_diff < 0] = 1
    
    # Create the datetime column while handling rollover
    df['datetime'] = df.DATE + \
                     pd.to_timedelta(df.TIME[0], unit='s') + \
                     pd.to_timedelta(np.cumsum(time_diff), unit='s')
    
    # Add column for total number of seconds after epoch to make it easier
    # to interpolate between readings of multiple datasets
    df['epoch_sec'] = (df['datetime'] - pd.Timestamp('1970-01-01')).dt.total_seconds()
    
    # Fix DATE column to handle rollover
    sec_offset = np.array(df.epoch_sec - df.epoch_sec[0] + df.TIME[0], dtype=int)
    day_offset = np.array(sec_offset / 86400, dtype=int)
    
    df.DATE = df.DATE + pd.to_timedelta(day_offset, unit='d')
    
    return df

def parse_GSMP(fname:     str,
               fast_mode: bool=True,
               chunk:     int=1000) -> pd.DataFrame:
    '''
    Parse the GSMP log file and return a
    pandas data frame with the resulting parsed data
    
    Parameters
    ----------
    fname
        File path/name to the GSMP log file to parse
    fast_mode
        Calculate IGRF values based only on the first corrdinate
        in the GSMP log file (speeds up parsing considerably)
    chunk
        If not parsing in `fast_mode`, set the number of samples
        to simultaneously calculate IGRF values for
    
    Returns
    -------
    pd.DataFrame
        Dataframe of data parsed from the given GSMP log
        file - includes the following columns/fields at
        a minimum:
        
        - DATE:      Date object (UTC)
        - TIME:      Number of seconds past UTC midnight
        - F:         Magnetic field measurement magnitude (nT)
        - datetime:  Datetime object (UTC)
        - epoch_sec: UNIX epoch timestamp (s)
        - LAT:       Latitude (dd)
        - LONG:      Longitude (dd)
        - ALT:       Altitude MSL (km)
        - IGRF_X:    IGRF magnetic field in the North direction (nT)
        - IGRF_Y:    IGRF magnetic field in the East direction (nT)
        - IGRF_Z:    IGRF magnetic field in the Down direction (nT)
        - IGRF_F:    IGRF magnetic field magnitude (nT)
    '''
    
    date = parse_date(fname)
    skip = skip_lines(fname)
    df   = pd.read_csv(fname, delim_whitespace=True, skiprows=skip)

    # Renaming columns for compatibility
    df.rename(columns={'nT':       'F'},    inplace=True)
    df.rename(columns={'hhmmss.s': 'TIME'}, inplace=True)
    df.rename(columns={'lat':      'LAT'},  inplace=True)
    df.rename(columns={'lon':      'LONG'}, inplace=True)
    df.rename(columns={'alt':      'ALT'},  inplace=True)
    
    # Add empty vector columns for compatibility
    df['X'] = np.ones(len(df.F)) * np.nan
    df['Y'] = np.ones(len(df.F)) * np.nan
    df['Z'] = np.ones(len(df.F)) * np.nan
    
    df['DATE'] = date

    fix_datetime(df)
    
    # Remove all rows of data with invalid GNSS fixes
    df = df[df.sat >= 4]
    
    return add_igrf_cols(df, fast_mode, chunk)