import os
import sys
from os.path import join, dirname, realpath

import pandas as pd

sys.path.append(dirname(realpath(__file__)))
sys.path.append(dirname(dirname(realpath(__file__))))

from Utils.ProcessingUtils import add_igrf_cols


def skip_lines(fname: str) -> int:
    '''
    Find number of header lines to skip when reading .sec file
    as a dataframe
    
    Parameters
    ----------
    fname
        File path/name to the .sec file
    
    Returns
    -------
    int
        Number of header lines to skip when reading .sec file
        as a dataframe
    '''

    with open(fname, 'r') as inFile:
        contents = inFile.readlines()
    
    for i, line in enumerate(contents):
        if '|' not in line:
            return i - 1
    
    return 0

def add_lla_cols(df:    pd.DataFrame,
                 fname: str) -> pd.DataFrame:
    '''
    Add latitude, longitude, and altitude columns for all samples
    in the dataframe (lat/lon in dd and alt in km above MSL)
    
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
        Dataframe of INTERMAGNET data with new latitude,
        longitude, and altitude columns
    '''
    
    with open(fname, 'r') as inFile:
        header = inFile.readlines()[:25]
        header = ''.join(header)
    
    lat = float(header.split()[header.split().index('Latitude')  + 1]) # (dd)
    lon = float(header.split()[header.split().index('Longitude') + 1]) # (dd)
    alt = float(header.split()[header.split().index('Elevation') + 1]) # (m)
    
    if lon > 180:
        lon -= 360
    
    df['LAT']  = lat
    df['LONG'] = lon
    df['ALT']  = alt
    
    return df

def parse_sec(fname:     str,
              fast_mode: bool=True,
              chunk:     int=1000) -> pd.DataFrame:
    '''
    Parse the INTERMAGNET (.sec) sensor file and return a
    pandas data frame with the resulting parsed data
    
    Parameters
    ----------
    fname
        File path/name to the .sec file to parse
    fast_mode
        Calculate IGRF values based only on the first corrdinate
        in the .sec file (speeds up parsing considerably)
    chunk
        If not parsing in `fast_mode`, set the number of samples
        to simultaneously calculate IGRF values for
    
    Returns
    -------
    pd.DataFrame
        Dataframe of INTERMAGNET data parsed from the given .sec
        file - includes the following columns/fields:
        
        - DATE:      Date object (UTC)
        - TIME:      Number of seconds past UTC midnight
        - DOY:       Julian day of year (UTC)
        - X:         Magnetic field measurement in the North direction (nT)
        - Y:         Magnetic field measurement in the East direction (nT)
        - Z:         Magnetic field measurement in the Down direction (nT)
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
    
    if fname.endswith('.sec'):
        skip = skip_lines(fname)
        
        df = pd.read_csv(fname,
                         header=skip,
                         delim_whitespace=True,
                         na_values='99999.0')
        
        del df['|'] # Get rid of extra column (artifact of how .sec file headers are formatted)
        
        df.columns = ['DATE', 'TIME', 'DOY', 'X', 'Y', 'Z', 'F']
        df['TIME'] = pd.to_timedelta(df['TIME']).dt.total_seconds()
        
        df = df.dropna() # Must drop NaNs to get datetime column creation to work

        # want some columns as string for timestamp parsing DOY, and day-seconds
        # want them as native types otherwise for easof use
        date = pd.to_datetime(df['DATE'])
        df['datetime'] = pd.to_datetime(date.dt.year.astype(str) + df['DOY'].astype(int).astype(str),
                                        format='%Y%j',
                                        errors='coerce') + pd.to_timedelta(df['TIME'], unit='s')
        
        # add column for total number of seconds after epoch to make it easier
        # to interpolate between readings of multiple datasets
        df['epoch_sec'] = (df['datetime'] - pd.Timestamp('1970-01-01')).dt.total_seconds()
        
        df = add_lla_cols(df, fname)
        df = add_igrf_cols(df, fast_mode, chunk)
             
        return df.sort_values(by=['datetime'])
    
    return None

def loadInterMagData(data_dir: str,
                     fast_mode: bool=True,
                     chunk:     int=1000) -> dict:
    '''
    Walks through all the files saved in the InterMagnet data storage folder,
    and saves the data from all '.sec' files in that folder/subfolders. For
    each '.sec' file, the function decodes the file's data into a Pandas
    dataframe, determines the location of the data source (i.e. data is from
    Boulder CO or Touscon AZ, etc.), combines the data from the current '.sec'
    file with the previously saved data for that location, sorts the entire
    dataset of that location by day of year (DOY/Julian Day) and by
    day-seconds, and drops all rows with NaNs. After all '.sec' files are
    processed, a dictionary with combined data from each location is returned.
    
    Parameters
    ----------
    data_dir
        Path to directory that holds all INTERMAGNET .sec files
        to be parsed
    fast_mode
        Calculate IGRF values based only on the first corrdinate
        in the .sec file (speeds up parsing considerably)
    chunk
        If not parsing in `fast_mode`, set the number of samples
        to simultaneously calculate IGRF values for
    
    Returns
    -------
    dict
        Dictionary that includes all INTERMAGNET data parsed from .sec files
        found in `data_dir`
    '''
    
    data = {}
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.sec'):
                id = file[:3].upper()
                
                if id not in data.keys():
                    data[id] = parse_sec(join(root, file),
                                            fast_mode,
                                            chunk)
                    
                else:
                    data[id] = pd.concat([data[id],
                                            parse_sec(join(root, file),
                                                    fast_mode,
                                                    chunk)]).sort_values(by=['DOY', 'TIME']).dropna()
                    
                print('Loaded', file)
    
    return data