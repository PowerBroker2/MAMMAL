import numpy as np
import pandas as pd


def parse_xyz(fname:      str,
              drop_dups:  bool=False,
              drop_cust:  list=[]) -> pd.DataFrame:
    '''
    Parse the xyz sensor file name and return a pandas data
    frame with the resulting parsed data
    
    Parameters
    ----------
    fname
        File path/name to the .xyz file to parse
    drop_dups
        Drop rows with duplicate 'TIME' values if set to True
    drop_cust
        String or list of strings of column names to drop

    Returns
    -------
    pd.DataFrame
        Dataframe of SGL data parsed from the given .xyz
        file - *usually* includes the following columns/fields:
        
        - LINE:      Line Number XXXX.YY where XXXX is line number and YY is segment number
        - FLT:       Flight Number
        - YEAR:      Year
        - DOY:       Julian day of year (UTC)
        - TIME:      Number of seconds past UTC midnight
        - UTM-X:     X coordinate, WGS-84 UTM ZONE 11N (m)
        - UTM-Y:     Y coordinate, WGS-84 UTM ZONE 11N (m)
        - UTM-Z:     GPS Elevation (above WGS-84 Ellipsoid) (m)
        - MSL-Z:     GPS Elevation (above EGM2008 Geoid) (m)
        - LAT:       Latitude, WGS-84 (dd)
        - LONG:      Longitude, WGS-84 (dd)
        - BARO:      Barometric Altimeter (m)
        - RADAR:     Filtered Radar Altimeter (m)
        - TOPO:      Radar Topography (above EGM2008 Geoid) (m)
        - DEM:       Digital Elevation Model from SRTM (above EGM2008 Geoid) (m)
        - PITCH:     System computed aircraft pitch (deg)
        - ROLL:      System computed aircraft roll (deg)
        - AZIMUTH:   System computed aircraft azimuth (deg)
        - DIURNAL:   Diurnal Magnetic Field from reference station (nT)
        - FLUX_X:    Fluxgate x-axis (nT)
        - FLUX_Y:    Fluxgate y-axis (nT)
        - FLUX_Z:    Fluxgate z-axis (nT)
        - FLUX_TOT:  Flugate total (nT)
        - UNCOMPMAG: Uncompensated Airborne Magnetic Field (nT)
        - COMPMAG:   Compensated Airborne Magnetic Field (nT)
        - DCMAG:     Diurnal Corrected Airborne Magnetic Field (nT)
        - IGRFMAG:   IGRF and Diurnal Corrected Airborne Magnetic Field (nT)
        - LVLDMAG:   Levelled Magnetic Anomaly (nT)
        - datetime:  Datetime object (UTC)
        - epoch_sec: UNIX epoch timestamp (s)
    '''
    
    df = pd.read_csv(fname, header=2, delim_whitespace=True, na_values='*')
    df.columns = np.hstack([df.columns[1:].values, np.array(['extra_header'])]) # shift over the header names and pad an extra for the N/A column of data
    df = df.iloc[:, :-1] # drop the last N/A column of data
    
    if 'FTIME' in df.columns:
        df.rename(columns={'FTIME': 'TIME'}, inplace=True)
    
    for col in drop_cust:
        df.drop(col, axis=1, inplace=True)
    
    # Remove rows with nan-date/time values for the datetime column creation to work
    df.drop(df.index[df['YEAR'].isnull()], axis=0, inplace=True)
    df.drop(df.index[df['DOY'].isnull()],  axis=0, inplace=True)
    df.drop(df.index[df['TIME'].isnull()], axis=0, inplace=True)
    
    # Reindex DataFrame
    df.index = range(len(df))

    # Want some columns as string for timestamp parsing YEAR, DOY, and day-seconds
    # Want them as native types otherwise for eas of use
    df['datetime'] = pd.to_datetime(df['YEAR'].astype(int).astype(str) + df['DOY'].astype(int).astype(str),
                                    format='%Y%j',
                                    errors='coerce') + pd.to_timedelta(df['TIME'], unit='s')
    
    # Fix these fields to be true UTC julian day/day-seconds
    df['DOY']  = df['datetime'].dt.dayofyear
    df['TIME'] = df['TIME'] % 86399
    
    # Add column for total number of seconds after epoch to make it easier
    # to interpolate between readings of multiple datasets
    df['epoch_sec'] = (df['datetime'] - pd.Timestamp('1970-01-01')).dt.total_seconds()

    # There are sometimes duplicated rows - they are associated with different lines, but the
    # other information is duplicate
    if drop_dups:
        df.drop_duplicates(subset='TIME', inplace=True, keep='first')

    return df.sort_values(by=['datetime'])