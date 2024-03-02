import datetime as dt

import pandas as pd
from scipy import interpolate
from tqdm import tqdm


LEAP_SEC = 18


def gps2utc(gpsweek, gpsseconds, leapseconds=LEAP_SEC):
    '''
    https://gist.github.com/jeremiahajohnson/eca97484db88bcf6b124
    '''
    
    epoch   = dt.datetime.strptime("1980-01-06 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    elapsed = dt.timedelta(days=(gpsweek * 7), seconds=(gpsseconds - leapseconds))
    
    return epoch + elapsed

def parsePix(fname, alt_min=None, leapsec=LEAP_SEC):
    '''
    Create Pandas DataFrame of timestamped LLA coordinates from log
    '''
    
    gps_dict = {'TimeUS':  [],
                'GMS':     [],
                'GWk':     []}
    ekf_dict = {'TimeUS':  [],
                'LAT':     [],
                'LONG':    [],
                'ALT':     [],
                'PITCH':   [],
                'ROLL':    [],
                'AZIMUTH': []}
    
    with open(fname, 'r') as inFile:
        lines = inFile.readlines()
    
    for line in tqdm(lines):
        if line.startswith('GPS'):
            pieces = line.split(', ')
            
            if len(pieces) == 16:
                gps_dict['TimeUS'].append(float(pieces[1]))
                gps_dict['GMS'].append(float(pieces[4]))
                gps_dict['GWk'].append(int(pieces[5]))
        
        elif line.startswith('AHR2'):
            pieces = line.split(', ')
            
            if len(pieces) == 12:
                ekf_dict['TimeUS'].append(float(pieces[1]))
                ekf_dict['ROLL'].append(float(pieces[2]))
                ekf_dict['PITCH'].append(float(pieces[3]))
                ekf_dict['AZIMUTH'].append(float(pieces[4]))
                ekf_dict['ALT'].append(float(pieces[5]))
                ekf_dict['LAT'].append(float(pieces[6]))
                ekf_dict['LONG'].append(float(pieces[7]))
                
    gps_df = pd.DataFrame(gps_dict)
    ekf_df = pd.DataFrame(ekf_dict)
    
    gps_timeus = gps_df.TimeUS
    gms        = gps_df.GMS
    gwk        = gps_df.GWk

    datetime  = [gps2utc(float(gwk[i]), float(gms[i]) / 1000.0, leapsec) for i in range(len(gps_df))]
    epoch_sec = [datetime[i].timestamp() for i in range(len(gps_df))]
    
    ekf_timeus = ekf_df.TimeUS

    f = interpolate.interp1d(gps_timeus, epoch_sec, fill_value='extrapolate')
    ekf_epoch_sec = f(ekf_timeus)

    ekf_df['epoch_sec'] = ekf_epoch_sec
    ekf_df['datetime']  = [dt.datetime.utcfromtimestamp(ts) for ts in ekf_epoch_sec]
    
    ekf_df = ekf_df[['epoch_sec', 'datetime', 'LAT', 'LONG', 'ALT', 'PITCH', 'ROLL', 'AZIMUTH']]
    
    if alt_min is not None:
        ekf_df = ekf_df[ekf_df.ALT >= alt_min]
    
    return ekf_df