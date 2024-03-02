import sys
import datetime as dt
from copy import deepcopy
from os.path import dirname, join, exists, realpath
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from GeoScraper import OSM_URL_Wizard, GeoScraper, uri_validator
from numpy import fft
from osgeo import gdal
from osgeo import osr
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from simplekml import Kml, Color
from ppigrf import igrf

sys.path.append(dirname(realpath(__file__)))
sys.path.append(dirname(dirname(realpath(__file__))))

import coordinateUtils as cu
import Filters


WGS84_EPSG = "EPSG:4326" # The EPSG for WGS-84 lat/lon is 4326 

# Enumerate band numbers
SCALAR = 0
VEC_X  = 1
VEC_Y  = 2
VEC_Z  = 3
ALT    = 4
STD    = 5


def plt_freqs(map:        rxr.rioxarray.raster_dataset.xarray.DataArray,
              map_name:   str='_',
              mag_thresh: float=1e-2):
    '''
    
    '''
    
    map_utm      = map.rio.reproject(map.rio.estimate_utm_crs())
    map_utm.data = np.nan_to_num(num_to_nan(map_utm).data)
    
    map_fft = fft.fft2(np.nan_to_num(np.squeeze(map_utm.data)))
    
    kx = fft.fftfreq(n=len(map_utm.x), d=np.diff(map_utm.x).mean()) # Cycles/m
    ky = fft.fftfreq(n=len(map_utm.y), d=np.diff(map_utm.y).mean()) # Cycles/m
    
    _, (ax1, ax2) = plt.subplots(1, 2)
    ew_freqs = np.mean(abs(fft.fftshift(map_fft)), axis=0)
    shift_kx = fft.fftshift(kx)
    ax1.set_title('{} Map E/W Marginal FFT'.format(map_name))
    ax1.set_xlabel('Wavenumber (cycles/m)')
    ax1.set_ylabel('Magnitude (nT)')
    ax1.plot(shift_kx, ew_freqs, label='E/W Map Freq Content')
    ax1.legend()
    ax1.grid()

    ns_freqs = np.mean(abs(fft.fftshift(map_fft)), axis=1)
    shift_ky = fft.fftshift(ky)
    ax2.set_title('{} Map N/S Marginal FFT'.format(map_name))
    ax2.set_xlabel('Wavenumber (cycles/m)')
    ax2.set_ylabel('Magnitude (nT)')
    ax2.plot(shift_ky, ns_freqs, label='N/S Map Freq Content')
    ax2.legend()
    ax2.grid()

def ned2body(ned_vecs: np.ndarray,
             eulers:   np.ndarray,
             rot_seq:  str='zyx',
             degrees:  bool=True) -> np.ndarray:
    '''
    Rotate vectors from the NED to sensor body frame
    using an array of euler angles
    
    Parameters
    ----------
    ned_vecs
        Nx3 array of vectors in the NED frame to be
        rotated -> [North, East, Down] <- 1st vector
                   [.      .     .   ]
                   [.      .     .   ]
                   [.      .     .   ] <- Nth vector
    eulers
        Nx3 array of sensor pitch, roll, yaw euler angles
        (rotates sensor frame - _not measurement vector_ - from NED to body)
        where N is the number of vectors
        to be rotated -> [roll, pitch, yaw] <- 1st vector
                         [.     .      .  ]
                         [.     .      .  ]
                         [.     .      .  ] <- Nth vector
    rot_seq
        Rotation sequence for the euler angles
    degrees
        Whether the euler angles are in degrees or not
    
    Returns
    -------
    np.ndarray
        Nx3 array of the original vectors rotated
        into sensor's body frame -> [x, y, z] <- 1st vector
                                    [.  .  .]
                                    [.  .  .]
                                    [.  .  .] <- Nth vector
        where N is the number of vectors
    '''
    
    flpd_angs = np.flip(-eulers, axis=1) # Negate angles because we want to rotate NED vectors into body frame and flip angle order cause scipy rotations are stupid
    dcms      = R.from_euler(rot_seq, flpd_angs, degrees=degrees).as_matrix()
    
    return np.array([dcms[i] @ vec for i, vec in enumerate(ned_vecs)])

def num_to_nan(map:      np.ndarray,
               nan_lims: list=[-1e6, 1e6]) -> np.ndarray:
    '''
    Set all values of a map's data array that exceed the values
    given in `nan_lims` to NaNs
    
    Parameters
    ----------
    map
        MxN magnetic map values
    nan_lims
        Min and max value thresholds for num to NaN
        conversion --> [min, max]
    
    Returns
    -------
    map
        MxN magnetic map values with applied NaN mask
    '''
    
    map.data[np.logical_or(map.data < nan_lims[0],
                           map.data > nan_lims[1])] = np.nan
    
    return map

def stack_bands(bands: list) -> rxr.rioxarray.raster_dataset.xarray.DataArray:
    '''
    Create a combined rioxarray with bands as listed in `bands`
    
    Parameters
    ----------
    bands
        List of rioxarrays whos data/bands are to be combined. Note
        that the coordinates of each rioxarray MUST match for this
        to work
    
    Returns
    -------
    rxr.rioxarray.raster_dataset.xarray.DataArray
        Combined rioxarray
    '''
    
    return xr.concat(bands, dim='band')

def igrf_WGS84(map_WGS84:   rxr.rioxarray.raster_dataset.xarray.DataArray,
               map_alt_m:   float,
               survey_date: Union[dt.date, dt.datetime]) -> np.ndarray:
    '''
    Create a rioxarray object that holds both scalar and vector IGRF
    values that correspond to a given WGS-84 anomaly survey. The
    coordinates will be in lat/lon (dd), and the band list is as
    follows:
    
    - Band 0 --> x/North IGRF Component Values (nT)
    - Band 1 --> y/East IGRF Component Values (nT)
    - Band 2 --> z/Down IGRF Component Values (nT)
    - Band 3 --> f/Magnitude IGRF Values (nT)
    
    Parameters
    ----------
    map_WGS84
        Anomaly map rioxarray with WGS-84 coordinates
    map_alt_m
        Altitude MSL survey was collected (m)
    survey_date
        Date survey was collected
    
    Returns
    -------
    igrf_map_WGS84
        4 Band rioxarray of vector and scalar IGRF map
        with WGS-84 coordinates
    '''
    
    igrf_Be, igrf_Bn, igrf_Bu = igrf(np.array(map_WGS84.x),
                                     np.array(map_WGS84.y)[:, np.newaxis],
                                     map_alt_m / 1000,
                                     survey_date)
    
    igrf_mag = np.sqrt(igrf_Be**2 + igrf_Bn**2 + igrf_Bu**2)
    
    if map_WGS84.data.shape[0] == 1:
        igrf_Bn_map = deepcopy(map_WGS84)
        igrf_Bn_map.data = igrf_Bn
        
        igrf_Be_map = deepcopy(map_WGS84)
        igrf_Be_map.data = igrf_Be
        
        igrf_Bd_map = deepcopy(map_WGS84)
        igrf_Bd_map.data = -igrf_Bu
        
        igrf_mag_map = deepcopy(map_WGS84)
        igrf_mag_map.data = igrf_mag
        
        igrf_map_WGS84 = stack_bands([igrf_Bn_map,
                                      igrf_Be_map,
                                      igrf_Bd_map,
                                      igrf_mag_map])
    
    else:
        igrf_Bn_map = deepcopy(map_WGS84[0])
        igrf_Bn_map.data = igrf_Bn.squeeze()
        
        igrf_Be_map = deepcopy(map_WGS84[0])
        igrf_Be_map.data = igrf_Be.squeeze()
        
        igrf_Bd_map = deepcopy(map_WGS84[0])
        igrf_Bd_map.data = -igrf_Bu.squeeze()
        
        igrf_mag_map = deepcopy(map_WGS84[0])
        igrf_mag_map.data = igrf_mag.squeeze()
        
        igrf_map_WGS84 = stack_bands([igrf_mag_map,
                                      igrf_Bn_map,
                                      igrf_Be_map,
                                      igrf_Bd_map])
    
    return igrf_map_WGS84
    
def upcontinue(map:     np.ndarray,
               delta_x: float,
               delta_y: float,
               delta_h: float) -> np.ndarray:
    '''
    This function upward continues a scalar magnetic anomaly map
    by an altitude different delta_h. 

    Parameters
    ----------
    map
        MxN array of scalar magnetic anomaly map values
    delta_x
        Size of pixels in x-dimension (m)
    delta_y
        Size of pixels in y-dimension (m)
    delta_h
        Number of meters (MSL) to upward continue the original
        map by
    
    Returns
    -------
    upmap
        MxN array of upward continued scalar anomaly map values
    '''
    
    upmap  = deepcopy(map)
    mean   = upmap[~np.isnan(upmap)].mean()
    upmap -= mean # Remove DC bias
    
    kx2d, ky2d = Filters.compute_wavevectors(delta_x,
                                             delta_y,
                                             upmap.shape[1],
                                             upmap.shape[0])

    fftMap    = np.fft.fft2(np.nan_to_num(upmap))
    scaledMap = fftMap * (np.e**(-delta_h * np.sqrt(kx2d**2 +  ky2d**2))) 
    upmap     = np.real(np.fft.ifft2(scaledMap))

    upmap[map == np.nan] = np.nan # Keep original NaNs
    upmap += mean # Keep DC bias

    return upmap

def drape2lvl(drape_map:        np.ndarray,
              drape_heights:    np.ndarray,
              delta_x:          float,
              delta_y:          float,
              cartesian_height: float) -> np.ndarray:
    '''
    This function takes a draped scalar magnetic anomaly map and 
    converts it to a cartesian grid.

    Parameters
    ----------
    drape_map
        MxN array of drape scalar magnetic anomaly map values (nT)
    drape_heights
        MxN array of pixel drape heights MSL (m)
    delta_x
        Size of pixels in x-dimension (m)
    delta_y
        Size of pixels in y-dimension (m)
    cartesian_height
        Height of output grid - must be greater than max(drape_height) MSL (m)
    
    Returns
    -------
    cartesian_grid
        MxN array of level scalar magnetic anomaly map values (nT)
    '''
    
    maxHeight = cartesian_height - np.min(drape_heights[~np.isnan(drape_heights)])
    minHeight = cartesian_height - np.max(drape_heights[~np.isnan(drape_heights)])

    topLim = upcontinue(drape_map,
                        delta_x,
                        delta_y,
                        maxHeight)
    topHeights = drape_heights + maxHeight

    bottomLim = upcontinue(drape_map,
                           delta_x,
                           delta_y,
                           minHeight)
    bottomHeights = drape_heights + minHeight
    
    # Flatten all arrays and then turn them into column vectors
    topLimFlat        = topLim.flatten()[:, np.newaxis]
    topHeightsFlat    = topHeights.flatten()[:, np.newaxis]
    bottomLimFlat     = bottomLim.flatten()[:, np.newaxis]
    bottomHeightsFlat = bottomHeights.flatten()[:, np.newaxis]

    # Horizontally tack the upward continued map readings
    lims    = np.hstack([bottomLimFlat, topLimFlat])
    heights = np.hstack([bottomHeightsFlat, topHeightsFlat])

    cartesian_grid = np.array([np.interp(cartesian_height, heights[i, :], lims[i, :]) for i in range(lims.shape[0])])
    
    return cartesian_grid.reshape(drape_map.shape)

def sample_map(map:  rxr.rioxarray.raster_dataset.xarray.DataArray,
               x:    float,
               y:    float,
               band: int=SCALAR) -> Union[float, np.ndarray]:
    '''
    Evaluate map at a given coordinate using
    the 2D cubic interpolation method. Set either
    x or y to None to get all map values in that
    direction (i.e. y=200 and x=None will return
    an array with all map values in the row where
    the Northing=200m)

    Parameters
    ----------
    map
        DataArray of the magnetic map
    x
        x location of the point on
        the map to estimate/interpolate in (m)
    y
        y location of the point on
        the map to estimate/interpolate (m)
    band
        Index of the band to interpolate
    
    Returns
    -------
    float | np.array:
        Estimated/interpolated magnetic
        field strength(s) of the map at the
        given coordinate and band (nT)
    '''
    
    if x is None and y is not None: # Grab row of data
        return map[band].interp(y=y, method='cubic').data.squeeze()
    
    elif x is not None and y is None: # Grab row of data
        return map[band].interp(x=x, method='cubic').data.squeeze()
    
    elif x is not None and y is not None: # Grab single sample
        return map[band].interp(x=x, y=y, method='cubic').data.item()

def find_corrugation_fom(map:     rxr.rioxarray.raster_dataset.xarray.DataArray,
                         cutoffs: list[float],
                         dp:      float,
                         band:    int=SCALAR,
                         axis:    int=1) -> float:
    '''
    Estmiate the severity of corrugation present in a given map
    by producing a FOM (Figure of Merit) based on 
    
    Parameters
    ----------
    map
        DataArray of the magnetic map
    cutoffs
        List of wavelength cutoffs in the tie line
        direction -> [lowest wavelength (m), highest wavelength (m)]
    dp
        Pixel length in the tie line direction (m)
    band
        Band of the map DataArray of which to calculate the FOM
    axis
        Array axis in the tie line direction (i.e. if the map is
        North up and the tie lines are in the East direction,
        `axis` should be set to 1)
    
    Returns
    -------
    corrugation_fom
        A figure of merit where the larger the value, the more corrugation is
        present in the given map
    '''
    
    map_cpy = deepcopy(map)
    data    = map_cpy[band].data
    
    high_cutoff = 1 / cutoffs[0]
    low_cutoff  = 1 / cutoffs[1]
    
    if axis == 1:
        rev_axis = 0
    else:
        rev_axis = 1
    
    data_zero_mean = data - data.mean()
    data_hpf       = Filters.hpf(data_zero_mean, low_cutoff, dp, 10, axis)
    hpf_zero_mean  = data_hpf - data_hpf.mean()
    hpf_compressed = np.mean(hpf_zero_mean, axis=rev_axis)
    
    hpf_compressed_zero_mean = hpf_compressed - hpf_compressed.mean()
    hpf_compressed_fft       = np.fft.fft(np.nan_to_num(hpf_compressed_zero_mean))
    
    fft_mag   = np.abs(hpf_compressed_fft)
    n         = hpf_compressed_zero_mean.size
    fft_mag   = fft_mag[:int(n/2)]
    fft_freqs = np.fft.fftfreq(n, d=dp)[:int(n/2)] # Only use positive freqs
    f         = interp1d(fft_freqs, fft_mag)
    
    corrugation_fom = f(np.linspace(low_cutoff, high_cutoff, 100)).max()
    
    return corrugation_fom

def add_kml_flight_path(kml:         Kml,
                        flight_path: np.ndarray=None) -> Kml:
    '''
    Credits:
        https://simplekml.readthedocs.io/en/latest/geometries.html#gxtrack
    
    Using the given flight path, add a gxtrack to the given Kml object
    
    Parameters
    ----------
    kml
        Kml object
    flight_path
        Nx4 array of timestamped survey sample
        geolocations -> [latitude (dd), longitude (dd), altitude above MSL (m), UTC timestamp (s)]
    
    Returns
    -------
    kml_cpy
        An updated Kml object with the flightpath added (if given)
    '''
    
    kml_cpy = deepcopy(kml)
    
    if flight_path is not None:
        lats       = flight_path[:, 0]
        lons       = flight_path[:, 1]
        alts       = flight_path[:, 2]
        timestamps = flight_path[:, 3]
        datetimes  = [dt.datetime.fromtimestamp(stamp).strftime('%Y-%m-%dT%H:%M:%S.%fZ') for stamp in timestamps]
        
        when  = list(datetimes)
        coord = list(zip(lons,
                         lats,
                         alts))
        
        trk = kml_cpy.newgxtrack(name='Flight Path')
        
        trk.newwhen(when)
        trk.newgxcoord(coord)
        
        trk.altitudemode = 'absolute'
        
        trk.stylemap.normalstyle.iconstyle.icon.href    = 'http://earth.google.com/images/kml-icons/track-directional/track-0.png'
        trk.stylemap.normalstyle.linestyle.color        = '99ffac59'
        trk.stylemap.normalstyle.linestyle.width        = 6
        trk.stylemap.highlightstyle.iconstyle.icon.href = 'http://earth.google.com/images/kml-icons/track-directional/track-0.png'
        trk.stylemap.highlightstyle.iconstyle.scale     = 1.2
        trk.stylemap.highlightstyle.linestyle.color     = '99ffac59'
        trk.stylemap.highlightstyle.linestyle.width     = 8
    
    return kml_cpy

def add_kml_areas(kml:        Kml,
                  area_polys: list[dict]=None) -> Kml:
    '''
    Add the sub-survey area polygons to the given Kml object
    
    Parameters
    ----------
    kml
        Kml object
    area_polys
        List of dicts where each dict corresponds to
        a single sub-survey area with the following
        contents:
        
        - NAME:    str         - Name of sub-survey area
        - FL_DIR:  float       - Direction (degrees) of flight
                                 lines in the area (must be
                                 within the range [0, 180))
        - FL_DIST: float       - Distance (m) between flight lines
        - TL_DIR:  float       - Direction (degrees) of tie
                                 lines in the area (must be
                                 within the range [0, 180)). Set
                                 to -90 if tie lines not flown in
                                 the area
        - TL_DIST: float       - Distance (m) between tie lines.
                                 Set to 0 if tie lines not flown in
                                 the area
        - LAT:     list[float] - Boundary latitudes (dd)
        - LONG:    list[float] - Boundary longitudes (dd)
        - ALT:     list[float] - Boundary altitudes (m) above MSL
        
        Default value of None will cause a single polygon
        generated with boundary points at the map extent
        with the average altitude of the pixel points. Line
        directions will be set to -90 and line distances
        will be set to 0
    
    Returns
    -------
    kml_cpy
        An updated Kml object with the sub-survey area polygons added (if given)
    '''
    
    kml_cpy = deepcopy(kml)
    
    if area_polys is not None:
        fol = kml_cpy.newfolder(name='SubSurveyAreas')
        
        for area in area_polys:
            name     = area['NAME']
            fl_dir   = area['FL_DIR'] % 180.0 # [0, 180)
            fl_dist  = area['FL_DIST']
            tl_dir   = area['TL_DIR'] % 180.0 # [0, 180)
            tl_dist  = area['TL_DIST']
            lats     = area['LAT']
            lons     = area['LONG']
            alts     = area['ALT']
            avg_alts = np.array(alts).mean()
            
            if np.isnan(fl_dir):
                fl_dir = -90
            
            if np.isnan(tl_dir):
                tl_dir = -90
            
            coords = list(zip(lons,
                              lats,
                              alts))
            
            pol = fol.newpolygon(name=name)
            
            pol.outerboundaryis = coords
            pol.altitudemode    = 'absolute'
            pol.description     = 'FL Dir: {}°, FL Dist: {}m, TL Dir: {}°, TL Dist: {}m, Alt: {}m above MSL'.format(fl_dir,
                                                                                                                    fl_dist,
                                                                                                                    tl_dir,
                                                                                                                    tl_dist,
                                                                                                                    avg_alts)
            pol.style.linestyle.color = Color.green
            pol.style.linestyle.width = 5
            pol.style.polystyle.color = Color.changealphaint(100, Color.green)
    
    return kml_cpy

def add_kml_features(kml:      Kml,
                     osm_path: str=None) -> Kml:
    '''
    Using OpenSourceMap (OSM), add road/power line linestrings and
    substation polygons within the survey extent to the given
    Kml object
    
    Parameters
    ----------
    kml
        Kml object
    osm_path
        Either path to an OpenStreetMap *.osm XML file or valid
        OpenStreetMap API query URL. Devault value of None
        will cause the function to query the OpenStreetMap API
        with a bbox set to the map extent
    
    Returns
    -------
    kml_cpy
        An updated Kml object with the road/power line
        linestrings and substation polygons added (if valid OSM
        path given)
    '''
    
    kml_cpy = deepcopy(kml)
    
    if osm_path is not None:
        scraper = GeoScraper()
        
        if uri_validator(osm_path):
            scraper.from_url(osm_path)
        else:
            scraper.from_file(osm_path)
        
        kml_roads        = kml_cpy.newmultigeometry(name='Roads')
        kml_power_lines  = kml_cpy.newmultigeometry(name='PowerLines')
        kml_sub_stations = kml_cpy.newmultigeometry(name='Substations')
        
        for road in scraper.highways():
            kml_roads.newlinestring(coords=road['coords'])
        
        for power in scraper.power_locations():
            if 'line' in power['tag_vals']:
                kml_power_lines.newlinestring(coords=power['coords'])
                
            elif 'substation' in power['tag_vals']:
                kml_sub_stations.newpolygon(outerboundaryis=power['coords'])
        
        kml_roads.style.linestyle.color        = Color.black
        kml_roads.style.linestyle.width        = 2
        kml_power_lines.style.linestyle.color  = Color.yellow
        kml_power_lines.style.linestyle.width  = 2
        kml_sub_stations.style.polystyle.color = Color.orange
    
    return kml_cpy

def read_map_metadata(map_fname: str) -> dict:
    '''
    Read map GeoTIFF metadata fields and return as a dictionary with
    the following top-level keys:
    
    - AREA_OR_POINT
    - CSV
    - Description
    - ExtentDD
    - FinalFiltCut
    - FinalFiltOrder
    - InterpType
    - KML
    - LevelType
    - POC
    - SampleDistM
    - ScalarSensorVar
    - ScalarType
    - SurveyDateUTC
    - TLCoeffs
    - TLCoeffTypes
    - VectorSensorVar
    - VectorType
    - xResolutionM
    - yResolutionM
    - Band_1
    - Band_2
    - Band_3
    - Band_4
    - Band_5
    - Band_6
    - Band_7
    - Band_8
    
    Parameters
    ----------
    map_fname
        File path/name to the map GeoTIFF
    
    Returns
    -------
    metadata
        Dictionary containing all metadata fields of the
        given map GeoTIFF
    '''
    
    gdal.UseExceptions()
    map = gdal.Open(map_fname)
    
    metadata = map.GetMetadata()
    
    metadata['Band_1'] = map.GetRasterBand(1).GetMetadata()
    metadata['Band_2'] = map.GetRasterBand(2).GetMetadata()
    metadata['Band_3'] = map.GetRasterBand(3).GetMetadata()
    metadata['Band_4'] = map.GetRasterBand(4).GetMetadata()
    metadata['Band_5'] = map.GetRasterBand(5).GetMetadata()
    metadata['Band_6'] = map.GetRasterBand(6).GetMetadata()
    metadata['Band_7'] = map.GetRasterBand(7).GetMetadata()
    metadata['Band_8'] = map.GetRasterBand(8).GetMetadata()
    
    return metadata

def export_map(out_dir:          str,
               location:         str,
               date:             Union[dt.date, dt.datetime],
               lats:             np.ndarray,
               lons:             np.ndarray,
               scalar:           np.ndarray,
               heights:          Union[np.ndarray, float],
               process_df:       pd.DataFrame,
               process_app:      str,
               stds:             np.ndarray=None,
               vector:           np.ndarray=None,
               scalar_type:      str='Not Given',
               vector_type:      str='Not Given',
               scalar_var:       float=np.nan,
               vector_var:       float=np.nan,
               poc:              str='Not Given',
               flight_path:      np.ndarray=None,
               area_polys:       list[dict]=None,
               osm_path:         str=None,
               level_type:       str='Not Given',
               tl_coeff_types:   list=[],
               tl_coeffs:        np.ndarray=np.array([]),
               interp_type:      str='Not Given',
               final_filt_cut:   float=0,
               final_filt_order: int=1) -> rxr.rioxarray.raster_dataset.xarray.DataArray:
    '''
    Credits:
        https://stackoverflow.com/a/33950009/9860973
    
    Exports survey data to a multi-band GeoTIFF file in WGS-84
    coordinates. Bands include:
    
    - Band 0: Scalar anomaly values (nT)
    - Band 1: x/North anomaly vector values (nT)
    - Band 2: y/East anomaly vector values (nT)
    - Band 3: z/Down anomaly vector values (nT)
    - Band 4: Pixel height values (m)
    - Band 5: Pixel scalar standard deviation values
    - Band 6: Pixel scalar x/North gradient values (nT)
    - Band 7: Pixel scalar y/East gradient values (nT)
    
    **NOTE**: rioxarray can't read GeoTIFF metadata correctly. In
    order to read and parse both the top-level and band-level
    metadata, you must use gdal like so::
    
    >>> from pprint import pprint
    >>> from osgeo import gdal
    >>> gdal.UseExceptions()
    >>> map = gdal.Open('map.tif') # Read in map
    >>> pprint(map.GetMetadata()) # Print top-level metadata
    >>> pprint(map.GetRasterBand(1).GetMetadata()) # Print first band metadata
    
    Parameters
    ----------
    out_dir
        Path to directory where the GeoTIFF will be exported to
    location
        Description of survey area
    date
        Date of when the survey was collected (UTC)
    lats
        1xM array of pixel latitudes (dd)
    lons
        1xN array of pixel longitudes (dd)
    scalar
        MxN scalar magnetic anomaly values (nT)
    heights
        MxN array of heights (if map is drape) or float
        map height MSL (m)
    process_df
        Pandas DataFrame of all pertinent survey data points and
        data processing steps. Minimum required columns include:
        
        - TIMESTAMP:  UTC timestamps (s)
        - LAT:        Latitudes (dd)
        - LONG:       Longitudes (dd)
        - ALT:        Altitudes above MSL (m)
        - DC_X:       Direction cosine X-Components (n/a)
        - DC_Y:       Direction cosine Y-Components (n/a)
        - DC_Z:       Direction cosine Z-Components (n/a)
        - F:          Raw scalar measurements (nT)
        - F_CAL:      Calibrated scalar measurements (nT)
        - F_CAL_IGRF: Calibrated scalar measurements without
                      core field (nT)
        - F_CAL_IGRF_TEMPORAL: Calibrated scalar measurements
                               without core field or temporal
                               corrections (nT)
        - F_CAL_IGRF_TEMPORAL_FILT: Calibrated scalar measurements
                                    without core field or temporal
                                    corrections after low pass
                                    filtering (nT)
        - F_CAL_IGRF_TEMPORAL_FILT_LEVEL: Calibrated scalar
                                          measurements without
                                          core field or temporal
                                          corrections after low
                                          pass filtering and map
                                          leveling (nT)
        
    process_app
        String describing the application name and version used
        to generate the map file
    stds
        Pixel standard deviation values
    vector
        3xMxN array of NED vector magnetic anomaly values
        (first page is x/North values, etc.) (nT)
    scalar_type
        String describing the make/model/type of scalar
        magnetometer used
    vector_type
        String describing the make/model/type of vector
        magnetometer used
    scalar_var
        Scalar magnetometer's assessed noise variance
    vector_var
        Vector magnetometer's assessed noise variance
    poc
        Point of contact information
    flight_path
        Nx4 array of timestamped survey sample
        geolocations -> [latitude (dd), longitude (dd), altitude above MSL (m), UTC timestamp (s)]
    area_polys
        List of dicts where each dict corresponds to
        a single sub-survey area with the following
        contents:
        
        - NAME:    str         - Name of sub-survey area
        - FL_DIR:  float       - Direction (degrees) of flight
                                 lines in the area (must be
                                 within the range [0, 180))
        - FL_DIST: float       - Distance (m) between flight lines
        - TL_DIR:  float       - Direction (degrees) of tie
                                 lines in the area (must be
                                 within the range [0, 180)). Set
                                 to -90 if tie lines not flown in
                                 the area
        - TL_DIST: float       - Distance (m) between tie lines.
                                 Set to 0 if tie lines not flown in
                                 the area
        - LAT:     list[float] - Boundary latitudes (dd)
        - LONG:    list[float] - Boundary longitudes (dd)
        - ALT:     list[float] - Boundary altitudes (m) above MSL
        
        Default value of None will cause a single polygon
        generated with boundary points at the map extent
        with the average altitude of the pixel points. Line
        directions will be set to -90 and line distances
        will be set to 0
        
    osm_path
        Either path to an OpenStreetMap *.osm XML file or valid
        OpenStreetMap API query URL. Devault value of None
        will cause the function to query the OpenStreetMap API
        with a bbox set to the map extent and a value of -1 will
        skip the OpenStreetMap data processing completely
    level_type
        String describing how the flight lines were leveled
    tl_coeff_types
        List of TL coefficient types used in the same order as
        listed in `tl_coeffs`
    tl_coeffs
        1xN array of TL coefficients in teh same order as listed
        in `tl_coeff_types`
    interp_type
        String describing algorithm used to interpolate pixel values
    final_filt_cut
        Cutoff wavelength of the 2D LPF applied to the interpolated
        scalar pixel values. The output of this filter are the pixel
        values in raster band 1
    final_filt_order
        Order of the 2D LPF applied to the interpolated
        scalar pixel values. The output of this filter are the pixel
        values in raster band 1
    
    Returns
    -------
    rxr.rioxarray.raster_dataset.xarray.DataArray
        Map as a rioxarray
    '''
    
    ##################################################
    # Handle sample distance
    ##################################################
    samp_dist = np.nan
    
    if flight_path is not None:
        lats_1 = flight_path[:, 0]
        lons_1 = flight_path[:, 1]
        
        lats_2 = np.roll(lats_1, -1)
        lons_2 = np.roll(lons_1, -1)
        
        lats_1 = lats_1[1:-1]
        lons_1 = lons_1[1:-1]
                
        lats_2 = lats_2[1:-1]
        lons_2 = lons_2[1:-1]
        
        samp_dist = cu.coord_dist(lats_1,
                                  lons_1,
                                  lats_2,
                                  lons_2).mean() * 1000 # km to m conversion
    
    ##################################################
    # Determine number of bands, image size, and
    # verify rasters and image size all match
    ##################################################
    lats = np.sort(np.unique(lats))
    lons = np.sort(np.unique(lons))
    
    num_bands  = 8 # 1 scalar, 3 vector, 1 height, 1 scalar std, and 2 scalar gradient bands
    image_size = (len(lats), len(lons))
    
    assert scalar.shape == image_size, 'Map scalar dimensions must match coordinate dimensions'
    
    ##################################################
    # Handle various input stuff
    ##################################################
    if type(heights) == np.ndarray:
        height = int(heights.mean())
    else:
        height  = int(heights)
        heights = np.ones(image_size) * height
    
    if vector is None:
        vector = np.ones((3, *image_size)) * np.nan
    
    if stds is None:
        stds = np.ones(scalar.shape) * np.nan
    
    grad_x = np.gradient(scalar, axis=1)
    grad_y = np.gradient(scalar, axis=0)
    
    vec_x = vector[0, :, :]
    vec_y = vector[1, :, :]
    vec_z = vector[2, :, :]
    
    ##################################################
    # Handle pixel and transform stuff
    ##################################################
    ny = image_size[0]
    nx = image_size[1]
    
    extent = [min(lons), min(lats), max(lons), max(lats)]
    xmin, ymin, xmax, ymax = extent
    
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    
    ##################################################
    # Handle file name stuff
    ##################################################
    i = 0
    fname = '{loc}_{height}m_{year}_{month}_{day}_{num}.tiff'.format(loc    = location,
                                                                     height = height,
                                                                     year   = date.year,
                                                                     month  = date.month,
                                                                     day    = date.day,
                                                                     num    = i)
    full_path = join(out_dir, fname)
    
    while exists(full_path):
        i += 1
        fname = '{loc}_{height}m_{year}_{month}_{day}_{num}.tiff'.format(loc    = location,
                                                                         height = height,
                                                                         year   = date.year,
                                                                         month  = date.month,
                                                                         day    = date.day,
                                                                         num    = i)
        full_path = join(out_dir, fname)
    
    ##################################################
    # Handle KML stuff
    ##################################################
    kml_fname     = fname.split('.')[0] + '.kml' # Use same name as GeoTIFF, but .kml extension
    kml_full_path = join(out_dir, kml_fname)
    
    kml = Kml(name = kml_fname,
              open = 1)
    
    kml = add_kml_flight_path(kml, flight_path)
    
    if area_polys is None:
        area_polys = [{'NAME':    'Main Survey Area',
                       'FL_DIR':  np.nan,
                       'FL_DIST': 0,
                       'TL_DIR':  np.nan,
                       'TL_DIST': 0,
                       'LAT':     [ymax, ymax, ymin, ymin],
                       'LONG':    [xmin, xmax, xmax, xmin],
                       'ALT':     [height, height, height, height]}]
    
    kml = add_kml_areas(kml, area_polys)
    
    if osm_path is None:
        url_wiz  = OSM_URL_Wizard()
        osm_path = url_wiz.bbox_url(*extent)
    
    if osm_path != -1:
        kml = add_kml_features(kml, osm_path)
    
    kml_text = kml.kml()
    kml.save(kml_full_path)
    
    ##################################################
    # Handle CSV stuff
    ##################################################
    csv_fname     = fname.split('.')[0] + '.csv' # Use same name as GeoTIFF, but .csv extension
    csv_full_path = join(out_dir, csv_fname)
    process_df.to_csv(csv_full_path, index=False)
    
    ##################################################
    # Handle GeoTIFF stuff
    ##################################################
    dst_ds = gdal.GetDriverByName('GTiff').Create(full_path,
                                                  nx,
                                                  ny,
                                                  num_bands,
                                                  gdal.GDT_Float64)

    # Note that when writing out the bands, we
    # have to reverse the data's row order for
    # the GeoTIFF to be correct
    dst_ds.SetGeoTransform(geotransform)    # Specify coords
    srs = osr.SpatialReference()            # Establish encoding
    srs.ImportFromEPSG(4326)                # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt()) # Export coords to file
    
    dst_ds.SetMetadata({'Description':     'MagNav Aeromagnetic Anomaly Map', # Set general map metadata fields
                        'ProcessingApp':   str(process_app),
                        'SurveyDateUTC':   str(date.isoformat()),
                        'SampleDistM':     str(samp_dist),
                        'xResolutionM':    str(xres),
                        'yResolutionM':    str(yres),
                        'ExtentDD':        str(extent),
                        'ScalarType':      str(scalar_type),
                        'VectorType':      str(vector_type),
                        'ScalarSensorVar': str(scalar_var),
                        'VectorSensorVar': str(vector_var),
                        'POC':             str(poc),
                        'KML':             str(kml_text),
                        'LevelType':       str(level_type),
                        'TLCoeffTypes':    str(tl_coeff_types),
                        'TLCoeffs':        str(tl_coeffs),
                        'CSV':             str(process_df.to_csv(index=False)),
                        'InterpType':      str(interp_type),
                        'FinalFiltCut':    str(final_filt_cut),
                        'FinalFiltOrder':  str(final_filt_order)})

    dst_ds.GetRasterBand(1).WriteArray(scalar[::-1, :])    # Write scalar anomaly values to the raster (nT)
    dst_ds.GetRasterBand(1).SetMetadata({'Type':      'F', # Set band-specific metadata
                                         'Units':     'nT',
                                         'Direction': 'n/a'})
    
    dst_ds.GetRasterBand(2).WriteArray(vec_x[::-1, :])     # Write x/North anomaly vector values to the raster (nT)
    dst_ds.GetRasterBand(2).SetMetadata({'Type':      'X', # Set band-specific metadata
                                         'Units':     'nT',
                                         'Direction': 'North'})
    
    dst_ds.GetRasterBand(3).WriteArray(vec_y[::-1, :])     # Write y/East anomaly vector values to the raster (nT)
    dst_ds.GetRasterBand(3).SetMetadata({'Type':      'Y', # Set band-specific metadata
                                         'Units':     'nT',
                                         'Direction': 'East'})
    
    dst_ds.GetRasterBand(4).WriteArray(vec_z[::-1, :])     # Write z/Down anomaly vector values to the raster (nT)
    dst_ds.GetRasterBand(4).SetMetadata({'Type':      'Z', # Set band-specific metadata
                                         'Units':     'nT',
                                         'Direction': 'Down'})
    
    dst_ds.GetRasterBand(5).WriteArray(heights[::-1, :])     # Write height MSL values to the raster (m)
    dst_ds.GetRasterBand(5).SetMetadata({'Type':      'ALT', # Set band-specific metadata
                                         'Units':     'm MSL',
                                         'Direction': 'n/a'})
    
    dst_ds.GetRasterBand(6).WriteArray(stds[::-1, :])        # Write standard deviation values to the raster (nT)
    dst_ds.GetRasterBand(6).SetMetadata({'Type':      'STD', # Set band-specific metadata
                                         'Units':     'nT',
                                         'Direction': 'n/a'})
    
    dst_ds.GetRasterBand(7).WriteArray(grad_x[::-1, :])     # Write scalar x-gradient values to the raster (nT)
    dst_ds.GetRasterBand(7).SetMetadata({'Type':      'dX', # Set band-specific metadata
                                         'Units':     'nT',
                                         'Direction': 'East'})
    
    dst_ds.GetRasterBand(8).WriteArray(grad_y[::-1, :])     # Write scalar y-gradient values to the raster (nT)
    dst_ds.GetRasterBand(8).SetMetadata({'Type':      'dY', # Set band-specific metadata
                                         'Units':     'nT',
                                         'Direction': 'North'})
    
    dst_ds.FlushCache() # Write to disk
    dst_ds = None
    
    while not exists(full_path):
        pass
    
    return rxr.open_rasterio(full_path)