from typing import Union

import numpy as np
from numpy import degrees, radians, sqrt, sin, cos, arcsin, arccos, arctan2, pi
from scipy.spatial import distance


EARTH_RADIUS_KM = 6378.137


def coord_bearing(lat_1: Union[float, np.ndarray],
                  lon_1: Union[float, np.ndarray],
                  lat_2: Union[float, np.ndarray],
                  lon_2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Credits:
        - http://www.movable-type.co.uk/scripts/latlong.html
        - https://github.com/PowerBroker2/WarThunder/blob/master/WarThunder/mapinfo.py
    
    Find the bearing (in degrees) between two lat/lon coordinates (dd)
    
    Parameters
    ----------
    lat_1
        First point's latitude (dd)
    lon_1
        First point's longitude (dd)
    lat_2
        Second point's latitude (dd)
    lon_2
        Second point's longitude (dd)
    
    Returns
    -------
    float | np.ndarray
        Bearing between point 1 and 2 (Degrees)
    '''
    
    deltaLon_r = radians(lon_2 - lon_1)
    lat_1_r    = radians(lat_1)
    lat_2_r    = radians(lat_2)

    x = cos(lat_2_r) * sin(deltaLon_r)
    y = cos(lat_1_r) * sin(lat_2_r) - sin(lat_1_r) * cos(lat_2_r) * cos(deltaLon_r)

    return (degrees(arctan2(x, y)) + 360) % 360

def coord_dist(lat_1: Union[float, np.ndarray],
               lon_1: Union[float, np.ndarray],
               lat_2: Union[float, np.ndarray],
               lon_2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Credits:
        - http://www.movable-type.co.uk/scripts/latlong.html
        - https://github.com/PowerBroker2/WarThunder/blob/master/WarThunder/mapinfo.py
    
    Find the total distance (in km) between two lat/lon coordinates (dd)
    
    Parameters
    ----------
    lat_1
        First point's latitude (dd)
    lon_1
        First point's longitude (dd)
    lat_2
        Second point's latitude (dd)
    lon_2
        Second point's longitude (dd)
    
    Returns
    -------
    float | np.ndarray
        Distance between point 1 and 2 (km)
    '''
    
    lat_1_rad = radians(lat_1)
    lon_1_rad = radians(lon_1)
    lat_2_rad = radians(lat_2)
    lon_2_rad = radians(lon_2)
    
    d_lat = lat_2_rad - lat_1_rad
    d_lon = lon_2_rad - lon_1_rad
    
    a = (sin(d_lat / 2) ** 2) + cos(lat_1_rad) * cos(lat_2_rad) * (sin(d_lon / 2) ** 2)
    
    return 2 * EARTH_RADIUS_KM * arctan2(sqrt(a), sqrt(1 - a))

def coord_coord(lat:     Union[float, np.ndarray],
                lon:     Union[float, np.ndarray],
                dist:    Union[float, np.ndarray],
                bearing: Union[float, np.ndarray]) -> np.ndarray:
    '''
    Credits:
        - http://www.movable-type.co.uk/scripts/latlong.html
        - https://github.com/PowerBroker2/WarThunder/blob/master/WarThunder/mapinfo.py
    
    Finds the lat/lon coordinates "dist" km away from the given "lat" and "lon"
    coordinate along the given compass "bearing"
    
    Parameters
    ----------
    lat
        First point's latitude (dd)
    lon
        First point's longitude (dd)
    dist
        Distance in km the second point should be from the first point
    bearing
        Bearing in degrees from the first point to the second
    
    Returns
    -------
    np.ndarray
        Latitude and longitude in DD of the second point -> [lat (dd), lon (dd)]
    '''
    
    brng  = radians(bearing)
    lat_1 = radians(lat)
    lon_1 = radians(lon)
    
    lat_2 = arcsin(sin(lat_1) * cos(dist / EARTH_RADIUS_KM) + cos(lat_1) * sin(dist / EARTH_RADIUS_KM) * cos(brng))
    lon_2 = lon_1 + arctan2(sin(brng) * sin(dist / EARTH_RADIUS_KM) * cos(lat_1), cos(dist / EARTH_RADIUS_KM) - sin(lat_1) * sin(lat_2))
    
    try:
        return np.hstack([degrees(lat_2)[:, np.newaxis], degrees(lon_2)[:, np.newaxis]])
    except IndexError:
        return np.array([degrees(lat_2), degrees(lon_2)])

def coord_intercept(lat_1:     Union[float, np.ndarray],
                    lon_1:     Union[float, np.ndarray],
                    bearing_1: Union[float, np.ndarray],
                    lat_2:     Union[float, np.ndarray],
                    lon_2:     Union[float, np.ndarray],
                    bearing_2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Credits:
        - http://www.movable-type.co.uk/scripts/latlong.html
    
    Given two points and two bearings, find the great circle intersection coordinate
    
    Parameters
    ----------
    lat_1
        First point's latitude (dd)
    lon_1
        First point's longitude (dd)
    bearing_1
        First point's bearing to intersection point (degrees)
    lat_2
        Second point's latitude (dd)
    lon_2
        Second point's longitude (dd)
    bearing_2
        Second point's bearing to intersection point (degrees)
    
    Returns
    -------
    np.ndarray
        Latitude and longitude in DD of the intersection point -> [lat (dd), lon (dd)]
    '''
    
    brng_1 = radians(bearing_1)
    lat_1  = radians(lat_1)
    lon_1  = radians(lon_1)
    
    brng_2 = radians(bearing_2)
    lat_2  = radians(lat_2)
    lon_2  = radians(lon_2)
    
    del_lat = lat_2 - lat_1
    del_lon = lon_2 - lon_1
    
    del_12  = 2 * arcsin(sqrt(sin(del_lat / 2.0)**2 + (cos(lat_1) * cos(lat_2) * (sin(del_lon / 2.0)**2))))
    theta_a = arccos((sin(lat_2) - (sin(lat_1) * cos(del_12))) / (sin(del_12) * cos(lat_1)))
    theta_b = arccos((sin(lat_1) - (sin(lat_2) * cos(del_12))) / (sin(del_12) * cos(lat_2)))
    
    if sin(del_lon) > 0:
        theta_12 = theta_a
        theta_21 = (2 * pi) - theta_b
    
    else:
        theta_12 = (2 * pi) - theta_a
        theta_21 = theta_b
    
    alpha_1 = brng_1 - theta_12
    alpha_2 = theta_21 - brng_2
    
    alpha_3    = arccos((-cos(alpha_1) * cos(alpha_2)) + (sin(alpha_1) * sin(alpha_2) * cos(del_12)))
    del_13     = arctan2(sin(del_12) * sin(alpha_1) * sin(alpha_2), cos(alpha_2) + (cos(alpha_1) * cos(alpha_3)))
    lat_3      = arcsin((sin(lat_1) * cos(del_13)) + (cos(lat_1) * sin(del_13) * cos(brng_1)))
    del_lon_13 = arctan2(sin(brng_1) * sin(del_13) * cos(lat_1), cos(del_13) - (sin(lat_1) * sin(lat_3)))
    lon_3      = lon_1 + del_lon_13
    
    try:
        return np.hstack([degrees(lat_3)[:, np.newaxis], degrees(lon_3)[:, np.newaxis]])
    except IndexError:
        return np.array([degrees(lat_3), degrees(lon_3)])

def path_intersection(line_1_lats: np.ndarray,
                      line_1_lons: np.ndarray,
                      line_2_lats: np.ndarray,
                      line_2_lons: np.ndarray) -> np.ndarray:
    '''
    Given two path, find the great circle intersection coordinate
    
    Parameters
    ----------
    lat_1
        First path's latitude (dd)
    lon_1
        First path's longitude (dd)
    lat_2
        Second path's latitude (dd)
    lon_2
        Second path's longitude (dd)
    
    Returns
    -------
    np.ndarray
        Latitude and longitude in DD of the intersection point -> [lat (dd), lon (dd)]
    '''
    
    if type(line_1_lats) is not np.ndarray:
        line_1_lats = np.array(line_1_lats)
    
    if type(line_1_lons) is not np.ndarray:
        line_1_lons = np.array(line_1_lons)
    
    if type(line_2_lats) is not np.ndarray:
        line_2_lats = np.array(line_2_lats)
    
    if type(line_2_lons) is not np.ndarray:
        line_2_lons = np.array(line_2_lons)
    
    line_1_coords = np.hstack([np.array(line_1_lons)[:, np.newaxis],
                               np.array(line_1_lats)[:, np.newaxis]])
    line_2_coords = np.hstack([np.array(line_2_lons)[:, np.newaxis],
                               np.array(line_2_lats)[:, np.newaxis]])
    
    dists = distance.cdist(line_1_coords,
                           line_2_coords,
                           'euclidean')

    row, col = np.where(dists == dists.min())
    row_idx  = row[0]
    col_idx  = col[0]
    
    
    line_1_lat_1 = line_1_lats[row_idx]
    line_1_lon_1 = line_1_lons[row_idx]
    
    try:
        line_1_lat_2 = line_1_lats[row_idx + 1]
        line_1_lon_2 = line_1_lons[row_idx + 1]
    except IndexError:
        line_1_lat_2 = line_1_lats[row_idx - 1]
        line_1_lon_2 = line_1_lons[row_idx - 1]
    
    line_2_lat_1 = line_2_lats[col_idx]
    line_2_lon_1 = line_2_lons[col_idx]
    
    try:
        line_2_lat_2 = line_2_lats[col_idx + 1]
        line_2_lon_2 = line_2_lons[col_idx + 1]
    except IndexError:
        line_2_lat_2 = line_2_lats[col_idx - 1]
        line_2_lon_2 = line_2_lons[col_idx - 1]
    
    line_1_bearing_1 = coord_bearing(line_1_lat_1,
                                     line_1_lon_1,
                                     line_1_lat_2,
                                     line_1_lon_2)
    line_1_bearing_2 = (line_1_bearing_1 + 180) % 360
    
    line_2_bearing_1 = coord_bearing(line_2_lat_1,
                                     line_2_lon_1,
                                     line_2_lat_2,
                                     line_2_lon_2)
    line_2_bearing_2 = (line_2_bearing_1 + 180) % 360
    
    intercept_coord_1 = coord_intercept(line_1_lat_1,
                                        line_1_lon_1,
                                        line_1_bearing_1,
                                        line_2_lat_1,
                                        line_2_lon_1,
                                        line_2_bearing_1)
    intercept_coord_2 = coord_intercept(line_1_lat_1,
                                        line_1_lon_1,
                                        line_1_bearing_2,
                                        line_2_lat_1,
                                        line_2_lon_1,
                                        line_2_bearing_2)

    test_dist_1 = distance.cdist([[line_1_lat_1, line_1_lon_1]],
                                 [intercept_coord_1],
                                 'euclidean').item()
    test_dist_2 = distance.cdist([[line_1_lat_1, line_1_lon_1]],
                                 [intercept_coord_2],
                                 'euclidean').item()
    
    if test_dist_1 < test_dist_2:
        return intercept_coord_1
    return intercept_coord_2

def arc_angle(dist: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Find Arc angle between two points at the given distance
    on Earth's surface (or near it)
    
    Parameters
    ----------
    dist
        Arc distance between two points on Earth (km)
    
    Returns
    -------
    float | np.ndarray
        Arc angle between two points at the given distance (Degrees)
    '''
    
    return degrees(dist / EARTH_RADIUS_KM)