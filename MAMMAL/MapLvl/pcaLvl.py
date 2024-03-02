import sys
from copy import deepcopy
from os.path import dirname

import numpy as np
import pandas as pd
from scipy import interpolate

SRC_DIR = dirname(dirname(__file__))
sys.path.append(SRC_DIR)

from Utils import coordinateUtils as cu


def pca_lvl(survey_df:      pd.DataFrame,
            num_ptls:       int=5,
            ptl_locs:       np.ndarray=None,
            percent_thresh: float=0.85) -> pd.DataFrame:
    '''
    Credits:
        - Zhang, Q., Peng, C., Lu, Y., Wang, H., & Zhu, K. (2018).
        Airborne electromagnetic data levelling using principal
        component analysis based on flight line difference.
        Journal of Applied Geophysics, 151, 290–297.
        https://doi.org/10.1016/j.jappgeo.2018.02.023
    
    Level survey flight lines using principal component analysis
    on pseudo tie lines (ptls) of differenced flight line data
    
    Parameters
    ----------
    survey_df
        Dataframe containing magnetic anomaly flight data from the survey
        Minimum required columns include:
        
        - LAT
        - LONG
        - F (magnetic anomaly scalar values, NOT RAW!)
        - LINE
        - LINE_TYPE
        
    num_ptls
        Number of pseudo tie lines to use for leveling
    ptl_locs
        Kx1 array of relative locations of the pseudo tie lines.
        Each relative location is a percent distance from the
        edge of the survey area where the first sample of the
        first flight line was taken. For example, in order to
        set two pseudo tie lines at opposite ends of the dataset,
        set ptl_locs = [0.0, 1.0]. Values must be between 0 and 1
    percent_thresh
        Value ranging from 0 to 1 (not inclusive) that
        specifies the minimum cumulative contribution
        rate of the components to use for the PCA
        reconstruction
    
    Returns
    -------
    lvld_survey_df
        Dataframe containing leveled flight data from the survey
    '''
    
    assert num_ptls >= 2, 'Must use at least 2 pseudo tie-lines'
    assert (ptl_locs is None) or \
           ((ptl_locs.min() >= 0) and (ptl_locs.max() <= 1)), 'Invalid pseudo tie-line locations detected'

    if ptl_locs is not None:
        assert num_ptls == len(ptl_locs), 'Number of pseudo tie-lines must equal number of given pseudo tie-line locations indicies'
    
    # Calculate coordinate to coordinate azimuths for all data points
    num_samples = len(survey_df.F)
    azimuths    = np.zeros(num_samples)
    
    orig_pts   = np.hstack([np.array(survey_df.LAT)[:, np.newaxis],
                            np.array(survey_df.LONG)[:, np.newaxis]])
    rolled_pts = np.roll(orig_pts, 1, axis=0)
    azimuths   = cu.coord_bearing(orig_pts[:, 0], orig_pts[:, 1], rolled_pts[:, 0], rolled_pts[:, 1])
    
    # Sample Flight Lines at Pseudo Tie Line Locations Before Leveling
    unique_fl_nums = np.unique(survey_df.LINE[survey_df.LINE_TYPE == 1])
    
    avg_line_azs   = np.zeros(len(unique_fl_nums))
    avg_line_lens  = np.zeros(len(unique_fl_nums))
    avg_samp_dists = np.zeros(len(unique_fl_nums))

    for i, line in enumerate(unique_fl_nums):
        mask = survey_df.LINE == line
        
        lats = np.array(survey_df.LAT[mask])
        lons = np.array(survey_df.LONG[mask])
        azs  = np.array(azimuths[mask])[1:-1]
        
        avg_line_azs[i]   = azs.mean()
        avg_line_lens[i]  = cu.coord_dist(lats[0], lons[0], lats[-1], lons[-1])
        avg_samp_dists[i] = avg_line_lens[i] / len(lats)

    avg_line_az  = (avg_line_azs % 180).mean() # Keep flight line azimuth between 0 to 180 degrees for simplicity
    avg_line_len = avg_line_lens.mean()

    ptl_locs = (ptl_locs * avg_line_len) - (avg_line_len / 2.0)

    sample_shape = (len(unique_fl_nums), len(ptl_locs))

    sampled_fl_scalar = np.zeros(sample_shape)
    sampled_fl_lats   = np.zeros(sample_shape)
    sampled_fl_lons   = np.zeros(sample_shape)

    for i, loc in enumerate(ptl_locs):
        for j, line in enumerate(unique_fl_nums):
            mask = survey_df.LINE == line
            
            scalar = np.array(survey_df.F[mask], dtype=np.float64)
            lats   = np.array(survey_df.LAT[mask], dtype=np.float64)
            lons   = np.array(survey_df.LONG[mask], dtype=np.float64)
            
            center_lat = (lats.max() + lats.min()) / 2
            center_lon = (lons.max() + lons.min()) / 2
            
            dist = loc
            az   = avg_line_az
            
            if dist < 0: # Negative distance means 180 degree azimuth shift
                dist *= -1
                az   += 180
            
            sampled_lat, sampled_lon = cu.coord_coord(center_lat, center_lon, dist, az)
            
            rbfi_scalar = interpolate.Rbf(lons, # TODO: Pull this out of the loop somehow?
                                          lats,
                                          scalar,
                                          function='linear',
                                          smooth=0)
        
            interp_scalar = rbfi_scalar(sampled_lon, sampled_lat)
            
            sampled_fl_scalar[j, i] = interp_scalar
            sampled_fl_lats[j, i]   = sampled_lat
            sampled_fl_lons[j, i]   = sampled_lon
    
    # Find the approximate error between adjacent flight lines
    fl_diff = np.zeros(sample_shape).T
    fl_diff[:, 1:] = (sampled_fl_scalar.T - np.roll(sampled_fl_scalar.T, -1))[:, :-1]

    # Create pseudo tie-lines of the flight line difference data
    ptls = fl_diff

    # Find pseudo tie-line covariance matrix
    diff_cov = np.cov(ptls)

    # Find SVD of pseudo tie-line covariance matrix
    R, eigvals, _ = np.linalg.svd(diff_cov)

    # Find minimum required components for PCA reconstruction
    val_cum_sum       = np.cumsum(np.real(np.abs(eigvals)) / np.sum(np.real(np.abs(eigvals))))
    last_comp_idx     = np.where(val_cum_sum >= percent_thresh)[0].min()
    components        = R.T @ ptls
    reconstruct_comps = components[:last_comp_idx + 1, :]

    # Calculate pseudo tie-line correction terms
    ptl_corrs = R[:, :last_comp_idx + 1] @ reconstruct_comps
    ptl_corrs = np.cumsum(ptl_corrs, axis=1)

    # Interpolate pseudo tie-line correction terms and evaluate flight
    # line corrections at all flight line sample locations and apply
    # interpolated corrections to flight line data and return output
    ptl_lats = sampled_fl_lats
    ptl_lons = sampled_fl_lons
    
    anomaly_F = np.array(survey_df.F)

    for i, fl_num in enumerate(unique_fl_nums):
        mask    = survey_df.LINE == fl_num
        fl_lats = survey_df.loc[mask]['LAT']
        fl_lons = survey_df.loc[mask]['LONG']
        
        rbf = interpolate.Rbf(ptl_lons[i, :],
                              ptl_lats[i, :],
                              ptl_corrs.T[i, :],
                              function='linear',
                              smooth=0)
        interp_ptl_corrs = rbf(fl_lons, fl_lats)
        anomaly_F[mask] += interp_ptl_corrs

    lvld_survey_df      = deepcopy(survey_df)
    lvld_survey_df['F'] = anomaly_F

    return lvld_survey_df