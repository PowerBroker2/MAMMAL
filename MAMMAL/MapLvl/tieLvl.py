from copy import deepcopy
import sys
from os.path import dirname, realpath

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.stats as stats
from scipy.spatial import distance
from scipy import interpolate

sys.path.append(dirname(realpath(__file__)))
sys.path.append(dirname(dirname(realpath(__file__))))

from Utils import coordinateUtils as cu


def tie_lvl(survey_df: pd.DataFrame,
            approach:  str='lsq') -> pd.DataFrame:
    '''
    
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    '''
    
    fl_mask        = survey_df.LINE_TYPE == 1
    unique_fl_nums = np.unique(survey_df.LINE[fl_mask])
    num_fls        = len(unique_fl_nums)
    fl_scalar      = np.array(survey_df.F[fl_mask],    dtype=np.float64)
    fl_lats        = np.array(survey_df.LAT[fl_mask],  dtype=np.float64)
    fl_lons        = np.array(survey_df.LONG[fl_mask], dtype=np.float64)
    
    tl_mask        = survey_df.LINE_TYPE == 2
    unique_tl_nums = np.unique(survey_df.LINE[tl_mask])
    num_tls        = len(unique_tl_nums)
    tl_scalar      = np.array(survey_df.F[tl_mask],    dtype=np.float64)
    tl_lats        = np.array(survey_df.LAT[tl_mask],  dtype=np.float64)
    tl_lons        = np.array(survey_df.LONG[tl_mask], dtype=np.float64)
    
    # Find intersection coordinates
    num_ints = num_fls * num_tls
    int_lats = np.zeros(num_ints)
    int_lons = np.zeros(num_ints)
    
    combinations = np.array(np.meshgrid(unique_fl_nums, unique_tl_nums)).T.reshape(-1, 2)
    
    for i, combination in enumerate(combinations):
        fl, tl = combination
        
        int_fl_lats = survey_df.LAT[fl_mask & (survey_df.LINE == fl)]
        int_fl_lons = survey_df.LONG[fl_mask & (survey_df.LINE == fl)]
        
        int_tl_lats = survey_df.LAT[tl_mask & (survey_df.LINE == tl)]
        int_tl_lons = survey_df.LONG[tl_mask & (survey_df.LINE == tl)]
        
        int_lat, int_lon = cu.path_intersection(int_fl_lats,
                                                int_fl_lons,
                                                int_tl_lats,
                                                int_tl_lons)
        
        int_lats[i] = int_lat
        int_lons[i] = int_lon
    
    # Interpolate flight and tie lines at the intersection locations
    interp_fl = interpolate.Rbf(fl_lons,
                                fl_lats,
                                fl_scalar,
                                function='linear')
    fl_interp = interp_fl(int_lons, int_lats)
    
    interp_tl = interpolate.Rbf(tl_lons,
                                tl_lats,
                                tl_scalar,
                                function='linear')
    tl_interp = interp_tl(int_lons, int_lats)
    
    # Find the difference between the flight line and interpolated flight line data
    diff      = (fl_interp - tl_interp).reshape(num_fls, num_tls)
    diff_lats = int_lats.reshape(num_fls, num_tls)
    diff_lons = int_lons.reshape(num_fls, num_tls)
    
    lvld_survey_df = deepcopy(survey_df)
    
    if approach.lower() == 'lsq':
        # Find least squares optomized plane of best fit of the difference data at the intercept points
        A = np.hstack([int_lons[:, np.newaxis],
                       int_lats[:, np.newaxis],
                       np.ones((len(int_lons), 1))])
        C, _, _, _ = la.lstsq(A, diff.flatten())
        
        # Find the corrections for all flight line data points based on the optomized plane of best fit
        B = np.hstack([fl_lons[:, np.newaxis],
                       fl_lats[:, np.newaxis],
                       np.ones((len(fl_lons), 1))])
        corrections = B @ C
        
        # Apply the leveling corrections to all flight line data
        lvld_survey_df['F'].loc[fl_mask] = fl_scalar - corrections
    
    else:
        # Find the first order line of best fit of the difference data
        # for each flight line
        for i, k in enumerate(range(diff.shape[0])):
            fl_num         = unique_fl_nums[i]
            fl_samp_mask   = (fl_mask) & (survey_df.LINE == fl_num)
            fl_samp_lats   = survey_df.LAT[fl_samp_mask]
            fl_samp_lons   = survey_df.LONG[fl_samp_mask]
            fl_samp_coords = np.hstack([np.array(fl_samp_lats)[:, np.newaxis],
                                        np.array(fl_samp_lons)[:, np.newaxis]])
            ref_coord      = fl_samp_coords[0]
            fl_samp_dists  = distance.cdist(fl_samp_coords,
                                            [ref_coord],
                                            'euclidean').flatten()
            
            diff_line      = diff[k]
            diff_line_lats = diff_lats[k]
            diff_line_lons = diff_lons[k]
            diff_coords    = np.hstack([diff_line_lats[:, np.newaxis],
                                        diff_line_lons[:, np.newaxis]])
            diff_dists     = distance.cdist(diff_coords,
                                            [ref_coord],
                                            'euclidean').flatten()
            
            res = stats.linregress(diff_dists, diff_line)
            
            corrections = res.intercept + (res.slope * fl_samp_dists)
            
            # Apply the leveling corrections for the given flight line
            lvld_survey_df['F'].loc[fl_samp_mask] = survey_df['F'].loc[fl_samp_mask] - corrections
    
    return lvld_survey_df


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    
    # Flight lines are stacked column-wise:
    flight_lines = np.array([[1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1]])

    lats = np.array([[0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2, 2],
                     [3, 3, 3, 3, 3, 3],
                     [4, 4, 4, 4, 4, 4]])
        
    lons = np.array([[0, 1, 2, 3, 4, 5],
                     [0, 1, 2, 3, 4, 5],
                     [0, 1, 2, 3, 4, 5],
                     [0, 1, 2, 3, 4, 5],
                     [0, 1, 2, 3, 4, 5]])

    tie_lines = np.array([[1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1]])

    tie_lats = np.array([[-0.005, 1, 2, 3, 4.005],
                         [-0.005, 1, 2, 3, 4.005],
                         [-0.005, 1, 2, 3, 4.005]])

    tie_lons = np.array([[0.005, 0.005, 0.005, 0.005, 0.005],
                         [2.005, 2.005, 2.005, 2.005, 2.005],
                         [4.005, 4.005, 4.005, 4.005, 4.005]])
    
    survey_df = pd.DataFrame({'F':         np.hstack([flight_lines.flatten(), tie_lines.flatten()]),
                              'LAT':       np.hstack([lats.flatten(), tie_lats.flatten()]),
                              'LONG':      np.hstack([lons.flatten(), tie_lons.flatten()]),
                              'LINE':      np.hstack([np.tile(np.arange(flight_lines.shape[0]) + 1, [flight_lines.shape[1], 1]).T.flatten(), np.tile(np.arange(tie_lines.shape[0]) + 1, [tie_lines.shape[1], 1]).T.flatten()]),
                              'LINE_TYPE': np.array(np.hstack([np.ones(len(flight_lines.flatten())), np.ones(len(tie_lines.flatten())) * 2]), dtype=int)})
    
    lvld_df = tie_lvl(survey_df = survey_df,
                      approach  ='lobf')
    
    plt.figure()
    plt.scatter(lons, lats, c=flight_lines)
    plt.clim(0, 1)
    
    plt.figure()
    plt.scatter(lvld_df.LONG[lvld_df.LINE_TYPE == 1], lvld_df.LAT[lvld_df.LINE_TYPE == 1], c=lvld_df.F[lvld_df.LINE_TYPE == 1])
    plt.clim(0, 1)
    
    plt.show()