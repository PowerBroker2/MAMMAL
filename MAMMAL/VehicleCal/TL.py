import sys
from os.path import dirname, join, realpath

import numpy as np
import scipy.linalg as la
import scipy.signal as sps
import scipy.interpolate as interp
from sklearn.linear_model import Ridge

sys.path.append(dirname(realpath(__file__)))
sys.path.append(dirname(dirname(realpath(__file__))))

from Utils import Filters


# Enumerate A matrix terms
ALL_TERMS = 0
PERMANENT = 1
INDUCED   = 2
EDDY      = 3


def direction_cosines(vec: np.ndarray) -> np.ndarray:
    '''
    Compute direction cosines from vector magnetometer values
    
    Parameters
    ----------
    vec
        Nx3 array of vector measurements (nT)

    Returns
    -------
    np.ndarray
        Nx3 array of direction cosines
    '''

    magnitude = la.norm(vec, axis=1)[:, np.newaxis] # Use np.newaxis to keep magnitude a col vector
    
    return vec / magnitude

def A_matrix(dir_cos:    np.ndarray,
             b_external: np.ndarray,
             delta_t:    float=1.0,
             terms:      int=ALL_TERMS) -> np.ndarray:
    '''
    Compute the A_matrix from direction cosines. Filtering
    the colums with the input filter.
    
    Parameters
    ----------
    direction_cosines
        Nx3 numpy array of direciton cosines
    b_external
        External mag field magnitude (nT) to use for a-matrix, if none,
        try to compute from average of b_scalar
    terms
        Terms to include in A-matrix. Options include:
        
        - ALL_TERMS
        - PERMANENT
        - INDUCED
        - EDDY

    Returns
    -------
    A

    '''

    if terms == ALL_TERMS:
        terms = [PERMANENT, INDUCED, EDDY]

    A = np.array([]).reshape([len(dir_cos), 0])
    
    # Direction cosines
    cx = dir_cos[:, [0]]
    cy = dir_cos[:, [1]]
    cz = dir_cos[:, [2]]
    
    # Compute the gradient of the direction cosines in the time direction
    t = np.linspace(0, int((A.shape[0] + delta_t) * delta_t), A.shape[0]) # Relative timestamps
    
    cx_prime = interp.splev(t, interp.splrep(t, cx), der=1)[:, np.newaxis]
    cy_prime = interp.splev(t, interp.splrep(t, cy), der=1)[:, np.newaxis]
    cz_prime = interp.splev(t, interp.splrep(t, cz), der=1)[:, np.newaxis]

    if PERMANENT in terms:
        A = np.hstack([A, cx, cy, cz])
    
    if INDUCED in terms:
        A = np.hstack([A, 
                       b_external * cx**2,
                       b_external * cy**2,
                       b_external * cz**2,
                       b_external * cx * cy,
                       b_external * cx * cz,
                       b_external * cy * cz])

    if EDDY in terms:
        A = np.hstack([A,
                       b_external * cx * cx_prime,
                       b_external * cy * cy_prime,
                       b_external * cz * cz_prime,
                       b_external * cx_prime * cy,
                       b_external * cx_prime * cz,
                       b_external * cy_prime * cx,
                       b_external * cy_prime * cz,
                       b_external * cz_prime * cx,
                       b_external * cz_prime * cy])

    return A

def tlc(b_scalar:   np.ndarray,
        b_vector:   np.ndarray,
        delta_t:    float,
        use_filter: bool=False,
        fstart:     float=0.1,
        fstop:      float=1.0,
        order:      int=5,
        b_external: float=None,
        alpha:      float=0,
        fit_intcpt: bool=False,
        terms:      int=ALL_TERMS):
    '''
    Solve for Tolles-Lawson coefficients using all of the input data
    
    Parameters
    ----------
    b_scalar
        Nx1 array of scalar measurements (nT)
    b_vector
        Nx3 array of vector measurements (nT)
    delta_t
        Time between data samples (s)
    use_filter
        Use band pass filter if true
    fstart
        Start frequency for the band (Hz)
    fstop
        Stop frequency for the band (Hz)
    order
        Order of the filter
    b_external
        External mag field magnitude (nT) to use for a-matrix, if none,
        try to compute from average of b_scalar
    alpha
        Ridge regression alpha tuning parameter - set to
        0 to disable ridge regression
    fit_intcpt
        Ridge regression fit intercept flag
    terms
        Terms to include in A-matrix. Options include:
        
        - ALL_TERMS
        - PERMANENT
        - INDUCED
        - EDDY

    Returns
    -------
    np.ndarray
        1xK T-L calibration coefficients where K is the
        number of A matrix terms to use for calibration
    '''
    
    if terms == ALL_TERMS:
        terms = [PERMANENT, INDUCED, EDDY]
    
    if b_external is None:
        b_external = np.mean(b_scalar)

    dc = direction_cosines(b_vector)
    A  = A_matrix(dc,
                  b_external,
                  delta_t,
                  terms)

    if use_filter:
        fs = 1.0 / delta_t
        
        if (PERMANENT in terms) and (len(terms) > 1):
            A[:, 3:] = Filters.bpf(data   = A[:, 3:],
                                   fstart = fstart,
                                   fstop  = fstop,
                                   fs     = fs,
                                   order  = order,
                                   axis   = 0)
            
        elif PERMANENT not in terms:
            A = Filters.bpf(data   = A,
                            fstart = fstart,
                            fstop  = fstop,
                            fs     = fs,
                            order  = order,
                            axis   = 0)
        
        else:
            A = Filters.lpf(data    = A,
                            cutoff  = fstart,
                            fs      = fs,
                            axis    = 0)
        
        b_scalar = Filters.bpf(b_scalar, fstart, fstop, fs, order)
    
    if alpha == 0:
        return np.linalg.lstsq(A, b_scalar, rcond=None)[0]
    
    ridge = Ridge(alpha=alpha, fit_intercept=fit_intcpt)
    ridge.fit(A, b_scalar)
        
    return ridge.coef_

def apply_tlc(c:          np.ndarray,
              b_scalar:   np.ndarray,
              b_vector:   np.ndarray,
              delta_t:    float,
              b_external: float=None,
              terms:      int=ALL_TERMS):
    '''
    Apply T-L calibration coefficients to data
    
    Parameters
    ----------
    c
        1xK T-L calibration coefficients where K is the
        number of A matrix terms to use for calibration
    b_scalar
        Nx1 array of scalar measurements (nT)
    b_vector
        Nx3 array of vector measurements (nT)
    delta_t
        Time between data samples (s)
    b_external
        External mag field magnitude (nT) to use for a-matrix, if none,
        try to compute from average of b_scalar
    terms
        Terms to include in A-matrix. Options include:
        
        - ALL_TERMS
        - PERMANENT
        - INDUCED
        - EDDY

    Returns
    -------
    np.ndarray
        Nx1 array of calibrated scalar measurements (nT)
    '''
    
    if b_external is None:
        b_external = np.mean(b_scalar)

    dc = direction_cosines(b_vector)
    A  = A_matrix(dc,
                  b_external,
                  delta_t,
                  terms)
    
    return A @ c

def apply_tl_dist(c:          np.ndarray,
                  b_scalar:   np.ndarray,
                  b_vector:   np.ndarray,
                  delta_t:    float,
                  b_external: float=None,
                  terms:      int=ALL_TERMS) -> np.ndarray:
    '''
    Apply distortion to the scalar magnetometer data given a set of Tolles-Lawson
    coefficients
    
    Parameters
    ----------
    c
        Tolles-Lawson coefficients used to distort the scalar magnetometer data. The number
        of coefficients must correspond to the terms specified by `terms`
    b_scalar
        Nx1 array of scalar measurements (nT)
    b_vector
        Nx3 array of vector measurements (nT)
    delta_t
        Time between data samples (s)
    b_external
        External mag field magnitude (nT) to use for a-matrix, if none,
        try to compute from average of b_scalar
    terms
        Specify which terms to use to distort the data. Options include:
        
        - ALL_TERMS
        - PERMANENT
        - INDUCED
        - EDDY
    
    Returns
    -------
    np.ndarray
        Nx1 array of distorted scalar measurements (nT)
    '''
    
    return b_scalar + apply_tlc(c,
                                b_scalar,
                                b_vector,
                                delta_t,
                                b_external,
                                terms)


if __name__ == '__main__':
    import sys
    from os.path import dirname, join
    
    import matplotlib.pyplot as plt

    SRC_DIR = dirname(dirname(__file__))
    sys.path.append(SRC_DIR)
    
    from Parse import parseSGL as psgl
    
    
    BASE_DIR = dirname(SRC_DIR)
    DATA_DIR = join(BASE_DIR, 'data')
    TEST_DIR = join(DATA_DIR, 'test')
    
    FNAME = join(TEST_DIR, '10Hz_Mag_INS_Aux - Flt1002.xyz')
    
    
    df = psgl.parse_xyz(FNAME,
                        drop_dups=False,
                        drop_cust=['DRAPE', 'OGS_MAG', 'OGS_HGT'])
    print(df)
    
    alt     = np.array(df.BARO)
    t       = np.array(df.epoch_sec)
    vec_x   = np.array(df.FLUXB_X)
    vec_y   = np.array(df.FLUXB_Y)
    vec_z   = np.array(df.FLUXB_Z)

    b_vector = np.vstack((vec_x, vec_y, vec_z)).T
    b_mag1   = np.array(df.UNCOMPMAG1)
    b_mag2   = np.array(df.UNCOMPMAG2)
    b_mag3   = np.array(df.UNCOMPMAG3)
    b_mag4   = np.array(df.UNCOMPMAG4)
    b_mag5   = np.array(df.UNCOMPMAG5)

    truth   = np.array(df.COMPMAG1)
    delta_t = 0.1
    
    cal_mask = np.logical_and(alt > 2925, t < 1.59266e9) # Calibration alt > 2925m and Unix epoch timestamp < 1.59266e9s
    
    t_cal     = t[cal_mask]
    b_mag_cal = b_mag3[cal_mask]
    b_vec_cal = b_vector[cal_mask, :]
    truth_cal = df.COMPMAG1[cal_mask]
    
    c = tlc(b_scalar   = b_mag_cal,
            b_vector   = b_vec_cal,
            delta_t    = delta_t,
            use_filter = True,
            fstart     = 0.1,
            fstop      = 1.0,
            order      = 5,
            b_external = None,
            alpha      = 0,
            fit_intcpt = False,
            terms      = ALL_TERMS)
    
    cal_data = apply_tlc(c        = c,
                         b_scalar = b_mag_cal,
                         b_vector = b_vec_cal,
                         delta_t  = delta_t,
                         terms    = ALL_TERMS)
    
    plt.figure()
    plt.title('T-L Example')
    plt.plot(t_cal, b_mag_cal, label='Original')
    plt.plot(t_cal, b_mag_cal-cal_data, label='Calibrated')
    plt.plot(t_cal, truth_cal, label='Truth')
    plt.grid()
    plt.legend()
    plt.show()