"""
tolles_lawson.py

Implement a standard Tolles-Lawson magnetometer compensation.

Aaron Nielsen, Didactex, LLC, apn@didactex.com
ANT Center, Air Force Institute of Technology
"""
import itertools
import typing
from enum import Enum

import numpy as np
import scipy.signal
from sklearn.linear_model import Ridge
from scipy import signal


class Filter:
    """ this is a simple wrapper class around the scipy.signal filter tools
    to provide plotting and application methods in a single object """

    def __init__(self, order, fcut, btype='low', ftype='butter', fs=1.0):
        """
        argument nomenclature follows scipy.signal.iirfilter
        """
        self._order = order
        self._fcut = fcut
        self._btype = btype
        self._ftype = ftype
        self._fs = fs
        self._gensos()

    @property
    def order(self):  # pylint: disable=missing-function-docstring
        return self._order

    @property
    def fcut(self):  # pylint: disable=missing-function-docstring
        return self._fcut

    @property
    def btype(self):  # pylint: disable=missing-function-docstring
        return self._btype

    @property
    def ftype(self):  # pylint: disable=missing-function-docstring
        return self._ftype

    @property
    def fs(self):  # pylint: disable=missing-function-docstring
        return self._fs

    @property
    def sos(self):  # pylint: disable=missing-function-docstring
        return self._sos

    @sos.setter
    def sos(self, sos):
        self._sos = sos

    def _gensos(self):
        sos = signal.iirfilter(self.order, self.fcut, btype=self.btype, ftype=self.ftype, fs=self.fs, output='sos')
        self.sos = sos

    def filter_response(self, npts=2048):
        """ return the filter response """
        w, h = signal.sosfreqz(self.sos, npts, fs=self.fs)
        return w, h

    def filtfilt(self, sig: np.ndarray, axis=0):
        """ filter the input signal forward and backward """
        return signal.sosfiltfilt(self.sos, sig, axis=axis)



# list of the default Tolles-Lawson terms
class TollesLawsonTerms(Enum):
    """
    Enumeration type for the Tolles Lawson Terms
    """
    PERMANENT = 0
    INDUCED = 1
    EDDY = 2
    INDUCED_REDUCED = 3
    EDDY_REDUCED = 4
    CUBIC = 5
    CUBIC_REDUCED = 6


DEFAULT_TL_TERMS = (TollesLawsonTerms.PERMANENT, TollesLawsonTerms.INDUCED, TollesLawsonTerms.EDDY)

# mapping of the Tolles Lawson term to the number of coefficients
TollesLawsonTermsToCoefficientNumbers = {
    TollesLawsonTerms.PERMANENT: 3,
    TollesLawsonTerms.INDUCED: 6,
    TollesLawsonTerms.INDUCED_REDUCED: 5,
    TollesLawsonTerms.EDDY: 9,
    TollesLawsonTerms.EDDY_REDUCED: 8,
    TollesLawsonTerms.CUBIC: 10,
    TollesLawsonTerms.CUBIC_REDUCED: 7
}


def get_number_of_tl_coefficient_from_terms(tolles_lawson_terms: typing.Iterable[TollesLawsonTerms]) -> int:
    """
    Get the total number of tolles lawson coefficient from a list of terms

    Parameters
    ----------
    tolles_lawson_terms
        the list of terms to use for calculating the total

    Returns
    -------
    int
        the total number of coefficients for the input list of terms
    """
    num_tl_coefficients = 0
    for tl_term in tolles_lawson_terms:
        num_tl_coefficients += TollesLawsonTermsToCoefficientNumbers[tl_term]
    return num_tl_coefficients


class TollesLawsonError(Exception):
    """
    TollesLawsonError is the exception class to be used when an error
    specfic to the tolles-lawson model is generated.
    """


class TLFilter(Filter):
    """
    default filter to use, band-pass cutoff at 0.1 Hz to 0.9 Hz.
    """

    def __init__(self, fs: float):
        order = 10
        fcut = [0.1, 0.9]
        btype = 'band'
        ftype = 'butter'
        super().__init__(order, fcut, btype=btype, ftype=ftype, fs=fs)


def calculate_direction_cosines(
    vector: np.ndarray,
    f_total: np.ndarray = None,
) -> np.ndarray:
    """
    compute the direction cosines from a vector array

    Parameters
    ----------
    vector
        Nx3 array of vector components. Also work with NxMx...x3
    f_total
       N size array of total magnetic field, it will be synthesized from vector if None
       this should normally be set to None for default value

    Returns
    -------
    np.ndarray:
        Nx3 array of direction cosine components. Also can return same shape as vector
    """

    # if no total field is input, use the vector magnitude
    if f_total is None:
        f_total = np.linalg.norm(vector, axis=-1)

    direction_cosines = vector / f_total[..., np.newaxis]

    return direction_cosines


def compute_A(
    direction_cosines: np.ndarray,
    diff_direction_cosines: typing.Optional[np.ndarray],
    B_t: np.ndarray,
    terms: typing.Sequence[TollesLawsonTerms] = DEFAULT_TL_TERMS
) -> np.ndarray:
    """
    compute the Tolles Lawson A for vector compensations.

    Parameters
    ----------
    direction_cosines
        Nx3 vector magnetometer direction cosine values (cos(x/B_t), cos(y/B_t), and cos(z/B_t)) named (cosX, cosY, cosZ)
    diff_direction_cosines
        Nx3 vector magnetometer direction cosine gradient values (d(cos(x/B_t))/dt, d(cos(y/B_t))/dt, and d(cos(z/B_t))/dt) named
        (cosX_dot, cosY_dot, cosZ_dot)
    B_t:
        N size array of the total measured magnetic intensity for each measurement
    terms:
        list of the terms in the Tolles-Lawson model to include, defaults to all three fields
        for the permanent, induced and eddy current terms mode

    Returns
    -------
    np.ndarray
        A calibration vector with order of first dimension dependent on the terms:
        PERMANENT
        0: cosX
        1: cosY
        2: cosZ

        INDUCED
        3: cosX ** 2
        4: cosX * cosY
        5: cosX * cosZ
        6: cosY ** 2
        7: cosY * cosZ
        8: cosZ ** 2

        EDDY
        9: cosX * cosX_dot
        10: cosX * cosY_dot
        11: cosX * cosZ_dot
        12: cosY * cosX_dot
        13: cosY * cosY_dot
        14: cosY * cosZ_dot
        15: cosZ * cosX_dot
        16: cosZ * cosY_dot
        17: cosZ * cosZ_dot

        INDUCED_REDUCED
        3: cosX ** 2
        4: cosX * cosY
        5: cosX * cosZ
        6: cosY ** 2
        7: cosY * cosZ

        EDDY_REDUCED
        9: cosX * cosX_dot
        10: cosX * cosY_dot
        11: cosX * cosZ_dot
        12: cosY * cosX_dot
        13: cosY * cosY_dot
        14: cosY * cosZ_dot
        15: cosZ * cosX_dot
        16: cosZ * cosY_dot

        CUBIC
        18: XXX
        19: XXY
        20: XXZ
        21: XYY
        22: XYZ
        23: XZZ
        24: YYY
        25: YYZ
        26: YZZ
        27: ZZZ

        CUBIC_REDUCED
        18: XXX
        19: XXY
        20: XXZ
        21: XYY
        22: XYZ
        24: YYY
        25: YYZ
    """
    # check that at least one output term is requested
    if len(terms) == 0:
        raise TollesLawsonError("At least one term must be included in Tolles-Lawson model")

    # check if requested terms are in list of supported terms
    for term in terms:
        if not isinstance(term, TollesLawsonTerms):
            raise TollesLawsonError(f"Requested model term {term} must be of type {TollesLawsonTerms}")
    if TollesLawsonTerms.INDUCED in terms and TollesLawsonTerms.INDUCED_REDUCED in terms:
        raise TollesLawsonError("Cannot have both TollesLawsonTerms.INDUCED and TollesLawsonTerms.INDUCED_REDUCED terms")
    if TollesLawsonTerms.EDDY in terms and TollesLawsonTerms.EDDY_REDUCED in terms:
        raise TollesLawsonError("Cannot have both TollesLawsonTerms.EDDY and TollesLawsonTerms.EDDY_REDUCED terms")
    if TollesLawsonTerms.CUBIC in terms and TollesLawsonTerms.CUBIC_REDUCED in terms:
        raise TollesLawsonError("Cannot have both TollesLawsonTerms.CUBIC and TollesLawsonTerms.CUBIC_REDUCED terms")

    # Create A matrix components of zero size to be replaced later if needed
    Ap: np.ndarray
    Ae: np.ndarray
    Ai: np.ndarray
    Ac: np.ndarray
    Ap = Ae = Ai = Ac = np.array([], dtype=np.float64).reshape(direction_cosines.shape[0], 0)

    # compute the requested components of the A matrix

    # permanent moments
    # cosX, cosY, cosZ
    if TollesLawsonTerms.PERMANENT in terms:
        Ap = direction_cosines

    # induced moments
    if TollesLawsonTerms.INDUCED in terms or TollesLawsonTerms.INDUCED_REDUCED in terms:
        Ai = np.zeros(shape=(direction_cosines.shape[0], 6))
        # cosX ** 2, cosX * cosY, cosX * cosZ
        Ai[..., 0:3] = direction_cosines[..., 0:1] * direction_cosines
        # cosY **2, cosY * cosZ
        Ai[..., 3:5] = direction_cosines[..., 1:2] * direction_cosines[..., 1:]
        # cosZ ** 2
        Ai[..., 5:6] = direction_cosines[..., 2:3]**2
        # multiply the induced and eddy currents by the measured magnetic intensity
        Ai *= B_t[..., np.newaxis]

        if TollesLawsonTerms.INDUCED_REDUCED in terms:
            # remove the last coefficient to get a reduced set of induced terms
            Ai = Ai[..., :-1]

    # eddy current moments
    if TollesLawsonTerms.EDDY in terms or TollesLawsonTerms.EDDY_REDUCED in terms:
        if diff_direction_cosines is None:
            raise TollesLawsonError("direction_cosines must be supplied to use EDDY or EDDY_REDUCED terms")
        Ae = np.zeros(shape=(direction_cosines.shape[0], 9))
        # cosX * cosX_dot, cosX * cosY_dot, cosX * cosZ_dot
        Ae[..., 0:3] = direction_cosines[..., 0:1] * diff_direction_cosines
        # cosY * cosX_dot, cosY * cosY_dot, cosY * cosZ_dot
        Ae[..., 3:6] = direction_cosines[..., 1:2] * diff_direction_cosines
        # cosZ * cosX_dot, cosZ * cosY_dot, cosZ * cosZ_dot
        Ae[..., 6:9] = direction_cosines[..., 2:3] * diff_direction_cosines
        # multiply the induced and eddy currents by the measured magnetic intensity
        Ae *= B_t[..., np.newaxis]
        if TollesLawsonTerms.EDDY_REDUCED in terms:
            Ae = Ae[..., :-1]

    # cubic terms
    if TollesLawsonTerms.CUBIC in terms or TollesLawsonTerms.CUBIC_REDUCED in terms:
        Ac = np.zeros(shape=(direction_cosines.shape[0], 10))
        combinations: typing.List[typing.Tuple[int, int,
                                               int]] = list(itertools.combinations_with_replacement([0, 1, 2], r=3))  # type: ignore
        for idx, comb in enumerate(combinations):
            Ac[..., idx] = np.product([direction_cosines[..., ii:ii + 1] for ii in comb])

        if TollesLawsonTerms.CUBIC_REDUCED in terms:
            # remove the redundant coefficient to get a reduced set of induced terms
            good_indices = list(range(10))
            remove_indices = [5, 8, 9]
            for remove_idx in remove_indices:
                good_indices.remove(remove_idx)
            Ac = Ac[..., good_indices]

    # put all the terms together
    A = np.concatenate([Ap, Ai, Ae, Ac], axis=1)

    return A


def tlc_matrix(
    vector: np.ndarray,
    f_total: np.ndarray = None,
    time_delta: float = 1.0,
    window_length: int = 3,
    norm: bool = False,
    terms: typing.Sequence[TollesLawsonTerms] = DEFAULT_TL_TERMS
) -> np.ndarray:
    """
    computes the matrix of values required to solve for the
    Tolles-Lawson coefficients

    Parameters
    ----------
    vector
       is a Nx3 matrix of the vector magnetometer data
    f_total
       N size array of total magnetic field, it will be synthesized from vector if None
       this should normally be set to None for default value
    time_delta
       is the time between samples defaults to 1.0
       this is only used for computing the numerical derivative for the eddy current
       terms. Sometimes this is ignore (set to 1.0), but setting it explicitly
       allows for comparison with data sampled at different rates.
    window_length
       is the window length to use for the differentiation filter in samples
       defaults to 3
    norm
        if True will normalized the mean total field to have an average value of 1
    terms
        list of Tolles-Lawson terms to include in the model, available options are
        permanent, induced, and/or eddy, at least one must be included.
        defaults to all three terms.

    Returns
    -------
    np.ndarray
        Tolles-Lawson A-matrix, Nx18 ndarray

    """

    # if no total field is input, use the vector magnitude
    if f_total is None:
        f_total = np.linalg.norm(vector, axis=-1)

    # find the mean total field and normalize the induced and eddy current
    # moments to the mean value if requested
    mean_total_field: typing.Union[float, np.ndarray]
    if norm:
        mean_total_field = typing.cast(np.ndarray, np.mean(f_total))
    else:
        mean_total_field = 1.0
    B_t: np.ndarray = f_total / mean_total_field

    # compute direction cosines
    direction_cosines = calculate_direction_cosines(vector, f_total=f_total)

    if TollesLawsonTerms.EDDY in terms or TollesLawsonTerms.EDDY_REDUCED in terms:
        # compute the time derivatives
        # This is a differentiating filter that is zero lag, using a
        # Savitzgy-Golay filter with padding on each end.
        diff_direction_cosines = scipy.signal.savgol_filter(
            direction_cosines, window_length=window_length, polyorder=2, deriv=1, axis=0, delta=time_delta
        )
    else:
        diff_direction_cosines = None

    # calculate the A matrix using the direction cosines and the diffs
    A = compute_A(direction_cosines=direction_cosines, diff_direction_cosines=diff_direction_cosines, B_t=B_t, terms=terms)

    return A


def tolles_lawson_coefficients(
    vector: np.ndarray,
    y_value: np.ndarray,
    f_total: typing.Optional[np.ndarray] = None,
    time_delta: float = 1.0,
    window_length: int = 3,
    apply_filter: bool = True,
    mag_filter: typing.Optional[Filter] = None,
    cut_length: typing.Optional[int] = None,
    ridge_parameters: typing.Optional[typing.Dict[str, typing.Any]] = None,
    terms: typing.Sequence[TollesLawsonTerms] = DEFAULT_TL_TERMS
) -> np.ndarray:
    """
    compute the Tolles Lawson coefficients using the input vector data and target y_value data

    Parameters
    ----------
    vector
        Nx3 array of vector data describing the direction of the external field.
        typically the values of a vector magnetometer
    y_value
        N size array of the total-field magnetometer that we wish
        to compensate with the resulting coefficients.
        This value can have the geomagnetic elements pre-removed to improve accuracy
    f_total
        N size array of the total field as recorded by a scalar magnetometer
        determined to be more accurate compared to using the lengths of vector
        defaults to None which will use the vector values
    time_delta
        the sample spacing between each sample in seconds
        required for eddy current coefficient generation
    window_length
        the window length in samples to use for the differentiation filter
    apply_filter
        if True (default) will apply the mag_filter. if False then no filter will be used
    mag_filter
        an optional filter to apply to the A matrix and y_value before computing
        the TLC, should be of type Filter from compensation.filter
        Ignored if apply_filter is False
    cut_length
        is the length of samples to cut from each end to avoid filter artifacts
        at the edges, will be set to 2 seconds on each end (empirical) + window_length if None
        defaults to None
    ridge_parameters
        The parameters given to the ridge regression
        if None is given defaults to the built in parameters with the following exceptions:
        alpha=0, do not do a ridge regression and just do a least squares fit
        fit_intercept=False, assume the user can add an intercept if desired
        for more info see: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    terms
        list of Tolles-Lawson terms to include in the model, available options are
        permanent, induced, and/or eddy, at least one must be included.
        defaults to all three terms.

    Returns
    -------
    np.ndarray
        array of size 18 which is the computed tolles lawson coefficients
    """
    augmented_values: np.ndarray = np.empty(shape=(vector.shape[0], 0))
    return augmented_tolles_lawson_coefficients(
        vector=vector,
        augmented_values=augmented_values,
        y_value=y_value,
        f_total=f_total,
        time_delta=time_delta,
        window_length=window_length,
        apply_filter=apply_filter,
        apply_filter_augmented_values=False,
        mag_filter=mag_filter,
        cut_length=cut_length,
        ridge_parameters=ridge_parameters,
        terms=terms
    )


def augmented_tolles_lawson_fitting_matrices(
    vector: np.ndarray,
    augmented_values: np.ndarray,
    y_value: np.ndarray,
    f_total: typing.Optional[np.ndarray] = None,
    time_delta: float = 1.0,
    window_length: int = 3,
    apply_filter: bool = True,
    apply_filter_augmented_values: bool = True,
    mag_filter: typing.Optional[Filter] = None,
    cut_length: typing.Optional[int] = None,
    terms: typing.Sequence[TollesLawsonTerms] = DEFAULT_TL_TERMS
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Compute the augmented Tolles Lawson fitting matrices (A, y)
    using the input vector data augmented data and target y_value data
    Does preprocessing on the data such as direction cosines calculation, clipping, filtering and rearranging

    Parameters
    ----------
    vector
        Nx3 array of vector data describing the direction of the external field.
        typically the values of a vector magnetometer
    augmented_values
        NxM array of augmented values to concat to the A matrix so the final fit uses
        augmented data to predict the y_values
    y_value
        N size array of the total-field magnetometer that we wish
        to compensate with the resulting coefficients.
        This value can have the geomagnetic elements pre-removed to improve accuracy
    f_total
        N size array of the total field as recorded by a scalar magnetometer
        determined to be more accurate compared to using the lengths of vector
        defaults to None which will use the vector values
    time_delta
        the sample spacing between each sample in seconds
        required for eddy current coefficient generation
    window_length
        the window length in samples to use for the differentiation filter
    apply_filter
        if True (default) will apply the mag_filter to the A matrix and y_values
    apply_filter_augmented_values
        if True (default) will apply the mag_filter to the augmented values
    mag_filter
        an optional filter to apply to the A matrix and y_value before computing
        the TLC, should be of type Filter from compensation.filter
        Ignored if apply_filter is False
    cut_length
        is the length of samples to cut from each end to avoid filter artifacts
        at the edges, will be set to 2 seconds on each end (empirical) + window_length if None
        defaults to None
    terms
        list of Tolles-Lawson terms to include in the model, available options are
        permanent, induced, and/or eddy, at least one must be included.
        defaults to all three terms.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray]
        the two matrices for fitting via some optimizer, the tuple is (A, y) where A is the A matrix of the tolles lawson algorithm
        with the direction cosines (and any augmented data) and y is the target values to fit to.
        This is meant to be used in solving for the matrix x in y=Ax
    """
    # if the filter is None get our default filter
    if mag_filter is None:
        mag_filter = TLFilter(fs=1.0 / time_delta)

    # compute the components of the A matrix
    A = tlc_matrix(vector=vector, f_total=f_total, time_delta=time_delta, window_length=window_length, terms=terms)

    if apply_filter:
        # apply the filter to the A matrix and y_values
        A = mag_filter.filtfilt(A, axis=0)
        y_value = mag_filter.filtfilt(y_value, axis=0)

    if apply_filter_augmented_values:
        # apply the filter to our augmented data
        augmented_values = mag_filter.filtfilt(augmented_values, axis=0)

    # concat the A matrix and augmented data
    A_augmented = np.hstack([A, augmented_values])

    # if cut_length is None use the default
    if cut_length is None:
        cut_length = int(2.0 / time_delta) + window_length

    # if cut_length is above 0 then apply the cut
    if cut_length > 0:
        A_augmented = A_augmented[cut_length:-cut_length]  # pylint: disable=invalid-unary-operand-type
        y_value = y_value[cut_length:-cut_length]  # pylint: disable=invalid-unary-operand-type

    return A_augmented, y_value


def augmented_tolles_lawson_coefficients(
    vector: np.ndarray,
    augmented_values: np.ndarray,
    y_value: np.ndarray,
    f_total: typing.Optional[np.ndarray] = None,
    time_delta: float = 1.0,
    window_length: int = 3,
    apply_filter: bool = True,
    apply_filter_augmented_values: bool = True,
    mag_filter: typing.Optional[Filter] = None,
    cut_length: typing.Optional[int] = None,
    ridge_parameters: typing.Optional[typing.Dict[str, typing.Any]] = None,
    terms: typing.Sequence[TollesLawsonTerms] = DEFAULT_TL_TERMS
) -> np.ndarray:
    """
    compute the augmented Tolles Lawson coefficients using the input vector data augmented data
    and target y_value data

    Parameters
    ----------
    vector
        Nx3 array of vector data describing the direction of the external field.
        typically the values of a vector magnetometer
    augmented_values
        NxM array of augmented values to concat to the A matrix so the final fit uses
        augmented data to predict the y_values
    y_value
        N size array of the total-field magnetometer that we wish
        to compensate with the resulting coefficients.
        This value can have the geomagnetic elements pre-removed to improve accuracy
    f_total
        N size array of the total field as recorded by a scalar magnetometer
        determined to be more accurate compared to using the lengths of vector
        defaults to None which will use the vector values
    time_delta
        the sample spacing between each sample in seconds
        required for eddy current coefficient generation
    window_length
        the window length in samples to use for the differentiation filter
    apply_filter
        if True (default) will apply the mag_filter to the A matrix and y_values
    apply_filter_augmented_values
        if True (default) will apply the mag_filter to the augmented values
    mag_filter
        an optional filter to apply to the A matrix and y_value before computing
        the TLC, should be of type Filter from compensation.filter
        Ignored if apply_filter is False
    cut_length
        is the length of samples to cut from each end to avoid filter artifacts
        at the edges, will be set to 2 seconds on each end (empirical) + window_length if None
        defaults to None
    ridge_parameters
        The parameters given to the ridge regression
        if None is given defaults to the built in parameters with the following exceptions:
        alpha=0, do not do a ridge regression and just do a least squares fit
        fit_intercept=False, assume the user can add an intercept if desired
        for more info see: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    terms
        list of Tolles-Lawson terms to include in the model, available options are
        permanent, induced, and/or eddy, at least one must be included.
        defaults to all three terms.

    Returns
    -------
    np.ndarray
        array of size 18+M which is the computed augmented tolles lawson coefficients
    """

    A_augmented, y_value = augmented_tolles_lawson_fitting_matrices(
        vector=vector,
        augmented_values=augmented_values,
        y_value=y_value,
        f_total=f_total,
        time_delta=time_delta,
        window_length=window_length,
        apply_filter=apply_filter,
        apply_filter_augmented_values=apply_filter_augmented_values,
        mag_filter=mag_filter,
        cut_length=cut_length,
        terms=terms
    )

    # make our ridge regression model
    if ridge_parameters is None:
        ridge_parameters = {
            'alpha': 0,
            'fit_intercept': False,
        }
    ridge = Ridge(**ridge_parameters)

    # fit our ridge regression
    ridge.fit(A_augmented, y_value)

    # return the coefficients which are the tolles lawson coefficients
    return ridge.coef_


def tlc_compensation(
    vector: np.ndarray,
    tlc: np.ndarray,
    f_total: typing.Optional[np.ndarray] = None,
    time_delta: float = 1.0,
    window_length: int = 3,
    terms: typing.Sequence[TollesLawsonTerms] = DEFAULT_TL_TERMS
) -> np.ndarray:
    """
    determine the Tolles-Lawson disturbance field to remove from the total field magnetometer

    Parameters
    ----------
    vector
        Nx3 array of vector data describing the direction of the external field.
        typically the values of a vector magnetometer
    tlc
        18 size array of the tolles lawson coefficients that correspond to the columns of the
        A matrix calculated from the tlc_matrix function
    f_total
        N sized array of the total field as recorded by a scalar magnetometer
        determined to be more accurate compared to using the lengths of vector
        defaults to None which will use the vector values
    time_delta
        the sample spacing between each sample in seconds
        required for eddy current coefficient generation
    window_length
        the window length in samples to use for the differentiation filter
    terms
        list of Tolles-Lawson terms to include in the model, available options are
        permanent, induced, and/or eddy, at least one must be included.
        defaults to all three terms.

    Returns
    -------
    np.ndarray
        returns the T-L field to be removed
    """
    if f_total is None:
        f_total = np.linalg.norm(vector, axis=-1)

    # calculate our A matrix
    A = tlc_matrix(vector=vector, f_total=f_total, time_delta=time_delta, window_length=window_length, terms=terms)

    # do the dot product
    return A @ tlc


def augmented_tlc_compensation_matrix(
    vector: np.ndarray,
    augmented_values: np.ndarray,
    f_total: typing.Optional[np.ndarray] = None,
    time_delta: float = 1.0,
    window_length: int = 3,
    terms: typing.Sequence[TollesLawsonTerms] = DEFAULT_TL_TERMS
) -> np.ndarray:
    """
    determine the Tolles-Lawson A matrix to use for removing the aircraft field

    Parameters
    ----------
    vector
        Nx3 array of vector data describing the direction of the external field.
        typically the values of a vector magnetometer
    augmented_values
        NxM array of augmented values to concat to the A matrix so the final fit uses
        augmented data to predict the y_values
    f_total
        N sized array of the total field as recorded by a scalar magnetometer
        determined to be more accurate compared to using the lengths of vector
        defaults to None which will use the vector values
    time_delta
        the sample spacing between each sample in seconds
        required for eddy current coefficient generation
    window_length
        the window length in samples to use for the differentiation filter
    terms
        list of Tolles-Lawson terms to include in the model, available options are
        permanent, induced, and/or eddy, at least one must be included.
        defaults to all three terms.

    Returns
    -------
    np.ndarray
        returns the T-L field to be removed
    """
    if f_total is None:
        f_total = np.linalg.norm(vector, axis=-1)

    # calculate our A matrix
    A = tlc_matrix(vector=vector, f_total=f_total, time_delta=time_delta, window_length=window_length, terms=terms)

    # concat the A matrix and augmented data
    A_augmented = np.hstack([A, augmented_values])

    return A_augmented


def augmented_tlc_compensation(
    vector: np.ndarray,
    augmented_values: np.ndarray,
    tlc: np.ndarray,
    f_total: typing.Optional[np.ndarray] = None,
    time_delta: float = 1.0,
    window_length: int = 3,
    terms: typing.Sequence[TollesLawsonTerms] = DEFAULT_TL_TERMS
) -> np.ndarray:
    """
    determine the Tolles-Lawson disturbance field to remove from the total field magnetometer

    Parameters
    ----------
    vector
        Nx3 array of vector data describing the direction of the external field.
        typically the values of a vector magnetometer
    augmented_values
        NxM array of augmented values to concat to the A matrix so the final fit uses
        augmented data to predict the y_values
    tlc
        18 size array of the tolles lawson coefficients that correspond to the columns of the
        A matrix calculated from the tlc_matrix function
    f_total
        N sized array of the total field as recorded by a scalar magnetometer
        determined to be more accurate compared to using the lengths of vector
        defaults to None which will use the vector values
    time_delta
        the sample spacing between each sample in seconds
        required for eddy current coefficient generation
    window_length
        the window length in samples to use for the differentiation filter
    terms
        list of Tolles-Lawson terms to include in the model, available options are
        permanent, induced, and/or eddy, at least one must be included.
        defaults to all three terms.

    Returns
    -------
    np.ndarray
        returns the T-L field to be removed
    """

    # get the augmented matrix
    A_augmented = augmented_tlc_compensation_matrix(
        vector=vector, augmented_values=augmented_values, f_total=f_total, time_delta=time_delta, window_length=window_length, terms=terms
    )

    # do the dot product
    return A_augmented @ tlc
