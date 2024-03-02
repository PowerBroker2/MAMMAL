import sys
import datetime as dt
from os.path import dirname, realpath, join

import numpy as np
import pandas as pd
import rioxarray as rxr

SRC_DIR      = dirname(realpath(__file__))
PROJ_DIR     = dirname(SRC_DIR)
DOCS_DIR     = join(PROJ_DIR, 'docs')
EXAMPLES_DIR = join(PROJ_DIR, 'examples')
DATA_DIR     = join(PROJ_DIR, 'data')
TEST_DIR     = join(DATA_DIR, 'test')

sys.path.append(SRC_DIR)

import Simulator as sim
from Parse import parseIM as pim
from Utils import ProcessingUtils as pu
from VehicleCal import TL as tl


class BaseClass():
    '''
    
    '''
    
    def __init__(self,
                 map_loc_name:  str='test',
                 center_lat_dd: float=0,
                 center_lon_dd: float=0,
                 alt_m_msl:     float=0,
                 alt_m_agl:     float=0) -> None:
        '''
        
        '''
        
        # Spin Test
        self.spin_lat      = center_lat_dd
        self.spin_lon      = center_lon_dd
        self.spin_height_m = 0
        self.spin_a        = np.eye(3)
        self.spin_b        = np.zeros(3)
        
        # Tolles-Lawson
        self.tl_center_lat = center_lat_dd
        self.tl_center_lon = center_lon_dd
        self.tl_height_m   = alt_m_msl
        self.tl_box_xlen_m = 100
        self.tl_box_ylen_m = 100
        self.tl_c          = np.zeros(18)
        self.tl_dither_hz  = 1
        self.tl_dither_amp = 10
        self.tl_terms      = tl.ALL_TERMS
        
        # Map
        self.map_loc_name     = map_loc_name
        self.map_center_lat   = center_lat_dd
        self.map_center_lon   = center_lon_dd
        self.map_height_m     = alt_m_msl
        self.map_height_agl_m = alt_m_agl
        self.map_dx_m         = self.map_height_agl_m / 20
        self.map_dy_m         = self.map_height_agl_m / 20


class SurveySim(BaseClass):
    '''
    
    '''
    
    def __init__(self,
                 map_loc_name:  str='MAMMAL',
                 center_lat_dd: float=0,
                 center_lon_dd: float=0,
                 start_dt_utc:  dt.datetime=dt.datetime.utcnow(),
                 alt_m_msl:     float=0,
                 alt_m_agl:     float=0,
                 vel_mps:       float=20,
                 samp_hz:       float=10,
                 data_dir:      str=None,
                 debug:         bool=False) -> None:
        '''
        
        '''
        
        super().__init__(map_loc_name,
                         center_lat_dd,
                         center_lon_dd,
                         alt_m_msl,
                         alt_m_agl)
        
        # General
        self.vel_mps  = vel_mps
        self.samp_hz  = samp_hz
        self.data_dir = data_dir
        self.debug    = debug
        
        # Spin Test
        self.spin_df         = None
        self.spin_start_dt   = start_dt_utc
        self.spin_headings   = np.linspace(0, 720, 1000)
        self.spin_elevations = np.linspace(0, 7200, 1000)
        
        # Tolles-Lawson
        self.tl_df         = None
        self.tl_start_dt   = start_dt_utc
        self.tl_vel_mps    = self.vel_mps
        self.tl_sample_hz  = self.samp_hz
        
        # Reference Station
        self.ref_df        = None
        self.ref_file_df   = None
        self.ref_lat       = center_lat_dd
        self.ref_lon       = center_lon_dd
        self.ref_height_m  = 0
        self.ref_start_dt  = start_dt_utc
        self.ref_dur_s     = 10000
        self.ref_scale     = 1
        self.ref_offset    = 0
        self.ref_awgn_std  = 0
        self.ref_sample_hz = 1
        self.ref_id        = 'FRD'
        
        # Map
        self.sim_map            = None
        self.map_upcontinue     = False
        self.map_x_dist_m       = 300
        self.map_y_dist_m       = 300
        self.map_start_dt       = start_dt_utc
        self.map_anomaly_locs   = np.array([[self.map_center_lat],  # dd
                                            [self.map_center_lon]]) # dd
        self.map_anomaly_scales = np.array([20]) # nT
        self.map_anomaly_covs   = np.zeros((1, 2, 2))
        self.map_anomaly_covs[0, :, :] = np.diag([0.000001, 0.000002])
        
        # Survey
        self.survey_height_m        = self.map_height_m
        self.survey_start_dt        = start_dt_utc
        self.survey_vel_mps         = self.vel_mps
        self.survey_e_buff_m        = 15
        self.survey_w_buff_m        = 15
        self.survey_n_buff_m        = 15
        self.survey_s_buff_m        = 15
        self.survey_sample_hz       = self.samp_hz
        self.survey_ft_line_dist_m  = self.map_height_agl_m / 2
        self.survey_ft_line_dir     = sim.HORIZ
        self.survey_scalar_awgn_std = 0
        self.survey_diurnal_dist    = np.array([0, 1]), # [offset (nT), scale]
        self.survey_use_tie_lines   = True
        self.survey_tie_dist_m      = self.survey_ft_line_dist_m * 5
    
    def gen_spin_data(self) -> pd.DataFrame:
        '''
        
        '''
        
        self.spin_df = sim.gen_spin_data(out_dir    = self.data_dir,
                                         lat        = self.spin_lat,
                                         lon        = self.spin_lon,
                                         height     = self.spin_height_m,
                                         date       = self.spin_start_dt,
                                         headings   = self.spin_headings,
                                         elevations = self.spin_elevations,
                                         a          = self.spin_a,
                                         b          = self.spin_b,
                                         debug      = self.debug)
        return self.spin_df
    
    def gen_TL_data(self) -> pd.DataFrame:
        '''
        
        '''
        
        self.tl_df = sim.gen_TL_data(out_dir    = self.data_dir,
                                     center_lat = self.tl_center_lat,
                                     center_lon = self.tl_center_lon,
                                     height     = self.tl_height_m,
                                     start_dt   = self.tl_start_dt,
                                     box_xlen_m = self.tl_box_xlen_m,
                                     box_ylen_m = self.tl_box_ylen_m,
                                     c          = self.tl_c,
                                     vel_mps    = self.tl_vel_mps,
                                     sample_hz  = self.tl_sample_hz,
                                     dither_hz  = self.tl_dither_hz,
                                     dither_amp = self.tl_dither_amp,
                                     terms      = self.tl_terms,
                                     a          = self.spin_a,
                                     b          = self.spin_b,
                                     debug      = self.debug)
        return self.tl_df
    
    def gen_ref_station_data(self) -> pd.DataFrame:
        '''
        
        '''
        
        if self.ref_id is not None and self.data_dir is not None:
            self.ref_file_df = pim.loadInterMagData(self.data_dir)[self.ref_id]
        else:
            self.ref_file_df = None
        
        self.ref_df = sim.gen_ref_station_data(out_dir   = self.data_dir,
                                               lat       = self.ref_lat,
                                               lon       = self.ref_lon,
                                               height    = self.ref_height_m,
                                               start_dt  = self.ref_start_dt,
                                               dur_s     = self.ref_dur_s,
                                               scale     = self.ref_scale,
                                               offset    = self.ref_offset,
                                               awgn_std  = self.ref_awgn_std,
                                               sample_hz = self.ref_sample_hz,
                                               file_df   = self.ref_file_df,
                                               debug     = self.debug)
        
        return self.ref_df
    
    def gen_sim_map(self) -> pd.DataFrame:
        '''
        
        '''
        
        self.sim_map = sim.gen_sim_map(out_dir        = self.data_dir,
                                       location       = self.map_loc_name,
                                       center_lat     = self.map_center_lat,
                                       center_lon     = self.map_center_lon,
                                       dx_m           = self.map_dx_m,
                                       dy_m           = self.map_dy_m,
                                       x_dist_m       = self.map_x_dist_m,
                                       y_dist_m       = self.map_y_dist_m,
                                       height         = self.map_height_m,
                                       date           = self.map_start_dt,
                                       anomaly_locs   = self.map_anomaly_locs,
                                       anomaly_scales = self.map_anomaly_scales,
                                       anomaly_covs   = self.map_anomaly_covs,
                                       upcontinue     = self.map_upcontinue,
                                       debug          = self.debug)
        return self.sim_map
    
    def gen_survey_data(self) -> pd.DataFrame:
        '''
        
        '''
        
        self.survey_df = sim.gen_survey_data(out_dir         = self.data_dir,
                                             map             = self.sim_map,
                                             survey_height_m = self.survey_height_m,
                                             survey_start_dt = self.survey_start_dt,
                                             survey_vel_mps  = self.survey_vel_mps,
                                             survey_e_buff_m = self.survey_e_buff_m,
                                             survey_w_buff_m = self.survey_w_buff_m,
                                             survey_n_buff_m = self.survey_n_buff_m,
                                             survey_s_buff_m = self.survey_s_buff_m,
                                             sample_hz       = self.survey_sample_hz,
                                             ft_line_dist_m  = self.survey_ft_line_dist_m,
                                             ft_line_dir     = self.survey_ft_line_dir,
                                             a               = self.spin_a,
                                             b               = self.spin_b,
                                             c               = self.tl_c,
                                             terms           = self.tl_terms,
                                             scalar_awgn_std = self.survey_scalar_awgn_std,
                                             diurnal_df      = self.ref_df,
                                             diurnal_dist    = self.survey_diurnal_dist,
                                             use_tie_lines   = self.survey_use_tie_lines,
                                             tie_dist_m      = self.survey_tie_dist_m,
                                             debug           = self.debug)
        return self.survey_df


class MapMaker(BaseClass):
    '''
    
    '''
    
    def __init__(self,
                 map_loc_name: str='MAMMAL',
                 alt_m_msl:    float=0,
                 alt_m_agl:    float=0,
                 data_dir:     str=None,
                 debug:        bool=False) -> None:
        '''
        
        '''
        
        super().__init__(map_loc_name,
                         0,
                         0,
                         alt_m_msl,
                         alt_m_agl)
        
        # General
        self.data_dir = data_dir
        self.debug    = debug
        
        # Spin Test
        self.spin_fname = None
        self.spin_df    = None
        self.a          = np.eye(3)
        self.b          = np.zeros(3)
        
        # Tolles-Lawson
        self.tl_fname   = None
        self.tl_df      = None
        self.c          = np.zeros(18)
        self.use_filter = True
        self.fstart     = 0.1
        self.fstop      = 1
        self.terms      = tl.ALL_TERMS
        
        # Reference Station
        self.ref_fname       = None
        self.ref_df          = None
        self.ref_scale       = 1
        self.ref_offset      = 0
        self.enable_lon_norm = False
        
        # Survey
        self.survey_fname = None
        self.survey_df    = None
        
        # Map
        self.map            = None
        self.lvl_type       = None
        self.num_ptls       = None
        self.ptl_locs       = None
        self.percent_thresh = 0.85
        self.sensor_sigma   = 0
        self.interp_type    = 'bicubic'
    
    def spin_params(self,
                    spin_df:      pd.DataFrame=None,
                    use_internal: bool=False) -> list:
        '''
        
        '''
        
        if (spin_df is None) and (use_internal is False):
            self.spin_df = pd.read_csv(self.spin_fname, parse_dates=['datetime'])
        elif spin_df is not None:
            self.spin_df = spin_df
        
        self.a, self.b = pu.cal_spin_df(self.spin_df)
        
        return self.a, self.b
    
    def tl_params(self,
                  tl_df:        pd.DataFrame=None,
                  use_internal: bool=False) -> np.ndarray:
        '''
        
        '''
        
        if (tl_df is None) and (use_internal is False):
            self.tl_df = pd.read_csv(self.tl_fname, parse_dates=['datetime'])
        elif tl_df is not None:
            self.tl_df = tl_df
        
        self.c = pu.cal_tl_df(self.tl_df,
                              self.use_filter,
                              self.fstart,
                              self.fstop,
                              self.a,
                              self.b,
                              self.terms)
        return self.c
    
    def gen_map(self,
                survey_df:           pd.DataFrame=None,
                survey_use_internal: bool=False,
                ref_df:              pd.DataFrame=None,
                ref_use_internal:    bool=False) -> rxr.rioxarray.raster_dataset.xarray.DataArray:
        '''
        
        '''
        
        if (survey_df is None) and (survey_use_internal is False):
            self.survey_df = pd.read_csv(self.survey_fname, parse_dates=['datetime'])
        elif survey_df is not None:
            self.survey_df = survey_df
        
        if (ref_df is None) and (ref_use_internal is False):
            self.ref_df = pd.read_csv(self.ref_fname, parse_dates=['datetime'])
        elif ref_df is not None:
            self.ref_df = ref_df
        
        self.map = pu.gen_map(out_dir         = self.data_dir,
                              map_name        = self.map_loc_name,
                              survey_df       = self.survey_df,
                              ref_df          = self.ref_df,
                              dx              = self.map_dx_m,
                              dy              = self.map_dy_m,
                              min_alt_agl     = self.map_height_agl_m,
                              ref_scale       = self.ref_scale,
                              ref_offset      = self.ref_offset,
                              enable_lon_norm = self.enable_lon_norm,
                              a               = self.a,
                              b               = self.b,
                              c               = self.c,
                              terms           = self.terms,
                              lvl_type        = self.lvl_type,
                              num_ptls        = self.num_ptls,
                              ptl_locs        = self.ptl_locs,
                              percent_thresh  = self.percent_thresh,
                              sensor_sigma    = self.sensor_sigma,
                              interp_type     = self.interp_type,
                              debug           = self.debug)
        return self.map