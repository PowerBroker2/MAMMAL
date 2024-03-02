# MAMMAL

MAMMAL - Magnetic Anomaly Map Matching Airborne and Land

A Python package for simulating and processing aeromagnetic anomaly survey data. It can be used to create magnetic anomaly maps for Magnetic Navigation solutions (MagNav).

## Install

To install MAMMAL, clone the repository to your machine and open a terminal in the folder containing `setup.py`. Lastly, run the following commands:

```bash
conda install gdal==3.4.3
python setup.py install
```

You will also need to download and install the GeoScraper package. Navigate to the [GeoScraper repository](https://git.antcenter.net/lbergeron/geoscraper), clone the repository to your machine, and open a terminal in the folder containing `setup.py`. Lastly, run the following commands:

```bash
python setup.py install
```

### If the osgeo (GDAL) package is not importing correctly on Windows:

1. Download and install GDAL core _and_ Python binding binaries from https://www.gisinternals.com/release.php
2. Find folder where GDAL was installed (usually `C:\Program Files (x86)\GDAL`)
3. Create a new environment variable named `GDAL` and set its value to the GDAL install folder path
4. Download the GDAL wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal for your CPU type _and_ Python version
5. Navigate to the folder the wheel was saved to
6. Open a command terminal and run the following:

```bash
pip install GDAL‑X.X.X‑cpXX‑cpXX‑winXXX.whl
```

7. Test installation by opening a python/ipython terminal and trying:

```Python
import osgeo
```

### If the rioxarray/rasterio packages are not importing correctly on Windows

If rioxarray is erroring on import, it might be because rasterio was installed incorrectly. If this is the case:

1. Install rasterio manually by downloading the rasterio wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio for your CPU type _and_ Python version
2. Navigate to the folder the wheel was saved to
3. Open a command terminal and run the following:

```bash
pip install rasterio‑X.X.X‑cpXX‑cpXX‑winXXX.whl
```

## Usage

### Parsing Log Files

---

To process a single MFAM Dev Kit log and save as a CSV:

```Python
from MAMMAL.Parse import parseGeometrics as pg


LOG_FNAME = r'dev_kit_log.txt'
CSV_FNAME = r'dev_kit_log.csv'


df = pg.parse_devLog(LOG_FNAME)
print(df)

df.to_csv(CSV_FNAME, index=False)
```

To process an entire acquisition of MFAM Dev Kit logs and save as a CSV:

```Python
from MAMMAL.Parse import parseGeometrics as pg


LOG_PATH  = r'dev_kit_acqu_folder_path'
CSV_FNAME = r'dev_kit_log.csv'


df = pg.parse_devACQU(LOG_FNAME)
print(df)

df.to_csv(CSV_FNAME, index=False)
```

To process a GSMP sensor log and save as a CSV:

```Python
from MAMMAL.Parse import parseGSMP as pgsmp


LOG_FNAME = r'gsmp_log.txt'
CSV_FNAME = r'gsmp_log.csv'
    
    
df = pgsmp.parse_GSMP(LOG_FNAME)
print(df)
    
df.to_csv(CSV_FNAME, index=False)
```

To process an INTERMAGNET ground reference station log and save as a CSV:

```Python
from MAMMAL.Parse import parseIM as pim


LOG_FNAME = r'intermagnet_log.sec'
CSV_FNAME = r'intermagnet_log.csv'
    
    
df = pim.parse_sec(LOG_FNAME)
print(df)
    
df.to_csv(CSV_FNAME, index=False)
```

To process a Pixhawk flight log and save as a CSV:

```Python
from MAMMAL.Parse import parsePixhawk as pp


LOG_FNAME = r'pix_log.txt'
CSV_FNAME = r'pix_log.csv'
    
    
df = pp.parsePix(LOG_FNAME)
print(df)
    
df.to_csv(CSV_FNAME, index=False)
```

To load a raster map:

```Python
from MAMMAL.Parse import parseRaster as praster


MAP_FNAME = r'map.tiff'


map = praster.parse_raster(MAP_FNAME)
print(map)
```

### Data Processing

---

To find temporal variations after reading-in flight and magnetic reference datasets:

```Python
import pandas as pd

from MAMMAL import Diurnal


REF_FNAME = r'ref_log.csv'
LOG_FNAME = r'flight_log.csv'


ref_df = pd.read_csv(REF_FNAME, parse_dates=['datetime'])

log_df      = pd.read_csv(LOG_FNAME, parse_dates=['datetime'])
timestamps  = np.array(log_df.epoch_sec)

_, ref_mag = Diurnal.interp_reference_df(df            = ref_df,
                                         timestamps    = timestamps,
                                         survey_lon    = log_df.LONG.mean(),
                                         subtract_core = True)
```

To calibrate airborne scalar data:

```Python
import pandas as pd

from MAMMAL.VehicleCal import magUtilsTL as magtl


LOG_FNAME = r'flight_log.csv'

TL_C     = np.array([-1.86687725e+01,  1.33975396e+02, -1.80762945e+02,  1.69023832e-01,
                     -3.92262356e-03, -1.84382741e-03,  1.71830230e-01, -1.61173781e-04,
                      1.72575427e-01, -4.31927864e-04, -8.21512835e-05, -4.37609432e-05,
                     -1.06838978e-04, -1.22444017e-04, -2.76294434e-04, -8.51727772e-05,
                      3.16374022e-05, -2.77441572e-05])
TL_TERMS = magtl.DEFAULT_TL_TERMS


log_df = pd.read_csv(LOG_FNAME, parse_dates=['datetime'])
f      = log_df.F

b_vector = np.hstack([np.array(log_df.X)[:, np.newaxis], 
                      np.array(log_df.Y)[:, np.newaxis],
                      np.array(log_df.Z)[:, np.newaxis]])

body_effects_scalar = magtl.tlc_compensation(vector = b_vector,
                                             tlc    = TL_C,
                                             terms  = TL_TERMS)
f_cal  = f - body_effects_scalar
f_cal += (f.mean() - f_cal.mean())
```

To level scalar anomaly data:

```Python
import pandas as pd

from MAMMAL.Utils import ProcessingUtils as pu



log_df = pd.Dataframe() # Replace with df where df.F are the scalar anomaly values

# PCA leveling
lvld_survey_df = pcaLvl.pca_lvl(survey_df = log_df,
                                num_ptls  = 2,
                                ptl_locs  = np.array([0.25, 0.75]))

# Per flight line leveling
lvld_survey_df = tieLvl.tie_lvl(survey_df = log_df,
                                approach  = 'lobf')

# Plane of best fit leveling
lvld_survey_df = tieLvl.tie_lvl(survey_df = log_df,
                                approach  = 'lsq')
```

To interpolate scalar anomaly data:

```Python
import pandas as pd

from MAMMAL.Utils import ProcessingUtils as pu


DX = 5 # meters
DY = 5 # meters

MAX_TERRAIN_MSL = 630 # meters


log_df = pd.Dataframe() # Replace with df where df.F are the scalar anomaly values

interp_type = 'RBF'
interp_df   = pu.interp_flight_lines(anomaly_df      = log_df,
                                     dx              = DX,
                                     dy              = DY,
                                     max_terrain_msl = MAX_TERRAIN_MSL,
                                     buffer          = 0,
                                     interp_type     = interp_type,
                                     neighbors       = None,
                                     skip_na_mask    = True)
```

To create and export a magnetic anomaly map:

```Python
from MAMMAL.Utils import mapUtils as mu


# Replace each argument with the appropriate value for your use-case
# **See export_map doc string for argument details**
map = mu.export_map(out_dir          = SURVEY_DIR,
                    location         = map_title,
                    date             = log_df.datetime[0],
                    lats             = interp_lats,
                    lons             = interp_lons,
                    scalar           = interp_scalar_LPF,
                    heights          = interp_heights,
                    process_df       = pd.DataFrame(process_dict),
                    process_app      = PROCESS_APP,
                    stds             = interp_std,
                    vector           = None,
                    scalar_type      = SCALAR_TYPE,
                    vector_type      = VECTOR_TYPE,
                    scalar_var       = np.nan,
                    vector_var       = np.nan,
                    poc              = POC,
                    flight_path      = flight_path,
                    area_polys       = area_polys,
                    osm_path         = None,
                    level_type       = 'No leveling',
                    tl_coeff_types   = TL_COEFF_TYPES,
                    tl_coeffs        = TL_C,
                    interp_type      = interp_type,
                    final_filt_cut   = FINAL_FILT_CUT,
                    final_filt_order = FINAL_FILT_ORDER)
```

### Map Metadata

---

Magnetic anomaly maps for magnetic navigation (MagNav) must be standardized
in a easy to use, common file format with consistent use of units. This will ensure
plug-and-play interoperability between all future MagNav filters and maps generated
by various sources.

The GeoTIFF format is a highly versatile extension designed to represent various
geospacial data and is ubiquitous in the geospacial data processing discipline with
many mapping tools and software already supporting the file format. For this reason,
all MagNav survey maps should be published as GeoTIFF files with the following
metadata and fields:

* Coordinate reference system:
  * WGS84
* Orientation of raster bands:
  * North up
* Invalid pixel value:
  * NaN
* Top level metadata:
  * Metadata field name: “Description”
    * Standardized value: “MagNav Aeromagnetic Anomaly Map”
  * Metadata field name: “ProcessingApp”
    * Description of the application name and version used to generate the map file
  * Metadata field name: “SurveyDateUTC”
    * Approximate UTC data of the survey in an ISO 8601 formatted string
  * Metadata field name: “SampleDistM”
    * Approximate distance between each magnetic reading along a given flight line in meters
  * Metadata field name: “xResolutionM”
    * Pixel width in meters
  * Metadata field name: “yResolutionM”
    * Pixel height in meters
  * Metadata field name: “ExtentDD”
    * Extent of map in degrees decimal
    * Example: “[-84.0958, 39.7617, -84.0484, 39.7823]”
  * Metadata field name: “ScalarType”
    * Description of the make/model/type of scalar magnetometer used
  * Metadata field name: “VectorType”
    * Description of the make/model/type of vector magnetometer used
  * Metadata field name: “ScalarSensorVar”
    * Survey scalar magnetometer variance in nT
  * Metadata field name: “VectorSensorVar”
    * Survey vector magnetometer variance in nT
  * Metadata field name: “POC”
    * Point of contact information about the organization who conducted the survey and produced the map (no standard format for the information in this metadata field)
  * Metadata field name: “KML”
    * Keyhole Markup Language (KML) document text that specifies the timestamped survey sample locations; flight/tie line average directions, distances, and altitudes for each sub-survey area; and location of roads, power lines, and substations
      * The timestamped survey sample locations must be represented by a top-level GxTimeSpan named “FlightPath” with UTC timestamps; WGS84 coordinates; and the altitude mode set to “absolute”.
      * The sub-survey areas must be represented by a top-level folder of polygons named “SubSurveyAreas”. Each sub-survey area polygon must have the following description: “FL Dir: (fldir)°, FL Dist: (fldist)m, TL Dir: (tldir)°, TL Dist: (tldist)m, Alt: (alt)m above MSL” where:
        * “(fldir)” is replaced with the average flight line direction in degrees off North
        * “(fldist)” is replaced with the average flight line distance in meters
        * “(tldir)” is replaced with the average tie line direction in degrees off North (if tie lines not present, set to -90)
        * “(tldist)” is replaced with the average tie line distance in meters (if tie lines not present, set to 0)
        * “(alt)” is replaced with the average altitude in meters above mean sea level (MSL)
        * Directions must within the range [0°, 180°) except for the tie line direction if tie lines are not present (set value to -90)
      * The road locations must be represented by a top-level multigeometry of line strings named “Roads” with WGS84 coordinates and the altitude mode to “clampToGround”
      * The power line locations must be represented by a top-level multigeometry of line strings named “PowerLines” with WGS84 coordinates and the altitude mode set to “clampToGround”
      * The substation locations must be represented by a top-level multigeometry of polygons named “Substations” with WGS84 coordinates and the altitude mode set to “clampToGround”
  * Metadata field name: “LevelType”
    * Description of the algorithm used for map leveling
  * Metadata field name: “TLCoeffTypes”
    * Ordered list of Tolles-Lawson coefficient types used
    * Example: “[Permanent, Induced, Eddy]”
  * Metadata field name: “TLCoeffs”
    * Ordered list of Tolles-Lawson coefficients used
    * Example: “[0.62, 0.70, 0.55, 0.24, 0.49, 0.28, 0.43, 0.57, 0.90, 0.80, 0.84, 0.14, 0.42, 0.58, 0.85, 0.86, 0.80, 0.73]”
  * Metadata field name: “CSV”
    * Comma-separated values (CSV) document text that includes all pertinent survey data points and data processing steps.
    * Minimum required columns include:
      * TIMESTAMP: Coordinated Universal Time (UTC) timestamps (s)
      * LAT: Latitudes (dd)
      * LONG: Longitudes (dd)
      * ALT: Altitudes above MSL (m)
      * DC_X: Direction cosine X-Components (dimensionless)
      * DC_Y: Direction cosine Y-Components (dimensionless)
      * DC_Z: Direction cosine Z-Components (dimensionless)
      * F: Raw scalar measurements (nT)
    * Suggested columns include (may vary depending on exact steps used to produce the original map values):
      * F_CAL: Calibrated scalar measurements (nT)
      * F_CAL_IGRF: Calibrated scalar measurements without core field (nT)
      * F_CAL_IGRF_TEMPORAL: Calibrated scalar measurements without core field or temporal corrections (nT)
      * F_CAL_IGRF_TEMPORAL_FILT: Calibrated scalar measurements without core field or temporal corrections after low pass filtering (nT)
      * F_CAL_IGRF_TEMPORAL_FILT_LEVEL: Calibrated scalar measurements without core field or temporal corrections after low pass filtering and map leveling (nT)
  * Metadata field name: “InterpType”
    * Description of the algorithm used for map pixel interpolation
  * Metadata field name: “FinalFiltCut”
    * Cutoff wavelength of the 2D low pass filter applied to the interpolated scalar pixel values
  * Metadata field name: “FinalFiltOrder”
    * Order number of the 2D low pass filter applied to the interpolated scalar pixel values
* Band 1:
  * NxM raster array of scalar magnetic anomaly values in nT
  * Band metadata:
    * Metadata field name: “Type”
· Standardized value: “F”
    * Metadata field name: “Units”
· Standardized value: “nT”
    * Metadata field name: “Direction”
· Standardized value: “n/a”
* Band 2:
  * NxM raster array of North magnetic anomaly component values in nT (optional - if no data provided, fill band with NaNs)
  * Band metadata:
    * Metadata field name: “Type”
· Standardized value: “X”
    * Metadata field name: “Units”
· Standardized value: “nT”
    * Metadata field name: “Direction”
· Standardized value: “North”
* Band 3:
  * NxM raster array of East magnetic anomaly component values in nT (optional - if no data provided, fill band with NaNs)
  * Band metadata:
    * Metadata field name: “Type”
      * Standardized value: “Y”
    * Metadata field name: “Units”
      * Standardized value: “nT”
    * Metadata field name: “Direction”
      * Standardized value: “East”
* Band 4
  * NxM raster array of downward magnetic anomaly component values in nT (optional - if no data provided, fill band with NaNs)
  * Band metadata:
    * Metadata field name: “Type”
      * Standardized value: “Z”
    * Metadata field name: “Units”
      * Standardized value: “nT”
    * Metadata field name: “Direction”
      * Standardized value: “Down”
* Band 5
  * NxM raster array of pixel altitudes in meters above MSL
  * Band metadata:
    * Metadata field name: “Type”
      * Standardized value: “ALT”
    * Metadata field name: “Units”
      * Standardized value: “m MSL”
    * Metadata field name: “Direction”
      * Standardized value: “n/a”
* Band 6
  * NxM raster array of pixel interpolation standard deviation values in nT (optional - if no data provided, fill band with NaNs)
  * Band metadata:
    * Metadata field name: “Type”
      * Standardized value: “STD”
    * Metadata field name: “Units”
      * Standardized value: “nT”
    * Metadata field name: “Direction”
      * Standardized value: “n/a”
* Band 7
  * NxM raster array of pixel easterly gradients
  * Band metadata:
    * Metadata field name: “Type”
      * Standardized value: “dX”
    * Metadata field name: “Units”
      * Standardized value: “nT”
    * Metadata field name: “Direction”
      * Standardized value: “East”
* Band 8
  * NxM raster array of pixel northerly gradients
  * Band metadata:
    * Metadata field name: “Type”
      * Standardized value: “dY”
    * Metadata field name: “Units”
      * Standardized value: “nT”
    * Metadata field name: “Direction”
      * Standardized value: “North”
