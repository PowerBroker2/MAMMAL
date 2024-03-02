import rioxarray as rxr


WGS84_EPSG = 'EPSG:4326' # The EPSG for WGS-84 lat/lon is 4326 


def parse_raster(fname: str,
                 x_lim: list=None,
                 y_lim: list=None) -> rxr.rioxarray.raster_dataset.xarray.DataArray:
    '''
    Read-in raster file, crop if x or y limits are provided and return
    as a rioxarray object

    Parameters
    ----------
    fname
        File name/path of the raster file
    x_lim
        Optional 1x2 array-like object that defines the minimum
        and maximum x coordinates the raster can have --> [min x, max x]
    y_lim
        Optional 1x2 array-like object that defines the minimum
        and maximum y coordinates the raster can have --> [min y, max y]

    Returns
    -------
    rxr.rioxarray.raster_dataset.xarray.DataArray
        DataArray of the (optionally) cropped raster
    '''

    ds = rxr.open_rasterio(fname, masked=True, decode_coords="all")

    if x_lim is not None:
        x_min  = x_lim[0]
        x_max  = x_lim[1]
        mask_x = (ds.x >= x_min) & (ds.x <= x_max)
    else:
        mask_x = ds.all()

    if y_lim is not None:
        y_min  = y_lim[0]
        y_max  = y_lim[1]
        mask_y = (ds.y >= y_min) & (ds.y <= y_max)
    else:
        mask_y = ds.all()

    return ds.where(mask_x & mask_y, drop=True)