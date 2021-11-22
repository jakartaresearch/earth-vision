"""
Script from: 
- https://github.com/CosmiQ/solaris
- https://github.com/avanetten/CosmiQ_SN7_Baseline/blob/master/src/sn7_baseline_prep_funcs.py
"""
import pandas as pd
import numpy as np
from skimage import io
import os
from osgeo import gdal
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry.base import BaseGeometry
from fiona._err import CPLE_OpenFailedError
from fiona.errors import DriverError
from warnings import warn


# from core.py
def _check_df_load(df):
    """Check if `df` is already loaded in, if not, load from file."""
    if isinstance(df, str):
        if df.lower().endswith("json"):
            return _check_gdf_load(df)
        else:
            return pd.read_csv(df)
    elif isinstance(df, pd.DataFrame):
        return df
    else:
        raise ValueError(f"{df} is not an accepted DataFrame format.")


def _check_rasterio_im_load(im):
    """Check if `im` is already loaded in; if not, load it in."""
    if isinstance(im, str):
        return rasterio.open(im)
    elif isinstance(im, rasterio.DatasetReader):
        return im
    else:
        raise ValueError("{} is not an accepted image format for rasterio.".format(im))


def _check_geom(geom):
    """Check if a geometry is loaded in.
    Returns the geometry if it's a shapely geometry object. If it's a wkt
    string or a list of coordinates, convert to a shapely geometry.
    """
    if isinstance(geom, BaseGeometry):
        return geom
    elif isinstance(geom, str):  # assume it's a wkt
        return loads(geom)
    elif isinstance(geom, list) and len(geom) == 2:  # coordinates
        return Point(geom)


def _check_gdf_load(gdf):
    """Check if `gdf` is already loaded in, if not, load from geojson."""
    if isinstance(gdf, str):
        # as of geopandas 0.6.2, using the OGR CSV driver requires some add'nal
        # kwargs to create a valid geodataframe with a geometry column. see
        # https://github.com/geopandas/geopandas/issues/1234
        if gdf.lower().endswith("csv"):
            return gpd.read_file(gdf, GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")
        try:
            return gpd.read_file(gdf)
        except (DriverError, CPLE_OpenFailedError):
            warn(
                f"GeoDataFrame couldn't be loaded: either {gdf} isn't a valid"
                " path or it isn't a valid vector file. Returning an empty"
                " GeoDataFrame."
            )
            return gpd.GeoDataFrame()
    elif isinstance(gdf, gpd.GeoDataFrame):
        return gdf
    else:
        raise ValueError(f"{gdf} is not an accepted GeoDataFrame format.")


# from mask.py


def df_to_px_mask(
    df,
    channels=["footprint"],
    out_file=None,
    reference_im=None,
    geom_col="geometry",
    do_transform=None,
    affine_obj=None,
    shape=(900, 900),
    out_type="int",
    burn_value=255,
    **kwargs,
):
    """Convert a dataframe of geometries to a pixel mask.
    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    channels : list, optional
        The mask channels to generate. There are three values that this can
        contain:
        - ``"footprint"``: Create a full footprint mask, with 0s at pixels
            that don't fall within geometries and `burn_value` at pixels that
            do.
        - ``"boundary"``: Create a mask with geometries outlined. Use
            `boundary_width` to set how thick the boundary will be drawn.
        - ``"contact"``: Create a mask with regions between >= 2 closely
            juxtaposed geometries labeled. Use `contact_spacing` to set the
            maximum spacing between polygons to be labeled.
        Each channel correspond to its own `shape` plane in the output.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to ``None``, in which case the function
        attempts to infer whether or not a transformation is required based on
        the presence or absence of a CRS in `df`. If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    burn_value : `int` or `float`
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`.
    kwargs
        Additional arguments to pass to `boundary_mask` or `contact_mask`. See
        those functions for requirements.
    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`. Shape will be
        ``(shape[0], shape[1], len(channels))``, with channels ordered per the
        provided `channels` `list`.
    """
    if isinstance(channels, str):  # e.g. if "contact", not ["contact"]
        channels = [channels]

    if out_file and not reference_im:
        raise ValueError("If saving output to file, `reference_im` must be provided.")

    mask_dict = {}
    if "footprint" in channels:
        mask_dict["footprint"] = footprint_mask(
            df=df,
            reference_im=reference_im,
            geom_col=geom_col,
            do_transform=do_transform,
            affine_obj=affine_obj,
            shape=shape,
            out_type=out_type,
            burn_value=burn_value,
        )
    if "boundary" in channels:
        mask_dict["boundary"] = boundary_mask(
            footprint_msk=mask_dict.get("footprint", None),
            reference_im=reference_im,
            geom_col=geom_col,
            boundary_width=kwargs.get("boundary_width", 3),
            boundary_type=kwargs.get("boundary_type", "inner"),
            burn_value=burn_value,
            df=df,
            affine_obj=affine_obj,
            shape=shape,
            out_type=out_type,
        )
    if "contact" in channels:
        mask_dict["contact"] = contact_mask(
            df=df,
            reference_im=reference_im,
            geom_col=geom_col,
            affine_obj=affine_obj,
            shape=shape,
            out_type=out_type,
            contact_spacing=kwargs.get("contact_spacing", 10),
            burn_value=burn_value,
            meters=kwargs.get("meters", False),
        )

    output_arr = np.stack([mask_dict[c] for c in channels], axis=-1)

    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=output_arr.shape[-1])
        meta.update(dtype="uint8")
        with rasterio.open(out_file, "w", **meta) as dst:
            # I hate band indexing.
            for c in range(1, 1 + output_arr.shape[-1]):
                dst.write(output_arr[:, :, c - 1], indexes=c)

    return output_arr


def footprint_mask(
    df,
    out_file=None,
    reference_im=None,
    geom_col="geometry",
    do_transform=None,
    affine_obj=None,
    shape=(900, 900),
    out_type="int",
    burn_value=255,
    burn_field=None,
):
    """Convert a dataframe of geometries to a pixel mask.
    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to ``None``, in which case the function
        attempts to infer whether or not a transformation is required based on
        the presence or absence of a CRS in `df`. If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    out_type : 'float' or 'int'
    burn_value : `int` or `float`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`. Ignored if `burn_field` is provided.
    burn_field : str, optional
        Name of a column in `df` that provides values for `burn_value` for each
        independent object. If provided, `burn_value` is ignored.
    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`.
    """
    # start with required checks and pre-population of values
    if out_file and not reference_im:
        raise ValueError("If saving output to file, `reference_im` must be provided.")
    df = _check_df_load(df)

    if len(df) == 0 and not out_file:
        return np.zeros(shape=shape, dtype="uint8")

    if do_transform is None:
        # determine whether or not transform should be done
        do_transform = _check_do_transform(df, reference_im, affine_obj)

    df[geom_col] = df[geom_col].apply(_check_geom)  # load in geoms if wkt
    if not do_transform:
        affine_obj = Affine(1, 0, 0, 0, 1, 0)  # identity transform

    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
        shape = reference_im.shape
        if do_transform:
            affine_obj = reference_im.transform

    # extract geometries and pair them with burn values
    if burn_field:
        if out_type == "int":
            feature_list = list(zip(df[geom_col], df[burn_field].astype("uint8")))
        else:
            feature_list = list(zip(df[geom_col], df[burn_field].astype("float32")))
    else:
        feature_list = list(zip(df[geom_col], [burn_value] * len(df)))

    if len(df) > 0:
        output_arr = features.rasterize(shapes=feature_list, out_shape=shape, transform=affine_obj)
    else:
        output_arr = np.zeros(shape=shape, dtype="uint8")
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        if out_type == "int":
            meta.update(dtype="uint8")
            meta.update(nodata=0)
        with rasterio.open(out_file, "w", **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr


def boundary_mask(
    footprint_msk=None,
    out_file=None,
    reference_im=None,
    boundary_width=3,
    boundary_type="inner",
    burn_value=255,
    **kwargs,
):
    """Convert a dataframe of geometries to a pixel mask.
    Note
    ----
    This function requires creation of a footprint mask before it can operate;
    therefore, if there is no footprint mask already present, it will create
    one. In that case, additional arguments for :func:`footprint_mask` (e.g.
    ``df``) must be passed.
    By default, this function draws boundaries *within* the edges of objects.
    To change this behavior, use the `boundary_type` argument.
    Arguments
    ---------
    footprint_msk : :class:`numpy.array`, optional
        A filled in footprint mask created using :func:`footprint_mask`. If not
        provided, one will be made by calling :func:`footprint_mask` before
        creating the boundary mask, and the required arguments for that
        function must be provided as kwargs.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored
    boundary_width : int, optional
        The width of the boundary to be created **in pixels.** Defaults to 3.
    boundary_type : ``"inner"`` or ``"outer"``, optional
        Where to draw the boundaries: within the object (``"inner"``) or
        outside of it (``"outer"``). Defaults to ``"inner"``.
    burn_value : `int`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`. Ignored if `burn_field` is provided.
    **kwargs : optional
        Additional arguments to pass to :func:`footprint_mask` if one needs to
        be created.
    Returns
    -------
    boundary_mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and the same value as the
        footprint mask `burn_value` for the boundaries of each object.
    """
    if out_file and not reference_im:
        raise ValueError("If saving output to file, `reference_im` must be provided.")
    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
    # need to have a footprint mask for this function, so make it if not given
    if footprint_msk is None:
        footprint_msk = footprint_mask(reference_im=reference_im, burn_value=burn_value, **kwargs)

    # perform dilation or erosion of `footprint_mask` to get the boundary
    strel = square(boundary_width)
    if boundary_type == "outer":
        boundary_mask = dilation(footprint_msk, strel)
    elif boundary_type == "inner":
        boundary_mask = erosion(footprint_msk, strel)
    # use xor operator between border and footprint mask to get _just_ boundary
    boundary_mask = boundary_mask ^ footprint_msk
    # scale the `True` values to burn_value and return
    boundary_mask = boundary_mask > 0  # need to binarize to get burn val right
    output_arr = boundary_mask.astype("uint8") * burn_value

    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        meta.update(dtype="uint8")
        with rasterio.open(out_file, "w", **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr


def contact_mask(
    df,
    contact_spacing=10,
    meters=False,
    out_file=None,
    reference_im=None,
    geom_col="geometry",
    do_transform=None,
    affine_obj=None,
    shape=(900, 900),
    out_type="int",
    burn_value=255,
):
    """Create a pixel mask labeling closely juxtaposed objects.
    Notes
    -----
    This function identifies pixels in an image that do not correspond to
    objects, but fall within `contact_spacing` of >1 labeled object.
    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    contact_spacing : `int` or `float`, optional
        The desired maximum distance between adjacent polygons to be labeled
        as contact. Will be in pixel units unless ``meters=True`` is provided.
    meters : bool, optional
        Should `width` be defined in units of meters? Defaults to no
        (``False``). If ``True`` and `df` is not in a CRS with metric units,
        the function will attempt to transform to the relevant CRS using
        ``df.to_crs()`` (if `df` is a :class:`geopandas.GeoDataFrame`) or
        using the data provided in `reference_im` (if not).
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to ``None``, in which case the function
        attempts to infer whether or not a transformation is required based on
        the presence or absence of a CRS in `df`. If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    out_type : 'float' or 'int'
    burn_value : `int` or `float`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`.
    Returns
    -------
    output_arr : :class:`numpy.array`
        A pixel mask with `burn_value` at contact points between polygons.
    """
    if out_file and not reference_im:
        raise ValueError("If saving output to file, `reference_im` must be provided.")
    df = _check_df_load(df)

    if len(df) == 0 and not out_file:
        return np.zeros(shape=shape, dtype="uint8")

    if do_transform is None:
        # determine whether or not transform should be done
        do_transform = _check_do_transform(df, reference_im, affine_obj)

    df[geom_col] = df[geom_col].apply(_check_geom)  # load in geoms if wkt
    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
    buffered_geoms = buffer_df_geoms(
        df,
        contact_spacing / 2.0,
        meters=meters,
        reference_im=reference_im,
        geom_col=geom_col,
        affine_obj=affine_obj,
    )
    buffered_geoms = buffered_geoms[geom_col]
    # create a single multipolygon that covers all of the intersections
    if len(df) > 0:
        intersect_poly = geometries_internal_intersection(buffered_geoms)
    else:
        intersect_poly = Polygon()

    # handle case where there's no intersection
    if intersect_poly.is_empty:
        output_arr = np.zeros(shape=shape, dtype="uint8")

    else:
        # create a df containing the intersections to make footprints from
        df_for_footprint = pd.DataFrame({"shape_name": ["overlap"], "geometry": [intersect_poly]})
        # catch bowties
        df_for_footprint["geometry"] = df_for_footprint["geometry"].apply(lambda x: x.buffer(0))
        # use `footprint_mask` to create the overlap mask
        contact_msk = footprint_mask(
            df_for_footprint,
            reference_im=reference_im,
            geom_col="geometry",
            do_transform=do_transform,
            affine_obj=affine_obj,
            shape=shape,
            out_type=out_type,
            burn_value=burn_value,
        )
        footprint_msk = footprint_mask(
            df,
            reference_im=reference_im,
            geom_col=geom_col,
            do_transform=do_transform,
            affine_obj=affine_obj,
            shape=shape,
            out_type=out_type,
            burn_value=burn_value,
        )
        contact_msk[footprint_msk > 0] = 0
        contact_msk = contact_msk > 0
        output_arr = contact_msk.astype("uint8") * burn_value

    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        if out_type == "int":
            meta.update(dtype="uint8")
        with rasterio.open(out_file, "w", **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr


def _check_do_transform(df, reference_im, affine_obj):
    """Check whether or not a transformation should be performed."""
    try:
        crs = getattr(df, "crs")
    except AttributeError:
        return False  # if it doesn't have a CRS attribute

    if not crs:
        return False  # return False for do_transform if crs is falsey
    elif crs and (reference_im is not None or affine_obj is not None):
        # if the input has a CRS and another obj was provided for xforming
        return True


# from image.py


def create_multiband_geotiff(
    array, out_name, proj, geo, nodata=0, out_format=gdal.GDT_Byte, verbose=False
):
    """Convert an array to an output georegistered geotiff.
    Arguments
    ---------
    array  : :class:`numpy.ndarray`
        A numpy array with a the shape: [Channels, X, Y] or [X, Y]
    out_name : str
        The output name and path for your image
    proj : :class:`gdal.projection`
        A projection, can be extracted from an image opened with gdal with
        image.GetProjection(). Can be set to None if no georeferencing is
        required.
    geo : :class:`gdal.geotransform`
        A gdal geotransform which indicates the position of the image on the
        earth in projection units. Can be set to None if no georeferencing is
        required. Can be extracted from an image opened with gdal with
        image.GetGeoTransform()
    nodata : int
        A value to set transparent for GIS systems.
        Can be set to None if the nodata value is not required. Defaults to 0.
    out_format : str, gdalconst
        https://gdal.org/python/osgeo.gdalconst-module.html
        Must be one of the variables listed in the docs above. Defaults to
        gdal.GDT_Byte.
    verbose : bool
        A verbose output, printing all inputs and outputs to the function.
        Useful for debugging. Default to `False`
    """
    driver = gdal.GetDriverByName("GTiff")
    if len(array.shape) == 2:
        array = array[np.newaxis, ...]
    os.makedirs(os.path.dirname(os.path.abspath(out_name)), exist_ok=True)
    dataset = driver.Create(out_name, array.shape[2], array.shape[1], array.shape[0], out_format)
    if verbose is True:
        print("Array Shape, should be [Channels, X, Y] or [X,Y]:", array.shape)
        print("Output Name:", out_name)
        print("Projection:", proj)
        print("GeoTransform:", geo)
        print("NoData Value:", nodata)
        print("Bit Depth:", out_format)
    if proj is not None:
        dataset.SetProjection(proj)
    if geo is not None:
        dataset.SetGeoTransform(geo)
    if nodata is None:
        for i, image in enumerate(array, 1):
            dataset.GetRasterBand(i).WriteArray(image)
        del dataset
    else:
        for i, image in enumerate(array, 1):
            dataset.GetRasterBand(i).WriteArray(image)
            dataset.GetRasterBand(i).SetNoDataValue(nodata)
        del dataset


def map_wrapper(x):
    """For multi-threading"""
    return x[0](*(x[1:]))


def make_geojsons_and_masks(
    name_root, image_path, json_path, output_path_mask, output_path_mask_fbc=None
):
    """
    Make the stuffins
    mask_fbc is an (optional) three-channel fbc (footbrint, boundary, contact) mask
    """
    # filter out null geoms (this is always a worthy check)
    gdf_tmp = _check_gdf_load(json_path)
    if len(gdf_tmp) == 0:
        gdf_nonull = gdf_tmp
    else:
        gdf_nonull = gdf_tmp[gdf_tmp.geometry.notnull()]
        try:
            im_tmp = io.imread(image_path)
        except:
            print("Error loading image %s, skipping..." % (image_path))
            return

    # handle empty geojsons
    if len(gdf_nonull) == 0:
        # create masks
        # mask 1 has 1 channel
        # mask_fbc has 3 channel
        print("    Empty labels for name_root!", name_root)
        im = gdal.Open(image_path)
        proj = im.GetProjection()
        geo = im.GetGeoTransform()
        im = im.ReadAsArray()
        # set masks to 0 everywhere
        mask_arr = np.zeros((1, im.shape[1], im.shape[2]))
        create_multiband_geotiff(mask_arr, output_path_mask, proj, geo)
        if output_path_mask_fbc:
            mask_arr = np.zeros((3, im.shape[1], im.shape[2]))
            create_multiband_geotiff(mask_arr, output_path_mask_fbc, proj, geo)
        return

    # make masks (single channel)
    # https://github.com/CosmiQ/solaris/blob/master/docs/tutorials/notebooks/api_masks_tutorial.ipynb
    f_mask = df_to_px_mask(
        df=gdf_nonull,
        out_file=output_path_mask,
        channels=["footprint"],
        reference_im=image_path,
        shape=(im_tmp.shape[0], im_tmp.shape[1]),
    )

    # three channel mask (takes awhile)
    # https://github.com/CosmiQ/solaris/blob/master/docs/tutorials/notebooks/api_masks_tutorial.ipynb
    if output_path_mask_fbc:
        fbc_mask = df_to_px_mask(
            df=gdf_nonull,
            out_file=output_path_mask_fbc,
            channels=["footprint", "boundary", "contact"],
            reference_im=image_path,
            boundary_width=5,
            contact_spacing=10,
            meters=True,
            shape=(im_tmp.shape[0], im_tmp.shape[1]),
        )

    return
