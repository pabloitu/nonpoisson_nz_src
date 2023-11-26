import numpy as np
# import pyvista'
import vtk
from vtk import vtkImageData
from vtkmodules.util.numpy_support import numpy_to_vtk
import fiona
from fiona.crs import from_epsg
import os
from os.path import join
import datetime
from shapely.geometry import shape, mapping, Point
from shapely.geometry.multipolygon import MultiPolygon
import pandas
import geopandas
import rasterio
import rasterio.mask
import subprocess
import affine as affine_module
from pyproj import Proj, transform
from osgeo import ogr, osr, gdal
gdal.DontUseExceptions()

try:
    venv = os.environ['CONDA_PREFIX']
    proj_path = join(venv, 'share')

except KeyError:  # using python-venv
    venv = os.environ['VIRTUAL_ENV']
    proj_path = join(venv, 'share')

os.environ['PROJ_LIB'] = f'{proj_path}/proj'


def write_vtk(fname, vti_data):

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(fname)
    writer.SetCompressorTypeToZLib()
    writer.SetDataModeToAppended()
    writer.SetInputData(vti_data)
    writer.Write()


def parse_raster(fname, offset=0):

    raster = gdal.Open(fname)
    affine = raster.GetGeoTransform()
    dims = (raster.RasterXSize + 1 , raster.RasterYSize + 1, 1)
    spacing = (affine[1], -affine[5], 0)
    origin = (affine[0], affine[3] - spacing[1]*dims[1], offset)

    return raster, affine, dims, spacing, origin

def rasterize_hazard(storedir, filename, array,
                      grid, res, srs='EPSG:4326'):
    """ Load ssm results from lon/lat results and save it into a multi-band
    raster. Creates a points-shapefile as middle step.
    Input:
        storedir (str): Directory to save the middle steps
        filename (str): Name of the files to be created
        array (ndarray): n-D array, in columns the different results
        array_names (list): contains the names of each array
        grid (ndarra): must contain the same rows of the array
        res (tuple): floats containing the x and y resolution

    Output:
        shapefile of name shp_fn
        tiff raster of name raster_fn


    """

    shp_fn = os.path.join(storedir, filename + '.geojson')
    raster_fn = os.path.join(storedir, filename + '.tiff')

    ### Read CSV file, convert and write it into shapefile
    df = pandas.DataFrame(array, columns=[str(i) for i in range(array.shape[1])])

    gdf = geopandas.GeoDataFrame(df,
                                 crs=srs,
                                 geometry=[Point(xy) for xy
                                           in zip(grid[:, 0], grid[:, 1])])

    gdf.to_file(shp_fn, driver='GeoJSON')

    # 1) Opens the shape_file

    source_ds = ogr.Open(shp_fn)
    source_layer = source_ds.GetLayer()
    schema = []
    ldefn = source_layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        schema.append(fdefn.name)

    # 2) Creating the destination raster data source
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    cols = int((x_max - x_min) / res[0])
    rows = int((y_max - y_min) / res[1])

    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, cols, rows,
                                                     array.shape[1],
                                                     gdal.GDT_Float32)

    target_ds.SetGeoTransform((x_min, res[0], 0, y_max, 0, -res[1]))
    target_dsSRS = osr.SpatialReference()
    target_dsSRS.ImportFromEPSG(4326)
    target_ds.SetProjection(target_dsSRS.ExportToWkt())

    for i in range(array.shape[1]):
        band = target_ds.GetRasterBand(i + 1)
        band.SetNoDataValue(-9999)  ##COMMENT 5
        gdal.RasterizeLayer(target_ds, [i + 1], source_layer,
                            options=['ATTRIBUTE=' + schema[i], 'at'])
    target_ds = None

    return shp_fn, raster_fn


def rasterize_results(input_fn, raster_fn, shp_fn, attributes,
                      res, crs='EPSG:4326', del_shp=True):
    """ Load ssm figures from lon/lat figures and save it into a multi-band
    raster. Creates a points-shapefile as middle step.
    Input:
        input (str): name of the CSV file to read
        raster_fn (str): Name of the TIFF file to be created
        shp_fn (str): Name of the intermediate SHAPE file to be created
        attributes (list): contains the names of each column in the csv file
        res (tuple): floats containing the (roughly) x and y resolution
        crs (str): Original CRS of the data
        del_shape (bool): flag to delete intermediate shapefiles
    Output:
        shapefile of name shp_fn
        tiff raster of name raster_fn


    """


    ### Read CSV file, convert and write it into shapefile
    # df = pandas.DataFrame(array,  columns=[str(i) for i in range(array.shape[1])])   #old
    df = pandas.read_csv(input_fn, delimiter=',', header=0)
    gdf = geopandas.GeoDataFrame(df[attributes],
                                     crs=crs,
                                     geometry=[Point(xy) for xy
                                           in zip(df.lon, df.lat)])

    gdf.to_file(shp_fn)  # , driver='GeoJSON'

    # 1) Opens the shape_file
    source_ds = ogr.Open(shp_fn)
    source_layer = source_ds.GetLayer()
    schema = []
    ldefn = source_layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        schema.append(fdefn.name)

    # 2) Creating the destination raster data source
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    cols = int((x_max - x_min) / res[0])
    rows = int((y_max - y_min) / res[1])

    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, cols, rows,
                                                     len(attributes),
                                                     gdal.GDT_Float32)

    target_ds.SetGeoTransform((x_min, res[0], 0, y_max, 0, -res[1]))
    target_dsSRS = osr.SpatialReference()
    target_dsSRS.ImportFromEPSG(int(crs.split(':')[-1]))
    target_ds.SetProjection(target_dsSRS.ExportToWkt())

    for i, j in enumerate(attributes):
        band = target_ds.GetRasterBand(i + 1)
        band.SetNoDataValue(-9999)  ##COMMENT 5
        gdal.RasterizeLayer(target_ds, [i + 1], source_layer,
                            options=['ATTRIBUTE=' + schema[i], 'at'])
    target_ds = None

    if del_shp:
        for extension in ['.shp', '.cpg', '.dbf', '.prj', '.shx']:
            os.remove(os.path.splitext(shp_fn)[0] + extension)


    return shp_fn, raster_fn


def rasterize_polygons(shp_fn, raster_fn, attributes,
                       res, crs='EPSG:4326', del_shp=True):
    """ Load ssm figures from lon/lat figures and save it into a multi-band
    raster. Creates a points-shapefile as middle step.
    Input:
        input (str): name of the CSV file to read
        raster_fn (str): Name of the TIFF file to be created
        shp_fn (str): Name of the intermediate SHAPE file to be created
        attributes (list): contains the names of each column in the csv file
        res (tuple): floats containing the (roughly) x and y resolution
        crs (str): Original CRS of the data
        del_shape (bool): flag to delete intermediate shapefiles
    Output:
        shapefile of name shp_fn
        tiff raster of name raster_fn


    """

    # 1) Opens the shape_file

    source_ds = ogr.Open(shp_fn)
    source_layer = source_ds.GetLayer()
    schema = []
    ldefn = source_layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        schema.append(fdefn.name)

    # 2) Creating the destination raster data source
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    cols = int((x_max - x_min) / res[0])
    rows = int((y_max - y_min) / res[1])

    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, cols, rows,
                                                     len(attributes),
                                                     gdal.GDT_Float32)

    target_ds.SetGeoTransform((x_min, res[0], 0, y_max, 0, -res[1]))
    target_dsSRS = osr.SpatialReference()
    target_dsSRS.ImportFromEPSG(int(crs.split(':')[-1]))
    target_ds.SetProjection(target_dsSRS.ExportToWkt())

    for i, j in enumerate(attributes):
        band = target_ds.GetRasterBand(i + 1)
        band.SetNoDataValue(-9999)  ##COMMENT 5
        gdal.RasterizeLayer(target_ds, [i + 1], source_layer,
                            options=['ATTRIBUTE=' + schema[i], 'at'])
    target_ds = None

    if del_shp:
        for extension in ['.shp', '.cpg', '.dbf', '.prj', '.shx']:
            os.remove(os.path.splitext(shp_fn)[0] + extension)


    return shp_fn, raster_fn


def reproject_coordinates(array, crs_0, crs_f):
    inProj = Proj(init=crs_0)
    outProj = Proj(init=crs_f)
    x1, y1 = inProj(array[:, 0], array[:, 1])
    x2, y2 = transform(inProj, outProj, x1, y1)

    reprojected = np.vstack((x2, y2)).T
    return reprojected


def read_raster(fn_raster, indexes, nodata_val=-9999):

    """
    Creates a data dictionary from a multi-band raster, including the data and raster properties
    :param fn_raster: path of the raster file
    :param indexes: index of the raster bands
    :param nodata_val: no data values of the raster
    :return: Dictionary with data and parameters
    """

    raster = gdal.Open(fn_raster)
    affine = raster.GetGeoTransform()
    dims = (raster.RasterXSize + 1, raster.RasterYSize + 1, 1)
    Data = []
    mask = None
    for name, index in enumerate(indexes):

        array = np.flipud(raster.GetRasterBand(index + 1). \
                          ReadAsArray()).flatten(order='C')
        array[array == nodata_val] = np.nan
        Data.append(array)
        mask = np.isnan(array)
    data = np.ascontiguousarray(np.array(Data).T)
    dict_ = {'data': data,
             'mask': mask,
             'affine': affine,
             'dims': dims}

    return dict_


## CHECK (write vti uniareas)
def numpy2raster(fname, data, affine, dims, nodata_val=-9999):

    target_ds = gdal.GetDriverByName('GTiff').Create(fname, dims[0], dims[1],
                                                     data.shape[1],
                                                     gdal.GDT_Float32)
    target_ds.SetGeoTransform(tuple(affine))
    target_dsSRS = osr.SpatialReference()
    target_dsSRS.ImportFromEPSG(4326)
    target_ds.SetProjection(target_dsSRS.ExportToWkt())
    for i, j in enumerate(data.T):
        band = target_ds.GetRasterBand(i + 1)
        band.SetNoDataValue(nodata_val)

        target_ds.GetRasterBand(i+1).WriteArray(np.flipud(j.reshape(dims[1]-1, dims[0]-1)))
        target_ds.GetRasterBand(i+1).SetNoDataValue(nodata_val)
    target_ds = None


def reproject_rio(input_fn, output_fn, dst_CRS, res=None, bounds=None,
                  resample='nearest'):
    
    """
    Call rio warp, using basic arguments
    Input:  -input_fn (string): Filename of the input raster
            -output_fn (string): Filename of the input raster
            -dst_CRS (string): Coordinate Reference System (e.g. 'EPSG:3857')
            -res (int or tuple): xResolution (,yRes) of the results raster
                                 in results CRS
            -bounds (list or tuple): Bounding box of the results raster in
                                     results CRS
            -resample(string): Resampling method (e.g. bilinear, cubic, etc.)
    """


    rio_args = " ".join(['rio warp', input_fn, output_fn,
                         '--overwrite --dst-crs', dst_CRS,
                         '--resampling', resample])
    if isinstance(res, tuple):  ## Not working properly
        rio_args += " --res %f --res %f" % tuple(res)
    elif isinstance(res, int) or isinstance(res, float):
        rio_args += " --res %f" % (res)      
    else:
        pass
        
    if bounds:
        rio_args += " --bounds %f %f %f %f" % tuple(bounds)

    rio_process = subprocess.Popen(rio_args,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    rio_log, rio_error = rio_process.communicate()

    
    if not os.path.isfile(output_fn):
        #file does not exist
        print(rio_args)
        print(rio_error)
        raise AssertionError('Reprojection failed. Output not created')
    modtime=datetime.datetime.fromtimestamp(os.path.getmtime(output_fn))
    if ((datetime.datetime.now()-modtime).total_seconds()) > 6: 
        #file exists, but old
        print(rio_args)
        print(rio_error)
        raise AssertionError('Reprojection failed. Output not created')


def polygonize_array(Array, raw, epsg='EPSG:4326', noval=-9990, savepath=False):

    aff = raw['affine']
    transfrm = affine_module.Affine(aff[1], aff[2], aff[0], aff[4], aff[5],
                                     aff[3])
    dims = raw['dims']

    array = noval*np.ones(Array.shape)
    array[~raw['mask']] = Array[~raw['mask']]
    unique_values_noval = np.unique(array[~np.isnan(array)])
    unique_values = unique_values_noval[~(unique_values_noval == noval)]
    array = np.flipud(array.reshape((dims[1] - 1,
                                     dims[0] - 1)).astype('float32'))
    mask = 1-np.flipud(raw['mask'].reshape((dims[1] - 1,
                                            dims[0] - 1)).astype('uint8'))
    shape_list = list(rasterio.features.shapes(array, mask=mask,
                                               transform=transfrm))
    shp_schema = {'geometry': 'MultiPolygon',
                  'properties': {'pixelvalue': 'float'}}
    # crs = rasterio.crs.CRS.from_epsg(epsg.split(':')[-1])
    crs = from_epsg(epsg.split(':')[-1])
    Shapes = [poly for poly in shape_list if poly[1] in unique_values]

    if savepath:
        with fiona.open(savepath, 'w', 'ESRI Shapefile',
                        shp_schema, crs) as shp:
            for poly, value in [(shape(geom), value)
                                for geom, value in Shapes]:
                multipolygon = MultiPolygon([poly])
                shp.write({
                    'geometry': mapping(multipolygon),
                    'properties': {'pixelvalue': float(value)}
                })

    return Shapes


def mask_raster(shapefile, input_fn, output_fn, all_touched=False,
                crop=False):
    """
    Mask a raster (.tiff) using a Polygon shapefile. Both should be in the 
    same CRS
    
    Input:  -shapefile(str):    Path of the shapefile
            -input_fn(str):     Path of the input raster to crop
            -output_fn(str):    Path of the results raster
            -all_touched(bool): Include (or not) pixels that falls within the
                                polygon's boundaries
            -crop(bool):        Crop the results raster bounds down to the
                                shapefile boundaries
    """

    with fiona.open(shapefile, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]        
    with rasterio.open(input_fn) as src:
        out_meta = src.meta
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=crop,
                                                      all_touched=all_touched)
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(output_fn, "w", **out_meta) as dest:
        dest.write(out_image)


def raster2vtk3d(fname_vtk, array_raster=None, array_names=[], basemap=None, dem=None,
                 dem_scale=1., offset=0, precision=3, nodata_val=-9999.):
    """
    Creates 3D Structured vtk. All arrays, basemap and dem must be in the same 
    projection and extent.
    
    Input:  -fname_vtk(str):  Filename of the results vtk file
            -array_raster(str/raster): one(multi)-band raster to be casted onto 
                                       the vtk
            -basemap(str/raster): gdal raster/fname contained RGB bands
                                 band(1)>red, band(2)>blue, band(3)>green.
                                 

    Output:     f_warp: Filename of warped raster
                raster_output: gdal raster object 
    """
    if type(dem) == str: # raster path or object
        dem = gdal.Open(dem)        
    
    affine_d = dem.GetGeoTransform()           
    height = dem_scale*dem.GetRasterBand(1).ReadAsArray() + offset  
        
        
    nx = dem.RasterXSize
    ny = dem.RasterYSize    
    
    x,y = np.meshgrid(affine_d[1]*np.arange(nx) + affine_d[0],
                       affine_d[5]*np.arange(ny) + affine_d[3])   


    grid = pyvista.StructuredGrid(x, y, height)
    if basemap:
        if type(basemap) == str: # raster path or object
            basemap = gdal.Open(basemap)

        affine_c = basemap.GetGeoTransform()
        if np.round(affine_c,precision).all() != \
            np.round(affine_d,precision).all():
            raise Exception("Affine of DEM & RGB Bands do not match")
        nx = basemap.RasterXSize
        ny = basemap.RasterYSize    
        
        r = np.abs(basemap.GetRasterBand(1).ReadAsArray().flatten(order='F'))
        g = np.abs(basemap.GetRasterBand(2).ReadAsArray().flatten(order='F'))
        b = np.abs(basemap.GetRasterBand(3).ReadAsArray().flatten(order='F'))
        rgb = np.vstack((r,g,b)).T
        grid.point_arrays["RGB"] = rgb

    if array_raster:
        if type(array_raster) == str: # raster path or object
            array_raster = gdal.Open(array_raster)

        affine_c = array_raster.GetGeoTransform()
        if np.round(affine_c,precision).all() !=\
            np.round(affine_d,precision).all():
            raise Exception("Affine of Array & DEM Bands do not match")
        nx = array_raster.RasterXSize
        ny = array_raster.RasterYSize    
        
        
        for i in range(array_raster.RasterCount):
            array = array_raster.GetRasterBand(i+1).ReadAsArray().flatten(order='F')
            array[array==nodata_val] = np.nan
            if array_names:
                name = array_names[i]
            else:
                name = str(i)
            grid.point_arrays[name] = array
    grid.save(fname_vtk)            
            
    
    return grid


def basemap2vti(fname_vti, raster,
                 offset=0, mask_rgb=[0,0,0]):
    """
    Creates 2D vtkImage. All rasters must be in the same 
    projection, extent and dimension.
    
    Input:  fname_vti(str):  Filename of the results vti file
            rasters(list): list of string, pointing the rasters to be casted
                           onto the vtk
            arrays_names(list): List of lists of strings. Each sublists contains
                        the array names of each raster band.
                        e.g. [['PGA_a','SA(0.1)'_a], ['PGV_b','SA(0.4)'_b]]
            offset (int): Vertical elevation upon which to cast the vti.
            nodata_val(float): Nodata value of the raster
            mask_rgb(list of int): Color value to set transparency
                                 
    Last mod. 07/08/2020
    """
    
    raster = gdal.Open(raster)        
    
    affine = raster.GetGeoTransform()
    dims = (raster.RasterXSize +1 , raster.RasterYSize +1, 1)
    spacing = (affine[1], -affine[5], 0)
    origin = (affine[0], affine[3] - spacing[1]*dims[1], offset)
    
    image = pyvista.UniformGrid(dims, spacing, origin)

    r = np.flipud(np.abs(raster.GetRasterBand(1).ReadAsArray())).flatten(order='C')
    g = np.flipud(np.abs(raster.GetRasterBand(2).ReadAsArray())).flatten(order='C')
    b = np.flipud(np.abs(raster.GetRasterBand(3).ReadAsArray())).flatten(order='C')
    
    rgb = np.ascontiguousarray(np.vstack((r,g,b)).T)
    dim = rgb.astype('uint16').max(0)+1
    mask = np.in1d(np.ravel_multi_index(rgb.T,dim),
                   np.ravel_multi_index(np.array(mask_rgb).T,
                                        dim)).astype('int')
    image.cell_arrays["basemap"] =  rgb
    image.cell_arrays["mask"] = mask
    image.save(fname_vti)

    return image


def rasters2vti(fname_vti, rasters, names,
               offset=0, nodata_val=-9999):
    """
    Creates 2D vtkImage. All rasters must be in the same
    projection, extent and dimension.

    Input:  fname_vti(str):  Filename of the results vti file
            rasters(list): list of string, pointing the rasters to be casted
                           onto the vtk
            arrays_names(list): List of lists of strings. Each sublists contains
                        the array names of each raster band.
                        e.g. [['PGA_a','SA(0.1)'_a], ['PGV_b','SA(0.4)'_b]]
            offset (int): Vertical elevation upon which to cast the vti.
            nodata_val(float): Nodata value of the raster
            mask_rgb(list of int): Color value to set transparency

    Last mod. 07/08/2020
    """

    for raster_path, name in zip(rasters, names):
        raster = gdal.Open(raster_path)
        n_bands = raster.RasterCount
        affine = raster.GetGeoTransform()

        dims = (raster.RasterXSize + 1, raster.RasterYSize + 1, 1)
        spacing = (affine[1], -affine[5], 0)
        origin = (affine[0], affine[3] - spacing[1] * dims[1], offset)
        mask = None


        if raster_path == rasters[0]:
            image = pyvista.UniformGrid(dims=dims, spacing=spacing, origin=origin)

        Array = []
        for i in range(n_bands):
            array = np.flipud(raster.GetRasterBand(i + 1). \
                              ReadAsArray()).flatten(order='C')
            array[array == nodata_val] = np.nan
            Array.append(array)
            if i == 0:
                mask = np.isnan(array)

            else:
                mask += np.isnan(array)
        Array = np.ascontiguousarray(np.array(Array).T)
        image.cell_data[name] = Array
        if raster_path == rasters[0]:
            image.cell_data["mask"] = 1 - mask * 1
        image.save(fname_vti)


def raster2vti(fname_vti, raster, data_struct,
                 offset=0, nodata_val=-9999):
    """
    Creates 2D vtkImage. All rasters must be in the same
    projection, extent and dimension.

    Input:  fname_vti(str):  Filename of the results vti file
            rasters(list): list of string, pointing the rasters to be casted
                           onto the vtk
            arrays_names(list): List of lists of strings. Each sublists contains
                        the array names of each raster band.
                        e.g. [['PGA_a','SA(0.1)'_a], ['PGV_b','SA(0.4)'_b]]
            offset (int): Vertical elevation upon which to cast the vti.
            nodata_val(float): Nodata value of the raster
            mask_rgb(list of int): Color value to set transparency

    Last mod. 07/08/2020
    """

    raster, affine, dims, spacing, origin = parse_raster(raster, offset)
    mask = None

    image_data = vtkImageData()
    image_data.SetDimensions(*dims)
    image_data.SetOrigin(*origin)
    image_data.SetSpacing(*spacing)

    # Write the image data to a VTI file
    iterator = data_struct.items()
    for name, index in iterator:
        Array = []
        for i in index:
            array = np.flipud(
                raster.GetRasterBand(i+1).ReadAsArray()).flatten(order='C')
            array[array == nodata_val] = np.nan
            Array.append(array)
            if name == [i[0] for i in iterator][0]:
                mask = np.isnan(array)
            else:
                mask += np.isnan(array)
        Array = np.ascontiguousarray(np.array(Array).T)
        vtk_array = numpy_to_vtk(Array, deep=True,
                                 array_type=vtk.VTK_FLOAT)
        vtk_array.SetName(name)
        image_data.GetCellData().AddArray(vtk_array)

    vtk_mask = numpy_to_vtk(1 - mask.ravel() * 1, deep=True,
                            array_type=vtk.VTK_INT)
    vtk_mask.SetName('mask')
    image_data.GetCellData().AddArray(vtk_mask)
    write_vtk(fname_vti, image_data)


def model2raster(array, storedir, filename, grid, res, srs='EPSG:4326'):
    """ Load ssm results from lon/lat results and save it into a multi-band
    raster. Creates a points-shapefile as middle step.
    Input:
        storedir (str): Directory to save the middle steps
        filename (str): Name of the files to be created
        array (ndarray): n-D array, in columns the different results
        array_names (list): contains the names of each array
        grid (ndarra): must contain the same rows of the array
        res (tuple): floats containing the x and y resolution

    Output:
        shapefile of name shp_fn
        tiff raster of name raster_fn


    """

    shp_fn = join(storedir, filename + '.GeoJSON')
    raster_fn = join(storedir, filename + '.tiff')

    ### Read CSV file, convert and write it into shapefile
    df = pandas.DataFrame(array, columns=[str(i) for i in range(array.shape[1])])

    gdf = geopandas.GeoDataFrame(df,
                                 crs=srs,
                                 geometry=[Point(xy) for xy
                                           in zip(grid[:, 0], grid[:, 1])])

    gdf.to_file(shp_fn, driver='GeoJSON')
    # gdf.to_file(shp_fn)
    # 1) Opens the shape_file

    source_ds = ogr.Open(shp_fn)
    source_layer = source_ds.GetLayer()
    schema = []
    ldefn = source_layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        schema.append(fdefn.name)

    # 2) Creating the destination raster data source
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    cols = int((x_max - x_min) / res[0])
    rows = int((y_max - y_min) / res[1])

    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, cols, rows,
                                                     array.shape[1],
                                                     gdal.GDT_Float32)

    target_ds.SetGeoTransform((x_min, res[0], 0, y_max, 0, -res[1]))
    target_dsSRS = osr.SpatialReference()
    target_dsSRS.ImportFromEPSG(int(srs.split(':')[-1]))
    target_ds.SetProjection(target_dsSRS.ExportToWkt())

    for i in range(array.shape[1]):
        band = target_ds.GetRasterBand(i + 1)
        band.SetNoDataValue(-9999)  ##COMMENT 5
        gdal.RasterizeLayer(target_ds, [i + 1], source_layer,
                            options=['ATTRIBUTE=' + schema[i], 'at'])
    target_ds = None

    return shp_fn, raster_fn


def hazard_model2raster(array, storedir, filename, grid, res, srs='EPSG:4326'):
    """ Load ssm results from lon/lat results and save it into a multi-band
    raster. Creates a points-shapefile as middle step.
    Input:
        storedir (str): Directory to save the middle steps
        filename (str): Name of the files to be created
        array (ndarray): n-D array, in columns the different results
        array_names (list): contains the names of each array
        grid (ndarra): must contain the same rows of the array
        res (tuple): floats containing the x and y resolution

    Output:
        shapefile of name shp_fn
        tiff raster of name raster_fn


    """

    shp_fn = join(storedir, filename + '.GeoJSON')
    raster_fn = join(storedir, filename + '.tiff')

    ### Read CSV file, convert and write it into shapefile
    df = pandas.DataFrame(array, columns=[str(i) for i in range(array.shape[1])])

    gdf = geopandas.GeoDataFrame(df,
                                 crs=srs,
                                 geometry=[Point(xy) for xy
                                           in zip(grid[:, 0], grid[:, 1])])

    gdf.to_file(shp_fn, driver='GeoJSON')
    # gdf.to_file(shp_fn)
    # 1) Opens the shape_file

    source_ds = ogr.Open(shp_fn)
    source_layer = source_ds.GetLayer()
    schema = []
    ldefn = source_layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        schema.append(fdefn.name)

    # 2) Creating the destination raster data source
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    cols = int((x_max - x_min) / res[0])
    rows = int((y_max - y_min) / res[1])

    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, cols, rows,
                                                     array.shape[1],
                                                     gdal.GDT_Float32)

    target_ds.SetGeoTransform((x_min, res[0], 0, y_max, 0, -res[1]))
    target_dsSRS = osr.SpatialReference()
    target_dsSRS.ImportFromEPSG(int(srs.split(':')[-1]))
    target_ds.SetProjection(target_dsSRS.ExportToWkt())

    for i in range(array.shape[1]):
        band = target_ds.GetRasterBand(i + 1)
        band.SetNoDataValue(-9999)  ##COMMENT 5
        gdal.RasterizeLayer(target_ds, [i + 1], source_layer,
                            options=['ATTRIBUTE=' + schema[i]])
    target_ds = None

    return shp_fn, raster_fn


def source_model2raster(array, datatype, storedir, filename, grid, res, srs='EPSG:4326'):
    """ Load ssm results from lon/lat results and save it into a multi-band
    raster. Creates a points-shapefile as middle step.
    Input:
        storedir (str): Directory to save the middle steps
        filename (str): Name of the files to be created
        array (ndarray): n-D array, in columns the different results
        array_names (list): contains the names of each array
        grid (ndarra): must contain the same rows of the array
        res (tuple): floats containing the x and y resolution

    Output:
        shapefile of name shp_fn
        tiff raster of name raster_fn


    """

    array = np.array(array)

    if array.ndim == 1:
        nlayers = 1
    if array.ndim == 2:
        nlayers = array.shape[1]


    shp_fn = join(storedir, filename + '.GeoJSON')
    raster_fn = join(storedir, filename + '.tiff')

    ### Read CSV file, convert and write it into shapefile

    df = pandas.DataFrame(array, columns=[f'{i}' for i in range(nlayers)])

    gdf = geopandas.GeoDataFrame(df,
                                 crs=srs,
                                 geometry=[Point(xy) for xy
                                           in zip(grid[:, 0], grid[:, 1])])

    gdf.to_file(shp_fn, driver='GeoJSON')
    # gdf.to_file(shp_fn)
    # 1) Opens the shape_file+

    source_ds = ogr.Open(shp_fn)
    source_layer = source_ds.GetLayer()
    schema = []
    ldefn = source_layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        schema.append(fdefn.name)

    # 2) Creating the destination raster data source
    # x_min, x_max, y_min, y_max = source_layer.GetExtent()
    grid[grid[:, 0] < 0, 0] += 360
    x_min, x_max, y_min, y_max = [np.min(grid[:, 0]), np.max(grid[:, 0]), np.min(grid[:, 1]), np.max(grid[:, 1])]

    cols = int((x_max - x_min) / res[0])
    rows = int((y_max - y_min) / res[1])
    dt = gdal.GDT_Float32
    if isinstance(datatype, float):
        dt = gdal.GDT_Float32
    elif isinstance(datatype, str):
        dt = gdal.GFT_String
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, cols, rows,
                                                     nlayers,
                                                     dt)

    target_ds.SetGeoTransform((x_min, res[0], 0, y_max, 0, -res[1]))
    target_ds_srs = osr.SpatialReference()
    target_ds_srs.ImportFromEPSG(int(srs.split(':')[-1]))
    target_ds.SetProjection(target_ds_srs.ExportToWkt())


    for i in range(nlayers):
        band = target_ds.GetRasterBand(i + 1)
        band.SetNoDataValue(-9999)  ##COMMENT 5
        gdal.RasterizeLayer(target_ds, [i + 1], source_layer,
                            options=['ATTRIBUTE=' + schema[i], 'at'])
    target_ds = None

    return shp_fn, raster_fn


def source_raster2vti(fname_vti, rasters, names,
               offset=0, nodata_val=-9999):
    """
    Creates 2D vtkImage. All rasters must be in the same
    projection, extent and dimension.

    Input:  fname_vti(str):  Filename of the results vti file
            rasters(list): list of string, pointing the rasters to be casted
                           onto the vtk
            arrays_names(list): List of lists of strings. Each sublists contains
                        the array names of each raster band.
                        e.g. [['PGA_a','SA(0.1)'_a], ['PGV_b','SA(0.4)'_b]]
            offset (int): Vertical elevation upon which to cast the vti.
            nodata_val(float): Nodata value of the raster
            mask_rgb(list of int): Color value to set transparency
    Last mod. 07/08/2020
    """

    for raster_path, name in zip(rasters, names):

        raster = gdal.Open(raster_path)
        n_bands = raster.RasterCount
        affine = raster.GetGeoTransform()

        dims = (raster.RasterXSize + 1, raster.RasterYSize + 1, 1)
        spacing = (affine[1], -affine[5], 0)
        origin = (affine[0] + offset[0], affine[3] - spacing[1] * (dims[1] - 1) + offset[1], offset[2])
        mask = None
        if raster_path == rasters[0]:
            image = pyvista.UniformGrid(dims=dims, spacing=spacing, origin=origin)

        Array = []
        for i in range(n_bands):
            array = np.flipud(raster.GetRasterBand(i + 1). \
                              ReadAsArray()).flatten(order='C')
            array[array == nodata_val] = np.nan
            Array.append(array)
            if i == 0:
                mask = np.isnan(array)

            else:
                mask += np.isnan(array)
        Array = np.ascontiguousarray(np.array(Array).T)
        image.cell_data[name] = Array
        if raster_path == rasters[0]:
            image.cell_data["mask"] = 1 - mask * 1
        image.save(fname_vti)
    # return image


if __name__ == '__main__':
    pass
