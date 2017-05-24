# Copyright 2017 Jeffrey A. Wolf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#-------------------------------------------------------------------------
"""Metadata class and raster input and output functions
"""
import os
import numpy as np
import gdal
import osr

class Metadata(object):
    """Geographical metadata for `load_image()` and `save_image()`
    """
    # Band types supported for reading and writing of GeoTiff images 
    # www.gdal.org/frmt_gtiff.html
    GDAL_DATATYPES = [gdal.GDT_Byte,
                      gdal.GDT_UInt16,
                      gdal.GDT_Int16,
                      gdal.GDT_UInt32,
                      gdal.GDT_Int32,
                      gdal.GDT_Float32,
                      gdal.GDT_Float64,
                      gdal.GDT_CInt16,
                      gdal.GDT_CInt32,
                      gdal.GDT_CFloat32,
                      gdal.GDT_CFloat64]

    def __init__(self, filename=None):
        """Construct geographical metadata for an image

        Args:
            filename: None (default) or a file name path
        Returns:
            A `Metadata` object
        Raises:
            AssertionError
        """
        self._x = None
        self._y = None
        self._projection = None
        self._geotransform = None
        self._datatype = None
        self_ndv = None
        # initialize with metadata from <filename> file
        if filename is not None:
            # assertions
            assert os.path.exists(filename), (
                "Invalid path to <filename> file.")
            dataset = gdal.Open(filename, gdal.GA_ReadOnly)
            assert dataset # assert that dataset is not None
            self._projection = dataset.GetProjectionRef() # wkt
            # use osr.SpatialReference class to validate projection
            # http://www.gdal.org/osr_tutorial.html
            spatial_reference = osr.SpatialReference()
            spatial_reference.ImportFromWkt(self._projection)
            isprojected = spatial_reference.IsProjected() 
            assert isprojected, ("WKT projection not parsed by OGR")
            self._geotransform = dataset.GetGeoTransform()
            assert len(self._geotransform) == 6, (
                "geotransform must have 6 elements")
            assert False not in [
                isinstance(x, float) for x in self._geotransform], (
                "geotransform elements must be float type")
            band = dataset.GetRasterBand(1)
            assert band # assert that band is not None
            # what is this if there is no ndv?
            self._ndv = band.GetNoDataValue()
            # assert ndv
            self._x = band.XSize
            assert isinstance(self._x, int)
            self._y = band.YSize
            assert isinstance(self._y, int)
            self._datatype = band.DataType
            assert self._datatype in Metadata.GDAL_DATATYPES

    @property
    def x(self):
        """The size of the image along the x-axis (width).
        """
        return self._x

    @property
    def y(self):
        """The size of the image along the y-axis (height)
        """
        return self._y

    @property
    def projection(self):
        """The well-known text projection
        """
        return self._projection

    @property
    def geotransform(self):
        """The geotransform list
        """
        return self._geotransform

    @property
    def datatype(self):
        """The GDAL DataType
        """
        return self._datatype

    @property
    def ndv(self):
        """The no data value
        """
        return self._ndv

    def set(self, x, y, projection, geotransform, 
        datatype=gdal.GDT_Float32, ndv=None):
        """Set the metadata for new file

        Args:
            x: `int` x size
            y: `int` y size
            projection: a valid WKT projection
            geotransform: a list of six floating point numbers representing an 
                affine GeoTransform as described in the GDAL data model 
                (http://www.gdal.org/gdal_datamodel.html)
            datatype: gdal.GDT_Float32 (default) or any GeoTiff supported
                data type (www.gdal.org/frmt_gtiff.html).
            ndv: None (default) or any number representable in datatype
        Returns:
            None
        Raises:
            AssertonError if invalid types are used to initialize the Metadata
        """
        # assert's about x
        assert isinstance(x, int)
        self._x = x
        # assert's about y
        assert isinstance(y, int)
        self._y = y
        # use osr.SpatialReference class to validate projection
        # http://www.gdal.org/osr_tutorial.html
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromWkt(projection)
        isprojected = spatial_reference.IsProjected() 
        assert isprojected, ("WKT projection not parsed by OGR")
        self._projection = projection
        assert len(geotransform) == 6, ("geotransform must have 6 elements")
        assert False not in [isinstance(x, float) for x in geotransform],(
            "geotransform elements must be float type")
        self._geotransform = geotransform
        assert datatype in Metadata.GDAL_DATATYPES, (
            "datatype is not recognized as a valid GDAL datatype for GeoTiff.")
        self._datatype = datatype
        # assert's about ndv
        self._ndv = ndv

    def create(self):
        """
        Args:
            None
        Returns:
            x: x-dimension size
            y: y-dimension size
            datatype: GDAL DataType
        Raises:
            AssertionError
        """
        assert isinstance(self.x, int), ("Metadata is uninitialized")
        assert isinstance(self.y, int), ("Metadata is uninitialized")
        assert self.datatype in Metadata.GDAL_DATATYPES, (
            "Invalid GDAL DataType")
        return self.x, self.y, self.datatype

    
    def __eq__(self, other):
        """Test two Metadata objects for spatial equality.

        Test if two `Metadata` objects have the geotransform and 
        projection, x, and y properties. This test is used to 
        evaluate whether two images can use the same horizontal 
        indexing operations and maintain spatial consistency. 

        Args:
            other: another `Metadata` object
        Returns:
            `bool` object
        Raises:
            AssertionError
        """
        assert isinstance(other, Metadata), (
            "other is not a valid Metadata object")
        
        return (repr(self.geotransform) == repr(other.geotransform) and
            self.projection == other.projection and self.x == other.x 
            and self.y == other.y)


def load_image(filename=None):
    """Load an image

    Loads an image as an `numpy.ndarray` with dim's (H,W,D). H is the
    height (y size), W is the width (x size), and D is the depth (the
    number of channels in the image).

    Args:
        filename: `string` path to a gdal compatable image file
    Returns:
        numpy representation of an image with array shape (H,W,D)
    Raises:
        AssertionError
    """
    # assertions
    assert filename is not None, ("Filename cannot be None")
    assert os.path.exists(filename), (
        "Invalid path to <filename> file.")
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    assert dataset, ("GDAL could not open {}".format(filename))
    # read image
    bands = [dataset.GetRasterBand(i+1).ReadAsArray() for i in range(
        dataset.RasterCount)]
    dataset = None
    return np.stack(bands, 2)

def load_mask(filename):
    """Load a mask

    Loads a mask as a `np.ndarray` with dtype `np.float32`. Invalid
    values are stored as `np.nan`
    
    Args:
        filename: `string` path to a gdal compatable image file
    Returns:
        numpy representation of an image mask with array shape (H,W,D)
    Raises:
        AssertionError from `load_image` or `Metadata` initialization
    """
    mask_in = load_image(filename)
    mask = mask_in.astype(np.float32)
    metadata = Metadata(filename)
    mask[mask == metadata.ndv] = np.NAN
    return mask

def save_image(filename, image, metadata):
    """Save an image

    Saves an image as a GeoTiff.

    Args:
        image: a numpy `ndarray` with array shape (H,W,D)
        metadata: object of class `Metadata`
        filename: `string` a valid system path
    Returns:
        None
    Raises:
        AssertionError
    """
    # assertions
    path = os.path.dirname(filename)
    assert path == "" or os.path.exists(path), ("Invalid directory name")
    assert isinstance(image, np.ndarray), ("image must be a numpy.ndarray")
    assert len(image.shape) == 3, ("image must be an numpy.ndarray with shape (H,W,D)")
    rows = image.shape[0]
    cols = image.shape[1]
    n_bands = image.shape[2]
    assert isinstance(metadata, Metadata)
    geotransform = metadata.geotransform
    assert len(geotransform) == 6, ("Geotransform must be 6 elements")
    # use osr.SpatialReference class to validate projection
    # http://www.gdal.org/osr_tutorial.html
    projection = metadata.projection
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(projection)
    isprojected = spatial_reference.IsProjected()
    assert isprojected, ("WKT projection not parsed by OGR")
    x, y, datatype = metadata.create()
    assert y == rows
    assert x == cols
    # could check that datatype is appropriate for dtype of image
    assert datatype in Metadata.GDAL_DATATYPES, (
        "datatype is not recognized as a valid GDAL datatype for GeoTiff.")   
    ndv = metadata.ndv

    # save image
    format = "GTiff"
    driver = gdal.GetDriverByName(format)    
    dataset = driver.Create(filename, x, y, n_bands, datatype)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    depth_axis_len = image.shape[2]
    for depth in range(depth_axis_len):
        band = depth + 1
        dataset.GetRasterBand(band).WriteArray(image[:,:, depth])
        if band == 1 and ndv is not None:
            dataset.GetRasterBand(1).SetNoDataValue(ndv)
    dataset = None