# -*- coding: utf-8 -*-
"""Test suite for `raster` module from the `landscape` package

Test the API

More comprehensive tests would include rasters with different
types (e.g. Byte, Float32, etc.)
"""
import os
import unittest
import numpy as np
from landscape.raster import raster

class TestMetadata(unittest.TestCase):
    """Test Metadata class in raster.py
    """
    def setUp(self):
        """Set up method
        """
        self.valid_filename = "test/data/test_image.tif"
        # This could be an iterable of invalid filenames instead
        self.invalid_filename = ""
        # Metadata
        width = 100
        height = 100    
        # A WKT projection
        projection = 'PROJCS["NAD_1983_StatePlane_Puerto_Rico_Virgin_Islands\
_FIPS_5200",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID\
["GRS 1980",6378137,298.2572221010042,AUTHORITY["EPSG","7019"]],AUTHORI\
TY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.01745329251994\
33],AUTHORITY["EPSG","4269"]],PROJECTION["Lambert_Conformal_Conic_2SP"]\
,PARAMETER["standard_parallel_1",18.03333333333334],PARAMETER["standard\
_parallel_2",18.43333333333333],PARAMETER["latitude_of_origin",17.83333\
333333333],PARAMETER["central_meridian",-66.43333333333334],PARAMETER["\
false_easting",200000],PARAMETER["false_northing",200000],UNIT["metre"\
,1,AUTHORITY["EPSG","9001"]]]'
        geotransform = (266842.0, 0.2999681224099458, 0.0, 
            249454.0, 0.0, -0.29997741133950756)
        datatype = 1 # equivalent to gdal.GDT_Byte
        ndv = None # there is no ndv so assign to None
        self.valid_metadata_dict = {
            "x": width,
            "y": height,
            "projection":projection,
            "geotransform":geotransform,
            "datatype":datatype,
            "ndv":ndv
        }
        self.invalid_metadata_dict = {
            "x": None,
            "y": None,
            "projection": None,
            "geotransform": None,
            "datatype": None,
            "ndv": None
        }

    def tearDown(self):
        """Tear down method
        """
        del self.valid_filename
        del self.invalid_filename

    def test_no_filename(self):
        """Test instantiation of Metadata object without filename
        """
        metadata = raster.Metadata()
        self.assertTrue(isinstance(metadata, raster.Metadata))

    def test_valid_filename(self):
        """Test instantiation of Metadata object with valid filename
        """
        metadata = raster.Metadata(self.valid_filename)
        self.assertTrue(isinstance(metadata, raster.Metadata))

    def test_invalid_filename(self):
        """Test failure to instantiate Metadata object with invalid filename
        """
        with self.assertRaises(AssertionError):
            metadata = raster.Metadata(self.invalid_filename)

class TestLoadImage(unittest.TestCase):
    """Test the load_image() function in raster.py
    """
    def setUp(self):
        """Set up method
        """      
        self.valid_filename = "test/data/test_image.tif"
        self.invalid_filename = ""

    def tearDown(self):
        """Tear down method
        """
        del self.valid_filename
        del self.invalid_filename

    def test_valid_filename(self):
        """Test image loads with valid filename
        """
        image = raster.load_image(self.valid_filename)
        self.assertTrue(isinstance(image, np.ndarray))

    def test_invalid_filename(self):
        """Test failure to load with invalid filename
        """
        with self.assertRaises(AssertionError):
            image = raster.load_image(self.invalid_filename)


class TestLoadMask(unittest.TestCase):
    """Test the load_mask() function in raster.py
    """
    def setUp(self):
        """Set up method
        """    
        self.valid_filename = "test/data/test_mask.tif"
        self.invalid_filename = ""

    def tearDown(self):
        """Tear down method
        """
        del self.valid_filename
        del self.invalid_filename

    def test_valid_filename(self):
        """Test load success with valid filename
        """
        image = raster.load_image(self.valid_filename)
        self.assertTrue(isinstance(image, np.ndarray))

    def test_invalid_filename(self):
        """Test load failure with invalid filename
        """
        with self.assertRaises(AssertionError):
            image = raster.load_image(self.invalid_filename)

class TestSaveImage(unittest.TestCase):
    """Test the save_image() function in raster.py
    """
    def setUp(self):
        """Set up method
        """
        self.valid_filename = "test/data/temp/test_output.tif"
        try:
            if os.path.isfile(self.valid_filename):
                os.remove(self.valid_filename)
        except Exception as e:
            print("Unable to remove the temporary file used", 
                " to test raster.save_image().")
        # could use an iterable of invalid filenames instead
        self.invalid_filename = "test/data/directory/test_output.tif"
        # height, width, and depth of ndarray
        width = 100
        height = 100
        depth = 4        
        self.valid_image = np.zeros((height, width, depth))
        self.invalid_image = None
        # Metadata
        # A WKT projection
        projection = 'PROJCS["NAD_1983_StatePlane_Puerto_Rico_Virgin_Islands\
_FIPS_5200",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID\
["GRS 1980",6378137,298.2572221010042,AUTHORITY["EPSG","7019"]],AUTHORI\
TY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.01745329251994\
33],AUTHORITY["EPSG","4269"]],PROJECTION["Lambert_Conformal_Conic_2SP"]\
,PARAMETER["standard_parallel_1",18.03333333333334],PARAMETER["standard\
_parallel_2",18.43333333333333],PARAMETER["latitude_of_origin",17.83333\
333333333],PARAMETER["central_meridian",-66.43333333333334],PARAMETER["\
false_easting",200000],PARAMETER["false_northing",200000],UNIT["metre"\
,1,AUTHORITY["EPSG","9001"]]]'
        #geotransform = None  # fill
        geotransform = (266842.0, 0.2999681224099458, 0.0, 
            249454.0, 0.0, -0.29997741133950756)
        #datatype = None # gdal.GDT_Byte # fill
        datatype = 1 # equivalent to gdal.GDT_Byte
        ndv = None # there is no ndv so assign to None
        self.valid_metadata_dict = {
            "x": width,
            "y": height,
            "projection":projection,
            "geotransform":geotransform,
            "datatype":datatype,
            "ndv":ndv
        }
        # new empty metadata
        self.valid_metadata = raster.Metadata() # empty
        self.valid_metadata.set(**self.valid_metadata_dict)
        self.invalid_metadata = None

    def tearDown(self):
        """Tear down method
        """
        try:
            if os.path.isfile(self.valid_filename):
                os.remove(self.valid_filename)
        except Exception as e:
            print("Unable to remove the temporary file used", 
                " to test raster.save_image().")
        del self.valid_filename
        del self.invalid_filename
        del self.valid_image
        del self.invalid_image
        del self.valid_metadata_dict 
        del self.valid_metadata
        del self.invalid_metadata

    def test_valid_filename(self):
        """Test write output with valid filename
        """
        raster.save_image(self.valid_filename, 
            self.valid_image, self.valid_metadata)
        self.assertTrue(os.path.exists(self.valid_filename))

    def test_invalid_filename(self):
        """Test load failure with invalid filename
        """
        with self.assertRaises(AssertionError):
            image = raster.save_image(self.invalid_filename, 
                self.valid_image, self.valid_metadata)

class TestFullCircle(unittest.TestCase):
    """Test that reading input, writing output, then reloading works
    """
    def setUp(self):
        """Set up method
        """
        self.input_filename = "test/data/test_image.tif"
        self.output_filename = (
            "test/data/temp/test_image_output.tif")
        try:
            if os.path.isfile(self.output_filename):
                os.remove(self.output_filename)
        except Exception as e:
            print("Unable to remove the temporary file used", 
                " to test raster.save_image().")        

    def tearDown(self):
        """Tear down method
        """
        try:
            if os.path.isfile(self.output_filename):
                os.remove(self.output_filename)
        except Exception as e:
            print("Unable to remove the temporary file used", 
                " to test raster.save_image().")
        del self.input_filename
        del self.output_filename     

    def test_full_circle(self):
        """Test load-save-load cycle maintains image and metadata integrity
        """
        image = raster.load_image(self.input_filename)
        metadata = raster.Metadata(self.input_filename)
        raster.save_image(self.output_filename, image, 
            metadata)
        image_reloaded = raster.load_image(self.output_filename)
        metadata_reloaded = raster.Metadata(self.output_filename)
        self.assertTrue(np.array_equal(image, image_reloaded))
        self.assertTrue(metadata == metadata_reloaded)
        self.assertTrue(metadata.datatype == metadata_reloaded.datatype)
