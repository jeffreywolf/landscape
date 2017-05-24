# -*- coding: utf-8 -*-
"""Test suite for `topography` module from the `landscape` package
"""
from landscape.topography import topography
import unittest
import os
import shutil
import numpy as np

class TestGaussianTopographyInitialize(unittest.TestCase):
    """
    """
    def setUp(self):
        """Setup method
        """
        self.valid_input_filename = "test/data/test_dem.tif"
        self.invalid_input_filename = "test/data/no_such_file.tif"

    def tearDown(self):
        """Tear down method
        """
        del self.valid_input_filename
        del self.invalid_input_filename

    def test_invalid_dem_filename(self):
        """Test instantiation failure with invalid filename
        """
        with self.assertRaises(AssertionError):
            gt = topography.GaussianTopography(self.invalid_input_filename)

    def test_valid_dem_filename(self):
        """Test instantiation success with valid filename
        """
        gt = topography.GaussianTopography(self.valid_input_filename)
        self.assertTrue(isinstance(gt, topography.GaussianTopography))

# Remaining test cases subclass TestGaussianTopographyMethod
class TestGaussianTopographyMethod(unittest.TestCase):
    """Superclass
    """
    def setUp(self):
        """Setup method
        """
        self.valid_input_filename = "test/data/test_dem.tif"
        self.valid_output_filename = "test/data/temp/test.tif"
        self.invalid_output_filename = (
            "test/data/no_such_directory/test.tif")
        self.gt = topography.GaussianTopography(self.valid_input_filename)
        self.bw = 5
        try:
            if os.path.isfile(self.valid_output_filename):
                os.remove(self.valid_output_filename)
        except Exception as e:
            print("Unable to remove the temporary file used", 
                " to test raster.save_image().")

    def tearDown(self):
        """Tear down method
        """
        try:
            if os.path.isfile(self.valid_output_filename):
                os.remove(self.valid_output_filename)
        except Exception as e:
            print("Unable to remove the temporary file used", 
                " to test raster.save_image().")
        del self.valid_input_filename
        del self.valid_output_filename
        del self.invalid_output_filename
        del self.gt
        del self.bw


class TestGaussianTopographyFilter(TestGaussianTopographyMethod):
    """Subclass of `TestGaussianTopographyMethod`
    """
    def test_valid_filename(self):
        """Test write success with valid output filename
        """
        self.gt.filter(self.bw, self.valid_output_filename)
        self.assertTrue(os.path.exists(self.valid_output_filename))

    def test_no_filename(self):
        """Test valid return of array without an output filename
        """
        array = self.gt.filter(self.bw)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_invalid_filename(self):
        """Test failure of output with invalid filename
        """
        with self.assertRaises(AssertionError):
            self.gt.filter(self.bw, self.invalid_output_filename)

    def test_image_returned(self):    
        """Test an ndarray returned
        """
        array_no_filename = self.gt.filter(self.bw)
        self.assertTrue(isinstance(array_no_filename, np.ndarray))
        array_filename = self.gt.filter(self.bw, self.valid_output_filename)
        self.assertTrue(isinstance(array_filename, np.ndarray))


class TestGaussianTopographyGradientMagnitude(TestGaussianTopographyMethod):
    """Subclass of `TestGaussianTopographyMethod`
    """
    def test_valid_filename(self):
        """Test write success with valid output filename
        """
        self.gt.gradient_magnitude(self.bw, self.valid_output_filename)
        self.assertTrue(os.path.exists(self.valid_output_filename))

    def test_no_filename(self):
        """Test valid return of array without an output filename
        """
        array = self.gt.gradient_magnitude(self.bw)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_invalid_filename(self):
        """Test failure of output with invalid filename
        """
        with self.assertRaises(AssertionError):
            self.gt.gradient_magnitude(self.bw, self.invalid_output_filename)

    def test_image_returned(self):    
        """Test an ndarray returned
        """
        array_no_filename = self.gt.gradient_magnitude(self.bw)
        self.assertTrue(isinstance(array_no_filename, np.ndarray))
        array_filename = self.gt.gradient_magnitude(
            self.bw, self.valid_output_filename)
        self.assertTrue(isinstance(array_filename, np.ndarray))


class TestGaussianTopographyGradientDx(TestGaussianTopographyMethod):
    """Subclass of `TestGaussianTopographyMethod`
    """
    def test_valid_filename(self):
        """Test write success with valid output filename
        """
        self.gt.gradient_dx(self.bw, self.valid_output_filename)
        self.assertTrue(os.path.exists(self.valid_output_filename))

    def test_no_filename(self):
        """Test valid return of array without an output filename
        """
        array = self.gt.gradient_dx(self.bw)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_invalid_filename(self):
        """Test failure of output with invalid filename
        """
        with self.assertRaises(AssertionError):
            self.gt.gradient_dx(self.bw, self.invalid_output_filename)

    def test_image_returned(self):    
        """Test an ndarray returned
        """
        array_no_filename = self.gt.gradient_dx(self.bw)
        self.assertTrue(isinstance(array_no_filename, np.ndarray))
        array_filename = self.gt.gradient_dx(
            self.bw, self.valid_output_filename)
        self.assertTrue(isinstance(array_filename, np.ndarray))


class TestGaussianTopographyGradientDy(TestGaussianTopographyMethod):
    """Subclass of `TestGaussianTopographyMethod`
    """
    def test_valid_filename(self):
        """Test write success with valid output filename
        """
        self.gt.gradient_dy(self.bw, self.valid_output_filename)
        self.assertTrue(os.path.exists(self.valid_output_filename))

    def test_no_filename(self):
        """Test valid return of array without an output filename
        """
        array = self.gt.gradient_dy(self.bw)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_invalid_filename(self):
        """Test failure of output with invalid filename
        """
        with self.assertRaises(AssertionError):
            self.gt.gradient_dy(self.bw, self.invalid_output_filename)

    def test_image_returned(self):    
        """Test an ndarray returned
        """
        array_no_filename = self.gt.gradient_dy(self.bw)
        self.assertTrue(isinstance(array_no_filename, np.ndarray))
        array_filename = self.gt.gradient_dy(
            self.bw, self.valid_output_filename)
        self.assertTrue(isinstance(array_filename, np.ndarray))


class TestGaussianTopographyLaplacian(TestGaussianTopographyMethod):
    """Subclass of `TestGaussianTopographyMethod`
    """
    def test_valid_filename(self):
        """Test write success with valid output filename
        """
        self.gt.laplacian(self.bw, self.valid_output_filename)
        self.assertTrue(os.path.exists(self.valid_output_filename))

    def test_no_filename(self):
        """Test valid return of array without an output filename
        """
        array = self.gt.laplacian(self.bw)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_invalid_filename(self):
        """Test failure of output with invalid filename
        """
        with self.assertRaises(AssertionError):
            self.gt.laplacian(self.bw, self.invalid_output_filename)

    def test_image_returned(self):    
        """Test an ndarray returned
        """
        array_no_filename = self.gt.laplacian(self.bw)
        self.assertTrue(isinstance(array_no_filename, np.ndarray))
        array_filename = self.gt.laplacian(
            self.bw, self.valid_output_filename)
        self.assertTrue(isinstance(array_filename, np.ndarray))

class TestGaussianTopographyAspect(TestGaussianTopographyMethod):
    """Subclass of `TestGaussianTopographyMethod`
    """
    def test_valid_filename(self):
        """Test write success with valid output filename
        """
        self.gt.aspect(self.bw, self.valid_output_filename)
        self.assertTrue(os.path.exists(self.valid_output_filename))

    def test_no_filename(self):
        """Test valid return of array without an output filename
        """
        array = self.gt.aspect(self.bw)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_invalid_filename(self):
        """Test failure of output with invalid filename
        """
        with self.assertRaises(AssertionError):
            self.gt.aspect(self.bw, self.invalid_output_filename)

    def test_image_returned(self):    
        """Test an ndarray returned
        """
        array_no_filename = self.gt.aspect(self.bw)
        self.assertTrue(isinstance(array_no_filename, np.ndarray))
        array_filename = self.gt.aspect(
            self.bw, self.valid_output_filename)
        self.assertTrue(isinstance(array_filename, np.ndarray))                           