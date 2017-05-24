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
"""Surface topography analysis using Gaussian kernels
"""
import os
import argparse
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_gradient_magnitude
from scipy.ndimage.filters import gaussian_laplace
from landscape.raster import raster

class GaussianTopography:
    """A class to analyze digital elevation model's using Gaussian kernels
    and their derivatives.
    """
    def __init__(self, filename):
        """Initialize an instance of GaussianTopography class.

        Parameters
        ----------
        filename : `str`
            An input file path to a digital elevation model (DEM)

        Returns
        -------
        result : ndarray
            The result of calculating aspect on the digital elevation model.

        Raises
        ------
        AssertionError            
        """
        assert os.path.isfile(filename), ("Invalid filename for DEM")
        self.data = raster.load_image(filename)
        self.metadata = raster.Metadata(filename)
        self.shape = self.data.shape

    def _assert_filename(self, filename):
        """Check if filename is valid for output filename
        Args:
            filename:  `str` or `None`
        Raises:
            AssertionError
        """
        assert (filename is None or os.path.exists(
            os.path.dirname(filename)) or 
            os.path.dirname(filename) == ""), (
            "Invalid filename argument")

    def filter(self, bw, filename=None):
        """Convolution with a Gaussian Kernel of specified bandwidth.

        This applies a Gaussian Kernel to mean filter the DEM.

        Parameters
        ----------
        bw : `float` 
            a bandwidth parameter for the 2D Gaussian kernel
        filename : `str` , optional. 
            A `str` file path. The directory of the file must exist. 
            Default is `None`.

        Returns
        -------
        result : ndarray
            The result of mean filtering of the digital elevation model with 
            a Gaussian Kernel.

        Raises
        ------
        AssertionError
        """
        self._assert_filename(filename)
        image = np.zeros_like(self.data)
        image[:,:,0] = gaussian_filter(self.data[:,:,0], bw, order=0)
        if filename:
            raster.save_image(filename, image, self.metadata)
        return image

    def gradient_magnitude(self, bw, filename=None):
        """Convolution with the first derivative of 2D Gaussian Kernel of 
        specified bandwidth.

        Returns the 2D gradient (i.e. slope) of the DEM.

        Parameters
        ----------
        bw : `float` 
            a bandwidth parameter for the 2D Gaussian kernel
        filename : `str` , optional. 
            A `str` file path. The directory of the file must exist. 
            Default is `None`.

        Returns
        -------
        result : ndarray
            The result of convolution of the digital elevation model with 
            the gradient of a Gaussian Kernel.

        Raises
        ------
        AssertionError
        """ 
        self._assert_filename(filename)
        image = np.zeros_like(self.data)
        #image[:,:,0] = np.absolute(
        #    gaussian_filter(self.data[:,:,0], bw, order=1))
        image[:,:,0] = gaussian_gradient_magnitude(
            self.data[:,:,0], bw)
        if filename:
            raster.save_image(filename, image, self.metadata)
        return image

    def gradient_dx(self, bw, filename=None):
        """Convolution with the first derivative of 1D Gaussian Kernel of 
        specified bandwidth along the x-axis.

        Returns the first partial derivative along the x-axis.

        Parameters
        ----------
        bw : `float` 
            a bandwidth parameter for the 1D Gaussian kernel
        filename : `str` , optional. 
            A `str` file path. The directory of the file must exist. 
            Default is `None`.

        Returns
        -------
        result : ndarray
            The result of convolution of the digital elevation model with
            the first partial derivative along the x-axis.

        Raises
        ------
        AssertionError
        """
        self._assert_filename(filename)
        image = np.zeros_like(self.data)
        image[:,:,0] = gaussian_filter1d(self.data[:,:,0], bw, axis=1, order=1)
        if filename:
            raster.save_image(filename, image, self.metadata)
        return image

    def gradient_dy(self, bw, filename=None):
        """Convolution with the first derivative of 1D Gaussian Kernel of 
        specified bandwidth along the y-axis.

        Returns the first partial derivative along the y-axis.

        Parameters
        ----------
        bw : `float` 
            a bandwidth parameter for the 2D Gaussian kernel
        filename : `str` , optional. 
            A `str` file path. The directory of the file must exist. 
            Default is `None`.

        Returns
        -------
        result : ndarray
            The result of convolution of the digital elevation model with
            the first partial derivative along the y-axis.

        Raises
        ------
        AssertionError
        """
        self._assert_filename(filename)
        image = np.zeros_like(self.data)
        # multiplied by -1.0 because want the gradient relative to
        # north-south axis instead of the "image gradient" which is
        # in image space - i.e. with row origin at top of image. 
        image[:,:,0] = -1.0 * gaussian_filter1d(self.data[:,:,0], 
            bw, axis=0, order=1)
        if filename:
            raster.save_image(filename, image, self.metadata)
        return image

    def aspect(self, bw, filename=None):
        """Aspect (radians) from the angular direction of the vector of 
        first partial derivatives.

        Angle of the vector (dz/dx, dz/dy) using arctan2

        Parameters
        ----------
        bw : `float` 
            a bandwidth parameter for the 2D Gaussian kernel
        filename : `str` , optional. 
            A `str` file path. The directory of the file must exist. 
            Default is `None`.

        Returns
        -------
        result : ndarray
            The result of calculating aspect on the digital elevation model.

        Raises
        ------
        AssertionError
        """
        self._assert_filename(filename)
        image = np.zeros_like(self.data)
        # aspect calculation here
        # gradient dx must be multiplied by -1 so positive downslope
        # T = np.array([[-1.,0.],[0,-1]])
        # because the gradients already are facing in opposite directions
        # of the downslope direction no multiplication by T is needed
        image[:,:,0] = np.arctan2(self.gradient_dx(bw)[:,:,0], 
            self.gradient_dy(bw)[:,:,0])*180.0/np.pi + 180.0
        if filename:
            raster.save_image(filename, image, self.metadata)
        return image

    def laplacian(self, bw, filename=None):
        """Convolution with the Laplacian of a 2D Gaussian Kernel of 
        specified bandwidth.

        Parameters
        ----------
        bw : `float` 
            a bandwidth parameter for the 2D Gaussian kernel
        filename : `str` , optional. 
            A `str` file path. The directory of the file must exist. 
            Default is `None`.

        Returns
        -------
        result : ndarray
            The result of calculating aspect on the digital elevation model.

        Raises
        ------
        AssertionError
        """
        self._assert_filename(filename)
        image = np.zeros_like(self.data)
        image[:,:,0] = gaussian_laplace(self.data[:,:,0], bw)
        if filename:
            raster.save_image(filename, image, self.metadata)
        return image

def get_args(): 
    """Parse command line arguments when run as a script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("raster", 
        help="`str` path to a digital elevation model (DEM)",
        type=str)
    parser.add_argument("bandwidth", 
        help="`float` bandwidth of the Gaussian Kernel",
        type=float)
    parser.add_argument("directory", help="`str` path to an output directory",
        type=str)
    args = parser.parse_args()
    return args

def main():
    "Command line script"
    args = get_args()
    assert os.path.exists(args.raster), ("DEM not found")
    assert args.bandwidth > 0, ("The bandwidth must be > 0")
    assert os.path.exists(args.directory), (
        "Output directory does not exist")
    topography = GaussianTopography(args.raster)
    topography.filter(args.bandwidth, os.path.join(args.directory, 
        "filter.tif"))
    topography.gradient_magnitude(args.bandwidth, os.path.join(args.directory, 
        "gradient.tif"))
    topography.gradient_dx(args.bandwidth, os.path.join(args.directory, 
        "gradient_dx.tif"))
    topography.gradient_dy(args.bandwidth, os.path.join(args.directory, 
        "gradient_dy.tif"))
    topography.aspect(args.bandwidth, os.path.join(args.directory, 
        "aspect.tif"))
    topography.laplacian(args.bandwidth, os.path.join(args.directory, 
        "laplacian.tif"))    

if __name__ == "__main__":
    main()