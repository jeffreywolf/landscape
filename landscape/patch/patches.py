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
"""Create small cropped images
"""
import numpy as np
import os
from landscape.raster import raster

class Patches(object):
    """Patches class is a sequence of Patch objects
    """
    def __init__(self, image_filename, labels_filename, size):
        """Initialize a Patches object

        The image at image_filename and the labels at labels_filename 
        must have the same projection, geotransform, and extent.

        Args:
            image_filename: filename representing path to image 
            labels_filename: filename representing path to labels
            size: `int` size of window side
        Returns:
            None
        Raises:
            AssertionError
        """
        self._image = None
        self._labels = None
        self._image_metadata = None
        self._labels_metadata = None
        self._size = None
        self._offset = None
        self._labeled_indices = None # initialized in __iter__()
        self._count = None # initialized in __iter__()
        self._max_iter = None # initialized in __iter__()
        # assert valid files
        assert os.path.exists(image_filename), ("image file not found")
        assert os.path.exists(labels_filename), ("labels file not found")
        # assert equal metadata 
        image_metadata = raster.Metadata(image_filename)
        labels_metadata = raster.Metadata(labels_filename)     
        assert image_metadata == labels_metadata, (
            "Metadata are not equivalent. " + 
            "Try `gdalinfo` on the files. " + 
            "Look at the docstring for `raster.Metadata.__eq__()`.")
        assert labels_metadata.ndv is not None, (
            "labels metadata ndv is None")
        self._image_metadata = image_metadata
        self._labels_metadata = labels_metadata
        # asserts on image and labels np.ndarrays
        image = raster.load_image(image_filename)
        labels = raster.load_image(labels_filename)
        assert isinstance(image, np.ndarray), (
            "image must be a numpy.ndarray")
        assert len(image.shape) == 3, (
            "image must be an numpy.ndarray with shape (H,W,D)")
        assert isinstance(labels, np.ndarray), (
            "labels must be a numpy.ndarray")
        assert len(labels.shape) == 3, (
            "lables must be an numpy.ndarray with shape (H,W,D)")
        # test if shape of both is equal on H,W axes
        assert image.shape[0] == labels.shape[0], (
            "Image and label height is different")
        assert image.shape[1] == labels.shape[1], (
            "Image and label height is different")        
        self._image = image 
        self._labels = labels
        # assert on size
        assert isinstance(size, int), ("size must be an integer")
        assert size % 2 == 1, ("size must be an odd integer")
        assert size > 1, ("size must be an integer >1")
        self._size = size
        self._offset = self.size // 2

    @property
    def image(self):
        """The image `np.ndarray` with shape (H,W,D)
        """
        return self._image

    @property 
    def labels(self):
        """The labels `np.ndarray` with shape (H,W,D)
        """
        return self._labels
    
    @property 
    def image_metadata(self):
        """The image `Metadata` object
        """
        return self._image_metadata

    @property 
    def labels_metadata(self):
        """The labels `Metadata` object
        """
        return self._labels_metadata

    @property 
    def size(self):
        """The `int` size of the side length.

        Must be an odd `int`
        """
        return self._size

    @property 
    def offset(self):
        """The `int` offset derived from self._size//2

        An even integer
        """
        return self._offset

    @property 
    def labeled_indices(self):
        """An indices iterator to access labeled pixels
        """
        return self._labeled_indices

    def _calculate_origin(self, origin, resolution, offset, index):
        """Calculate new origin

        Args:
            origin: `float`
            resolution: `float` that can be positive or negative
            offset: `int` pixel offset
            index: `int` index
        Returns:
            new origin `float`
        Raises:
            AssertionError
        """
        assert isinstance(index, int)
        assert isinstance(offset, int)
        resolution_string = str(resolution)
        parts = resolution_string.split(".")
        if len(parts) == 2:
            precision = len(parts[1])
        else:
            precision = 0
        # calculate difference
        difference = (index - offset) * resolution
        origin += difference
        return round(origin, precision)

    def _build_geotransform(self, i, j):
        """Build geotransform for an image patch

        Args:
            i: `int` row index
            j: `int` column index
        Returns:
            GDAL geotransform for `Metadata` object
        Raises:
            AssertionError
        """
        assert isinstance(i, int), ("i is not an integer")
        assert isinstance(j, int), ("j is not an integer")
        x_origin, x_res, x_ignore, y_origin, y_ignore, y_res = (
            self.image_metadata.geotransform)
        # integer conversion to reduce floating point error
        new_x_origin = self._calculate_origin(x_origin, x_res, self.offset, j)
        new_y_origin = self._calculate_origin(y_origin, y_res, self.offset, i)
        geotransform = (new_x_origin, x_res, x_ignore, new_y_origin, 
            y_ignore, y_res)            
        return geotransform 

    def _patch_metadata(self, i, j):
        """Build metadata for an image patch
        
        Uses self.image_metadata as the metadata source. Modifies
        the geotransform, x, and y size. Keeps the same projection,
        datatype, and ndv. 

        Args:
            i: `int`  row index into image and labels `np.ndarray`
            j: `int` col index into image and labels `np.ndarray`
        Returns:
            `raster.Metadata` object 
        Raises:

        """
        assert isinstance(i, int), ("i is not an integer")
        assert i >= 0, ("i must be >= 0")
        assert isinstance(j, int), ("j is not an integer")
        assert j >= 0, ("j must be >= 0")
        # modify the geotransform
        geotransform = self._build_geotransform(i, j)
        # modify the x and y size
        x, y = self.size, self.size
        # projection
        projection = self.image_metadata.projection
        # datatype
        datatype = self.image_metadata.datatype
        # ndv
        ndv = self.image_metadata.ndv
        metadata = raster.Metadata()
        metadata.set(x, y, projection, geotransform, datatype, ndv)
        return metadata

    def _patch_image(self, i, j):
        """Build an image patch

        Args:
            i: `int` row index into image and labels `nd.ndarray`
            j: `int` row index into image and labels `nd.ndarray`
            size:
        Returns:
            `np.ndarray`
        Raises:
            AssertionError
        """
        assert isinstance(i, int), ("i is not an integer")
        assert i >= 0, ("i must be >= 0")
        assert isinstance(j, int), ("j is not an integer")
        assert j >= 0, ("j must be >= 0")
        imin, imax = i - self.offset, i + self.offset + 1
        jmin, jmax = j - self.offset, j + self.offset + 1
        image = self.image[imin:imax, jmin:jmax, :]
        return image

    def _patch_label(self, i, j):
        """Get patch label 

        Args:
            i: index i
            j: index j
        Returns:
            label
        """
        assert isinstance(i, int), ("i is not an integer")
        assert i >= 0, ("i must be >= 0")
        assert isinstance(j, int), ("j is not an integer")
        assert j >= 0, ("j must be >= 0")
        band = 0 # currently supports 1 band labels
        label = self.labels[i, j, band]
        return label

    def __iter__(self):
        """Initialize an iterator
        """
        # height and width
        shape = self.labels.shape[:2] # equivalently use self.image.shape[:2]
        # rows (H,W) `np.ndarray` and columns (H,W) `np.ndarray`
        rows, columns = np.indices(shape)
        ndv = self.labels_metadata.ndv
        # an (H,W,D) `np.ndarray`, labels must be 1 band
        band = 0
        valid = self.labels[:,:,band] != ndv
        # valid rows
        valid_rows = rows[valid]
        valid_columns = columns[valid]
        # randomize - should use seed
        # equivalently could use valid_columns.shape
        n_valid_rows = valid_rows.shape[0]
        indices = np.arange(n_valid_rows) 
        np.random.shuffle(indices)
        self._labeled_indices = np.vstack(
            (valid_rows[indices], valid_columns[indices])).T
        self._labeled_indices.astype(int)
        self._count = 0
        self._max_iter = n_valid_rows
        return self

    def __next__(self):
        """Next patch from the iterator

        Args:
            None
        Returns:
            `Patch` object
        Raises:
            StopIteration
        """
        if self._count == self._max_iter:
            raise StopIteration
        i_npint64, j_npint64 = self._labeled_indices[self._count,:]
        # alternative to explicit casting is to 
        # broaden the integer types accepted by the assert clauses
        i, j = int(i_npint64), int(j_npint64) 
        image = self._patch_image(i, j)
        label = self._patch_label(i, j)
        metadata = self._patch_metadata(i, j)
        patch = Patch(image, label, metadata, self.size)
        self._count += 1
        return patch

    def __len__(self):
        """The number of `Patch` objects

        Args:
            None
        Returns:
            `int` number of `Patch` objects in `Patches` object
        """
        #initialize self._max_iter
        if self._max_iter is None:
            # self._max_iter is initialized in __iter__()
            iter(self)   
        return self._max_iter

class Patch(object):
    """Patch
    """
    def __init__(self, image, label, metadata, size):
        """Initialize a `Patch` object 

        Args:
            image: a `np.ndarray` of shape (H,W,D)
            label: an `int` or `float` type
            metadata: a `raster.Metadata` object
            size: `int` number of pixels along one axis
        Returns:
            None
        Raises:
            AssertionError

        """
        self._image = None
        self._label = None
        self._metadata = None
        self._size = None
        assert isinstance(image, np.ndarray), ("image must be a numpy.ndarray")
        assert len(image.shape) == 3, (
            "image must be an numpy.ndarray with shape (H,W,D)")
        self._image = image
        # label assertion
        # need to figure out how to better support np dtypes
        #assert isinstance(label, float) or isinstance(label, int), (
        #    "Patch class currently supports only float or int labels")
        self._label = label
        # metadata assertion
        assert isinstance(metadata, raster.Metadata)
        self._metadata = metadata
        # size 
        height, width = self._image.shape[:2]
        assert size == width, ("Size and width of image are not equal")
        assert size == height, ("Size and height of image are not equal")
        self._size = size

    @property
    def image(self):
        """The image
        """
        return self._image

    @property
    def label(self):
        """The label
        """
        return self._label

    @property
    def metadata(self):
        """The metadata
        """        
        return self._metadata

    @property
    def size(self):
        """The size
        """        
        return self._size

    def save_image(self, filename):
        """Save a patch as a raster file

        Args:
            filename: a valid path for a new file
        Returns:
            None
        """
        raster.save_image(filename, self.image, self.metadata)

    def __str__(self):
        """String containing image and label
        """
        return str(self.image) + "\n" + str(self.label)