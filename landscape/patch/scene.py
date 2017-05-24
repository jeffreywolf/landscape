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
"""Scene for predictive mapping
"""

import collections
import time
import numpy as np

Batch = collections.namedtuple("Batch", ["rows","columns","images", 
    "proportion", "time", "number"])

class Scene(object):
    """A Scene class to manage predictive mapping
    """
    def __init__(self, image, patch_size, batch_size, mask=None):
        """Initialize a `Scene` object

        Args:
            image: `np.ndarray` image
            patch_size: patch_size `int`
            batch_size: batch size `int`. Adjust based on available memory.
            mask: `np.ndarray` image with 1 where data and ndv where no data
        Returns:
            Scene object
        Raises:
            AssertionError
        """
        self._image = None
        self._shape = None
        self._indices = None
        self._num_examples = None
        self._patch_size = None
        self._offset = None
        self._index_in_epoch = None # initialized in __iter__()
        # patch size and offset
        assert isinstance(patch_size, int), ("Patch size must be an integer")
        assert patch_size > 0, ("Patch size must be >0")
        assert patch_size % 2 == 1, ("Patch size must be an odd integer")
        self._patch_size = patch_size
        self._offset = patch_size // 2
        # image and shape
        shape = image.shape
        assert len(shape) == 3, ("Not an image with HxWxD shape")
        self._shape = shape
        assert isinstance(image, np.ndarray), (
            "Image is not an np.ndarray")
        self._image = image
        # indices for batch iteration
        valid_rows = shape[0] - 2 * self.offset 
        valid_columns = shape[1] - 2 * self.offset
        assert valid_rows > 0 and valid_columns > 0, (
            "The border due to patch_size removes all valid data")       
        rows, columns = np.indices((valid_rows, valid_columns))
        # mask - use mask to subset the rows and columns
        if mask is not None:
            assert isinstance(mask, np.ndarray), (
                "Mask is not a np.ndarray")
            assert len(mask.shape) == 3, (
                "Mask is not an image with HxWxD shape")
            assert mask.shape[0] == image.shape[0], (
                "Number of rows in mask and image differ")
            assert mask.shape[1] == image.shape[1], (
                "Number of columns in mask and image differ")            
            mask_valid = mask[self.offset:-self.offset, 
                self.offset:-self.offset,0]
            assert mask_valid.shape[0] == rows.shape[0], (
                "Number of rows of subset mask and image differ")
            assert mask_valid.shape[1] == rows.shape[1], (
                "Number of columns of subset mask and image differ")            
            mask_rows = rows[~np.isnan(mask_valid)]
            mask_columns = columns[~np.isnan(mask_valid)]
            rows_flat, columns_flat = (mask_rows.flatten(), 
                mask_columns.flatten())
        else:
            rows_flat, columns_flat = rows.flatten(), columns.flatten()
        rows_flat += self.offset 
        columns_flat += self.offset
        self._rows, self._columns = rows_flat, columns_flat
        assert len(rows_flat) == len(columns_flat), (
            "row and column lengths differ")
        self._num_examples = len(rows_flat)
        # batch size
        assert isinstance(batch_size, int), ("Batch size must be an integer")
        assert batch_size > 0, ("Batch size must be a positive integer")
        #assert batch_size <= 10000, ("Try a batch size <=10000.")
        assert batch_size <= 25000, ("Try a batch size <=10000.")
        self._batch_size = batch_size 
        # determine the boundary and use indices only within bounds
     
    @property
    def image(self):
        """image"""
        return self._image

    @property
    def shape(self):
        """shape"""
        return self._shape

    @property
    def patch_size(self):
        """patch_size"""
        return self._patch_size

    @property
    def offset(self):
        """offset"""
        return self._offset

    @property
    def batch_size(self):
        """batch_size"""
        return self._batch_size

    @property
    def rows(self):
        """rows"""
        return self._rows
    
    @property
    def columns(self):
        """columns"""
        return self._columns        

    def _patch(self, i, j):
        """Build an image patch

        Args:
            i: `int` row index into image and labels `nd.ndarray`
            j: `int` row index into image and labels `nd.ndarray`
        Returns:
            image, an `np.ndarray`
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

    def __iter__(self):
        """initialize iterator

        Args:
            None
        Returns:
            self
        """
        self._index_in_epoch = 0
        self._batch_number = 0
        return self

    def __next__(self):
        """next method

        Args:
            None
        Returns:
            tuple of three `np.ndarray` representing row index, column index,
            and a set of image patches.
        Raises:
            StopIteration
        """
        t_i = time.time()
        start = self._index_in_epoch
        if start >= self._num_examples:
            raise StopIteration
        self._index_in_epoch += self.batch_size
        end = self._index_in_epoch
        if end > self._num_examples:
            end = self._num_examples
        batch_rows, batch_columns = (self.rows[start:end], 
            self.columns[start:end])
        patches = []
        for i, j in zip(batch_rows, batch_columns):
            #print(i, j)
            i_int, j_int = int(i), int(j)
            patch = self._patch(i_int, j_int)
            patches.append(patch)
        patches_array = np.array(patches)
        proportion_complete = float(start)/self._num_examples
        t_f = time.time()
        dt = t_f - t_i
        batch = Batch(batch_rows, batch_columns, patches_array, 
            proportion_complete, dt, self._batch_number)
        self._batch_number += 1
        return batch