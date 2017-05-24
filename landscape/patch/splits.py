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
"""Code to create the dataset
"""
import sys
import os
import collections
import numpy as np
from landscape.patch import patches

Proportions = collections.namedtuple('Proportions', 
    ["train", "validate", "test"])

class Splits(object):
    """Splits class
    """
    def __init__(self, patches_object, train=0.7, validate=0.15, test=0.15):
        """Initialize a Splits object
        
        The sum of train, validate, and test must equal 1. 

        Args:
            patches:
            train: a `float` proportion with default 0.7
            validate: a `float` proportion with default 0.15
            test: a `float` proportion with default 0.15
        Returns:
            None
        Raises:
            AssertionError
        """
        assert isinstance(patches_object, patches.Patches)
        self._patches = patches_object
        EPSILON = 0.00001
        assert abs(train + validate + test - 1.0) < EPSILON, (
            "Sum of train, validate, and test not within {} of 1.0".format(
                EPSILON))
        self._proportions = Proportions(train, validate, test)    

    @property
    def patches(self):
        """A `Patches` object
        """
        return self._patches 

    @property
    def proportions(self):
        """A `namedtuple` containing `train`, `validate`, and `test`
        """
        return self._proportions

    def save(self, path, prefix=""):
        """Save Splits

        Args:
            path: a new directory to create
        Returns:
            None
        Raises:
            AssertionError 
            IndexError
        """
        assert isinstance(prefix, str), ("prefix is not an `str`")
        # get dirname of path
        dirname = os.path.dirname(path)
        assert os.path.isdir(dirname)
        # check if new directory exists
        # if not then create new directory
        if not os.path.isdir(path):
            os.mkdir(path)
        # check if subdirectories exists
        # if not create subdirectories
        train_dir = os.path.join(path, "train")
        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)       
        validate_dir = os.path.join(path, "validate")
        if not os.path.isdir(validate_dir):
            os.mkdir(validate_dir)
        test_dir = os.path.join(path, "test")
        if not os.path.isdir(test_dir):
            os.mkdir(test_dir)
        
        # determine how to randomize the Patches order 
        # for train, validation, and test
        length = len(self.patches)
        indices = np.arange(length)
        np.random.shuffle(indices)
        #print(indices)
        # determine indices to use for each split
        train_max = int(self.proportions.train * length)
        validate_max = int((self.proportions.train + self.proportions.validate)
            * length)
        # use set for membership testing
        train_set = set(indices[:train_max])
        validate_set = set(indices[train_max:validate_max])
        test_set = set(indices[validate_max:])
        
        for i, patch in enumerate(self.patches):
            #print(i)
            if i in train_set:
                patch_dir = train_dir
            elif i in validate_set:
                patch_dir = validate_dir
            elif i in test_set:
                patch_dir = test_dir
            else:
                raise IndexError
            # name should include label and identifier              
            patch_name = "{prefix}{name}_{size}_{label}.tif".format(
                prefix=prefix,
                name=i,
                size=patch.size,
                label=patch.label)
            #print(patch_name)
            patch.save_image(os.path.join(patch_dir, patch_name))