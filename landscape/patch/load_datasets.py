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
"""Load a dataset 
"""
import os
import glob
import pickle
import numpy as np
from landscape.patch import datasets
from landscape.raster import raster
#import datasets

def _parse_category(filename):
    """Parse the category from filename

    Args:
        filename:
    Returns:
        `int` category label
    Raises:
        AssertionError
    """
    assert os.path.exists(filename), ("Invalid filename")
    basename = os.path.basename(filename)
    return int(basename.split(".")[0].split("_")[2])

def _parse_files(directory):
    """Parse dataset files

    Args:
        directory:
    Returns:
        tuple of `np.ndarray` containing labels and images
    """
    files = glob.glob(os.path.join(directory, "*.tif"))
    labels = []
    images = []
    for f in files:
        labels.append(_parse_category(f))
        images.append(raster.load_image(f))
    return np.array(labels, dtype=int), np.array(images)    

def _onehot(y, categories):
    """One-hot encode.

    One-hot encode an N element `ndarray` containing `int` elements
    with M classes to produce an N x M `ndarray` where in a row
    the element belonging to the class is 1 and otherwise is 0.

    Args:
        y: `ndarray` holding `int` representing class categories.
        categories: `int` number of categories
    Returns:
        Matrix with one-hot encoding
    Raises:
        AssertionError
    """
    assert np.amax(y) < categories, (
        "Label exceeds the number of categories")
    rows = len(y)
    y_mat = np.zeros((rows, categories))
    y_mat[np.arange(rows), y] = 1
    return y_mat

def _parse_dataset(directory, onehot):
    """Converts directory from `Splits.save()` to a `datasets.Datasets` object

    args:
        directory: valid path `str` of dataset directory
        onehot: `bool` one-hot vector encoding of class categories
    Returns:
        `Datasets` object
    Raises:
        AssertionError
    """
    assert os.path.exists(directory), (
        "Invalid path to datasets")
    train_directory = os.path.join(directory, "train")
    assert os.path.exists(train_directory), (
        "Invalid path to train dataset directory")    
    validate_directory = os.path.join(directory, "validate")
    assert os.path.exists(validate_directory), (
        "Invalid path to validate dataset directory")    
    test_directory = os.path.join(directory, "test")
    assert os.path.exists(test_directory), (
        "Invalid path to test dataset directory") 
    # parse data from files
    labels_train_flat, images_train = _parse_files(train_directory)
    labels_validate_flat, images_validate = _parse_files(validate_directory)
    labels_test_flat, images_test = _parse_files(test_directory)
    # number of label categories
    labels = np.hstack((labels_train_flat, labels_validate_flat,
        labels_test_flat))
    categories = len(set(labels))
    # create one-hot encoding
    labels_train = _onehot(labels_train_flat, categories)
    labels_validate = _onehot(labels_validate_flat, categories)
    labels_test = _onehot(labels_test_flat, categories)
    # datasets
    train = datasets.DataSet(images_train, labels_train)
    validate = datasets.DataSet(images_validate, labels_validate)
    test = datasets.DataSet(images_test, labels_test)
    data = datasets.DataSets(train, validate, test)
    return data

def _serialize(data, filename):
    """Serialize a dataset

    Args:
        data: data to serialize
        filename: filename to contain serialized data
    Returns:
        None
    Raises:
        AssertionError
    """
    dirname = os.path.dirname(filename)
    assert os.path.isdir(dirname), ("directory part of filename must be valid")
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def _deserialize(filename):
    """Deserialize a dataset
    
    Args:   
        filename: `str` name of file to unpickle
    Returns:
        `datasets.DataSets` object
    Raises:
        AssertionError
    """
    assert os.path.exists(filename), ("Must use a valid filename")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load(directory, onehot=True, serialized=True):
    """Load a new DataSets object

    Args:
        directory:  directory path containing data from `splits.Split` 
                    object `save` method.
        onehot: `bool` with default `True`. Encodes each label as a vector 
                of length N number of label categories with the element 
                representing the class label coded 1 and all others 0.
        serialized: saves/loads a `pickle`'d dataset for fast reloading
    Returns:
        `datasets.DataSets` object
    Raises:
        AssertionError
    """
    assert os.path.exists(directory), (
        "Invalid path to datasets")
    if serialized == True:
        pickled = os.path.join(directory, "dataset.pickle")
        # try loading from pickle
        if os.path.exists(pickled):
            data = _deserialize(pickled)
        else:
            data = _parse_dataset(directory, onehot)
            # save pickle
            _serialize(data, pickled)
    else:
        data = _parse_dataset(directory, onehot)
    return data       