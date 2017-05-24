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
"""DataSets and DataSet class

The DataSet class was derived from the MNIST TensorFlow example which
has an Apache 2.0 license. 
"""

from collections import namedtuple
import numpy as np

# Store data splits inside of a Datasets object
DataSets = namedtuple("DataSets", ["train", "validate", "test"])

class DataSet(object):

    def __init__(self, 
                 images,
                 labels, 
                 seed=None):
        """Initialize a DataSet

        Args:
            images:
            labels:
            seed: Not implemented
        Returns:
            None
        Exceptions:
            raises an AssertionError if the number of examples in 
            `images` and `labels` are different.
        """
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._epochs_completed = 0 
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images 

    @property 
    def labels(self):
        return self._labels

    @property 
    def num_examples(self):
        return self._num_examples

    @property 
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Get the next mini-batch

        Args:
            batch_size: `int` size of mini-batch 
            shuffle: if True (default) randomly permutes the Dataset order
        Returns:
            `next_batch` of examples.
        """
        start = self._index_in_epoch
        # shuffle first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # next epoch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            # get remaining examples in current epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # reshuffle
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # restart
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return (np.concatenate((images_rest_part, images_new_part), axis=0),
                np.concatenate((labels_rest_part, labels_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]