# landscape 

`landscape` is a package to help you create geospatial 
datasets and use machine learning models on top of a 
dataset API.

### Introduction
`landscape` provides simple interfaces to raster data and
a dataset API for machine learning pipelines. Geospatial 
functionality is provided by GDAL and the dataset API 
uses numpy's ndarray.  `landscape` is compatible with 
Python 3.5+.

### Installation

`landscape` is available on GitHub. This version of landscape requires Python 3.5 or later. `landscape` is dependent on `numpy`, `scipy`, and `gdal`.

The simplest way to get a Python environment set up to use this library is through using Anaconda.
1. Download Anaconda (https://www.continuum.io/downloads). `landscape` requires Python >=3.5.
2. Create a conda environment for using `landscape`. GDAL from the conda-forge repository is recommended.  The `landscape` package is designed for use as part of a machine learning pipeline. Here is an example of a typical `landscape` environment.
```
conda create -c conda-forge -n landscape python=3.6.1 tensorflow gdal keras ipython jupyter hdf5 h5py scikit-learn
```
3. Activate your environment `source activate landscape` (or `activate landscape` on Windows).
4. Download and install `landscape` by downloading the GitHub repository (`git clone https://www.github.com/jeffreywolf/landscape`) and then from within the repository directory running `python setup.py install`.
5. You now have a working environment for `landscape`. 


To exit the landscape environment use `source deactivate landscape` (or on Windows `deactivate landscape`). To completely remove the landscape enviroment use `conda remove --name landscape --all`. 

If you have `gdal`, `numpy`, and `scipy` in Python 3.5+ then a simple `python setup.py install` from the landscape package directory will install `landscape`.  However, the Anaconda install also avoids potential conflicts with system level GDAL installations that may be used by geospatial applications already on your computer. GDAL (`gdal`) can also be difficult to install through PyPI.  


### Example

Use the `raster` subpackage to read a raster as an array, load metadata,
and write out the image as a new file.

```python
import numpy as np
from landscape.raster import raster

# Going "full circle" is easy
filename = "test/data/test_image.tif" # from within the main directory
image = raster.load_image(filename)
metadata = raster.Metadata(filename)
output_filename = "test/data/temp/test_image_output.tif"
raster.save_image(output_filename, image, metadata)
image_reloaded = raster.load_image(output_filename)
metadata_reloaded = raster.Metadata(output_filename)
# Tests
print(np.array_equal(image, image_reloaded))
print(metadata == metadata_reloaded)
# the datatype is not used in the metadata comparison, 
# so lets test this separately
print(metadata.datatype == metadata_reloaded.datatype)
```

Use the `topography` subpackage to analyze topography metrics of a digital elevation model.

```python
import os
from landscape.topography import topography

bandwidth = 5 # pixels
filename = "test/data/test_dem.tif"
dirname = "test/data/temp"
# create an GaussianTopography object
gt = topography.GaussianTopography(filename)
# compute all sorts of useful topography statistics in a jiffy
gt.filter(bandwidth, os.path.join(dirname, "filter.tif"))
gt.gradient_magnitude(bandwidth, os.path.join(dirname, "gradient_mag.tif"))
gt.gradient_dx(bandwidth, os.path.join(dirname, "gradient_dx.tif"))
gt.gradient_dy(bandwidth, os.path.join(dirname, "gradient_dy.tif"))
gt.laplacian(bandwidth, os.path.join(dirname, "laplacian.tif"))
gt.aspect(bandwidth, os.path.join(dirname, "aspect.tif"))
```

There is also a command line program called `topography` that is installed when you install `landscape`.  The `topography` program will calculate the same statistics on your DEM as done above from within a Python script.

```bash
$ topography -h # information about usage
$ # topography <dem.tif> <bandwidth in pixels> <output directory>
$ topography test/data/test_dem.tif 5 test/data/temp
```
------------------------------------------------------------------------
## Package Organization

LICENSE.txt  
setup.py  
landscape/  
	__init__.py  
	patch/  
		__init__.py  
		datasets.py  
		load_datasets.py  
		patches.py  
		scene.py  
		splits.py  
	raster/  
		__init__.py  
		raster.py  
	topography/  
		__init__.py  
		topography.py  
test  
    __init__.py  
    data/  
        test_image.tif  
        test_mask.tif  
        test_dem.tif  
        temp/  
    test_landscape.py  
    test_patch.py  
    test_raster.py  	

---------------------------------------------------------------------------

Contributing

Pull requests are welcome. 