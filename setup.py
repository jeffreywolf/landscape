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
"""landscape setup script
"""
from setuptools import setup
import landscape

with open("README.md") as f:
    readme = f.read()

setup(
    name="landscape",
    version=landscape.__version__,    
    description="Simplify geospatial modeling",
    license="Apache License 2.0",
    keywords="gis raster model",
    author="Jeffrey A. Wolf",
    author_email="iamjwolf@gmail.com",
    url="https://www.github.com/jeffreywolf/landscape",
    long_description=readme,
    packages=["landscape","landscape.raster","landscape.patch",
        "landscape.topography"],
    install_requires = ['numpy', 'gdal', 'scipy'],
    test_suite = "test.test_landscape",
    data_files = [("test/data", [
        "test/data/test_image.tif", 
        "test/data/test_mask.tif", 
        "test/data/test_dem.tif"])],
    include_package_data = True,
    entry_points={"console_scripts":[
        "topography = landscape.topography.topography:main"
    ]},
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)