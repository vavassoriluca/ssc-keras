from distutils.core import setup
from Cython.Build import cythonize

import numpy

setup(
    ext_modules = cythonize("feature_extractor_cython.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)