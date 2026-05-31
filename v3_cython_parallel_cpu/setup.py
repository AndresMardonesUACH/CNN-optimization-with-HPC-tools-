from setuptools import setup
from setuptools import Extension

from Cython.Build import cythonize

import numpy as np


extensions = [
    Extension(
        "layers.conv",
        ["layers/conv.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3"
        }
    ),
    include_dirs=[np.get_include()],
)