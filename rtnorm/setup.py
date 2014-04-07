import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy.distutils.system_info import get_info
import cython_gsl

ext_modules = [
    Extension("pyrtnorm", ["pyrtnorm.pyx"],
        libraries=cython_gsl.get_libraries(),
        library_dirs=[cython_gsl.get_library_dir()],
        include_dirs=[cython_gsl.get_cython_include_dir()],
    )
]

setup(
    name = "pyrtnorm",
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include(), cython_gsl.get_include()],
)