from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[ Extension("FTCS_Cython",
              ["FTCS_Cython.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "FTCS_Cython",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)
