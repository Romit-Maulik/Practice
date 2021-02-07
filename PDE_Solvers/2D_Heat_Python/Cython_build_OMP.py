from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[ Extension("FTCS_Cython_OMP",
              ["FTCS_Cython_OMP.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math","-fopenmp"],extra_link_args=["-fopenmp"])]

setup(
  name = "FTCS_Cython_OMP",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)
