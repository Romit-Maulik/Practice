echo -n "Do you want to build Pythran and Cython again? 0[No], 1[Yes]: "
read build_val

if [ $build_val -eq 1 ]; then
	# Build pythran 
	pythran -DUSE_XSIMD -fopenmp -march=native FTCS_Pythran.py
	# Build Cython
	python3 Cython_build.py build_ext --inplace	
	# Build Cython_OMP
	python3 Cython_build_OMP.py build_ext --inplace
fi

# Run the profiler 
python3 FTCS_Profiling.py

