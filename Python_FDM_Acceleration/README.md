# 2D_FDM_Optimization
Assessing different code speed up in Python for the 2D Heat diffusion equation

You can execute the shell script in the terminal ("sh build_and_run.sh")

The prompt gives you 5 options for solving the same heat diffusion problem with periodic boundary conditions

0 - Native python for loops

1 - Vectorized python

2 - Cython

3 - Pythran (builds with full utilization of machine hardware for parallelization - really easy!)

4 - Cython with OMP
