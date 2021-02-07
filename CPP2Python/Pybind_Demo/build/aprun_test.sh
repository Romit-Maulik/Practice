#!/bin/bash -f
#COBALT -t 01:00:00
#COBALT -n 1
#COBALT -A datascience
#COBALT -q debug-cache-quad

# Load the right compiler and stuff
source /home/projects/OpenFOAM/OpenFOAM-5.x/etc/bashrc.ThetaIcc
module load datascience/tensorflow-2.2

export CRAYPE_LINK_TYPE=dynamic
#module load valgrind4hpc
echo "The python package is:"
echo $(which python)

export KMP_BLOCKTIME=0
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export OMP_PROC_BIND='spread,close'
export OMP_NESTED='TRUE'

#aprun -n $((nodes*rpn)) -N ${rpn} -j ${hpc} -cc depth -d ${OMP_Threads} --env OMP_NUM_THREADS=${OMP_Threads} $FOAM_APPBIN/${solver}  >> ${solverLogFile}
aprun -n 1 -cc depth ./app
