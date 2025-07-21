#!/bin/bash

#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 40
#SBATCH -p skx-dev
#SBATCH --account=DS-Cybershake
#SBATCH --output=basin-%x.%j.out
#SBATCH --error=basin-%x.%j.err
#SBATCH --mail-user=maechlin@usc.edu
#SBATCH --mail-type=ALL
#SBATCH --export=ALL

####################
## Configuration ##
####################

source /work2/00329/tg456034/stampede3/ucvm_257/conf/ucvm_env.sh

CVM=sfcvm

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m sfcvm -i 10 -v 1000 -l 36.0,-123.8 -s 0.01 -x 425 -y 425

CVM=canvas

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m canvas -i 10 -v 1000 -l 36.0,-123.8 -s 0.01 -x 425 -y 425

CVM=uwlinca

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m uwlinca -i 10 -v 1000 -l 36.0,-123.8 -s 0.01 -x 425 -y 425

CVM=uwsfbcvm

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m uwsfbcvm -i 10 -v 1000 -l 36.0,-123.8 -s 0.01 -x 425 -y 425

CVM=cs248

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -m cs248 -v 1000 -l 36.0,-123.8 -i 10 -s 0.01 -x 425 -y 425 -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf

RET=$?

echo date
echo "Simulation complete. Exit code: $RET"
