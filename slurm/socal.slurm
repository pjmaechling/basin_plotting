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

CVM=cvmh

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m cvmh -i 10 -v 2500 -l 32.0,-120.7 -s 0.01 -x 600 -y 450

CVM=sjfz

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m sjfz -i 10 -v 2500 -l 32.0,-120.7 -s 0.01 -x 600 -y 450

CVM=cvms

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m cvms -i 10 -v 2500 -l 32.0,-120.7 -s 0.01 -x 600 -y 450

CVM=cvms5

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m cvms5 -i 10 -v 1000 -l 32.0,-120.7 -s 0.01 -x 600 -y 450

CVM=cvmsi

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m cvmsi -i 10 -v 2500 -l 32.0,-120.7 -s 0.01 -x 600 -y 450

CVM=ssip

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m cvlsu,ivlsu -i 10 -v 2500 -l 32.0,-120.7 -s 0.01 -x 600 -y 450

CVM=albacore

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m albacore -i 10 -v 2500 -l 32.0,-120.7 -s 0.01 -x 600 -y 450

CVM=basins

ibrun ${UCVM_INSTALL_PATH}/bin/basin_query_mpi_complete -o ${SCRATCH}/${CVM}.txt,${SCRATCH}/${CVM}_meta.txt \
  -f ${UCVM_INSTALL_PATH}/conf/ucvm.conf -m cvmhsmbn,cvmhsbcbn,cvmhsbbn,cvmhstbn,cvmhrbn,cvmhibbn,cvmhvbn,cvmhsgbn,cvmhlabn -i 10 -v 2500 -l 32.0,-120.7 -s 0.01 -x 600 -y 450


RET=$?

echo date
echo "Simulation complete. Exit code: $RET
