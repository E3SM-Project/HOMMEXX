module add intel/compilers/17.0.098 openmpi/1.10.4/intel/17.0.098 cmake/3.5.2 git/2.8.2
export CC=mpicc
export CXX=mpicxx
export FC=mpif90
export CCLD=${CXX}
export CXXLD=${CXX}
export FCLD=${FC}
export PREFIX="/home/ikalash/mdeakin_prefix"
export PATH="${PREFIX}/bin:${PATH}"
