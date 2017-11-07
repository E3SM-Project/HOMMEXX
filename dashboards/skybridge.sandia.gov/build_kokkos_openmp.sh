#!/bin/bash

mkdir build
cd build
mkdir KokkosBuildOpenMP
cd KokkosBuildOpenMP
/projects/hommexx/nightlyHOMMEXXCDash/repos/Trilinos/packages/kokkos/generate_makefile.bash --with-openmp --with-serial --compiler=icpc --with-options=aggressive_vectorization --prefix=/projects/hommexx/nightlyHOMMEXXCDash/build/KokkosInstallOpenMP --arch=KNL --cxxflags="-O3"
make -j 24
make install -j 24

