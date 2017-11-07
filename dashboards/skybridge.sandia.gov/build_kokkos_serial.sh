#!/bin/bash

mkdir build
cd build
mkdir KokkosBuild
cd KokkosBuild
/projects/hommexx/nightlyHOMMEXXCDash/repos/Trilinos/packages/kokkos/generate_makefile.bash --with-serial --compiler=icpc --with-options=aggressive_vectorization --prefix=/projects/hommexx/nightlyHOMMEXXCDash/build/KokkosInstall --arch=HSW --cxxflags="-O3"
make -j 24
make install -j 24

