#!/bin/bash

mkdir build
cd build
mkdir KokkosBuild
cd KokkosBuild

jenkins_trilinos_dir=/home/projects/hommexx/nightlyCDash/repos/Trilinos
${jenkins_trilinos_dir}/packages/kokkos/generate_makefile.bash --with-openmp --with-serial --compiler=icpc --with-options=aggressive_vectorization,disable_profiling --prefix=/home/projects/hommexx/nightlyCDash/build/KokkosInstall --arch=KNL --cxxflags="-O3"
make -j 24
make install -j 24

