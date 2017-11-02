
#!/bin/bash

cd /home/projects/hommexx/nightlyCDash
mkdir build
cd build 
mkdir KokkosBuild
cd KokkosBuild
sed s/default_arch=\"sm_35\"/default_arch=\"sm_60\"/ -i ${jenkins_trilinos_dir}/packages/kokkos/config/nvcc_wrapper 
${jenkins_trilinos_dir}/packages/kokkos/generate_makefile.bash --prefix=/home/projects/hommexx/nightlyCDash/build/KokkosInstall --with-openmp --with-serial --with-cuda=${CUDA_ROOT} --with-cuda-options=enable_lambda --with-options=aggressive_vectorization --arch=Pascal60
make -j 24
make install -j 24 
