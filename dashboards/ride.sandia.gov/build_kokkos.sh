
#!/bin/bash

cd /home/projects/hommexx/nightlyCDash
mkdir build
cd build 
mkdir KokkosBuild
cd KokkosBuild
sed -e s/default_arch=\"sm_35\"/default_arch=\"sm_60\"/ -e s/cuda_args=\"\"/cuda_args=\"--fmad=false\"/ nvcc_wrapper -i ${jenkins_trilinos_dir}/packages/kokkos/bin/nvcc_wrapper 
${jenkins_trilinos_dir}/packages/kokkos/generate_makefile.bash --compiler=${jenkins_trilinos_dir}/packages/kokkos/bin/nvcc_wrapper --prefix=/home/projects/hommexx/nightlyCDash/build/KokkosInstall --with-serial --with-cuda=${CUDA_ROOT} --with-cuda-options=enable_lambda --with-options=disable_deprecated_code,aggressive_vectorization --arch=Pascal60
make -j 24
make install -j 24 
