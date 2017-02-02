
#!/bin/bash

cd ${WORKSPACE}/build
mkdir KokkosBuild
cd KokkosBuild 
${jenkins_trilinos_dir}/packages/kokkos/generate_makefile.bash --prefix=${WORKSPACE}/build/TrilinosInstall --with-openmp --with-serial --with-cuda=${CUDA_ROOT} --with-cuda-options=enable_lambda 
make -j 24
make install -j 24 
