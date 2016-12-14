
#!/bin/bash

cd /ascldap/users/ikalash/nightlyHOMMEXXCDash/build
mkdir KokkosBuild
cd KokkosBuild 
/ascldap/users/ikalash/nightlyHOMMEXXCDash/repos/Trilinos/packages/kokkos/generate_makefile.bash --prefix=/ascldap/users/ikalash/nightlyHOMMEXXCDash/build/TrilinosInstall --with-openmp --with-serial --with-cuda=${CUDA_ROOT} --with-cuda-options=enable_lambda 
make -j 24
make install -j 24 
