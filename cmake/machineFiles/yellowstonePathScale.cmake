# CMake initial cache file for yellowstone pathscale
SET (CMAKE_Fortran_COMPILER mpief90 CACHE FILEPATH "")
SET (CMAKE_C_COMPILER mpiecc CACHE FILEPATH "")
SET (CMAKE_CXX_COMPILER mpiecc CACHE FILEPATH "")
SET (NETCDF_DIR $ENV{NETCDF} CACHE FILEPATH "")
SET (PNETCDF_DIR $ENV{PNETCDF} CACHE FILEPATH "")
SET (PREFER_SHARED TRUE CACHE FILEPATH "")
