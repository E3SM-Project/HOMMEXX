/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_DIMENSIONS_HPP
#define HOMMEXX_DIMENSIONS_HPP

#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif

#include <Hommexx_config.h>

#include <Kokkos_Core.hpp>

namespace Homme {

// Until whenever CUDA supports constexpr properly
#ifdef CUDA_BUILD

#define VECTOR_SIZE         1

#define NUM_PHYSICAL_LEV    PLEV
#define NUM_TIME_LEVELS     3
#define Q_NUM_TIME_LEVELS   2

#define NUM_LEV             NUM_PHYSICAL_LEV
#define NUM_LEV_P           (NUM_LEV + 1)
#define NUM_INTERFACE_LEV   NUM_LEV_P

#else

#if   (AVX_VERSION == 0)
// Vector<VectorTag<SIMD<T, SpT>, l> > can use this for good results
// on, e.g., Power9, where AVX doesn't exist.
static constexpr const int VECTOR_SIZE = 8;
#elif (AVX_VERSION == 1 || AVX_VERSION == 2)
static constexpr const int VECTOR_SIZE = 4;
#elif (AVX_VERSION == 512)
static constexpr const int VECTOR_SIZE = 8;
#endif
static_assert(VECTOR_SIZE>0, "Error: VECTOR_SIZE=0!");

static constexpr const int NUM_PHYSICAL_LEV = PLEV;
static constexpr const int NUM_LEV =
    (NUM_PHYSICAL_LEV + VECTOR_SIZE - 1) / VECTOR_SIZE;

static constexpr const int NUM_INTERFACE_LEV = NUM_PHYSICAL_LEV + 1;
static constexpr const int NUM_LEV_P =
    (NUM_INTERFACE_LEV + VECTOR_SIZE - 1) / VECTOR_SIZE;

static constexpr const int NUM_TIME_LEVELS = 3;
static constexpr const int Q_NUM_TIME_LEVELS = 2;

#endif // CUDA_BUILD

} // namespace TinMan

#endif // HOMMEXX_DIMENSIONS_HPP
