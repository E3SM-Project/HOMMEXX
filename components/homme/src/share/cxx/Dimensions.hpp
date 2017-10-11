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
#define NUM_LEV_P           (NUM_LEV + 1)
#define NUM_INTERFACE_LEV   NUM_LEV_P
#define NUM_TIME_LEVELS     3
#define Q_NUM_TIME_LEVELS   2

#define NUM_LEV             NUM_PHYSICAL_LEV
#define LEVEL_PADDING       0

#else

#if   (AVX_VERSION == 0)
static constexpr const int VECTOR_SIZE = 1;
#elif (AVX_VERSION == 1 || AVX_VERSION == 2)
static constexpr const int VECTOR_SIZE = 4;
#elif (AVX_VERSION == 512)
static constexpr const int VECTOR_SIZE = 8;
#endif

static constexpr const int NUM_PHYSICAL_LEV = PLEV;
static constexpr const int LEVEL_PADDING =
    (VECTOR_SIZE - NUM_PHYSICAL_LEV % VECTOR_SIZE) % VECTOR_SIZE;
static constexpr const int NUM_LEV =
    (NUM_PHYSICAL_LEV + LEVEL_PADDING) / VECTOR_SIZE;

static constexpr const int NUM_INTERFACE_LEV = NUM_PHYSICAL_LEV + 1;
static constexpr const int INTERFACE_PADDING =
    (VECTOR_SIZE - NUM_INTERFACE_LEV % VECTOR_SIZE) % VECTOR_SIZE;
static constexpr const int NUM_LEV_P =
    (NUM_INTERFACE_LEV + INTERFACE_PADDING) / VECTOR_SIZE;

static constexpr const int NUM_TIME_LEVELS = 3;
static constexpr const int Q_NUM_TIME_LEVELS = 2;

#endif // CUDA_BUILD

} // namespace TinMan

#endif // HOMMEXX_DIMENSIONS_HPP
