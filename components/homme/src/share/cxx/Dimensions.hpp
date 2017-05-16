#ifndef HOMMEXX_DIMENSIONS_HPP
#define HOMMEXX_DIMENSIONS_HPP

#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif

namespace Homme {

// Until whenever CUDA supports constexpr properly
#ifdef CUDA_BUILD

#define NUM_LEV           PLEV
#define NUM_LEV_P         (NUM_LEV+1)
#define NUM_TIME_LEVELS   3
#define Q_NUM_TIME_LEVELS 2

#else

static constexpr int NUM_LEV           = PLEV;
static constexpr int NUM_LEV_P         = NUM_LEV+1;
static constexpr int NUM_TIME_LEVELS   = 3;
static constexpr int Q_NUM_TIME_LEVELS = 2;

#endif

} // namespace TinMan

#endif // HOMMEXX_DIMENSIONS_HPP
