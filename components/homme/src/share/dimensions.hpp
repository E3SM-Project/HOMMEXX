#ifndef HOMMEXX_DIMENSIONS_HPP
#define HOMMEXX_DIMENSIONS_HPP

#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif
#include "Hommexx_config.h"

namespace Homme {

// Until whenever CUDA supports constexpr properly
#ifdef HOMMEXX_CUDA_SPACE

#define np                      NP
#define npsq                    (np * np)
#define max_neigh_edges         8
#define nc                      NC
#define dim                     2
#ifdef QSIZE_D
#define qsize_d                 QSIZE_D
#else
#define qsize_d                 4
#endif
#define nlev                    PLEV
#define nlevp                   (nlev + 1)
#define timelevels              3

#else // HOMMEXX_CUDA_BUILD

constexpr int np              = NP;
constexpr int npsq            = np * np;
constexpr int max_neigh_edges = 8;
constexpr int nc              = NC;
constexpr int dim             = 2;
#ifdef QSIZE_D
constexpr int qsize_d         = QSIZE_D;
#else
constexpr int qsize_d         = 4;
#endif
constexpr int nlev            = PLEV;
constexpr int nlevp           = nlev + 1;
constexpr int timelevels      = 3;

#endif // HOMMEXX_CUDA_BUILD

} // namespace Homme

#endif // HOMMEXX_DIMENSIONS_HPP
