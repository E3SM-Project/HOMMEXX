
#ifndef _DIMENSIONS_HPP_
#define _DIMENSIONS_HPP_

namespace Homme {

#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif

constexpr const int np = NP;
constexpr const int npsq = np * np;
constexpr const int max_neigh_edges = 8;
constexpr const int nc = NC;

constexpr const int dim = 2;

#ifdef QSIZE_D
constexpr const int qsize_d = QSIZE_D;
#else
constexpr const int qsize_d = 4;
#endif

constexpr const int nlev = PLEV;
constexpr const int nlevp = nlev + 1;

constexpr const int timelevels = 3;

}  // namespace Homme

#endif
