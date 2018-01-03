#ifndef HOMMEXX_DIMENSIONS_HPP
#define HOMMEXX_DIMENSIONS_HPP

#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif
#include "Hommexx_config.h"

namespace Homme {

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

} // namespace Homme

#endif // HOMMEXX_DIMENSIONS_HPP
