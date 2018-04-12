#ifndef REMAP_HPP
#define REMAP_HPP

#include "dimensions_remap_tests.hpp"
#include "Types.hpp"

using namespace Homme;

void compute_ppm_grids(const Real dx[NLEVP4],
                       Real rslt[NLEVP2][DIM10],
                       const int alg);

void compute_ppm(const Real a[NLEVP4],
                 const Real dx[NLEVP2][DIM10],
                 Real coefs[NLEV][DIM3], const int alg);

void remap_Q_ppm(
    Real Qdp[][NLEV][NP][NP],  //[qsize] is the leading dim
    const int qsize, const Real dp1[NLEV][NP][NP],
    const Real dp2[NLEV][NP][NP], const int alg);

#endif  // REMAP_HPP
