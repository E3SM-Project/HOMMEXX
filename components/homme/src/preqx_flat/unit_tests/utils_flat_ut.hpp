#ifndef UTILS_FLAT_UT_HPP
#define UTILS_FLAT_UT_HPP

#include "Types.hpp"
#include <random>

namespace Homme{

using rngAlg = std::mt19937_64;

Real compare_answers(Real target, Real computed,
                     Real relative_coeff = 1.0);

void genRandArray(
    Real *arr, int arr_len, rngAlg &engine,
    std::uniform_real_distribution<Real> pdf);
}
#endif
