#ifndef UTILS_FLAT_UT_CPP
#define UTILS_FLAT_UT_CPP

#include <catch/catch.hpp>

#include <limits>

#include "utils_flat_ut.hpp"

#include "Types.hpp"

#include <assert.h>
#include <stdio.h>
#include <random>

namespace Homme{


Real compare_answers(Real target, Real computed,
                     Real relative_coeff) {
  Real denom = 1.0;
  if(relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }
  return std::fabs(target - computed) / denom;
}  // end of definition of compare_answers()

void genRandArray(
    Real *arr, int arr_len, rngAlg &engine,
    std::uniform_real_distribution<Real> pdf) {
  for(int i = 0; i < arr_len; ++i) {
    arr[i] = pdf(engine);
  }
}  // end of definition of genRandArray()
}
#endif
