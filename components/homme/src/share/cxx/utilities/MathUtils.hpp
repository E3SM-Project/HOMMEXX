/*********************************************************************************
 *
 * Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 *
 * For five (5) years from  the United States Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide
 * license in this data to reproduce, prepare derivative works, and perform
 * publicly and display publicly, by or on behalf of the Government. There is
 * provision for the possible extension of the term of this license. Subsequent
 * to that period or any extension granted, the United States Government is
 * granted for itself and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable worldwide license in this data to reproduce, prepare derivative
 * works, distribute copies to the public, perform publicly and display publicly,
 * and to permit others to do so. The specific term of the license can be
 * identified by inquiry made to National Technology and Engineering Solutions of
 * Sandia, LLC or DOE.
 *
 * NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
 * ENERGY, NOR NATIONAL TECHNOLOGY AND ENGINEERING SOLUTIONS OF SANDIA, LLC, NOR
 * ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY
 * LEGAL RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
 * INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS
 * USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
 *
 * Any licensee of this software has the obligation and responsibility to abide
 * by the applicable export control laws, regulations, and general prohibitions
 * relating to the export of technical data. Failure to obtain an export control
 * license or other authority from the Government may result in criminal
 * liability under U.S. laws.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 *     - Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *     - Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimers in the documentation
 *       and/or other materials provided with the distribution.
 *     - Neither the name of Sandia Corporation,
 *       nor the names of its contributors may be used to endorse or promote
 *       products derived from this Software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ********************************************************************************/

#ifndef HOMMEXX_MATH_UTILS_HPP
#define HOMMEXX_MATH_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <cmath>

namespace Homme
{

template <typename FPType>
KOKKOS_INLINE_FUNCTION constexpr FPType min(const FPType val_1,
                                            const FPType val_2) {
  return val_1 < val_2 ? val_1 : val_2;
}

template <typename FPType, typename... FPPack>
KOKKOS_INLINE_FUNCTION constexpr FPType min(const FPType val, FPPack... pack) {
  return val < min(pack...) ? val : min(pack...);
}

template <typename FPType>
KOKKOS_INLINE_FUNCTION constexpr FPType max(const FPType val_1,
                                            const FPType val_2) {
  return val_1 > val_2 ? val_1 : val_2;
}

template <typename FPType, typename... FPPack>
KOKKOS_INLINE_FUNCTION constexpr FPType max(const FPType val, FPPack... pack) {
  return val > max(pack...) ? val : max(pack...);
}

// Computes the greatest common denominator of a and b with Euclid's algorithm
KOKKOS_INLINE_FUNCTION constexpr int gcd(const int a, const int b) {
	return (a % b == 0) ? b : gcd(b, a % b);
}

static_assert(gcd(1, 6) == 1, "gcd is broken");
static_assert(gcd(25, 20) == 5, "gcd is broken");
static_assert(gcd(29, 20) == 1, "gcd is broken");
static_assert(gcd(24, 16) == 8, "gcd is broken");

template <typename... int_pack>
KOKKOS_INLINE_FUNCTION constexpr int gcd(const int a, const int b, int_pack... pack) {
	return gcd(gcd(a, b), pack...);
}

static_assert(gcd(16, 24, 28) == 4, "gcd is broken");
static_assert(gcd(24, 16, 28) == 4, "gcd is broken");

// Computes the least common multiple of a and b
// Divide b by gcd(a, b) before multiplying to prevent overflows
KOKKOS_INLINE_FUNCTION constexpr int lcm(const int a, const int b) {
	return a * (b / gcd(a, b));
}

static_assert(lcm(1, 6) == 6, "lcm is broken");
static_assert(lcm(25, 20) == 100, "lcm is broken");
static_assert(lcm(29, 20) == 29 * 20, "lcm is broken");
static_assert(lcm(24, 16) == 48, "lcm is broken");

template <typename... int_pack>
KOKKOS_INLINE_FUNCTION constexpr int lcm(const int a, const int b, int_pack... pack) {
	return lcm(a, lcm(b, pack...));
}

static_assert(lcm(16, 24, 28) == 336, "lcm is broken");
static_assert(lcm(24, 16, 28) == 336, "lcm is broken");

template <typename ViewType>
typename std::enable_if<
    !std::is_same<typename ViewType::non_const_value_type, Scalar>::value, Real>::type
frobenius_norm(const ViewType view, bool ignore_nans = false) {
  typename ViewType::pointer_type data = view.data();

  size_t length = view.size();

  // Note: use Kahan algorithm to increase accuracy
  Real norm = 0;
  Real c = 0;
  Real temp, y;
  for (size_t i = 0; i < length; ++i) {
    if (std::isnan(data[i]) && ignore_nans) {
      continue;
    }
    y = data[i] * data[i] - c;
    temp = norm + y;
    c = (temp - norm) - y;
    norm = temp;
  }

  return std::sqrt(norm);
}

template <typename ViewType>
typename std::enable_if<
    std::is_same<typename ViewType::non_const_value_type, Scalar>::value, Real>::type
frobenius_norm(const ViewType view, bool ignore_nans = false) {
  typename ViewType::pointer_type data = view.data();

  size_t length = view.size();

  // Note: use Kahan algorithm to increase accuracy
  Real norm = 0;
  Real c = 0;
  Real temp, y;
  for (size_t i = 0; i < length; ++i) {
    for (int v = 0; v < VECTOR_SIZE; ++v) {
      if (std::isnan(data[i][v]) && ignore_nans) {
        continue;
      }
      y = data[i][v] * data[i][v] - c;
      temp = norm + y;
      c = (temp - norm) - y;
      norm = temp;
    }
  }

  return std::sqrt(norm);
}

} // namespace Homme

#endif // HOMMEXX_MATH_UTILS_HPP
