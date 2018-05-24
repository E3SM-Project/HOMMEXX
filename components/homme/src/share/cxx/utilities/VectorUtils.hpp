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

#ifndef HOMMEXX_VECTOR_UTILS_HPP
#define HOMMEXX_VECTOR_UTILS_HPP

#include "vector/KokkosKernels_Vector.hpp"
#include "utilities/MathUtils.hpp"

namespace KokkosKernels {
namespace Batched {
namespace Experimental {

template <typename SpT>
inline
Vector<VectorTag<AVX<double, SpT>, 4> >
max (const Vector<VectorTag<AVX<double, SpT>, 4> >& a,
     const Vector<VectorTag<AVX<double, SpT>, 4> >& b)
{
  return _mm256_max_pd (a, b);
}

template <typename SpT>
inline
Vector<VectorTag<AVX<double, SpT>, 4> >
min (const Vector<VectorTag<AVX<double, SpT>, 4> >& a,
     const Vector<VectorTag<AVX<double, SpT>, 4> >& b)
{
  return _mm256_min_pd (a, b);
}

template <typename SpT>
inline
Vector<VectorTag<AVX<double, SpT>, 8> >
max (const Vector<VectorTag<AVX<double, SpT>, 8> >& a,
     const Vector<VectorTag<AVX<double, SpT>, 8> >& b)
{
  return _mm512_max_pd (a, b);
}

template <typename SpT>
inline
Vector<VectorTag<AVX<double, SpT>, 8> >
min (const Vector<VectorTag<AVX<double, SpT>, 8> >& a,
     const Vector<VectorTag<AVX<double, SpT>, 8> >& b)
{
  return _mm512_min_pd (a, b);
}

template <typename SpT, int l>
KOKKOS_INLINE_FUNCTION
Vector<VectorTag<SIMD<double, SpT>, l> >
max (const Vector<VectorTag<SIMD<double, SpT>, l> >& a,
     const Vector<VectorTag<SIMD<double, SpT>, l> >& b)
{
  Vector<VectorTag<SIMD<double, SpT>, l> > r_val;
#pragma ivdep
  for (int i = 0; i < Vector<VectorTag<SIMD<double, SpT>, l>>::vector_length; i++) {
    r_val[i] = Homme::max(a[i],b[i]);
  }

  return r_val;
}

template <typename SpT, int l>
KOKKOS_INLINE_FUNCTION
Vector<VectorTag<SIMD<double, SpT>, l> >
min (const Vector<VectorTag<SIMD<double, SpT>, l> >& a,
     const Vector<VectorTag<SIMD<double, SpT>, l> >& b)
{
  Vector<VectorTag<SIMD<double, SpT>, l> > r_val;
#pragma ivdep
  for (int i = 0; i < Vector<VectorTag<SIMD<double, SpT>, l>>::vector_length; i++) {
    r_val[i] = Homme::min(a[i],b[i]);
  }

  return r_val;
}

} // namespace KokkosKernels
} // namespace Batched
} // namespace Experimental

#endif // HOMMEXX_VECTOR_UTILS_HPP
