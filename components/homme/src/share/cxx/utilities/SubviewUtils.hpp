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

#ifndef HOMMEXX_SUBVIEW_UTILS_HPP
#define HOMMEXX_SUBVIEW_UTILS_HPP

#include "Types.hpp"
#include "ExecSpaceDefs.hpp"

#include <functional>

namespace Homme {

// ================ Subviews of several ranks views ======================= //
// Note: we template on ScalarType to allow both Real and Scalar case, and
//       also to allow const/non-const versions.
// Note: we assume to have exactly one runtime dimension.
template <typename ScalarType, int DIM1, typename MemSpace,
          typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM1], MemSpace>
subview(ViewType<ScalarType * [DIM1], MemSpace, Properties...> v_in, int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1], MemSpace>(
      &v_in.implementation_map().reference(ie, 0));
}

template <typename ScalarType, int DIM1, int DIM2, typename MemSpace,
          typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM2], MemSpace>
subview(ViewType<ScalarType[DIM1][DIM2], MemSpace, Properties...> v_in,
        int idx_1) {
  assert(v_in.data() != nullptr);
  assert(idx_1 < v_in.extent_int(0));
  assert(idx_1 >= 0);
  return ViewUnmanaged<ScalarType[DIM2], MemSpace>(
      &v_in.implementation_map().reference(idx_1, 0));
}

template <typename ScalarType, int DIM1, int DIM2, typename MemSpace,
          typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM1][DIM2], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2], MemSpace, Properties...> v_in,
        int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2], MemSpace>(
      &v_in.implementation_map().reference(ie, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, typename MemSpace,
          typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM2], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2], MemSpace, Properties...> v_in,
        int idx_0, const int idx_1) {
  assert(v_in.data() != nullptr);
  assert(idx_0 >= 0 && idx_0 < v_in.extent_int(0) );
  assert(idx_1 >= 0 && idx_1 < v_in.extent_int(1) );
  return ViewUnmanaged<ScalarType[DIM2], MemSpace>(
      &v_in.implementation_map().reference(idx_0, idx_1, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, typename MemSpace,
          typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM3], MemSpace>
subview(ViewType<ScalarType[DIM1][DIM2][DIM3], MemSpace, Properties...> v_in,
        int idx_1, int idx_2) {
  assert(v_in.data() != nullptr);
  assert(idx_1 < v_in.extent_int(0));
  assert(idx_1 >= 0);
  assert(idx_2 < v_in.extent_int(1));
  assert(idx_2 >= 0);
  return ViewUnmanaged<ScalarType[DIM3], MemSpace>(
      &v_in.implementation_map().reference(idx_1, idx_2, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, typename MemSpace,
          typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3], MemSpace, Properties...> v_in,
        int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3], MemSpace>(
      &v_in.implementation_map().reference(ie, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, typename MemSpace,
          typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM2][DIM3], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3], MemSpace, Properties...> v_in,
        int ie, int idim1) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  return ViewUnmanaged<ScalarType[DIM2][DIM3], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, typename MemSpace,
          typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM3], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3], MemSpace, Properties...> v_in,
        int ie, int idim1, int idim2) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  assert(idim2 < v_in.extent_int(2));
  assert(idim2 >= 0);
  return ViewUnmanaged<ScalarType[DIM3], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, idim2, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3][DIM4], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...>
            v_in, int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3][DIM4], MemSpace>(
      &v_in.implementation_map().reference(ie, 0, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4], MemSpace>
subview(
    ViewType<ScalarType[DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...> v_in,
    int var) {
  assert(v_in.data() != nullptr);
  assert(var < v_in.extent_int(0));
  assert(var >= 0);
  return ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4], MemSpace>(
      &v_in.implementation_map().reference(var, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM4], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...>
            v_in, int ie, int tl, int igp, int jgp) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(tl < v_in.extent_int(1));
  assert(tl >= 0);
  assert(igp < v_in.extent_int(2));
  assert(igp >= 0);
  assert(jgp < v_in.extent_int(3));
  assert(jgp >= 0);
  return ViewUnmanaged<ScalarType[DIM4], MemSpace>(
      &v_in.implementation_map().reference(ie, tl, igp, jgp, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...>
            v_in, int ie, int idim1) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  return ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM3][DIM4], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...>
            v_in, int ie, int idim1, int idim2) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  assert(idim2 < v_in.extent_int(2));
  assert(idim2 >= 0);
  return ViewUnmanaged<ScalarType[DIM3][DIM4], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, idim2, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4, int DIM5,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3][DIM4][DIM5], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4][DIM5], MemSpace,
                 Properties...> v_in,
        int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3][DIM4][DIM5], MemSpace>(
      &v_in.implementation_map().reference(ie, 0, 0, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4, int DIM5,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4][DIM5], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4][DIM5], MemSpace,
                 Properties...> v_in,
        int ie, int idim1) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  return ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4][DIM5], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, 0, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4, int DIM5,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM3][DIM4][DIM5], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4][DIM5], MemSpace,
                 Properties...> v_in,
        int ie, int idim1, int idim2) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  assert(idim2 < v_in.extent_int(2));
  assert(idim2 >= 0);
  return ViewUnmanaged<ScalarType[DIM3][DIM4][DIM5], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, idim2, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3], MemSpace>
subview(ViewType<ScalarType ** [DIM1][DIM2][DIM3], MemSpace,
                 Properties...> v_in,
        int ie, int remap_idx) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(remap_idx < v_in.extent_int(1));
  assert(remap_idx >= 0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3], MemSpace>(
    &v_in.implementation_map().reference(ie, remap_idx, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM3], MemSpace>
subview(ViewType<ScalarType ** [DIM1][DIM2][DIM3], MemSpace,
                 Properties...> v_in,
        int ie, int remap_idx, int idim1, int idim2) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(remap_idx < v_in.extent_int(1));
  assert(remap_idx >= 0);
  assert(idim1 < v_in.extent_int(2));
  assert(idim1 >= 0);
  assert(idim2 < v_in.extent_int(3));
  assert(idim2 >= 0);
  return ViewUnmanaged<ScalarType[DIM3], MemSpace>(
    &v_in.implementation_map().reference(ie, remap_idx, idim1, idim2, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM3][DIM4], MemSpace>
subview(ViewType<ScalarType ** [DIM1][DIM2][DIM3][DIM4], MemSpace,
                 Properties...> v_in,
        int ie, int remap_idx, int idim1, int idim2) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(remap_idx < v_in.extent_int(1));
  assert(remap_idx >= 0);
  assert(idim1 < v_in.extent_int(2));
  assert(idim1 >= 0);
  assert(idim2 < v_in.extent_int(3));
  assert(idim2 >= 0);
  return ViewUnmanaged<ScalarType[DIM3][DIM4], MemSpace>(
    &v_in.implementation_map().reference(ie, remap_idx, idim1, idim2, 0, 0));
}

} // namespace Homme

#endif // HOMMEXX_SUBVIEW_UTILS_HPP
