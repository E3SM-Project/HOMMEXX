#ifndef HOMMEXX_SUBVIEW_UTILS_HPP
#define HOMMEXX_SUBVIEW_UTILS_HPP

#include "Types.hpp"
#include "ExecSpaceDefs.hpp"

#include <functional>

namespace Homme {

// ================ Subviews of several ranks views with all compile-time dimensions ======================= //
// Note: we template on ScalarType to allow both Real and Scalar case, and
//       also to allow const/non-const versions.
template <typename ScalarType, int DIM0, int DIM1, typename MemSpace,
          typename... Properties>
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM1], MemSpace>
subview(ViewType<ScalarType [DIM0][DIM1], MemSpace, Properties...> v_in, int idim0) {
  assert(v_in.data() != nullptr);
  assert(idim0 >= 0 && idim0 < DIM0);
  return ViewUnmanaged<ScalarType[DIM1], MemSpace>(
      &v_in.implementation_map().reference(idim0, 0));
}

template <typename ScalarType, int DIM0, int DIM1, int DIM2,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1][DIM2], MemSpace>
subview(ViewType<ScalarType [DIM0][DIM1][DIM2], MemSpace, Properties...>
            v_in, const int idim0) {
  assert(v_in.data() != nullptr);
  assert(idim0 >= 0 && idim0 < DIM0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2], MemSpace>(
      &v_in.implementation_map().reference(idim0, 0, 0));
}

template <typename ScalarType, int DIM0, int DIM1, int DIM2,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM2], MemSpace>
subview(ViewType<ScalarType [DIM0][DIM1][DIM2], MemSpace, Properties...>
            v_in, const int idim0, const int idim1) {
  assert(v_in.data() != nullptr);
  assert(idim0 >= 0 && idim0 < DIM0);
  assert(idim1 >= 0 && idim1 < DIM1);
  return ViewUnmanaged<ScalarType[DIM2], MemSpace>(
      &v_in.implementation_map().reference(idim0, idim1, 0));
}

template <typename ScalarType, int DIM0, int DIM1, int DIM2, int DIM3,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3], MemSpace>
subview(ViewType<ScalarType [DIM0][DIM1][DIM2][DIM3], MemSpace, Properties...>
            v_in, const int idim0) {
  assert(v_in.data() != nullptr);
  assert(idim0 >= 0 && idim0 < DIM0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3], MemSpace>(
      &v_in.implementation_map().reference(idim0, 0, 0, 0));
}

template <typename ScalarType, int DIM0, int DIM1, int DIM2, int DIM3,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM3], MemSpace>
subview(ViewType<ScalarType [DIM0][DIM1][DIM2][DIM3], MemSpace, Properties...>
            v_in, const int idim0, const int idim1, const int idim2) {
  assert(v_in.data() != nullptr);
  assert(idim0 >= 0 && idim0 < DIM0);
  assert(idim1 >= 0 && idim1 < DIM1);
  assert(idim2 >= 0 && idim2 < DIM2);
  return ViewUnmanaged<ScalarType[DIM3], MemSpace>(
      &v_in.implementation_map().reference(idim0, idim1, idim2, 0));
}

// TODO: figure out why this routine is needed. The one currently around line 141
//       should already cover the needs for a rank5 view and 1 passed index.
//       Still, without this routine, tests failed. After adding it, they passed.
template <typename ScalarType, int DIM0, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3][DIM4], MemSpace>
subview(ViewType<ScalarType [DIM0][DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...>
            v_in, const int idim0) {
  assert(v_in.data() != nullptr);
  assert(idim0 >= 0 && idim0 < DIM0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3][DIM4], MemSpace>(
      &v_in.implementation_map().reference(idim0, 0, 0, 0, 0));
}

template <typename ScalarType, int DIM0, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4], MemSpace>
subview(ViewType<ScalarType [DIM0][DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...>
            v_in, const int idim0, const int idim1) {
  assert(v_in.data() != nullptr);
  assert(idim0 >= 0 && idim0 < DIM0);
  assert(idim1 >= 0 && idim1 < DIM1);
  return ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4], MemSpace>(
      &v_in.implementation_map().reference(idim0, idim1, 0, 0, 0));
}

// ================ Subviews of several ranks views with runtime 1st dimension ======================= //
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
KOKKOS_INLINE_FUNCTION ViewUnmanaged<ScalarType[DIM1][DIM2], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2], MemSpace, Properties...> v_in,
        int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2], MemSpace>(
      &v_in.implementation_map().reference(ie, 0, 0));
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
