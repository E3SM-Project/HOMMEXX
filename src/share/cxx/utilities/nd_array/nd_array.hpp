
#ifndef _NDARRAY_HPP_
#define _NDARRAY_HPP_

#include <type_traits>

namespace ND_Array_internals {

#include "ct_array.hpp"

template <typename A_Type, typename Dims_CT_Array>
class _ND_Array {
 public:
   constexpr _ND_Array() noexcept {}

  template <typename... int_t>
   A_Type &operator()(int_t... indices) noexcept {
    static_assert(sizeof...(int_t) == DIMS::len(),
                  "Number of indices passed is incorrect");
    return vals[DIMS::slice_idx(indices...)];
  }

  template <typename... int_t>
   constexpr A_Type operator()(int_t... indices) const
      noexcept {
    static_assert(sizeof...(int_t) == DIMS::len(),
                  "Number of indices passed is incorrect");
    return vals[DIMS::slice_idx(indices...)];
  }

  template <typename... int_t>
   constexpr _ND_Array<
      A_Type, typename forward_truncate_array<
                  sizeof...(int_t), Dims_CT_Array>::type>
      &outer_slice(int_t... indices) const noexcept {
    using truncated_dims =
        typename forward_truncate_array<sizeof...(int_t),
                                        DIMS>::type;
    using ret_type = _ND_Array<A_Type, truncated_dims>;
    return *(reinterpret_cast<ret_type *const>(
        &vals[0] + DIMS::slice_idx(indices...)));
  }

  template <typename... int_t>
   _ND_Array<A_Type,
            typename forward_truncate_array<
                sizeof...(int_t), Dims_CT_Array>::type>
      &outer_slice(int_t... indices) noexcept {
    using truncated_dims =
        typename forward_truncate_array<sizeof...(int_t),
                                        DIMS>::type;
    using ret_type = _ND_Array<A_Type, truncated_dims>;
    return *(reinterpret_cast<ret_type *>(
        &vals[0] + DIMS::slice_idx(indices...)));
  }

  template <typename Reshaped_Array>
   Reshaped_Array &reshape() noexcept {
    static_assert(
        Reshaped_Array::DIMS::product() == DIMS::product(),
        "Reshaped array is not the same size");
    return *reinterpret_cast<Reshaped_Array *>(this);
  }

	 static constexpr int extent(int dim) {
		return DIMS::value(dim);
	}

	 static constexpr int dimension() {
		return DIMS::len();
	}

  template <typename _A_Type, typename _Dims_CT_Array>
  friend class _ND_Array;

 private:
  using DIMS = Dims_CT_Array;
  A_Type vals[DIMS::product()];
};
}

template <typename A_Type, int... Dims>
using ND_Array = ND_Array_internals::_ND_Array<
    A_Type, ND_Array_internals::CT_Array<int, Dims...> >;

#endif
