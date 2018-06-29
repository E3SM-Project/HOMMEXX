
#ifndef _CTARRAY_HPP_
#define _CTARRAY_HPP_

#include <type_traits>

template <typename FieldT, FieldT leading, FieldT... others>
struct CT_Array {
  constexpr static const FieldT current = leading;

  static constexpr  int len() {
    return sizeof...(others) + 1;
  }

   static constexpr FieldT value(int idx) {
    return (idx == 0 ? current : Next::value(idx - 1));
  }

   static constexpr FieldT sum() {
    return leading + Next::sum();
  }

   static constexpr FieldT product() {
    return leading * Next::product();
  }

  template <typename... indices>
   static constexpr int slice_idx(int idx, indices... tail) {
    return idx * Next::product() + Next::slice_idx(tail...);
  }

   static constexpr int slice_idx(int idx) {
    return idx * Next::product();
  }

  template <typename Idx_Array,
            typename std::enable_if<Idx_Array::len() != 1,
                                    int>::type = 0>
   static constexpr FieldT slice_idx() {
    static_assert(Idx_Array::current < leading,
                  "Index array's indices are too large");
    static_assert(Idx_Array::len() <= Self::len(),
                  "Too many indices");
    return Idx_Array::current * Next::product() +
           Next::template slice_idx<
               typename Idx_Array::Next>();
  }

  template <typename Idx_Array,
            typename std::enable_if<Idx_Array::len() == 1,
                                    int>::type = 0>
   static constexpr FieldT slice_idx() {
    static_assert(Idx_Array::current < leading,
                  "Index array's indices are too large");
    static_assert(Idx_Array::len() <= Self::len(),
                  "Too many indices");
    return Idx_Array::current * Next::product();
  }

  using Self = CT_Array<FieldT, leading, others...>;
  using Next = CT_Array<FieldT, others...>;
};

template <typename FieldT, FieldT val>
struct CT_Array<FieldT, val> {
  constexpr static const FieldT current = val;

   static constexpr int len() { return 1; }
   static constexpr int value(int idx) { return current; }
   static constexpr FieldT sum() { return val; }
   static constexpr FieldT product() { return val; }

   static constexpr int slice_idx(int idx) { return idx; }

  template <typename Idx_Array>
   static constexpr FieldT slice_idx() {
    static_assert(Idx_Array::len() == 1,
                  "Index array not of length 1");
    static_assert(Idx_Array::current < val,
                  "Index array value overflow");
    return Idx_Array::current;
  }
};

/* These are needed to implement truncation of the array
 * without annoying extra specializations */
template <int to_remove, typename array>
struct forward_truncate_array {
  using type = typename forward_truncate_array<
      to_remove - 1, typename array::Next>::type;
};

template <typename array>
struct forward_truncate_array<0, array> {
  using type = array;
};

#endif
