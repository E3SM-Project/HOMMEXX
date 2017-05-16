#ifndef HOMMEXX_UTILITY_HPP
#define HOMMEXX_UTILITY_HPP

#include "Dimensions.hpp"
#include "Types.hpp"

namespace Homme {

namespace Impl
{

// No generic implementation
template<typename ViewOut, typename ViewIn, bool DoCopy>
struct DeepCopyImpl;

// Specialization for deep copy necessary
template<typename ViewOut, typename ViewIn>
struct DeepCopyImpl<ViewOut, ViewIn, true>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static void copy(ViewOut view_out, ViewIn view_in)
  {
    // Note: Kokkos will do some compile-time checks
    Kokkos::deep_copy(view_out, view_in);
  }
};

// Specialization for deep copy not necessary
template<typename ViewOut, typename ViewIn>
struct DeepCopyImpl<ViewOut, ViewIn, false>
{
  static void copy(ViewOut /*view_out*/, ViewIn /*view_in*/)
  {
    // At the very least, We can check that the underlying data type is the same
    // (apart from possible cv-qualifiers). This can help finding bugs at compile-time
    static_assert (std::is_same<typename ViewIn::non_const_data_type,
                                typename ViewOut::non_const_data_type
                               >::value,
                   "Error! Trying to copy views with intrinsically (not just cv-related) incompatible data types.");
  }
};

} // namespace Impl

 /*
  * This function will actually perform a deep copy only if the memory spaces
  * are different. Notice that this means that calling this function with two
  * views living on the same memory space will always end up in a no-op, even
  * if the two views are 'different'. Use this method only to perform a deep
  * copy between a view and its mirror view on (possibly) another memory space.
  * For all the other cases, use Kokkos::deep_copy.
  * Why we need this? Glad you asked. Right now, Kokkos::deep_copy has a compile
  * time check to ensure that the data type of the destination view is NOT const.
  * This can be a problem if you create view_out using create_mirror_view, passing
  * the output memory space. In particular, if you have a View view_in with
  * const data type on memory space MS, and you create a mirror view on the same
  * memory space (perhaps you're using OpenMP), the output view will be a shallow
  * copy of view_in, with also const data type. If you then try call deep_copy
  * on these two views, the compiler will bite you.
  */
template<typename ViewOut, typename ViewIn>
KOKKOS_FORCEINLINE_FUNCTION
void deep_copy_mirror_view(ViewOut view_out, ViewIn view_in)
{
  static constexpr bool do_copy = !std::is_same<typename ViewOut::memory_space,
                                                typename ViewIn::memory_space
                                               >::value;

  Impl::DeepCopyImpl<ViewOut,ViewIn,do_copy>::copy(view_out, view_in);
}

template<typename ViewType>
Real compute_view_norm (const ViewType view)
{
  typename ViewType::pointer_type data = view.data();

  size_t length = view.size();

  // Note: use Kahan algorithm to increase accuracy
  Real norm = 0;
  Real c = 0;
  Real temp, y;
  for (size_t i=0; i<length; ++i)
  {
    y = data[i]*data[i] - c;
    temp = norm + y;
    c = (temp - norm) - y;
    norm = temp;
  }

  return std::sqrt(norm);
}

} // namespace Homme

#endif // HOMMEXX_UTILITY_HPP
