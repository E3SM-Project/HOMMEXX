#ifndef HOMMEXX_MATH_UTILS_HPP
#define HOMMEXX_MATH_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <cmath>

namespace Homme
{

template <typename FPType>
KOKKOS_INLINE_FUNCTION constexpr FPType min(const FPType &val_1,
                                            const FPType &val_2) {
  return val_1 < val_2 ? val_1 : val_2;
}

template <typename FPType, typename... FPPack>
KOKKOS_INLINE_FUNCTION constexpr FPType min(const FPType &val, FPPack... pack) {
  return val < min(pack...) ? val : min(pack...);
}

template <typename FPType>
KOKKOS_INLINE_FUNCTION constexpr FPType max(const FPType &val_1,
                                            const FPType &val_2) {
  return val_1 > val_2 ? val_1 : val_2;
}

template <typename FPType, typename... FPPack>
KOKKOS_INLINE_FUNCTION constexpr FPType max(const FPType &val, FPPack... pack) {
  return val > max(pack...) ? val : max(pack...);
}

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
