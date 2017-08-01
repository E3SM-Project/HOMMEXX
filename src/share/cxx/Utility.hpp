#ifndef HOMMEXX_UTILITY_HPP
#define HOMMEXX_UTILITY_HPP

#include "Types.hpp"

namespace Homme {

template<typename MemSpace, typename ScalarType>
ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>
subview(ViewUnmanaged<ScalarType*[NP][NP][NUM_LEV],MemSpace> v_in, int ie)
{
  return ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>(&v_in(ie,0,0,0));
}

template<typename MemSpace, typename ScalarType>
ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>
subview(ViewUnmanaged<ScalarType*[NUM_TIME_LEVELS][NP][NP][NUM_LEV],MemSpace> v_in, int ie, int itl)
{
  return ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>(&v_in(ie,itl,0,0,0));
}

template<typename MemSpace, typename ScalarType>
ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>
subview(ViewUnmanaged<ScalarType*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV],MemSpace> v_in, int ie, int itl, int iq)
{
  return ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>(&v_in(ie,itl,iq,0,0,0));
}

template<typename ViewType>
typename std::enable_if<!std::is_same<typename ViewType::value_type,Scalar>::value,Real>::type frobenius_norm (const ViewType view)
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

template<typename ViewType>
typename std::enable_if<std::is_same<typename ViewType::value_type,Scalar>::value,Real>::type frobenius_norm (const ViewType view)
{
  typename ViewType::pointer_type data = view.data();

  size_t length = view.size();

  // Note: use Kahan algorithm to increase accuracy
  Real norm = 0;
  Real c = 0;
  Real temp, y;
  for (size_t i=0; i<length; ++i)
  {
    for (int v=0; v<VECTOR_SIZE; ++v)
    {
      y = data[i][v]*data[i][v] - c;
      temp = norm + y;
      c = (temp - norm) - y;
      norm = temp;
    }
  }

  return std::sqrt(norm);
}

} // namespace Homme

#endif // HOMMEXX_UTILITY_HPP
