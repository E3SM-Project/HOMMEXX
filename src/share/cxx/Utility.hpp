#ifndef HOMMEXX_UTILITY_HPP
#define HOMMEXX_UTILITY_HPP

#include "Types.hpp"

namespace Homme {

template<typename MemSpace, typename MemManagement, typename ScalarType>
ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>
subview(ViewType<ScalarType*[NP][NP][NUM_LEV],MemManagement,MemSpace> v_in, int ie)
{
  return ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>(&v_in(ie,0,0,0));
}

template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM>
ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>
subview(ViewType<ScalarType*[DIM][NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie, int idim)
{
  return ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>(&v_in(ie,idim,0,0,0));
}

template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM1, int DIM2>
ViewUnmanaged<ScalarType [DIM2][NP][NP][NUM_LEV],MemSpace>
subview(ViewType<ScalarType*[DIM1][DIM2][NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie, int idim1)
{
  return ViewUnmanaged<ScalarType [DIM2][NP][NP][NUM_LEV],MemSpace>(&v_in(ie,idim1,0,0,0,0));
}

template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM1, int DIM2>
ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>
subview(ViewType<ScalarType*[DIM1][DIM2][NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie, int idim1, int idim2)
{
  return ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>(&v_in(ie,idim1,idim2,0,0,0));
}

//template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM1, int DIM2, int DIM3>
//ViewUnmanaged<ScalarType [DIM3][NP][NP][NUM_LEV],MemSpace>
//subview(ViewType<ScalarType*[DIM1][DIM2][DIM3][NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie, int idim1, int idim2)
//{
//  return ViewUnmanaged<ScalarType [DIM3][NP][NP][NUM_LEV],MemSpace>(&v_in(ie,idim1,idim2,0,0,0,0));
//}

//template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM1, int DIM2, int DIM3>
//ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>
//subview(ViewType<ScalarType*[DIM1][DIM2][DIM3][NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie, int idim1, int idim2, int idim3)
//{
//  return ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>(&v_in(ie,idim1,idim2,idim3,0,0,0));
//}

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
