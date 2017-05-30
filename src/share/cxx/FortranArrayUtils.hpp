#ifndef HOMMEXX_F90_CXX_ARRAY_UTILS_HPP
#define HOMMEXX_F90_CXX_ARRAY_UTILS_HPP

#include "Dimensions.hpp"
#include "Types.hpp"

namespace Homme
{

template<int N1, int N2>
void flip_f90_array_2d_12 (CF90Ptr& array, HostViewUnmanaged<Real[N1][N2]> view)
{
  int idx = 0;
  for (int j=0; j<N2; ++j)
  {
    for (int i=0; i<N1; ++i, ++idx)
    {
      view(i,j) = array[idx];
    }
  }
}

template<int N1, int N2, int N3>
void flip_f90_array_3d_123 (CF90Ptr& array, HostViewUnmanaged<Real[N1][N2][N3]> view)
{
  int idx = 0;
  for (int k=0; k<N3; ++k)
  {
    for (int j=0; j<N2; ++j)
    {
      for (int i=0; i<N1; ++i, ++idx)
      {
        view(i,j,k) = array[idx];
      }
    }
  }
}

template<int N1, int N2, int N3>
void flip_f90_array_3d_312 (CF90Ptr& array, HostViewUnmanaged<Real[N3][N1][N2]> view)
{
  int idx = 0;
  for (int k=0; k<N3; ++k)
  {
    for (int j=0; j<N2; ++j)
    {
      for (int i=0; i<N1; ++i, ++idx)
      {
        view(k,i,j) = array[idx];
      }
    }
  }
}

template<int N1, int N2>
void flip_f90_array_3d_312 (CF90Ptr& array, HostViewUnmanaged<Real*[N1][N2]> view)
{
  int nelems = view.extent(0);
  int idx = 0;
  for (int k=0; k<nelems; ++k)
  {
    for (int j=0; j<N2; ++j)
    {
      for (int i=0; i<N1; ++i, ++idx)
      {
        view(k,i,j) = array[idx];
      }
    }
  }
}

template<int N1, int N2, int N3>
void flip_f90_array_3d_213 (CF90Ptr& array, HostViewUnmanaged<Real[N2][N1][N3]> view)
{
  int idx = 0;
  for (int k=0; k<N3; ++k)
  {
    for (int j=0; j<N2; ++j)
    {
      for (int i=0; i<N1; ++i, ++idx)
      {
        view(j,i,k) = array[idx];
      }
    }
  }
}

template<int N1, int N2, int N3>
void flip_f90_array_4d_4312 (CF90Ptr& array, HostViewUnmanaged<Real*[N3][N1][N2]> view)
{
  int nelems = view.extent(0);
  int idx = 0;
  for (int ie=0; ie<nelems; ++ie)
  {
    for (int k=0; k<N3; ++k)
    {
      for (int j=0; j<N2; ++j)
      {
        for (int i=0; i<N1; ++i, ++idx)
        {
          view(ie,k,i,j) = array[idx];
        }
      }
    }
  }
}

template<int N1, int N2, int N3, int N4>
void flip_f90_array_4d_3412 (CF90Ptr& array, HostViewUnmanaged<Real[N3][N4][N1][N2]> view)
{
  int idx = 0;
  for (int l=0; l<N4; ++l)
  {
    for (int k=0; k<N3; ++k)
    {
      for (int j=0; j<N2; ++j)
      {
        for (int i=0; i<N1; ++i, ++idx)
        {
          view(k,l,i,j) = array[idx];
        }
      }
    }
  }
}

template<int N1, int N2, int N3, int N4>
void flip_f90_array_5d_53412 (CF90Ptr& array, HostViewUnmanaged<Real*[N3][N4][N1][N2]> view)
{
  int nelems = view.extent(0);
  int idx = 0;
  for (int ie=0; ie<nelems; ++ie)
  {
    for (int l=0; l<N4; ++l)
    {
      for (int k=0; k<N3; ++k)
      {
        for (int j=0; j<N2; ++j)
        {
          for (int i=0; i<N1; ++i,++idx)
          {
            view(ie,k,l,i,j) = array[idx];
          }
        }
      }
    }
  }
}

} // namespace Homme

#endif // HOMME_F90_CXX_ARRAY_UTILS_HPP
