#ifndef HOMMEXX_UTILITY_HPP
#define HOMMEXX_UTILITY_HPP

#include "Types.hpp"
#include "Dimensions.hpp"
#include <array>

namespace Homme {

// These routines flip the order of dimensions, but keeps the relative order of the[NP][NP] block the same.
// E.g., we flip double[NUM_LEV][NPx][NPy] to  double[NPx][NPy][NUM_LEV]

// ------------ 3D View ------------- //

// Copy from f90 to cxx on execution space (dims_in={np,np,...})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==3 &&
                        ViewTypeOut::Rank==3 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     ExecMemSpace
                                    >::value
                       >::type
flip_view_f90_to_cxx_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,3>& dims_in)
{
  Kokkos::parallel_for (
    Kokkos::RangePolicy<>(0,dims_in[2]),
    KOKKOS_LAMBDA (const int ie)
    {
      for (int igp=0; igp<NP; ++igp)
        for (int jgp=0; jgp<NP; ++jgp)
          view_out(ie,igp,jgp) = view_in(igp,jgp,ie);
    }
  );
}

// Copy from f90 to cxx on host space (dims_in={np,np,...})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==3 &&
                        ViewTypeOut::Rank==3 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        !std::is_same<typename ViewTypeIn::traits::memory_space,
                                      ExecMemSpace
                                     >::value
                       >::type
flip_view_f90_to_cxx_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,3>& dims_in)
{
  for (int ie=0; ie<dims_in[2]; ++ie)
    for (int igp=0; igp<NP; ++igp)
      for (int jgp=0; jgp<NP; ++jgp)
        view_out(ie,igp,jgp) = view_in(igp,jgp,ie);
}

// Copy from cxx to f90 on execution space (dims_in={...,np,np})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==3 &&
                        ViewTypeOut::Rank==3 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     ExecMemSpace
                                    >::value
                       >::type
flip_view_cxx_to_f90_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,3>& dims_in)
{
  Kokkos::parallel_for (
    Kokkos::RangePolicy<>(0,dims_in[0]),
    KOKKOS_LAMBDA (const int ie)
    {
      for (int igp=0; igp<NP; ++igp)
        for (int jgp=0; jgp<NP; ++jgp)
          view_out(igp,jgp,ie) = view_in(ie,igp,jgp);
    }
  );
}

// Copy from cxx to f90 on host space (dims_in={...,np,np})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==3 &&
                        ViewTypeOut::Rank==3 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        !std::is_same<typename ViewTypeIn::traits::memory_space,
                                      ExecMemSpace
                                     >::value
                       >::type
flip_view_cxx_to_f90_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,3>& dims_in)
{
  for (int ie=0; ie<dims_in[0]; ++ie)
    for (int igp=0; igp<NP; ++igp)
      for (int jgp=0; jgp<NP; ++jgp)
        view_out(igp,jgp,ie) = view_in(ie,igp,jgp);
}

// ------------ 4D View ------------- //

// Copy from f90 to cxx on execution space (dims_in={np,np,...})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==4 &&
                        ViewTypeOut::Rank==4 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     ExecMemSpace
                                    >::value
                       >::type
flip_view_f90_to_cxx_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,4>& dims_in)
{
  Kokkos::parallel_for (
    Kokkos::RangePolicy<>(0,dims_in[3]*dims_in[2]),
    KOKKOS_LAMBDA (const int id)
    {
      const int ie = id % dims_in[0];
      const int k  = id / dims_in[0];
      for (int igp=0; igp<NP; ++igp)
        for (int jgp=0; jgp<NP; ++jgp)
          view_out(ie,k,igp,jgp) = view_in(igp,jgp,k,ie);
    }
  );
}

// Copy from f90 to cxx on host space (dims_in={np,np,...})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==4 &&
                        ViewTypeOut::Rank==4 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        !std::is_same<typename ViewTypeIn::traits::memory_space,
                                      ExecMemSpace
                                     >::value
                       >::type
flip_view_f90_to_cxx_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,4>& dims_in)
{
  for (int ie=0; ie<dims_in[3]; ++ie)
    for (int k=0; k<dims_in[2]; ++k)
      for (int igp=0; igp<NP; ++igp)
        for (int jgp=0; jgp<NP; ++jgp)
          view_out(ie,k,igp,jgp) = view_in(igp,jgp,k,ie);
}

// Copy from cxx to f90 on execution space (dims_in={...,np,np})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==4 &&
                        ViewTypeOut::Rank==4 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     ExecMemSpace
                                    >::value
                       >::type
flip_view_cxx_to_f90_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,4>& dims_in)
{
  Kokkos::parallel_for (
    Kokkos::RangePolicy<>(0,dims_in[0]*dims_in[1]),
    KOKKOS_LAMBDA (const int id)
    {
      const int ie = id % dims_in[0];
      const int k  = id / dims_in[0];
      for (int igp=0; igp<NP; ++igp)
        for (int jgp=0; jgp<NP; ++jgp)
          view_out(igp,jgp,k,ie) = view_in(ie,k,igp,jgp);
    }
  );
}

// Copy from cxx to f90 on host space (dims_in={...,np,np})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==4 &&
                        ViewTypeOut::Rank==4 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        !std::is_same<typename ViewTypeIn::traits::memory_space,
                                      ExecMemSpace
                                     >::value
                       >::type
flip_view_cxx_to_f90_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,4>& dims_in)
{
  for (int ie=0; ie<dims_in[0]; ++ie)
    for (int k=0; k<dims_in[1]; ++k)
      for (int igp=0; igp<NP; ++igp)
        for (int jgp=0; jgp<NP; ++jgp)
          view_out(igp,jgp,k,ie) = view_in(ie,k,igp,jgp);
}

// ------------ 5D View ------------- //

// Copy from f90 to cxx on execution space (dims_in={np,np,nlevnelemd})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==5 &&
                        ViewTypeOut::Rank==5 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     ExecMemSpace
                                    >::value
                       >::type
flip_view_f90_to_cxx_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,5>& dims_in)
{
  Kokkos::parallel_for (
    Kokkos::RangePolicy<>(0,dims_in[4]*dims_in[3]*dims_in[2]),
    KOKKOS_LAMBDA (const int id)
    {
      const int ie = id % dims_in[4];
      const int k  = (id / dims_in[4]) % dims_in[3];
      const int l  = (id / dims_in[4]) / dims_in[3];
      for (int igp=0; igp<NP; ++igp)
        for (int jgp=0; jgp<NP; ++jgp)
          view_out(ie,k,l,igp,jgp) = view_in(igp,jgp,l,k,ie);
    }
  );
}

// Copy from f90 to cxx on host space (dims_in={np,np,...})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==5 &&
                        ViewTypeOut::Rank==5 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        !std::is_same<typename ViewTypeIn::traits::memory_space,
                                      ExecMemSpace
                                     >::value
                       >::type
flip_view_f90_to_cxx_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,5>& dims_in)
{
  for (int ie=0; ie<dims_in[4]; ++ie)
    for (int k=0; k<dims_in[3]; ++k)
      for (int l=0; l<dims_in[2]; ++l)
        for (int igp=0; igp<NP; ++igp)
          for (int jgp=0; jgp<NP; ++jgp)
            view_out(ie,k,l,igp,jgp) = view_in(igp,jgp,l,k,ie);
}

// Copy from cxx to f90 on execution space (dims_in={...,np,np})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==5 &&
                        ViewTypeOut::Rank==5 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     ExecMemSpace
                                    >::value
                       >::type
flip_view_cxx_to_f90_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,5>& dims_in)
{
  Kokkos::parallel_for (
    Kokkos::RangePolicy<>(0,dims_in[0]*dims_in[1]*dims_in[2]),
    KOKKOS_LAMBDA (const int id)
    {
      const int ie = id % dims_in[0];
      const int k  = (id / dims_in[0]) % dims_in[1];
      const int l  = (id / dims_in[0]) / dims_in[1];
      for (int igp=0; igp<NP; ++igp)
        for (int jgp=0; jgp<NP; ++jgp)
          view_out(igp,jgp,l,k,ie) = view_in(ie,k,l,igp,jgp);
    }
  );
}

// Copy from cxx to f90 on host space (dims_in={...,np,np})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==5 &&
                        ViewTypeOut::Rank==5 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        !std::is_same<typename ViewTypeIn::traits::memory_space,
                                      ExecMemSpace
                                     >::value
                       >::type
flip_view_cxx_to_f90_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,5>& dims_in)
{
  for (int ie=0; ie<dims_in[0]; ++ie)
    for (int k=0; k<dims_in[1]; ++k)
      for (int l=0; l<dims_in[2]; ++l)
        for (int igp=0; igp<NP; ++igp)
          for (int jgp=0; jgp<NP; ++jgp)
            view_out(igp,jgp,l,k,ie) = view_in(ie,k,l,igp,jgp);
}

// ------------ 6D View ------------- //

// Copy from f90 to cxx on execution space (dims_in={np,np,nlevnelemd})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==6 &&
                        ViewTypeOut::Rank==6 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     ExecMemSpace
                                    >::value
                       >::type
flip_view_f90_to_cxx_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,6>& dims_in)
{
  Kokkos::parallel_for (
    Kokkos::RangePolicy<>(0,dims_in[5]*dims_in[4]*dims_in[3]*dims_in[2]),
    KOKKOS_LAMBDA (const int id)
    {
      const int ie = id % dims_in[5];
      const int k  = (id / dims_in[5]) % dims_in[4];
      const int l  = ((id / dims_in[5]) / dims_in[4]) % dims_in[3];
      const int m  = ((id / dims_in[5]) / dims_in[4]) / dims_in[3];
      for (int igp=0; igp<NP; ++igp)
        for (int jgp=0; jgp<NP; ++jgp)
          view_out(ie,k,l,m,igp,jgp) = view_in(igp,jgp,m,l,k,ie);
    }
  );
}

// Copy from f90 to cxx on host space (dims_in={np,np,...})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==6 &&
                        ViewTypeOut::Rank==6 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        !std::is_same<typename ViewTypeIn::traits::memory_space,
                                      ExecMemSpace
                                     >::value
                       >::type
flip_view_f90_to_cxx_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,6>& dims_in)
{
  for (int ie=0; ie<dims_in[5]; ++ie)
    for (int k=0; k<dims_in[4]; ++k)
      for (int l=0; l<dims_in[3]; ++l)
        for (int m=0; m<dims_in[2]; ++m)
          for (int igp=0; igp<NP; ++igp)
            for (int jgp=0; jgp<NP; ++jgp)
              view_out(ie,k,l,m,igp,jgp) = view_in(igp,jgp,m,l,k,ie);
}

// Copy from cxx to f90 on execution space (dims_in={...,np,np})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==6 &&
                        ViewTypeOut::Rank==6 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     ExecMemSpace
                                    >::value
                       >::type
flip_view_cxx_to_f90_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,6>& dims_in)
{
  Kokkos::parallel_for (
    Kokkos::RangePolicy<>(0,dims_in[0]*dims_in[1]*dims_in[2]*dims_in[3]),
    KOKKOS_LAMBDA (const int id)
    {
      const int ie = id % dims_in[0];
      const int k  = (id / dims_in[0]) % dims_in[1];
      const int l  = ((id / dims_in[0]) / dims_in[1]) % dims_in[2];
      const int m  = ((id / dims_in[0]) / dims_in[1]) / dims_in[2];
      for (int igp=0; igp<NP; ++igp)
        for (int jgp=0; jgp<NP; ++jgp)
          view_out(igp,jgp,m,l,k,ie) = view_in(ie,k,l,m,igp,jgp);
    }
  );
}

// Copy from cxx to f90 on host space (dims_in={...,np,np})
template<typename ViewTypeIn, typename ViewTypeOut>
typename std::enable_if<ViewTypeIn::Rank==6 &&
                        ViewTypeOut::Rank==6 &&
                        std::is_same<typename ViewTypeIn::traits::memory_space,
                                     typename ViewTypeOut::traits::memory_space
                                    >::value &&
                        !std::is_same<typename ViewTypeIn::traits::memory_space,
                                      ExecMemSpace
                                     >::value
                       >::type
flip_view_cxx_to_f90_impl (ViewTypeIn& view_in, ViewTypeOut& view_out, const std::array<int,6>& dims_in)
{
  for (int ie=0; ie<dims_in[0]; ++ie)
    for (int k=0; k<dims_in[1]; ++k)
      for (int l=0; l<dims_in[2]; ++l)
        for (int m=0; m<dims_in[3]; ++m)
          for (int igp=0; igp<NP; ++igp)
            for (int jgp=0; jgp<NP; ++jgp)
              view_out(igp,jgp,m,l,k,ie) = view_in(ie,k,l,m,igp,jgp);
}
/////////////////////////////////////////////////

// We hard copy a view flipping the order of its dimensions. We cannot use deep_copy,
// since we would not achieve our goal. Instead, we use the operator() to access entries.
template<typename ViewTypeIn, typename ViewTypeOut>
void flip_view_f90_to_cxx (ViewTypeIn view_in, ViewTypeOut view_out)
{
  constexpr size_t rank_in  = static_cast<size_t>(ViewTypeIn::rank);
  constexpr size_t rank_out = static_cast<size_t>(ViewTypeOut::rank);
  static_assert (rank_in==rank_out, "Error! Trying to copy views of different ranks.\n");

  std::array<int,rank_in> dims_in;
  for (size_t i=0; i<rank_in; ++i)
  {
    dims_in[i] = view_in.extent_int(i);
  }

  flip_view_f90_to_cxx_impl(view_in,view_out,dims_in);
}

template<typename ViewTypeIn, typename ViewTypeOut>
void flip_view_cxx_to_f90 (ViewTypeIn view_in, ViewTypeOut view_out)
{
  constexpr size_t rank_in  = static_cast<size_t>(ViewTypeIn::rank);
  constexpr size_t rank_out = static_cast<size_t>(ViewTypeOut::rank);
  static_assert (rank_in==rank_out, "Error! Trying to copy views of different ranks.\n");

  std::array<int,rank_in> dims_in;
  for (size_t i=0; i<rank_in; ++i)
  {
    dims_in[i] = view_in.extent_int(i);
  }

  flip_view_cxx_to_f90_impl(view_in,view_out,dims_in);
}

} // namespace Homme

#endif // HOMMEXX_UTILITY_HPP
