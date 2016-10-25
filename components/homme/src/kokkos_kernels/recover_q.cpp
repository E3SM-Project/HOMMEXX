
#include <Types.hpp>

#include <dimensions.hpp>
#include <kinds.hpp>

#include <iostream>

namespace Homme {

constexpr const int TIMELEVELS = 3;

// temporary until we have views in - column major
// multiplication with right dimensions
// FIXME: implement P_IDX as column major - Dan will do
// p(np,np,nlev,timelevels)
#define P_IDX(i, j, k, tl, ie) \
  (i + np * (j + np * (k + nlev * (tl + TIMELEVELS * ie))))

extern "C" {
#if 0
void recover_q(int &nets, int &nete, int &kmass, int nelems, int &n0,
               double *p) {
  if(kmass != -1) {
    for(int ie = nets - 1; ie < nete; ++ie) {
      for(int k = 0; k < nlev; ++k) {
        if(k != kmass) {
          for(int j = 0; j < np; ++j) {
            for(int i = 0; i < np; ++i) {
              p[P_IDX(i, j, k, n0, ie)] /=
                  p[P_IDX(i, j, kmass, n0, ie)];
            }
          }
        }
      }
    }
  }
}

#else

void recover_q(int nets, int nete, int kmass, int n0,
               int nelems, double *p_ptr) {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;
  P p(p_ptr, np, np, nlev, TIMELEVELS,
      nelems);  // unmanaged view - FIXME: implement this
                // typedef (Dan)
  if(kmass != -1) {
    Kokkos::Experimental::md_parallel_for(
        RangePolicy({0, nets - 1}, {nlev, nete}, {1, 1}),
        KOKKOS_LAMBDA(int k, int ie) {
          if(k != kmass) {
            for(int j = 0; j < np; ++j) {
              for(int i = 0; i < np; ++i) {
                p(i, j, k, n0, ie) /=
                    p(i, j, kmass, n0, ie);
              }
            }
          }
        });
  }
}
#endif
}
}  // namespace Homme
