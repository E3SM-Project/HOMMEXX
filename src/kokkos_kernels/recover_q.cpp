
#include <KokkosExp_MDRangePolicy.hpp>
#include <Kokkos_Core.hpp>

#include <config.h.c>

namespace Homme {

// temporary until we have views in - column major
// multiplication with right dimensions
// FIXME: implement P_IDX as column major - Dan will do
// p(np,np,nlev,timelevels)
#define P_IDX(i, j, k, tl, ie) \
  (i + NP * (j + NP * (k + PLEV * (tl + TIMELEVELS * ie))))

constexpr const int TIMELEVELS = 3;

extern "C" {
void recover_q(int &nets, int &nete, int &kmass, int &n0,
               double *p) {
  if(kmass != -1) {
    for(int ie = nets - 1; ie < nete; ++ie) {
      for(int k = 0; k < PLEV; ++k) {
        if(k != kmass) {
          for(int j = 0; j < NP; ++j) {
            for(int i = 0; i < NP; ++i) {
              p[P_IDX(i, j, k, n0, ie)] /=
                  p[P_IDX(i, j, kmass, n0, ie)];
            }
          }
        }
      }
    }
  }
}

// Disable this until the Kokkos defines are set up
#if 0
void recover_q_kokkos(int nets, int nete, int kmass, int n0,
                      double *p_ptr) {
  ViewP p(p_ptr);  // unmanaged view - FIXME: implement this
                   // typedef (Dan)
  if(kmass != -1) {
    md_parallel_for(
		    Kokkos::Experimental::MDRangePolicy({0, 0, 0, nets - 1},
                      {NP, NP, PLEV, nete}),
        KOKKOS_LAMBDA(int i, int j, int k, int ie) {
          if(k != kmass) {
            p(i, j, k, n0, ie) /= p(i, j, kmass, n0, ie);
          }
        });
  }
}
#endif
}
}  // namespace Homme
