
#include <Types.hpp>

#include <dimensions.hpp>
#include <kinds.hpp>

#include <fortran_binding.hpp>

#include <iostream>
#include <stdexcept>

namespace Homme {

extern "C" {
#if 0
// temporary until we have views in - column major
// multiplication with right dimensions
#define P_IDX(i, j, k, tl, ie) \
  (i + np * (j + np * (k + nlev * (tl + timelevels * ie))))

void recover_q_c(const int &nets, const int &nete,
                 const int &kmass, const int &n0,
                 const int &nelems, real *&p) noexcept {
  if(kmass != -1) {
    for(int ie = nets - 1; ie < nete; ++ie) {
      for(int k = 0; k < nlev; ++k) {
        if(k != kmass - 1) {
          for(int j = 0; j < np; ++j) {
            for(int i = 0; i < np; ++i) {
              p[P_IDX(i, j, k, n0 - 1, ie)] /=
                  p[P_IDX(i, j, kmass - 1, n0 - 1, ie)];
            }
          }
        }
      }
    }
  }
}

#define V_IDX(i, j, n, k, tl, ie) \
  (i +                            \
   np * (j +                      \
         np * (n +                \
               2 * (k + nlev * (tl + timelevels * ie)))))

#define D_IDX(i, j, m, n, ie) \
  (i + np * (j + np * (m + 2 * (n + 2 * ie))))

/* TODO: Give this a better name */
void loop3_c(const int &nets, const int &nete,
             const int &n0, const int &nelems,
             real *const &D, real *&v) noexcept {
  for(int ie = nets - 1; ie < nete; ++ie) {
    for(int k = 0; k < nlev; k++) {
      for(int j = 0; j < np; j++) {
        for(int i = 0; i < np; i++) {
          real v1 = v[V_IDX(i, j, 0, k, n0 - 1, ie)];
          real v2 = v[V_IDX(i, j, 1, k, n0 - 1, ie)];
          for(int h = 0; h < 2; h++) {
            v[V_IDX(i, j, h, k, n0 - 1, ie)] =
                D[D_IDX(i, j, h, 0, ie)] * v1 +
                D[D_IDX(i, j, h, 1, ie)] * v2;
          }
        }
      }
    }
  }
}

#else

void recover_q_c(const int &nets, const int &nete,
                 const int &kmass, const int &n0,
                 const int &nelems, real *&p_ptr) noexcept {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;
  P p(p_ptr, np, np, nlev, timelevels, nelems);
  if(kmass != -1) {
    try {
      Kokkos::Experimental::md_parallel_for(
          RangePolicy({0, nets - 1}, {nlev, nete}, {1, 1}),
          KOKKOS_LAMBDA(int k, int ie) {
            if(k != kmass - 1) {
              for(int j = 0; j < np; ++j) {
                for(int i = 0; i < np; ++i) {
                  p(i, j, k, n0 - 1, ie) /=
                      p(i, j, kmass - 1, n0 - 1, ie);
                }
              }
            }
          });
    } catch(std::exception &e) {
      std::cout << e.what() << std::endl;
      std::abort();
    } catch(...) {
      std::cout << "Unknown exception" << std::endl;
      std::abort();
    }
  }
}

/* TODO: Give this a better name */
void loop3_c(const int &nets, const int &nete,
             const int &n0, const int &nelems,
             real *const &d_ptr, real *&v_ptr) noexcept {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;
  constexpr const int dim = 2;
  V v(v_ptr, np, np, dim, nlev, timelevels, nelems);
  D d(d_ptr, np, np, dim, dim, nelems);

  try {
    Kokkos::Experimental::md_parallel_for(
        RangePolicy({0, nets - 1}, {nlev, nete}, {1, 1}),
        KOKKOS_LAMBDA(int k, int ie) {
          for(int j = 0; j < np; j++) {
            for(int i = 0; i < np; i++) {
              real v1 = v(i, j, 0, k, n0 - 1, ie);
              real v2 = v(i, j, 1, k, n0 - 1, ie);
              for(int h = 0; h < dim; h++) {
                v(i, j, h, k, n0 - 1, ie) =
                    d(i, j, h, 0, ie) * v1 +
                    d(i, j, h, 1, ie) * v2;
              }
            }
          }
        });
  } catch(std::exception &e) {
    std::cout << e.what() << std::endl;
    std::abort();
  } catch(...) {
    std::cout << "Unknown exception" << std::endl;
    std::abort();
  }
}

/* TODO: Deal with Fortran's globals in a better way */
extern real nu FORTRAN_VAR(control_mod, nu);
extern real nu_s FORTRAN_VAR(control_mod, nu_s);

/* TODO: Give this a better name */
void loop5_c(const int &nets, const int &nete,
             const int &nelems, real *const &spheremp_ptr,
             real *&ptens_ptr, real *&vtens_ptr) noexcept {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;
  constexpr const int dim = 2;

  SphereMP spheremp(spheremp_ptr, np, np, nelems);
  PTens ptens(ptens_ptr, np, np, nlev, nete - nets + 1);
  VTens vtens(vtens_ptr, np, np, dim, nlev,
              nete - nets + 1);
  try {
    Kokkos::Experimental::md_parallel_for(
        RangePolicy({0, nets - 1}, {nlev, nete}, {1, 1}),
        KOKKOS_LAMBDA(int k, int ie) {
          for(int j = 0; j < np; j++) {
            for(int i = 0; i < np; i++) {
              ptens(i, j, k, ie - nets + 1) =
                  -nu_s * ptens(i, j, k, ie - nets + 1) /
                  spheremp(i, j, ie);
              for(int h = 0; h < dim; h++) {
                vtens(i, j, h, k, ie - nets + 1) =
                    -nu * vtens(i, j, h, k, ie - nets + 1) /
                    spheremp(i, j, ie);
              }
            }
          }
        });
  } catch(std::exception &e) {
    std::cout << e.what() << std::endl;
    std::abort();
  } catch(...) {
    std::cout << "Unknown exception" << std::endl;
    std::abort();
  }
}

/* TODO: Give this a better name */
void loop6_c(const int &nets, const int &nete,
             const int &kmass, const int &n0,
             const int &numelems, real *&p_ptr) {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;
  P p(p_ptr, np, np, nlev, timelevels, numelems);
  try {
    Kokkos::Experimental::md_parallel_for(
        RangePolicy({0, nets - 1}, {nlev, nete}, {1, 1}),
        KOKKOS_LAMBDA(int k, int ie) {
          if(k != kmass - 1) {
            for(int j = 0; j < np; ++j) {
              for(int i = 0; i < np; ++i) {
                p(i, j, k, n0 - 1, ie) *=
                    p(i, j, kmass - 1, n0 - 1, ie);
              }
            }
          }
        });
  } catch(std::exception &e) {
    std::cout << e.what() << std::endl;
    std::abort();
  } catch(...) {
    std::cout << "Unknown exception" << std::endl;
    std::abort();
  }
}

/* TODO: Give this a better name */
void loop8_c(real *const &rspheremp_ptr,
             real *const &dinv_ptr, real *&ptens_ptr,
             real *&vtens_ptr) noexcept {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          3, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;
  constexpr const int dim = 2;

  SphereMP_noie rspheremp(rspheremp_ptr, np, np);
  D_noie dinv(dinv_ptr, np, np, dim, dim);
  PTens_noie ptens(ptens_ptr, np, np, nlev);
  VTens_noie vtens(vtens_ptr, np, np, dim, nlev);
  try {
    Kokkos::Experimental::md_parallel_for(
        RangePolicy({0, 0, 0}, {np, np, nlev}, {1, 1, 1}),
        KOKKOS_LAMBDA(int i, int j, int k) {
          ptens(i, j, k) *= rspheremp(i, j);
          real vtens1 = rspheremp(i, j) * vtens(i, j, 0, k);
          real vtens2 = rspheremp(i, j) * vtens(i, j, 1, k);
          for(int h = 0; h < dim; h++) {
            vtens(i, j, h, k) = dinv(i, j, h, 0) * vtens1 +
                                dinv(i, j, h, 1) * vtens2;
          }
        });
  } catch(std::exception &e) {
    std::cout << e.what() << std::endl;
    std::abort();
  } catch(...) {
    std::cout << "Unknown exception" << std::endl;
    std::abort();
  }
}

#endif
}
}  // namespace Homme
