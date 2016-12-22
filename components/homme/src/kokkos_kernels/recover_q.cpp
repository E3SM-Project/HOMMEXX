
#include <Types.hpp>

#include <dimensions.hpp>
#include <kinds.hpp>

#include <fortran_binding.hpp>

#include <iostream>
#include <stdexcept>

namespace Homme {

extern "C" {

#if 0

#define P_IDX(i, j, k, tl, ie) \
  (i + np * (j + np * (k + nlev * (tl + timelevels * ie))))

void recover_q_c(const int &nets, const int &nete,
                 const int &kmass, const int &n0,
                 const int &num_elems, real *&p) noexcept {
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

void contra2latlon_c(const int &nets, const int &nete,
                     const int &n0, const int &num_elems,
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
                 const int &num_elems,
                 real *&p_ptr) noexcept {
  if(kmass != -1) {
    using RangePolicy = Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<
            2, Kokkos::Experimental::Iterate::Left,
            Kokkos::Experimental::Iterate::Left>,
        Kokkos::IndexType<int> >;

    /* TODO: Improve name of p and it's label */
    HommeHostView5D<MemoryUnmanaged> p_host(
        p_ptr, np, np, nlev, timelevels, num_elems);
    HommeExecView5D p("p", np, np, nlev, timelevels,
                      num_elems);
    Kokkos::deep_copy(p, p_host);
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
    Kokkos::deep_copy(p_host, p);
  }
}

void contra2latlon_c(const int &nets, const int &nete,
                     const int &n0, const int &num_elems,
                     real *const &d_ptr,
                     real *&v_ptr) noexcept {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;

  HommeHostView6D<MemoryUnmanaged> v_host(
      v_ptr, np, np, dim, nlev, timelevels, num_elems);
  HommeExecView6D v("Lateral velocity", np, np, dim, nlev,
                    timelevels, num_elems);
  Kokkos::deep_copy(v, v_host);

  /* TODO: Improve name of d and it's label */
  HommeHostView5D<MemoryUnmanaged> d_host(
      d_ptr, np, np, dim, dim, num_elems);
  HommeExecView5D d("d", np, np, dim, dim, num_elems);
  Kokkos::deep_copy(d, d_host);

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
  Kokkos::deep_copy(v_host, v);
}

/* TODO: Deal with Fortran's globals in a better way */
extern real nu FORTRAN_VAR(control_mod, nu);
extern real nu_s FORTRAN_VAR(control_mod, nu_s);

/* TODO: Give this a better name */
void add_hv_c(const int &nets, const int &nete,
              const int &num_elems,
              real *const &sphere_mp_ptr, real *&ptens_ptr,
              real *&vtens_ptr) noexcept {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;

  const int elem_start = nets - 1;
  const int elem_end = nete;
  const int num_my_elems = elem_end - elem_start;

  /* TODO: Improve name of sphere_mp and it's label */
  HommeHostView3D<MemoryUnmanaged> sphere_mp_host(
      sphere_mp_ptr, np, np, num_elems);
  HommeExecView3D sphere_mp("sphere_mp", np, np, num_elems);
  Kokkos::deep_copy(sphere_mp, sphere_mp_host);

  /* TODO: Improve name of ptens and it's label */
  HommeHostView4D<MemoryUnmanaged> ptens_host(
      ptens_ptr, np, np, nlev, num_my_elems);
  HommeExecView4D ptens("ptens", np, np, nlev,
                        num_my_elems);
  Kokkos::deep_copy(ptens, ptens_host);

  /* TODO: Improve name of vtens and it's label */
  HommeHostView5D<MemoryUnmanaged> vtens_host(
      vtens_ptr, np, np, dim, nlev, num_my_elems);
  HommeExecView5D vtens("vtens", np, np, dim, nlev,
                        num_my_elems);
  Kokkos::deep_copy(vtens, vtens_host);

  real _nu = nu;
  real _nu_s = nu_s;
  try {
    Kokkos::Experimental::md_parallel_for(
        RangePolicy({0, elem_start}, {nlev, elem_end},
                    {1, 1}),
        KOKKOS_LAMBDA(int k, int ie) {
          for(int j = 0; j < np; j++) {
            for(int i = 0; i < np; i++) {
              ptens(i, j, k, ie - elem_start) =
                  -_nu_s * ptens(i, j, k, ie - elem_start) /
                  sphere_mp(i, j, ie);
              for(int h = 0; h < dim; h++) {
                vtens(i, j, h, k, ie - elem_start) =
                    -_nu *
                    vtens(i, j, h, k, ie - elem_start) /
                    sphere_mp(i, j, ie);
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

  // Copy the results back on the host views
  Kokkos::deep_copy(ptens_host, ptens);
  Kokkos::deep_copy(vtens_host, vtens);
}

/* TODO: Give this a better name */
void recover_dpq_c(const int &nets, const int &nete,
                   const int &kmass, const int &n0,
                   const int &num_elems, real *&p_ptr) {
  if(kmass != -1) {
    using RangePolicy = Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<
            2, Kokkos::Experimental::Iterate::Left,
            Kokkos::Experimental::Iterate::Left>,
        Kokkos::IndexType<int> >;
    /* TODO: Improve name of p and it's label */
    HommeHostView5D<MemoryUnmanaged> p_host(
        p_ptr, np, np, nlev, timelevels, num_elems);
    HommeExecView5D p("p", np, np, nlev, timelevels,
                      num_elems);
    Kokkos::deep_copy(p, p_host);
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
    Kokkos::deep_copy(p_host, p);
  }
}

void weighted_rhs_c(const int &nets, const int &nete,
                    const int &num_elems,
                    real *const &rsphere_mp_ptr,
                    real *const &dinv_ptr, real *&ptens_ptr,
                    real *&vtens_ptr) noexcept {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;

  const int elem_start = nets - 1;
  const int elem_end = nete;
  const int num_my_elems = elem_end - elem_start;

  /* TODO: Improve name of rsphere_mp and it's label */
  HommeHostView3D<MemoryUnmanaged> rsphere_mp_host(
      rsphere_mp_ptr, np, np, num_elems);
  HommeExecView3D rsphere_mp("rsphere_mp", np, np,
                             num_elems);
  Kokkos::deep_copy(rsphere_mp, rsphere_mp_host);

  /* TODO: Improve name of dinv and it's label */
  HommeHostView5D<MemoryUnmanaged> dinv_host(
      dinv_ptr, np, np, dim, dim, num_elems);
  HommeExecView5D dinv("dinv", np, np, dim, dim, num_elems);
  Kokkos::deep_copy(dinv, dinv_host);

  /* TODO: Improve name of ptens and it's label */
  HommeHostView4D<MemoryUnmanaged> ptens_host(
      ptens_ptr, np, np, nlev, num_my_elems);
  HommeExecView4D ptens("ptens", np, np, nlev,
                        num_my_elems);
  Kokkos::deep_copy(ptens, ptens_host);

  /* TODO: Improve name of vtens and it's label */
  HommeHostView5D<MemoryUnmanaged> vtens_host(
      vtens_ptr, np, np, dim, nlev, num_my_elems);
  HommeExecView5D vtens("vtens", np, np, dim, nlev,
                        num_my_elems);
  Kokkos::deep_copy(vtens, vtens_host);

  try {
    Kokkos::Experimental::md_parallel_for(
        RangePolicy({0, elem_start}, {nlev, elem_end},
                    {1, 1}),
        KOKKOS_LAMBDA(int k, int ie) {
          for(int j = 0; j < np; j++) {
            for(int i = 0; i < np; i++) {
              ptens(i, j, k, ie - elem_start) *=
                  rsphere_mp(i, j, ie);
              real vtens1 =
                  rsphere_mp(i, j, ie) *
                  vtens(i, j, 0, k, ie - elem_start);
              real vtens2 =
                  rsphere_mp(i, j, ie) *
                  vtens(i, j, 1, k, ie - elem_start);
              for(int h = 0; h < dim; h++) {
                vtens(i, j, h, k, ie - elem_start) =
                    dinv(i, j, h, 0, ie) * vtens1 +
                    dinv(i, j, h, 1, ie) * vtens2;
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

  // Copy the results back on the host views
  Kokkos::deep_copy(ptens_host, ptens);
  Kokkos::deep_copy(vtens_host, vtens);
}

void rk_stage_c(const int &nets, const int &nete,
                const int &n0, const int &np1, const int &s,
                const int &rkstages, const int &num_elems,
                real *&v_ptr, real *&p_ptr,
                real *const &alpha0_ptr,
                real *const &alpha_ptr,
                real *const &ptens_ptr,
                real *const &vtens_ptr) {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;

  const int elem_start = nets - 1;
  const int elem_end = nete;
  const int num_my_elems = elem_end - elem_start;

  if(rkstages < 1) {
    std::cout << "Uh-oh...\n";
  }

  HommeHostView6D<MemoryUnmanaged> v_host(
      v_ptr, np, np, dim, nlev, timelevels, num_elems);
  HommeExecView6D v("Lateral velocity", np, np, dim, nlev,
                    timelevels, num_elems);
  Kokkos::deep_copy(v, v_host);

  /* TODO: Improve name of p and it's label */
  HommeHostView5D<MemoryUnmanaged> p_host(
      p_ptr, np, np, nlev, timelevels, num_elems);
  HommeExecView5D p("p", np, np, nlev, timelevels,
                    num_elems);
  Kokkos::deep_copy(p, p_host);

  /* TODO: Improve name of alpha0 and it's label */
  HommeHostView1D<MemoryUnmanaged> alpha0_host(alpha0_ptr,
                                               rkstages);
  HommeExecView1D alpha0("alpha0", rkstages);
  Kokkos::deep_copy(alpha0, alpha0_host);

  /* TODO: Improve name of alpha and it's label */
  HommeHostView1D<MemoryUnmanaged> alpha_host(alpha_ptr,
                                              rkstages);
  HommeExecView1D alpha("alpha", rkstages);
  Kokkos::deep_copy(alpha, alpha_host);

  /* TODO: Improve name of ptens and it's label */
  HommeHostView4D<MemoryUnmanaged> ptens_host(
      ptens_ptr, np, np, nlev, num_my_elems);
  HommeExecView4D ptens("ptens", np, np, nlev,
                        num_my_elems);
  Kokkos::deep_copy(ptens, ptens_host);

  /* TODO: Improve name of vtens and it's label */
  HommeHostView5D<MemoryUnmanaged> vtens_host(
      vtens_ptr, np, np, dim, nlev, num_my_elems);
  HommeExecView5D vtens("vtens", np, np, dim, nlev,
                        num_my_elems);
  Kokkos::deep_copy(vtens, vtens_host);

  try {
    Kokkos::Experimental::md_parallel_for(
        RangePolicy({0, elem_start}, {nlev, elem_end},
                    {1, 1}),
        KOKKOS_LAMBDA(int k, int ie) {
          for(int j = 0; j < np; j++) {
            for(int i = 0; i < np; i++) {
              for(int h = 0; h < dim; h++) {
                v(i, j, h, k, n0 - 1, ie) =
                    alpha0(s - 1) *
                        v(i, j, h, k, np1 - 1, ie) +
                    alpha(s - 1) *
                        vtens(i, j, h, k, ie - elem_start);
              }
              p(i, j, k, n0 - 1, ie) =
                  alpha0(s - 1) * p(i, j, k, np1 - 1, ie) +
                  alpha(s - 1) *
                      ptens(i, j, k, ie - elem_start);
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

  // Copy the results back on the host views
  Kokkos::deep_copy(p_host, p);
  Kokkos::deep_copy(v_host, v);
}

void copy_timelevels_c(const int &nets, const int &nete,
                       const int &num_elems,
                       const int &n_src, const int &n_dist,
                       real *&p_ptr,
                       real *&v_ptr) noexcept {
  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;
  HommeHostView6D<MemoryUnmanaged> v_host(
      v_ptr, np, np, dim, nlev, timelevels, num_elems);
  HommeExecView6D v("Lateral velocity", np, np, dim, nlev,
                    timelevels, num_elems);
  Kokkos::deep_copy(v, v_host);

  /* TODO: Improve name of p and it's label */
  HommeHostView5D<MemoryUnmanaged> p_host(
      p_ptr, np, np, nlev, timelevels, num_elems);
  HommeExecView5D p("p", np, np, nlev, timelevels,
                    num_elems);
  Kokkos::deep_copy(p, p_host);

  try {
    Kokkos::Experimental::md_parallel_for(
        RangePolicy({0, nets - 1}, {nlev, nete}, {1, 1}),
        KOKKOS_LAMBDA(int k, int ie) {
          for(int j = 0; j < np; ++j) {
            for(int i = 0; i < np; ++i) {
              p(i, j, k, n_dist - 1, ie) =
                  p(i, j, k, n_src - 1, ie);
              v(i, j, 0, k, n_dist - 1, ie) =
                  v(i, j, 0, k, n_src - 1, ie);
              v(i, j, 1, k, n_dist - 1, ie) =
                  v(i, j, 1, k, n_src - 1, ie);
            }
          }
        });
  } catch(std::exception &e) {
    std::cout << e.what() << std::endl;
    std::abort();
  } catch(...) {
    std::cout << "Unknown exception in copy_timelevels_c"
              << std::endl;
    std::abort();
  }

  // Copy the results back on the host views
  Kokkos::deep_copy(v_host, v);
  Kokkos::deep_copy(p_host, p);
}

#endif
}  // extern "C"
}  // namespace Homme
