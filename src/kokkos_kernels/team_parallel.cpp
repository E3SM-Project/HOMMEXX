
#include <Types.hpp>

#include <dimensions.hpp>
#include <kinds.hpp>

#include <cmath>

namespace Homme {

constexpr const real rearth = 6.376E6;
constexpr const real rrearth = 1.0 / rearth;

template <typename Scalar_QP, typename Vector_QP>
void gradient_sphere_c(int ie, const Scalar_QP &s,
                       const Dvv &dvv, const D &dinv,
                       Vector_QP &grad) {
  Homme_Local<real *> dsd("Velocity Spatial Derivatives",
                          dim);
  Homme_Local<real ***> v("Velocity", np, np, dim);
  for(int j = 0; j < np; j++) {
    for(int l = 0; l < np; l++) {
      for(int k = 0; k < dim; k++) {
        dsd(k) = 0.0;
      }
      for(int i = 0; i < np; i++) {
        dsd(0) += dvv(i, l) * s(i, j);
        dsd(1) += dvv(i, l) * s(j, i);
      }
      v(l, j, 0) = dsd(0) * rrearth;
      v(j, l, 1) = dsd(1) * rrearth;
    }
  }
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int h = 0; h < dim; h++) {
        real di1 = dinv(i, j, 0, h, ie);
        real v1 = v(i, j, 0);
        real di2 = dinv(i, j, 1, h, ie);
        real v2 = v(i, j, 1);
        grad(i, j, h) = di1 * v1 + di2 * v2;
      }
    }
  }
}

template void gradient_sphere_c(int, const Scalar_Field &,
                                const Dvv &, const D &,
                                Vector_Field &);

template <typename Scalar_QP, typename Vector_QP>
void vorticity_sphere_c(int ie, const Vector_QP &v,
                        const Dvv &dvv, const D &d,
                        const MetDet &rmetdet,
                        Scalar_QP &vorticity) {
  Homme_Local<real *> dvd("Velocity Spatial Derivatives",
                          dim);
  Homme_Local<real ***> vco("buffer for metric adjustments",
                            np, np, dim);
  Homme_Local<real **> vtemp("buffer for performance", np,
                             np);

  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int h = 0; h < dim; h++) {
        vco(i, j, h) = d(i, j, 0, h, ie) * v(i, j, 0) +
                       d(i, j, 1, h, ie) * v(i, j, 1);
      }
    }
  }

  for(int j = 0; j < np; j++) {
    for(int l = 0; l < np; l++) {
      for(int h = 0; h < dim; h++) {
        dvd(h) = 0.0;
      }
      for(int i = 0; i < np; i++) {
        dvd(0) += dvv(i, l) * vco(i, j, 1);
        dvd(1) += dvv(i, l) * vco(j, i, 0);
      }
      vorticity(l, j) = dvd(0);
      vtemp(j, l) = dvd(1);
    }
  }

  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      vorticity(i, j) = (vorticity(i, j) - vtemp(i, j)) *
                        (rmetdet(i, j, ie) * rrearth);
    }
  }
}

template void vorticity_sphere_c(int, const Vector_Field &,
                                 const Dvv &, const D &,
                                 const MetDet &,
                                 Scalar_Field &);

template <typename Scalar_QP, typename Vector_QP>
void divergence_sphere_c(int ie, const Vector_QP &v,
                         const Dvv &dvv,
                         const MetDet &metdet,
                         const MetDet &rmetdet,
                         const D &dinv,
                         Scalar_QP &divergence) {
  Homme_Local<real *> dvd("Positional Velocity Derivatives",
                          dim);
  Homme_Local<real ***> gv("Contravariant Form", np, np,
                           dim);
  Homme_Local<real **> vvtemp(
      "Divergence Performance Buffer", np, np);

  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int h = 0; h < dim; h++) {
        gv(i, j, h) = metdet(i, j, ie) *
                      (dinv(i, j, h, 0, ie) * v(i, j, 0) +
                       dinv(i, j, h, 1, ie) * v(i, j, 1));
      }
    }
  }

  for(int j = 0; j < np; j++) {
    for(int l = 0; l < np; l++) {
      for(int h = 0; h < dim; h++) {
        dvd(h) = 0.0;
      }
      for(int i = 0; i < np; i++) {
        dvd(0) += dvv(i, l) * gv(i, j, 0);
        dvd(1) += dvv(i, l) * gv(j, i, 1);
      }
      divergence(l, j) = dvd(0);
      vvtemp(j, l) = dvd(1);
    }
  }
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      divergence(i, j) = (divergence(i, j) + vvtemp(i, j)) *
                         (rmetdet(i, j, ie) * rrearth);
    }
  }
}

template void divergence_sphere_c(int, const Vector_Field &,
                                  const Dvv &,
                                  const MetDet &,
                                  const MetDet &, const D &,
                                  Scalar_Field &);

extern "C" {

/* TODO: Give this a better name */
void loop7_c(const int &nets, const int &nete,
             const int &n0, const int &nelemd,
             const int &tracer_advection_formulation,
             const real &pmean, const real &dtstage,
             real *&dvv_ptr, real *&d_ptr, real *&dinv_ptr,
             real *&metdet_ptr, real *&rmetdet_ptr,
             real *&fcor_ptr, real *&p_ptr, real *&ps_ptr,
             real *&v_ptr, real *&ptens_ptr,
             real *&vtens_ptr) {
  Dvv_Host dvv_host(dvv_ptr, np, np);
  Dvv dvv("dvv", np, np);
  Kokkos::deep_copy(dvv, dvv_host);

  D_Host d_host(d_ptr, np, np, dim, dim, nelemd);
  D d("d", np, np, dim, dim, nelemd);
  Kokkos::deep_copy(d, d_host);

  D_Host dinv_host(dinv_ptr, np, np, dim, dim, nelemd);
  D dinv("dinv", np, np, dim, dim, nelemd);
  Kokkos::deep_copy(dinv, dinv_host);

  MetDet_Host metdet_host(metdet_ptr, np, np, nelemd);
  MetDet metdet("metdet", np, np, nelemd);
  Kokkos::deep_copy(metdet, metdet_host);

  MetDet_Host rmetdet_host(rmetdet_ptr, np, np, nelemd);
  MetDet rmetdet("rmetdet", np, np, nelemd);
  Kokkos::deep_copy(rmetdet, rmetdet_host);

  FCor_Host fcor_host(fcor_ptr, np, np, nelemd);
  FCor fcor("fcor", np, np, nelemd);
  Kokkos::deep_copy(fcor, fcor_host);

  P_Host p_host(p_ptr, np, np, nlev, timelevels, nelemd);
  P p("p", np, np, nlev, timelevels, nelemd);
  Kokkos::deep_copy(p, p_host);

  PS_Host ps_host(ps_ptr, np, np, nelemd);
  PS ps("ps", np, np, nelemd);
  Kokkos::deep_copy(ps, ps_host);

  V_Host v_host(v_ptr, np, np, dim, nlev, timelevels,
                nelemd);
  V v("Lateral velocity", np, np, dim, nlev, timelevels,
      nelemd);
  Kokkos::deep_copy(v, v_host);

  PTens_Host ptens_host(ptens_ptr, np, np, nlev,
                        nete - nets + 1);
  PTens ptens("ptens", np, np, nlev, nete - nets + 1);
  Kokkos::deep_copy(ptens, ptens_host);

  VTens_Host vtens_host(vtens_ptr, np, np, dim, nlev,
                        nete - nets + 1);
  VTens vtens("vtens", np, np, dim, nlev, nete - nets + 1);
  Kokkos::deep_copy(vtens, vtens_host);

  enum {
    TRACERADV_UGRADQ = 0,
    TRACERADV_TOTAL_DIVERGENCE = 1
  };

  using RangePolicy = Kokkos::Experimental::MDRangePolicy<
      Kokkos::Experimental::Rank<
          2, Kokkos::Experimental::Iterate::Left,
          Kokkos::Experimental::Iterate::Left>,
      Kokkos::IndexType<int> >;

  /* League size is arbitrary
   * Team size must fit in the hardware constraints
   */
  const int league_size = (nete - nets + 1) * nlev;
  const int team_size = 32;  // arbitrarily chosen for now

  const int vector_mem_needed =
      Vector_Field_Scratch::shmem_size(np, np, dim);
  const int scalar_mem_needed =
      Scalar_Field_Scratch::shmem_size(np, np);
  const int mem_needed =
      4 * vector_mem_needed + 3 * scalar_mem_needed;
  Kokkos::TeamPolicy<> policy(league_size, team_size);

  Kokkos::parallel_for(
      policy.set_scratch_size(0,
                              Kokkos::PerTeam(mem_needed)),
      KOKKOS_LAMBDA(const Team_State &team) {
        const int ie =
            (team.league_rank() / nlev) + nets - 1;
        const int k = team.league_rank() % nlev;
        if(ie < nete) {
          Vector_Field_Scratch ulatlon(team.team_scratch(0),
                                       np, np, dim);
          Scalar_Field_Scratch e(team.team_scratch(1), np,
                                 np);
          Vector_Field_Scratch pv(team.team_scratch(2), np,
                                  np, dim);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, np * np),
              KOKKOS_LAMBDA(const int index) {
                const int j = index / np;
                const int i = index % np;
                real v1 = v(i, j, 0, k, n0 - 1, ie);
                real v2 = v(i, j, 1, k, n0 - 1, ie);
                e(i, j) = 0.0;
                for(int h = 0; h < dim; h++) {
                  ulatlon(i, j, h) =
                      d(i, j, h, 0, ie) * v1 +
                      d(i, j, h, 1, ie) * v2;
                  pv(i, j, h) =
                      ulatlon(i, j, h) *
                      (pmean + p(i, j, k, n0 - 1, ie));
                  e(i, j) +=
                      ulatlon(i, j, h) * ulatlon(i, j, h);
                }
                e(i, j) /= 2.0;
                e(i, j) +=
                    p(i, j, k, n0 - 1, ie) + ps(i, j, ie);
              });

          team.team_barrier();

          Vector_Field_Scratch grade(team.team_scratch(3),
                                     np, np, dim);
          gradient_sphere_c(ie, e, dvv, dinv, grade);

          Scalar_Field_Scratch zeta(team.team_scratch(4),
                                    np, np);
          vorticity_sphere_c(ie, ulatlon, dvv, d, rmetdet,
                             zeta);

          Vector_Field_Scratch gradh(team.team_scratch(5),
                                     np, np, dim);
          Scalar_Field_Scratch div(team.team_scratch(6), np,
                                   np);
          if(tracer_advection_formulation ==
             TRACERADV_UGRADQ) {
            auto p_slice = Kokkos::subview(
                p, std::make_pair(0, np),
                std::make_pair(0, np), k, n0 - 1, ie);
            gradient_sphere_c(ie, p_slice, dvv, dinv,
                              gradh);
            for(int j = 0; j < np; j++) {
              for(int i = 0; i < np; i++) {
                div(i, j) =
                    ulatlon(i, j, 0) * gradh(i, j, 0) +
                    ulatlon(i, j, 1) * gradh(i, j, 1);
              }
            }
          } else {
            divergence_sphere_c(ie, pv, dvv, metdet,
                                rmetdet, dinv, div);
          }

          team.team_barrier();

          // ==============================================
          // Compute velocity
          // tendency terms
          // ==============================================
          // accumulate all RHS terms
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, np * np),
              KOKKOS_LAMBDA(const int index) {
                const int j = index / np;
                const int i = index % np;
                vtens(i, j, 0, k, ie - nets + 1) +=
                    (ulatlon(i, j, 1) *
                         (fcor(i, j, ie) + zeta(i, j)) -
                     grade(i, j, 0));

                vtens(i, j, 1, k, ie - nets + 1) +=
                    (-ulatlon(i, j, 0) *
                         (fcor(i, j, ie) + zeta(i, j)) -
                     grade(i, j, 1));

                ptens(i, j, k, ie - nets + 1) -= div(i, j);
              });

          for(int j = 0; j < np; j++) {
            for(int i = 0; i < np; i++) {
              // take the local element timestep
              for(int h = 0; h < dim; h++) {
                vtens(i, j, h, k, ie - nets + 1) =
                    ulatlon(i, j, h) +
                    dtstage *
                        vtens(i, j, h, k, ie - nets + 1);
              }
              ptens(i, j, k, ie - nets + 1) =
                  p(i, j, k, n0 - 1, ie - nets + 1) +
                  dtstage * ptens(i, j, k, ie - nets + 1);
            }
          }
        }
      });
  Kokkos::deep_copy(ptens_host, ptens);
  Kokkos::deep_copy(vtens_host, vtens);
}

}  // extern "C"

}  // Homme
