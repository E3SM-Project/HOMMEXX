
#include <Types.hpp>

#include <dimensions.hpp>
#include <kinds.hpp>

#include <cmath>

#include <assert.h>

namespace Homme {

constexpr const real rearth = 6.376E6;
constexpr const real rrearth = 1.0 / rearth;

template <typename Scalar_QP, typename Vector_QP,
          typename Vector_QP_Scratch>
struct gradient_sphere {
  int ie_m;
  const Scalar_QP s_m;
  const Dvv dvv_m;
  const D dinv_m;
  const Vector_QP_Scratch scratch_m;
  const Vector_QP grad_m;

  KOKKOS_INLINE_FUNCTION gradient_sphere(
      int ie, const Scalar_QP s, const Dvv &dvv,
      const D &dinv, const Vector_QP_Scratch &scratch,
      const Vector_QP &grad)
      : ie_m(ie),
        s_m(s),
        dvv_m(dvv),
        dinv_m(dinv),
        scratch_m(scratch),
        grad_m(grad) {}

  /* TODO: Rename these */
  struct loop1_tag {};
  struct loop2_tag {};

  KOKKOS_INLINE_FUNCTION void operator()(const loop1_tag,
                                         int idx) const {
    real dsd[2];
    const int j = idx / np;
    const int l = idx % np;
    for(int k = 0; k < dim; k++) {
      dsd[k] = 0.0;
    }
    for(int i = 0; i < np; i++) {
      dsd[0] += this->dvv_m(i, l) * this->s_m(i, j);
      dsd[1] += this->dvv_m(i, l) * this->s_m(j, i);
    }
    this->scratch_m(l, j, 0) = dsd[0] * rrearth;
    this->scratch_m(j, l, 1) = dsd[1] * rrearth;
  }

  KOKKOS_INLINE_FUNCTION void operator()(const loop2_tag,
                                         int idx) const {
    const int j = idx / np;
    const int i = idx % np;
    for(int h = 0; h < dim; h++) {
      real di1 = this->dinv_m(i, j, 0, h, this->ie_m);
      real v1 = this->scratch_m(i, j, 0);
      real di2 = this->dinv_m(i, j, 1, h, this->ie_m);
      real v2 = this->scratch_m(i, j, 1);
      this->grad_m(i, j, h) = di1 * v1 + di2 * v2;
    }
  }
};

/* This version should never be called from a Kokkos functor
 */
template <typename Scalar_QP, typename Vector_QP>
void gradient_sphere_c(int ie, const Scalar_QP &s,
                       const Dvv &dvv, const D &dinv,
                       Vector_QP &grad) {
  Vector_Field scratch("scratch", np, np, dim);
  using functor =
      gradient_sphere<Scalar_QP, Vector_QP, Vector_Field>;
  functor f(ie, s, dvv, dinv, scratch, grad);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<typename functor::loop1_tag>(
          0, np * np),
      f);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<typename functor::loop2_tag>(
          0, np * np),
      f);
}

template <typename Scalar_QP, typename Vector_QP>
KOKKOS_INLINE_FUNCTION void gradient_sphere_c(
    int ie, const Scalar_QP &s, const Dvv &dvv,
    const D &dinv, const Team_State &team,
    Vector_QP &grad) {
  Vector_Field_Scratch scratch(team.team_scratch(0), np, np,
                               dim);
  using functor = gradient_sphere<Scalar_QP, Vector_QP,
                                  Vector_Field_Scratch>;
  functor f(ie, s, dvv, dinv, scratch, grad);
  /* Somewhat hacky solution to lack of tag support for
   * TeamThreadRange */
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, np * np),
      KOKKOS_LAMBDA(const int idx) {
        f(typename functor::loop1_tag(), idx);
        team.team_barrier();
        f(typename functor::loop2_tag(), idx);
      });
}

template void gradient_sphere_c(int, const Scalar_Field &,
                                const Dvv &, const D &,
                                Vector_Field &);

template <typename Scalar_QP, typename Scalar_QP_Scratch,
          typename Vector_QP, typename Vector_QP_Scratch>
struct vorticity_sphere {
  int ie_m;
  const Vector_QP v_m;
  const Dvv dvv_m;
  const D d_m;
  const MetDet rmetdet_m;
  const Vector_QP_Scratch scratch_buffer_m;
  const Scalar_QP_Scratch scratch_cache_m;
  const Scalar_QP vorticity_m;

  struct loop1_tag {};
  struct loop2_tag {};
  struct loop3_tag {};

  KOKKOS_INLINE_FUNCTION vorticity_sphere(
      int ie, const Vector_QP &v, const Dvv &dvv,
      const D &d, const MetDet &rmetdet,
      const Vector_QP_Scratch &scratch_buffer,
      const Scalar_QP_Scratch &scratch_cache,
      const Scalar_QP &vorticity)
      : ie_m(ie),
        v_m(v),
        dvv_m(dvv),
        d_m(d),
        rmetdet_m(rmetdet),
        scratch_buffer_m(scratch_buffer),
        scratch_cache_m(scratch_cache),
        vorticity_m(vorticity) {}

  KOKKOS_INLINE_FUNCTION void operator()(const loop1_tag &,
                                         int idx) const {
    const int j = idx / dim / np;
    const int i = idx / dim % np;
    const int h = idx % dim;
    this->scratch_buffer_m(i, j, h) =
        this->d_m(i, j, 0, h, this->ie_m) *
            this->v_m(i, j, 0) +
        this->d_m(i, j, 1, h, this->ie_m) *
            this->v_m(i, j, 1);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const loop2_tag &,
                                         int idx) const {
    const int j = idx / np;
    const int l = idx % np;
    real dvd[dim];
    for(int h = 0; h < dim; h++) {
      dvd[h] = 0.0;
    }
    for(int i = 0; i < np; i++) {
      dvd[0] += this->dvv_m(i, l) *
                this->scratch_buffer_m(i, j, 1);
      dvd[1] += this->dvv_m(i, l) *
                this->scratch_buffer_m(j, i, 0);
    }
    this->vorticity_m(l, j) = dvd[0];
    this->scratch_cache_m(j, l) = dvd[1];
  }

  KOKKOS_INLINE_FUNCTION void operator()(const loop3_tag &,
                                         int idx) const {
    const int j = idx / np;
    const int i = idx % np;
    this->vorticity_m(i, j) =
        (this->vorticity_m(i, j) -
         this->scratch_cache_m(i, j)) *
        (this->rmetdet_m(i, j, this->ie_m) * rrearth);
  }
};

template <typename Scalar_QP, typename Vector_QP>
void vorticity_sphere_c(int ie, const Vector_QP &v,
                        const Dvv &dvv, const D &d,
                        const MetDet &rmetdet,
                        Scalar_QP &vorticity) {
  Vector_Field scratch_buffer("contravariant scratch space",
                              np, np, dim);
  Scalar_Field scratch_cache("scratch cache", np, np);

  using functor = vorticity_sphere<Scalar_QP, Scalar_Field,
                                   Vector_QP, Vector_Field>;
  functor f(ie, v, dvv, d, rmetdet, scratch_buffer,
            scratch_cache, vorticity);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<typename functor::loop1_tag>(
          0, np * np * dim),
      f);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<typename functor::loop2_tag>(
          0, np * np),
      f);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<typename functor::loop3_tag>(
          0, np * np),
      f);
}

template <typename Scalar_QP, typename Vector_QP>
KOKKOS_INLINE_FUNCTION void vorticity_sphere_c(
    int ie, const Vector_QP &v, const Dvv &dvv, const D &d,
    const MetDet &rmetdet, const Team_State &team,
    Scalar_QP &vorticity) {
  Vector_Field_Scratch scratch_buffer(team.team_scratch(0),
                                      np, np, dim);
  Scalar_Field_Scratch scratch_cache(team.team_scratch(0),
                                     np, np);
  using functor =
      vorticity_sphere<Scalar_QP, Scalar_Field_Scratch,
                       Vector_QP, Vector_Field_Scratch>;
  functor f(ie, v, dvv, d, rmetdet, scratch_buffer,
            scratch_cache, vorticity);
  /* Somewhat hacky solution to lack of tag support for
   * TeamThreadRange */
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, np * np),
      KOKKOS_LAMBDA(const int idx) {
        for(int i = 0; i < dim; i++) {
          f(typename functor::loop1_tag(), idx * dim + i);
        }
        team.team_barrier();
        f(typename functor::loop2_tag(), idx);
        team.team_barrier();
        f(typename functor::loop3_tag(), idx);
      });
}

template void vorticity_sphere_c(int, const Vector_Field &,
                                 const Dvv &, const D &,
                                 const MetDet &,
                                 Scalar_Field &);

template <typename Scalar_QP, typename Scalar_QP_Scratch,
          typename Vector_QP, typename Vector_QP_Scratch>
struct divergence_sphere {
  int ie_m;
  const Vector_QP v_m;
  const Dvv dvv_m;
  const MetDet metdet_m;
  const MetDet rmetdet_m;
  const D dinv_m;
  const Vector_QP_Scratch scratch_contra_m;
  const Scalar_QP_Scratch scratch_cache_m;
  const Scalar_QP divergence_m;

  KOKKOS_INLINE_FUNCTION divergence_sphere(
      int ie, const Vector_QP &v, const Dvv &dvv,
      const MetDet &metdet, const MetDet &rmetdet,
      const D &dinv, Vector_QP_Scratch &scratch_contra,
      Scalar_QP_Scratch &scratch_cache,
      Scalar_QP &divergence)
      : ie_m(ie),
        v_m(v),
        dvv_m(dvv),
        metdet_m(metdet),
        rmetdet_m(rmetdet),
        dinv_m(dinv),
        scratch_contra_m(scratch_contra),
        scratch_cache_m(scratch_cache),
        divergence_m(divergence) {}

  struct loop1_tag {};
  struct loop2_tag {};
  struct loop3_tag {};

  KOKKOS_INLINE_FUNCTION void operator()(
      const loop1_tag &, const int idx) const {
    const int j = idx / dim / np;
    const int i = (idx / dim) % np;
    const int h = idx % dim;
    assert(j < np);
    assert(i < np);
    assert(h < dim);

    scratch_contra_m(i, j, h) =
        metdet_m(i, j, ie_m) *
        (dinv_m(i, j, h, 0, ie_m) * v_m(i, j, 0) +
         dinv_m(i, j, h, 1, ie_m) * v_m(i, j, 1));
  }

  KOKKOS_INLINE_FUNCTION void operator()(
      const loop2_tag &, const int idx) const {
    const int j = idx / np;
    const int l = idx % np;
    assert(j < np);
    assert(l < np);
    real dvd[dim];
    for(int h = 0; h < dim; h++) {
      dvd[h] = 0.0;
    }
    for(int i = 0; i < np; i++) {
      dvd[0] += dvv_m(i, l) * scratch_contra_m(i, j, 0);
      dvd[1] += dvv_m(i, l) * scratch_contra_m(j, i, 1);
    }
    divergence_m(l, j) = dvd[0];
    scratch_cache_m(j, l) = dvd[1];
  }

  KOKKOS_INLINE_FUNCTION void operator()(
      const loop3_tag &, const int idx) const {
    const int j = idx / np;
    const int i = idx % np;
    divergence_m(i, j) =
        (divergence_m(i, j) + scratch_cache_m(i, j)) *
        (rmetdet_m(i, j, ie_m) * rrearth);
  }
};

template <typename Scalar_QP, typename Vector_QP>
void divergence_sphere_c(int ie, const Vector_QP &v,
                         const Dvv &dvv,
                         const MetDet &metdet,
                         const MetDet &rmetdet,
                         const D &dinv,
                         Scalar_QP &divergence) {
  Vector_Field scratch_contra("contravariant scratch space",
                              np, np, dim);
  Scalar_Field scratch_cache("scratch cache", np, np);
  using functor =
      divergence_sphere<Scalar_QP, Scalar_Field, Vector_QP,
                        Vector_Field>;
  functor f(ie, v, dvv, metdet, rmetdet, dinv,
            scratch_contra, scratch_cache, divergence);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<typename functor::loop1_tag>(
          0, np * np * dim),
      f);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<typename functor::loop2_tag>(
          0, np * np),
      f);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<typename functor::loop3_tag>(
          0, np * np),
      f);
}

template <typename Scalar_QP, typename Vector_QP>
KOKKOS_INLINE_FUNCTION void divergence_sphere_c(
    int ie, const Vector_QP &v, const Dvv &dvv,
    const MetDet &metdet, const MetDet &rmetdet,
    const D &dinv, const Team_State &team,
    Scalar_QP &divergence) {
  Vector_Field_Scratch scratch_contra(team.team_scratch(0),
                                      np, np, dim);
  Scalar_Field_Scratch scratch_cache(team.team_scratch(0),
                                     np, np);
  using functor =
      divergence_sphere<Scalar_QP, Scalar_Field_Scratch,
                        Vector_QP, Vector_Field_Scratch>;
  functor f(ie, v, dvv, metdet, rmetdet, dinv,
            scratch_contra, scratch_cache, divergence);
  /* Somewhat hacky solution to lack of tag support for
   * TeamThreadRange */
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, np * np),
      KOKKOS_LAMBDA(const int idx) {
        for(int i = 0; i < dim; i++) {
          f(typename functor::loop1_tag(), idx * dim + i);
        }
        team.team_barrier();
        f(typename functor::loop2_tag(), idx);
        team.team_barrier();
        f(typename functor::loop3_tag(), idx);
      });
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

  const int vector_mem_needed =
      Vector_Field_Scratch::shmem_size(np, np, dim);
  const int scalar_mem_needed =
      Scalar_Field_Scratch::shmem_size(np, np);
  const int mem_needed =
      7 * vector_mem_needed + 5 * scalar_mem_needed;
  Kokkos::TeamPolicy<> policy(league_size, 1);
  // Let Kokkos choose the team size with Kokkos::AUTO
  // Need to use Kokkos::single somewhere to protect something

  Kokkos::parallel_for(
      policy.set_scratch_size(0,
                              Kokkos::PerTeam(mem_needed)),
      KOKKOS_LAMBDA(const Team_State &team) {
        const int ie =
            (team.league_rank() / nlev) + nets - 1;
        const int k = team.league_rank() % nlev;
	assert(ie < nelemd);
        if(ie < nete) {
          Vector_Field_Scratch ulatlon(team.team_scratch(0),
                                       np, np, dim);
          Scalar_Field_Scratch e(team.team_scratch(0), np,
                                 np);
          Vector_Field_Scratch pv(team.team_scratch(0), np,
                                  np, dim);

          Vector_Field_Scratch grade(team.team_scratch(0),
                                     np, np, dim);
          Scalar_Field_Scratch zeta(team.team_scratch(0),
                                    np, np);
          Vector_Field_Scratch gradh(team.team_scratch(0),
                                     np, np, dim);
          Scalar_Field_Scratch div(team.team_scratch(0), np,
                                   np);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, np * np),
	      [&](const int index) {
		const int j = index / np;
		const int i = index % np;
		real v1 = v(i, j, 0, k, n0 - 1, ie);
		real v2 = v(i, j, 1, k, n0 - 1, ie);
		e(i, j) = 0.0;
		for(int h = 0; h < dim; h++) {
		  ulatlon(i, j, h) = d(i, j, h, 0, ie) * v1 +
		    d(i, j, h, 1, ie) * v2;
		  pv(i, j, h) =
		    ulatlon(i, j, h) *
		    (pmean + p(i, j, k, n0 - 1, ie));
		  e(i, j) += ulatlon(i, j, h) * ulatlon(i, j, h);
		}
		e(i, j) /= 2.0;
		e(i, j) +=
		  p(i, j, k, n0 - 1, ie) + ps(i, j, ie);
	      });

          team.team_barrier();

          gradient_sphere_c(ie, e, dvv, dinv, team, grade);

          vorticity_sphere_c(ie, ulatlon, dvv, d, rmetdet,
                             team, zeta);

          if(tracer_advection_formulation ==
             TRACERADV_UGRADQ) {
            auto p_slice = Kokkos::subview(
                p, std::make_pair(0, np),
                std::make_pair(0, np), k, n0 - 1, ie);
            gradient_sphere_c(ie, p_slice, dvv, dinv, team,
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
                                rmetdet, dinv, team, div);
          }

          team.team_barrier();

          // ==============================================
          // Compute velocity
          // tendency terms
          // ==============================================
          // accumulate all RHS terms
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, np * np),
	      [&](const int index) {
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
