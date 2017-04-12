#ifndef HOMMEXX_SPHERE_OPERATORS_HPP
#define HOMMEXX_SPHERE_OPERATORS_HPP

#include "Dimensions.hpp"
#include "Types.hpp"
#include "PhysicalConstants.hpp"

#include <Kokkos_Core.hpp>

namespace Homme {

KOKKOS_INLINE_FUNCTION void
gradient_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                ExecViewUnmanaged<Real[2][NP][NP]> grad_s);

// Sums into grad_s rather than replacing its values
KOKKOS_INLINE_FUNCTION void
gradient_sphere_update(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                       const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                       const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                       const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                       ExecViewUnmanaged<Real[2][NP][NP]> grad_s);

KOKKOS_INLINE_FUNCTION void
divergence_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                  const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                  const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Real[NP][NP]> metDet,
                  const ExecViewUnmanaged<const Real[NP][NP]> rmetDet,
                  const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                  ExecViewUnmanaged<Real[NP][NP]> div_v);

KOKKOS_INLINE_FUNCTION void
vorticity_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                 const ExecViewUnmanaged<const Real[NP][NP]> u,
                 const ExecViewUnmanaged<const Real[NP][NP]> v,
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Real[NP][NP]> rmetDet,
                 const ExecViewUnmanaged<const Real[2][2][NP][NP]> D,
                 ExecViewUnmanaged<Real[NP][NP]> vort);

// ============================ IMPLEMENTATION =========================== //

// Note that gradient_sphere requires scratch space of 2 x NP x NP Reals
// This must be called from the device space
KOKKOS_INLINE_FUNCTION void
gradient_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
  Real _tmp_viewbuf[2][NP][NP];
  ExecViewUnmanaged<Real[2][NP][NP]> v(&_tmp_viewbuf[0][0][0]);
  constexpr int contra_iters = NP * NP;

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, contra_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Real dsdx(0), dsdy(0);
    for (int i = 0; i < NP; ++i) {
      dsdx += dvv(i, l) * scalar(i, j);
      dsdy += dvv(i, l) * scalar(j, i);
    }
    v(0, l, j) = dsdx * PhysicalConstants::rrearth;
    v(1, j, l) = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, grad_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int i = loop_idx / NP;
    const int j = loop_idx % NP;
    grad_s(0, i, j) =
        DInv(0, 0, i, j) * v(0, i, j) + DInv(1, 0, i, j) * v(1, i, j);
    grad_s(1, i, j) =
        DInv(0, 1, i, j) * v(0, i, j) + DInv(1, 1, i, j) * v(1, i, j);
  });
}

KOKKOS_INLINE_FUNCTION void
gradient_sphere_update(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                       const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                       const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                       const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                       ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
  Real _tmp_viewbuf[2][NP][NP];
  ExecViewUnmanaged<Real[2][NP][NP]> v(&_tmp_viewbuf[0][0][0]);
  constexpr int contra_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, contra_iters),
                       [&](const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Real dsdx(0), dsdy(0);
    for (int i = 0; i < NP; ++i) {
      dsdx += dvv(i, l) * scalar(i, j);
      dsdy += dvv(i, l) * scalar(j, i);
    }

    v(0, l, j) = dsdx * PhysicalConstants::rrearth;
    v(1, j, l) = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, grad_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int i = loop_idx / NP;
    const int j = loop_idx % NP;
    grad_s(0, i, j) +=
        (DInv(0, 0, i, j) * v(0, i, j) + DInv(1, 0, i, j) * v(1, i, j));
    grad_s(1, i, j) +=
        (DInv(0, 1, i, j) * v(0, i, j) + DInv(1, 1, i, j) * v(1, i, j));
  });
}

// Note that divergence_sphere requires scratch space of 2 x NP x NP Reals
// This must be called from the device space
KOKKOS_INLINE_FUNCTION void
divergence_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                  const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                  const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Real[NP][NP]> metDet,
                  const ExecViewUnmanaged<const Real[NP][NP]> rmetDet,
                  const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                  ExecViewUnmanaged<Real[NP][NP]> div_v) {
  Real _tmp_viewbuf[2][NP][NP];
  ExecViewUnmanaged<Real[2][NP][NP]> gv(&_tmp_viewbuf[0][0][0]);
  constexpr int contra_iters = NP * NP * 2;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, contra_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / 2 / NP;
    const int jgp = (loop_idx / 2) % NP;
    const int kgp = loop_idx % 2;
    gv(kgp, igp, jgp) =
        metDet(igp, jgp) * (DInv(kgp, 0, igp, jgp) * v(0, igp, jgp) +
                            DInv(kgp, 1, igp, jgp) * v(1, igp, jgp));
  });

  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, div_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Real dudx = 0.0, dvdy = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dudx += dvv(kgp, igp) * gv(0, kgp, jgp);
      dvdy += dvv(kgp, jgp) * gv(1, igp, kgp);
    }

    div_v(igp, jgp) =
        (dudx + dvdy) * ( PhysicalConstants::rrearth * rmetDet(igp, jgp) ) ;
  });
}

// Note that divergence_sphere requires scratch space of 3 x NP x NP Reals
// This must be called from the device space
KOKKOS_INLINE_FUNCTION void
vorticity_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                 const ExecViewUnmanaged<const Real[NP][NP]> u,
                 const ExecViewUnmanaged<const Real[NP][NP]> v,
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Real[NP][NP]> rmetDet,
                 const ExecViewUnmanaged<const Real[2][2][NP][NP]> D,
                 ExecViewUnmanaged<Real[NP][NP]> vort) {
  Real _tmp_viewbuf[2][NP][NP];
  ExecViewUnmanaged<Real[2][NP][NP]> vcov(&_tmp_viewbuf[0][0][0]);
  constexpr int covar_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, covar_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    vcov(0, igp, jgp) =
        D(0, 0, igp, jgp) * u(igp, jgp) + D(1, 0, igp, jgp) * v(igp, jgp);
    vcov(1, igp, jgp) =
        D(0, 1, igp, jgp) * u(igp, jgp) + D(1, 1, igp, jgp) * v(igp, jgp);
  });

  constexpr int vort_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, vort_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Real dudy = 0.0, dvdx = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dvdx += dvv(kgp, igp) * vcov(1, kgp, jgp);
      dudy += dvv(kgp, jgp) * vcov(0, igp, kgp);
    }

    vort(igp, jgp) =
        (dvdx - dudy) * ( PhysicalConstants::rrearth * rmetDet(igp, jgp) );
  });
}

} // namespace Homme

#endif // HOMMEXX_SPHERE_OPERATORS_HPP
