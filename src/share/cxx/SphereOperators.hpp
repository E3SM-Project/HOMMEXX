#ifndef HOMMEXX_SPHERE_OPERATORS_HPP
#define HOMMEXX_SPHERE_OPERATORS_HPP

#include "Types.hpp"
#include "CaarRegion.hpp"
#include "Dimensions.hpp"
#include "KernelVariables.hpp"
#include "PhysicalConstants.hpp"

#include <Kokkos_Core.hpp>

namespace Homme {

// Single-level kernels
KOKKOS_INLINE_FUNCTION void
gradient_sphere_sl(const KernelVariables &kv,
                   const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                   const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                   const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                   ExecViewUnmanaged<Real[2][NP][NP]> grad_s);

KOKKOS_INLINE_FUNCTION void gradient_sphere_update_sl(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Real[NP][NP]> scalar,
    ExecViewUnmanaged<Real[2][NP][NP]> grad_s);

KOKKOS_INLINE_FUNCTION void
divergence_sphere_sl(const KernelVariables &kv,
                     const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                     const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                     const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                     const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                     ExecViewUnmanaged<Real[NP][NP]> div_v);

KOKKOS_INLINE_FUNCTION void
vorticity_sphere_sl(const KernelVariables &kv,
                    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> d,
                    const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                    const ExecViewUnmanaged<const Real[NP][NP]> u,
                    const ExecViewUnmanaged<const Real[NP][NP]> v,
                    ExecViewUnmanaged<Real[NP][NP]> vort);

// Multi-level kernels
KOKKOS_INLINE_FUNCTION void
gradient_sphere(const KernelVariables &kv,
                const ExecViewUnmanaged<const Scalar * [2][2][NP][NP]> dinv,
                const ExecViewUnmanaged<const Scalar[NP][NP]> dvv,
                const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> scalar,
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s);

KOKKOS_INLINE_FUNCTION void gradient_sphere_update(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Scalar * [2][2][NP][NP]> dinv,
    const ExecViewUnmanaged<const Scalar[NP][NP]> dvv,
    const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> scalar,
    ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s);

KOKKOS_INLINE_FUNCTION void
divergence_sphere(const KernelVariables &kv,
                  const ExecViewUnmanaged<const Scalar * [2][2][NP][NP]> dinv,
                  const ExecViewUnmanaged<const Scalar * [NP][NP]> metdet,
                  const ExecViewUnmanaged<const Scalar[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Scalar[2][NP][NP][NUM_LEV]> v,
                  ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> div_v);

KOKKOS_INLINE_FUNCTION void
vorticity_sphere(const KernelVariables &kv,
                 const ExecViewUnmanaged<const Scalar * [2][2][NP][NP]> d,
                 const ExecViewUnmanaged<const Scalar * [NP][NP]> metdet,
                 const ExecViewUnmanaged<const Scalar[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> u,
                 const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> v,
                 ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> vort);

// ================ SINGLE-LEVEL IMPLEMENTATION =========================== //

KOKKOS_INLINE_FUNCTION void
gradient_sphere_sl(const KernelVariables &kv,
                   const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                   const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                   const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                   ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
  constexpr int contra_iters = NP * NP;
  // TODO: Use scratch space for this
  Real temp_v[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Real dsdx(0), dsdy(0);
    for (int i = 0; i < NP; ++i) {
      dsdx += dvv(l, i) * scalar(j, i);
      dsdy += dvv(l, i) * scalar(i, j);
    }
    temp_v[0][j][l] = dsdx * PhysicalConstants::rrearth;
    temp_v[1][l][j] = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int h = (loop_idx / NP) / NP;
    const int i = (loop_idx / NP) % NP;
    const int j = loop_idx % NP;
    grad_s(h, j, i) = dinv(kv.ie, h, 0, j, i) * temp_v[0][j][i] +
                      dinv(kv.ie, h, 1, j, i) * temp_v[1][j][i];
  });
}

KOKKOS_INLINE_FUNCTION void gradient_sphere_update_sl(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Real[NP][NP]> scalar,
    ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
  constexpr int contra_iters = NP * NP;
  Real temp_v[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Real dsdx(0), dsdy(0);
    for (int i = 0; i < NP; ++i) {
      dsdx += dvv(l, i) * scalar(j, i);
      dsdy += dvv(l, i) * scalar(i, j);
    }
    temp_v[0][j][l] = dsdx * PhysicalConstants::rrearth;
    temp_v[1][l][j] = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int h = (loop_idx / NP) / NP;
    const int i = (loop_idx / NP) % NP;
    const int j = loop_idx % NP;
    grad_s(h, j, i) += dinv(kv.ie, h, 0, j, i) * temp_v[0][j][i] +
                       dinv(kv.ie, h, 1, j, i) * temp_v[1][j][i];
  });
}

KOKKOS_INLINE_FUNCTION void
divergence_sphere_sl(const KernelVariables &kv,
                     const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                     const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                     const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                     const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                     ExecViewUnmanaged<Real[NP][NP]> div_v) {
  constexpr int contra_iters = NP * NP * 2;
  Real gv[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int hgp = (loop_idx / NP) / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    gv[hgp][igp][jgp] = (dinv(kv.ie, 0, hgp, igp, jgp) * v(0, igp, jgp) +
                         dinv(kv.ie, 1, hgp, igp, jgp) * v(1, igp, jgp)) *
                        metdet(kv.ie, igp, jgp);
  });

  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, div_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Real dudx = 0.0, dvdy = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dudx += dvv(igp, kgp) * gv[0][jgp][kgp];
      dvdy += dvv(jgp, kgp) * gv[1][kgp][igp];
    }
    div_v(jgp, igp) = (dudx + dvdy) * ((1.0 / metdet(kv.ie, jgp, igp)) *
                                       PhysicalConstants::rrearth);
  });
}

KOKKOS_INLINE_FUNCTION void
vorticity_sphere_sl(const KernelVariables &kv,
                    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> d,
                    const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                    const ExecViewUnmanaged<const Real[NP][NP]> u,
                    const ExecViewUnmanaged<const Real[NP][NP]> v,
                    ExecViewUnmanaged<Real[NP][NP]> vort) {
  constexpr int covar_iters = 2 * NP * NP;
  Real vcov[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, covar_iters),
                       [&](const int loop_idx) {
    const int hgp = loop_idx / NP / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    vcov[hgp][igp][jgp] = d(kv.ie, hgp, 0, igp, jgp) * u(igp, jgp) +
                          d(kv.ie, hgp, 1, igp, jgp) * v(igp, jgp);
  });

  constexpr int vort_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, vort_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Real dudy = 0.0;
    Real dvdx = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dvdx += dvv(igp, kgp) * vcov[1][jgp][kgp];
      dudy += dvv(jgp, kgp) * vcov[0][kgp][igp];
    }

    vort(jgp, igp) = (dvdx - dudy) * ((1.0 / metdet(kv.ie, jgp, igp)) *
                                      PhysicalConstants::rrearth);
  });
}

// ================ MULTI-LEVEL IMPLEMENTATION =========================== //

KOKKOS_INLINE_FUNCTION void
gradient_sphere(const KernelVariables &kv,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> scalar,
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s) {
  constexpr int contra_iters = NP * NP;
  // TODO: Use scratch space for this
  Scalar temp_v[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Scalar dsdx(0.0), dsdy(0.0);
    for (int i = 0; i < NP; ++i) {
      dsdx += dvv(l, i) * scalar(j, i, kv.ilev);
      dsdy += dvv(l, i) * scalar(i, j, kv.ilev);
    }
    temp_v[0][j][l] = dsdx * PhysicalConstants::rrearth;
    temp_v[1][l][j] = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int h = (loop_idx / NP) / NP;
    const int i = (loop_idx / NP) % NP;
    const int j = loop_idx % NP;
    grad_s(h, j, i, kv.ilev) = dinv(kv.ie, h, 0, j, i) * temp_v[0][j][i] +
                               dinv(kv.ie, h, 1, j, i) * temp_v[1][j][i];
  });
}

KOKKOS_INLINE_FUNCTION void gradient_sphere_update(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> scalar,
    ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s) {
  constexpr int contra_iters = NP * NP;
  Scalar temp_v[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Scalar dsdx(0), dsdy(0);
    for (int i = 0; i < NP; ++i) {
      dsdx += dvv(l, i) * scalar(j, i, kv.ilev);
      dsdy += dvv(l, i) * scalar(i, j, kv.ilev);
    }
    temp_v[0][j][l] = dsdx * PhysicalConstants::rrearth;
    temp_v[1][l][j] = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int h = (loop_idx / NP) / NP;
    const int i = (loop_idx / NP) % NP;
    const int j = loop_idx % NP;
    grad_s(h, i, j, kv.ilev) += dinv(kv.ie, h, 0, i, j) * temp_v[0][i][j] +
                                dinv(kv.ie, h, 1, i, j) * temp_v[1][i][j];
  });
}

KOKKOS_INLINE_FUNCTION void
divergence_sphere(const KernelVariables &kv,
                  const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                  const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                  const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Scalar[2][NP][NP][NUM_LEV]> v,
                  ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> div_v) {
  constexpr int contra_iters = NP * NP * 2;
  Scalar gv[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int hgp = (loop_idx / NP) / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    gv[hgp][igp][jgp] =
        (dinv(kv.ie, 0, hgp, igp, jgp) * v(0, igp, jgp, kv.ilev) +
         dinv(kv.ie, 1, hgp, igp, jgp) * v(1, igp, jgp, kv.ilev)) *
        metdet(kv.ie, igp, jgp);
  });

  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, div_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Scalar dudx = 0.0, dvdy = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dudx += dvv(igp, kgp) * gv[0][jgp][kgp];
      dvdy += dvv(jgp, kgp) * gv[1][kgp][igp];
    }
    div_v(jgp, igp, kv.ilev) =
        (dudx + dvdy) *
        ((1.0 / metdet(kv.ie, jgp, igp)) * PhysicalConstants::rrearth);
  });
}

KOKKOS_INLINE_FUNCTION void
vorticity_sphere(const KernelVariables &kv,
                 const ExecViewUnmanaged<const Real * [2][2][NP][NP]> d,
                 const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> u,
                 const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> v,
                 ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> vort) {
  constexpr int covar_iters = 2 * NP * NP;
  Scalar vcov[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, covar_iters),
                       [&](const int loop_idx) {
    const int hgp = loop_idx / NP / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    vcov[hgp][igp][jgp] = d(kv.ie, hgp, 0, igp, jgp) * u(igp, jgp, kv.ilev) +
                          d(kv.ie, hgp, 1, igp, jgp) * v(igp, jgp, kv.ilev);
  });

  constexpr int vort_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, vort_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Scalar dudy = 0.0;
    Scalar dvdx = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dvdx += dvv(igp, kgp) * vcov[1][jgp][kgp];
      dudy += dvv(jgp, kgp) * vcov[0][kgp][igp];
    }

    vort(jgp, igp, kv.ilev) = (dvdx - dudy) * ((1.0 / metdet(kv.ie, jgp, igp)) *
                                               PhysicalConstants::rrearth);
  });
}

// analog of fortran's laplace_wk_sphere
// Single level implementation
KOKKOS_INLINE_FUNCTION void laplace_wk_sl(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real[NP][NP]> field,         // input
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,           // for grad, div
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> DInv, // for grad, div
    const ExecViewUnmanaged<const Real * [NP][NP]> metDet,     // for div
    ExecViewUnmanaged<Real[2][NP][NP]> gv,                     // temp for div
    ExecViewUnmanaged<Real[NP][NP]> div_v,     // temp to store div
    ExecViewUnmanaged<Real[2][NP][NP]> temp_v, // temp for grad
    ExecViewUnmanaged<Real[2][NP][NP]> grad_s, // temp to store grad
    // let's reduce num of temps later
    // output
    ExecViewUnmanaged<Real[NP][NP]> laplace) {
  // let's ignore var coef and tensor hv
  gradient_sphere_sl(kv, DInv, dvv, field, grad_s);
  divergence_sphere_sl(kv, DInv, metDet, dvv, grad_s, laplace);
}

/*
  function laplace_sphere_wk(s,deriv,elem,var_coef) result(laplace)
!
!   input:  s = scalar
!   ouput:  -< grad(PHI), grad(s) >   = weak divergence of grad(s)
!     note: for this form of the operator, grad(s) does not need to be made C0
!
    real(kind=real_kind), intent(in) :: s(np,np)
    logical, intent(in) :: var_coef
    type (derivative_t), intent(in) :: deriv
    type (element_t), intent(in) :: elem
    real(kind=real_kind)             :: laplace(np,np)
    real(kind=real_kind)             :: laplace2(np,np)
    integer i,j

    ! Local
    real(kind=real_kind) :: grads(np,np,2), oldgrads(np,np,2)

    grads=gradient_sphere(s,deriv,elem%Dinv)

    if (var_coef) then
       if (hypervis_power/=0 ) then
          ! scalar viscosity with variable coefficient
          grads(:,:,1) = grads(:,:,1)*elem%variable_hyperviscosity(:,:)
          grads(:,:,2) = grads(:,:,2)*elem%variable_hyperviscosity(:,:)
       else if (hypervis_scaling /=0 ) then
          ! tensor hv, (3)
          oldgrads=grads
          do j=1,np
             do i=1,np
!JMD                grads(i,j,1) = sum(oldgrads(i,j,:)*elem%tensorVisc(i,j,1,:))
!JMD                grads(i,j,2) = sum(oldgrads(i,j,:)*elem%tensorVisc(i,j,2,:))
                grads(i,j,1) = oldgrads(i,j,1)*elem%tensorVisc(i,j,1,1) + &
                               oldgrads(i,j,2)*elem%tensorVisc(i,j,1,2)
                grads(i,j,2) = oldgrads(i,j,1)*elem%tensorVisc(i,j,2,1) + &
                               oldgrads(i,j,2)*elem%tensorVisc(i,j,2,2)
             end do
          end do
       else
          ! do nothing: constant coefficient viscsoity
       endif
    endif

    ! note: divergnece_sphere and divergence_sphere_wk are identical *after*
bndry_exchange
    ! if input is C_0.  Here input is not C_0, so we should use
divergence_sphere_wk().
    laplace=divergence_sphere_wk(grads,deriv,elem)
end function
*/

} // namespace Homme

#endif // HOMMEXX_SPHERE_OPERATORS_HPP
