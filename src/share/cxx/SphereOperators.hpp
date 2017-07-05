#ifndef HOMMEXX_SPHERE_OPERATORS_HPP
#define HOMMEXX_SPHERE_OPERATORS_HPP

#include "CaarRegion.hpp"
#include "Dimensions.hpp"
#include "KernelVariables.hpp"
#include "PhysicalConstants.hpp"
#include "Types.hpp"

#include <Kokkos_Core.hpp>

#include <assert.h>

namespace Homme {

// Pass the temporary vector for contravariant gradient
KOKKOS_INLINE_FUNCTION void
gradient_sphere(const KernelVariables &kv, const CaarRegion &region,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Real[NP][NP][NUM_LEV]> scalar,
                ExecViewUnmanaged<Real[2][NP][NP][NUM_LEV]> grad_s);

KOKKOS_INLINE_FUNCTION void gradient_sphere_update(
    const KernelVariables &kv, const CaarRegion &region,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Real[NP][NP][NUM_LEV]> scalar,
    ExecViewUnmanaged<Real[2][NP][NP][NUM_LEV]> grad_s);

KOKKOS_INLINE_FUNCTION void
divergence_sphere(const KernelVariables &kv, const CaarRegion &region,
                  const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Real[2][NP][NP][NUM_LEV]> v,
                  ExecViewUnmanaged<Real[NP][NP][NUM_LEV]> div_v);

KOKKOS_FUNCTION void
vorticity_sphere(const KernelVariables &kv, const CaarRegion &region,
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Real[NP][NP][NUM_LEV]> u,
                 const ExecViewUnmanaged<const Real[NP][NP][NUM_LEV]> v,
                 ExecViewUnmanaged<Real[NP][NP][NUM_LEV]> vort);

// ============================ IMPLEMENTATION =========================== //

KOKKOS_INLINE_FUNCTION void
gradient_sphere(const KernelVariables &kv, const CaarRegion &region,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Real[NP][NP][NUM_LEV]> scalar,
                ExecViewUnmanaged<Real[2][NP][NP][NUM_LEV]> grad_s) {
  constexpr int contra_iters = NP * NP;
  // TODO: Use scratch space for this
  Real temp_v[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
                         const int j = loop_idx / NP;
                         const int l = loop_idx % NP;
                         Real dsdx(0), dsdy(0);
                         for (int i = 0; i < NP; ++i) {
                           dsdx += dvv(l, i) * scalar(j, i, kv.ilev);
                           dsdy += dvv(l, i) * scalar(i, j, kv.ilev);
                         }
                         temp_v[0][j][l] = dsdx * PhysicalConstants::rrearth;
                         temp_v[1][l][j] = dsdy * PhysicalConstants::rrearth;
                       });

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(kv.team, grad_iters), [&](const int loop_idx) {
        const int h = (loop_idx / NP) / NP;
        const int i = (loop_idx / NP) % NP;
        const int j = loop_idx % NP;
        grad_s(h, j, i, kv.ilev) = region.m_dinv(kv.ie, 0, h, j, i) * temp_v[0][j][i] +
                                   region.m_dinv(kv.ie, 1, h, j, i) * temp_v[1][j][i];
      });
}

KOKKOS_INLINE_FUNCTION void gradient_sphere_update(
    const KernelVariables &kv, const CaarRegion &region,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Real[NP][NP][NUM_LEV]> scalar,
    ExecViewUnmanaged<Real[2][NP][NP][NUM_LEV]> grad_s) {
  constexpr int contra_iters = NP * NP;
  Real temp_v[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
                         const int j = loop_idx / NP;
                         const int l = loop_idx % NP;
                         Real dsdx(0), dsdy(0);
                         for (int i = 0; i < NP; ++i) {
                           dsdx += dvv(l, i) * scalar(j, i, kv.ilev);
                           dsdy += dvv(l, i) * scalar(i, j, kv.ilev);
                         }
                         temp_v[0][j][l] = dsdx * PhysicalConstants::rrearth;
                         temp_v[1][l][j] = dsdy * PhysicalConstants::rrearth;
                       });

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(kv.team, grad_iters), [&](const int loop_idx) {
        const int h = (loop_idx / NP) / NP;
        const int i = (loop_idx / NP) % NP;
        const int j = loop_idx % NP;
        grad_s(h, j, i, kv.ilev) += region.m_dinv(kv.ie, 0, h, j, i) * temp_v[0][j][i] +
                                    region.m_dinv(kv.ie, 1, h, j, i) * temp_v[1][j][i];
      });
}

KOKKOS_INLINE_FUNCTION void
divergence_sphere(const KernelVariables &kv, const CaarRegion &region,
                  const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Real[2][NP][NP][NUM_LEV]> v,
                  ExecViewUnmanaged<Real[NP][NP][NUM_LEV]> div_v) {
  constexpr int contra_iters = NP * NP * 2;
  Real gv[2][NP][NP];
  Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(kv.team, contra_iters),
      [&](const int loop_idx) {
        const int hgp = (loop_idx / NP) / NP;
        const int igp = (loop_idx / NP) % NP;
        const int jgp = loop_idx % NP;
        gv[hgp][jgp][igp] =
            region.m_metdet(kv.ie, jgp, igp) *
            (region.m_dinv(kv.ie, hgp, 0, jgp, igp) * v(0, jgp, igp, kv.ilev) +
             region.m_dinv(kv.ie, hgp, 1, jgp, igp) * v(1, jgp, igp, kv.ilev));
      });

  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(kv.team, div_iters), [&](const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        Real dudx = 0.0, dvdy = 0.0;
        for (int kgp = 0; kgp < NP; ++kgp) {
          dudx += dvv(igp, kgp) * gv[0][jgp][kgp];
          dvdy += dvv(jgp, kgp) * gv[1][kgp][igp];
        }

        div_v(jgp, igp, kv.ilev) =
            (dudx + dvdy) *
            ((1.0 / region.m_metdet(kv.ie, jgp, igp)) * PhysicalConstants::rrearth);
      });
}

KOKKOS_FUNCTION void
vorticity_sphere(const KernelVariables &kv, const CaarRegion &region,
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Real[NP][NP][NUM_LEV]> u,
                 const ExecViewUnmanaged<const Real[NP][NP][NUM_LEV]> v,
                 ExecViewUnmanaged<Real[NP][NP][NUM_LEV]> vort) {
  constexpr int covar_iters = NP * NP;
  Real vcov[2][NP][NP];
  Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(kv.team, covar_iters), [&](const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        vcov[0][jgp][igp] = region.m_d(kv.ie, 0, 0, jgp, igp) * u(jgp, igp, kv.ilev) +
                            region.m_d(kv.ie, 1, 0, jgp, igp) * v(jgp, igp, kv.ilev);
        vcov[1][jgp][igp] = region.m_d(kv.ie, 0, 1, jgp, igp) * u(jgp, igp, kv.ilev) +
                            region.m_d(kv.ie, 1, 1, jgp, igp) * v(jgp, igp, kv.ilev);
      });

  constexpr int vort_iters = NP * NP;
  Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(kv.team, vort_iters), [&](const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        Real dudy = 0.0;
        Real dvdx = 0.0;
        for (int kgp = 0; kgp < NP; ++kgp) {
          dvdx += dvv(igp, kgp) * vcov[1][jgp][kgp];
          dudy += dvv(jgp, kgp) * vcov[0][kgp][igp];
        }

        vort(jgp, igp, kv.ilev) =
            (dvdx - dudy) *
            ((1.0 / region.m_metdet(kv.ie, jgp, igp)) * PhysicalConstants::rrearth);
      });
}

// analog of fortran's laplace_wk_sphere
KOKKOS_INLINE_FUNCTION void laplace_wk(
    const KernelVariables &kv,
    const CaarRegion &region,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,        // for grad, div
    const ExecViewUnmanaged<const Real[NP][NP][NUM_LEV]> field,      // input
    ExecViewUnmanaged<Real[2][NP][NP][NUM_LEV]> grad_s, // temp to store grad
    // let's reduce num of temps later
    // output
    ExecViewUnmanaged<Real[NP][NP][NUM_LEV]> laplace) {
  // let's ignore var coef and tensor hv
  gradient_sphere(kv, region, dvv, field, grad_s);
  divergence_sphere(kv, region, dvv, grad_s, laplace);
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
