#ifndef HOMMEXX_SPHERE_OPERATORS_HPP
#define HOMMEXX_SPHERE_OPERATORS_HPP

#include "Dimensions.hpp"
#include "Types.hpp"
#include "PhysicalConstants.hpp"

#include <Kokkos_Core.hpp>

#include <assert.h>

namespace Homme {

// Pass the temporary vector for contravariant gradient
KOKKOS_INLINE_FUNCTION void
gradient_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                ExecViewUnmanaged<Real[2][NP][NP]> temp_v,
                ExecViewUnmanaged<Real[2][NP][NP]> grad_s);

KOKKOS_INLINE_FUNCTION void
gradient_sphere_update(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                       const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                       const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                       const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                       ExecViewUnmanaged<Real[2][NP][NP]> temp_v,
                       ExecViewUnmanaged<Real[2][NP][NP]> grad_s);


KOKKOS_INLINE_FUNCTION void
divergence_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                  const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                  const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Real[NP][NP]> metDet,
                  const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                  ExecViewUnmanaged<Real[2][NP][NP]> gv,
                  ExecViewUnmanaged<Real[NP][NP]> div_v);

KOKKOS_FUNCTION void
vorticity_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                 const ExecViewUnmanaged<const Real[NP][NP]> u,
                 const ExecViewUnmanaged<const Real[NP][NP]> v,
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Real[NP][NP]> metDet,
                 const ExecViewUnmanaged<const Real[2][2][NP][NP]> D,
                 ExecViewUnmanaged<Real[2][NP][NP]> vcov,
                 ExecViewUnmanaged<Real[NP][NP]> vort);

// ============================ IMPLEMENTATION =========================== //

KOKKOS_INLINE_FUNCTION void
gradient_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                ExecViewUnmanaged<Real[2][NP][NP]> temp_v,
                ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
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
    temp_v(0, l, j) = dsdx * PhysicalConstants::rrearth;
    temp_v(1, j, l) = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, grad_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int h = (loop_idx / NP) / NP;
    const int i = (loop_idx / NP) % NP;
    const int j = loop_idx % NP;
    grad_s(h,i,j) = DInv(0,h,i,j) * temp_v(0,i,j) + DInv(1,h,i,j) * temp_v(1,i,j);
  });
}

KOKKOS_INLINE_FUNCTION void
gradient_sphere_update(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                       const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                       const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                       const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                       ExecViewUnmanaged<Real[2][NP][NP]> temp_v,
                       ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
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
    temp_v(0, l, j) = dsdx * PhysicalConstants::rrearth;
    temp_v(1, j, l) = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, grad_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int h = (loop_idx / NP) / NP;
    const int i = (loop_idx / NP) % NP;
    const int j = loop_idx % NP;
    grad_s(h,i,j) += DInv(0,h,i,j) * temp_v(0,i,j) + DInv(1,h,i,j) * temp_v(1,i,j);
  });
}

// Note that divergence_sphere requires scratch space of 2 x NP x NP Reals
// This must be called from the device space
KOKKOS_INLINE_FUNCTION void
divergence_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                  const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                  const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Real[NP][NP]> metDet,
                  const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                  ExecViewUnmanaged<Real[2][NP][NP]> gv,
                  ExecViewUnmanaged<Real[NP][NP]> div_v) {
  constexpr int contra_iters = NP * NP * 2;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, contra_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int hgp = (loop_idx / NP) / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    gv(hgp,igp,jgp) = metDet(igp,jgp) * (DInv(hgp,0,igp,jgp) * v(0,igp,jgp) +
                                         DInv(hgp,1,igp,jgp) * v(1,igp,jgp));
  });

  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, div_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Real dudx = 0.0, dvdy = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dudx += dvv(kgp, igp) * gv(0, kgp, jgp);
      dvdy += dvv(kgp, jgp) * gv(1, igp, kgp);
    }

    div_v(igp,jgp) = (dudx + dvdy) * ((1.0 / metDet(igp, jgp)) * PhysicalConstants::rrearth);
  });
}

// Note that divergence_sphere requires scratch space of 3 x NP x NP Reals
// This must be called from the device space
KOKKOS_INLINE_FUNCTION void
vorticity_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                 const ExecViewUnmanaged<const Real[NP][NP]> u,
                 const ExecViewUnmanaged<const Real[NP][NP]> v,
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Real[NP][NP]> metDet,
                 const ExecViewUnmanaged<const Real[2][2][NP][NP]> D,
                 ExecViewUnmanaged<Real[2][NP][NP]> vcov,
                 ExecViewUnmanaged<Real[NP][NP]> vort) {
  constexpr int covar_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, covar_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    vcov(0,igp,jgp) = D(0,0,igp,jgp) * u(igp,jgp) + D(1,0,igp,jgp) * v(igp,jgp);
    vcov(1,igp,jgp) = D(0,1,igp,jgp) * u(igp,jgp) + D(1,1,igp,jgp) * v(igp,jgp);
  });

  constexpr int vort_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, vort_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Real dudy = 0.0;
    Real dvdx = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dvdx += dvv(kgp, igp) * vcov(1, kgp, jgp);
      dudy += dvv(kgp, jgp) * vcov(0, igp, kgp);
    }

    vort(igp, jgp) =
        (dvdx - dudy) * ((1.0 / metDet(igp, jgp)) * PhysicalConstants::rrearth );
  });
}


//version of fortran's laplace_wk_sphere
KOKKOS_INLINE_FUNCTION void
laplace_wk(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                 const ExecViewUnmanaged<const Real[NP][NP]> field, //input
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,   //for grad, div
                 const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv, //for grad, div
                 const ExecViewUnmanaged<const Real[NP][NP]> metDet,//for div
                 ExecViewUnmanaged<Real[2][NP][NP]> gv,//temp for div
                 ExecViewUnmanaged<Real[NP][NP]> div_v, //temp to store div
                 ExecViewUnmanaged<Real[2][NP][NP]> temp_v, //temp for grad
                 ExecViewUnmanaged<Real[2][NP][NP]> grad_s, //temp to store grad
//let's reduce num of temps later
//output
                 ExecViewUnmanaged<Real[NP][NP]> laplace) {
//let's ignore var coef and tensor hv
  gradient_sphere(team, field, dvv, DInv, temp_v, grad_s);
  divergence_sphere(team, grad_s, dvv, metDet, DInv, gv, laplace);
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

    ! note: divergnece_sphere and divergence_sphere_wk are identical *after* bndry_exchange
    ! if input is C_0.  Here input is not C_0, so we should use divergence_sphere_wk().
    laplace=divergence_sphere_wk(grads,deriv,elem)
end function
*/










} // namespace Homme

#endif // HOMMEXX_SPHERE_OPERATORS_HPP
