
namespace Homme {

#include <Types.hpp>
#include <dimensions.hpp>
#include <kinds.hpp>

#include <cmath>

#include <string.h>

// tc1_velocity(ie, n0, k, spherep, d, v);
// 0 <= k < nlev
// 0 <= ie < nelemd
// elem_state_v(:,:,:,k,n0,ie)=tc1_velocity(elem(ie)%spherep,elem(ie)%Dinv)
// v(np, np, dim, nlev, timelevels, nelemd)
void tc1_velocity(int ie, int n0, int k, real u0,
                  SphereP &spherep, D &d, V &v) {
  constexpr const real alpha = 0.0;
  const real csalpha = std::cos(alpha);
  const real snalpha = std::sin(alpha);
  Kokkos::Array<Kokkos::Array<real, np>, np> snlon, cslon,
      snlat, cslat;
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      snlat[i][j] = std::sin(
          spherep(Spherical_Polar_e::Lat, i, j, ie));
      cslat[i][j] = std::cos(
          spherep(Spherical_Polar_e::Lat, i, j, ie));
      snlon[i][j] = std::sin(
          spherep(Spherical_Polar_e::Lon, i, j, ie));
      cslon[i][j] = std::cos(
          spherep(Spherical_Polar_e::Lon, i, j, ie));
    }
  }
  real V1, V2;
  for(int i = 0; i < np; i++) {
    for(int h = 0; h < np; h++) {
      V1 = u0 * (cslat[h][i] * csalpha +
                 snlat[h][i] * cslon[h][i] * snalpha);
      V2 = -u0 * (snlon[h][i] * snalpha);
      for(int g = 0; g < dim; g++) {
        v(h, i, g, k, n0, ie) =
            V1 * d(h, i, g, 0) + V2 * d(h, i, g, 1);
      }
    }
  }
}

void swirl_velocity(int ie, int n0, int k, real time,
                    SphereP &spherep, D &d, V &v) {
  /* TODO: Implement this */
}

real gradient_sphere(int ie, Energy &e) {
  /* TODO: Implement this */
  return 0.0 / 0.0;
}

void team_parallel_ex(const int &nets, const int &nete,
                      const int &n0, const int &nelemd,
                      const real &pmean, const real &u0,
                      const real &real_time,
                      const char *&topology,
                      const char *&test_case, real *&d_ptr,
                      real *&dinv_ptr, real *&fcor_ptr,
                      real *&spheremp_ptr, real *&p_ptr,
                      real *&ps_ptr, real *&v_ptr) {
  constexpr const unsigned dim = 2;

  D d(d_ptr, np, np, dim, dim, nelemd);
  D dinv(dinv_ptr, np, np, dim, dim, nelemd);
  FCor fcor(fcor_ptr, np, np, nelemd);
  SphereMP spheremp(np, np, nelemd);
  P p(p_ptr, np, np, nlev, timelevels, nelemd);
  PS ps(ps_ptr, np, np, nelemd);
  V v(v_ptr, np, np, dim, nlev, timelevels, nelemd);

  for(int ie = nets - 1; ie < nete; ie++) {
    if(strcmp(topology, "cube") == 0) {
      if(strcmp(test_case, "swtc1") == 0) {
        for(int k = 0; k < nlev; k++) {
          tc1_velocity(ie, n0, k, u0, spherep, d, v);
        }
      } else if(strcmp(test_case, "swirl") == 0) {
        for(int k = 0; k < nlev; k++) {
          swirl_velocity(ie, n0, k, time, spherep, dinv);
        }
      }
    }

    for(int k = 0; k < nlev; k++) {
      // ulatlon(np, np, dim), local variable
      ULatLon ulatlon(np, np, dim);
      // E(np, np), local variable
      Energy e(np.np);
      // pv(np, np, dim)
      PV pv(np, np, dim);

      for(int j = 0; j < np; j++) {
        for(int i = 0; i < np; i++) {
          real v1 = v(i, j, 0, k, n0, ie);
          real v2 = v(i, j, 1, k, n0, ie);
          e(i, j) = 0.0;
          for(int h = 0; h < dim; h++) {
            ulatlon(i, j, h) =
                d(i, j, h, 0) * v1 + d(i, j, h, 1) * v2;
            e(i, j) += ulatlon(i, j, h) * ulatlon(i, j, h);
          }
          e(i, j) /= 2.0;
          e(i, j) += p(i, j, k, n0, ie) + ps(i, j, ie);
          for(int h = 0; h < dim; h++) {
            pv(i, j, h) = ulatlon(i, j, h) *
                          (pmean + p(i, j, k, n0, ie));
          }
        }
      }

      // TODO: Implement gradient_sphere
      real grade = gradient_sphere(ie, e, deriv, dinv);
    }
  }
  // Fortran Implementation of the first team parallel loop
  //
  // do ie=nets,nete
  //   fcor   => elem(ie)%fcor
  //   spheremp     => elem(ie)%spheremp
  //
  //   use control_mod, only :  topology, test_case
  //   use element_mod, only : element_t
  //   use dimensions_mod, only : nlev
  //   use shallow_water_mod, only : tc1_velocity,
  //   vortex_velocity, swirl_velocity
  //   implicit none
  //   type (element_t)     , intent(inout) :: elem
  //   ! local
  //   integer :: n0,k
  //   real (kind=real_kind) :: time
  //   if (topology == "cube" .and. test_case=="swtc1") then
  //     do k=1,nlev
  //       elem%state%v(:,:,:,k,n0)=tc1_velocity(elem%spherep,elem%Dinv)
  //     end do

  // IGNORE THIS!!!
  //   else if (topology == "cube" .and.
  //   test_case=="vortex") then
  //     do k=1,nlev
  //       elem%state%v(:,:,:,k,n0)=vortex_velocity(time,elem%spherep,elem%Dinv)
  //     end do

  //   else if (topology == "cube" .and. test_case=="swirl")
  //   then
  //     do k=1,nlev
  //       elem%state%v(:,:,:,k,n0)=swirl_velocity(time,elem%spherep,elem%Dinv)
  //     end do
  //   end if
  //
  //   do k=1,nlev
  //     ! ==============================================
  //     ! Compute kinetic energy term
  //     ! ==============================================
  //     !IKT, 10/21/16: we wrote part of this code in
  //     AdvanceRK.cpp
  //     do j=1,np
  //       do i=1,np
  //         v1     = elem(ie)%state%v(i,j,1,k,n0)   !
  //         contra
  //         v2     = elem(ie)%state%v(i,j,2,k,n0)   !
  //         contra
  //         ulatlon(i,j,1)=elem(ie)%D(i,j,1,1)*v1 +
  //         elem(ie)%D(i,j,1,2)*v2   ! contra->latlon
  //         ulatlon(i,j,2)=elem(ie)%D(i,j,2,1)*v1 +
  //         elem(ie)%D(i,j,2,2)*v2   ! contra->latlon
  //         E(i,j) = 0.5D0*(ulatlon(i,j,1)**2 +
  //         ulatlon(i,j,2)**2)  +&
  //         elem(ie)%state%p(i,j,k,n0) +
  //         elem(ie)%state%ps(i,j)
  //         pv(i,j,1) =
  //         ulatlon(i,j,1)*(pmean+elem(ie)%state%p(i,j,k,n0))
  //         pv(i,j,2) =
  //         ulatlon(i,j,2)*(pmean+elem(ie)%state%p(i,j,k,n0))
  //       end do
  //     end do
  //     grade = gradient_sphere(E,deriv,elem(ie)%Dinv)
  //     ! scalar -> latlon vector
  //     zeta = vorticity_sphere(ulatlon,deriv,elem(ie)) !
  //     latlon vector -> scalar
  //     if (tracer_advection_formulation==TRACERADV_UGRADQ)
  //     then
  //       gradh =
  //       gradient_sphere(elem(ie)%state%p(:,:,k,n0),deriv,elem(ie)%Dinv)
  //       div =
  //       ulatlon(:,:,1)*gradh(:,:,1)+ulatlon(:,:,2)*gradh(:,:,2)
  //     else
  //       div = divergence_sphere(pv,deriv,elem(ie))      !
  //       latlon vector -> scalar
  //     endif
  //     ! ==============================================
  //     ! Compute velocity tendency terms
  //     ! ==============================================
  //     ! accumulate all RHS terms
  //     vtens(:,:,1,k,ie)=vtens(:,:,1,k,ie) +
  //     (ulatlon(:,:,2)*(fcor(:,:) + zeta(:,:))  -
  //     grade(:,:,1))
  //     vtens(:,:,2,k,ie)=vtens(:,:,2,k,ie) +
  //     (-ulatlon(:,:,1)*(fcor(:,:) + zeta(:,:)) -
  //     grade(:,:,2))
  //     ptens(:,:,k,ie) = ptens(:,:,k,ie) - div(:,:)
  //     ! take the local element timestep
  //     vtens(:,:,:,k,ie)=ulatlon(:,:,:) +
  //     dtstage*vtens(:,:,:,k,ie)
  //     ptens(:,:,k,ie) = elem(ie)%state%p(:,:,k,n0) +
  //     dtstage*ptens(:,:,k,ie)
  //   end do!end of loop over levels
  //   if ((limiter_option == 8))then
  //     call
  //     limiter_optim_wrap(ptens(:,:,:,ie),elem(ie)%spheremp(:,:),&
  //                             pmin(:,ie),pmax(:,ie),kmass)
  //   endif
  //   if ((limiter_option == 81))then
  //     pmax(:,ie)=pmax(:,ie)+9e19  ! disable max
  //     constraint
  //     call
  //     limiter_optim_wrap(ptens(:,:,:,ie),elem(ie)%spheremp(:,:),&
  //                             pmin(:,ie),pmax(:,ie),kmass)
  //   endif
  //   if ((limiter_option == 84))then
  // 	   pmin(:,ie)=0.0d0
  //     if (test_case=='swirl') then
  //       pmin(1,ie)=.1d0
  //       if (nlev>=3) then
  //         k=3; pmin(k,ie)=.1d0
  //       endif
  //     endif
  //     pmax(:,ie)=pmax(:,ie)+9e19  ! disable max
  //     constraint
  //     call
  //     limiter_optim_wrap(ptens(:,:,:,ie),elem(ie)%spheremp(:,:),&
  //     pmin(:,ie),pmax(:,ie),kmass)
  //   endif
  //   if ( (limiter_option == 4) ) then
  //     call
  //     limiter2d_zero(ptens(:,:,:,ie),elem(ie)%spheremp,
  //     kmass)
  //   endif
  //   do k=1,nlev
  //     ptens(:,:,k,ie) =
  //     ptens(:,:,k,ie)*elem(ie)%spheremp(:,:)
  //     vtens(:,:,1,k,ie) =
  //     vtens(:,:,1,k,ie)*elem(ie)%spheremp(:,:)
  //     vtens(:,:,2,k,ie) =
  //     vtens(:,:,2,k,ie)*elem(ie)%spheremp(:,:)
  //   enddo
  //   ! ===================================================
  //   ! Pack cube edges of tendencies, rotate velocities
  //   ! ===================================================
  //   kptr=0
  //   !IKT, 10/21/16: packing needs to be pulled out
  //   (separate loop)
  //   call edgeVpack(edge3, ptens(1,1,1,ie),nlev,kptr,ie)
  //   kptr=nlev
  //   call
  //   edgeVpack(edge3,vtens(1,1,1,1,ie),2*nlev,kptr,ie)
  // end do
}

}  // Homme
