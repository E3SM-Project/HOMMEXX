
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!#define _DBG_ print *,"File:",__FILE__," at ",__LINE__
!#define _DBG_ !DBG
!
!
module prim_advance_hypervis_mod

  use control_mod,    only: qsplit,rsplit
  use derivative_mod, only: derivative_t, vorticity, divergence, gradient, gradient_wk
  use dimensions_mod, only: np, nlev, nlevp, nvar, nc, nelemd
  use edgetype_mod,   only: EdgeDescriptor_t, EdgeBuffer_t
  use element_mod,    only: element_t
  use hybrid_mod,     only: hybrid_t
  use hybvcoord_mod,  only: hvcoord_t
  use kinds,          only: real_kind, iulog
  use perf_mod,       only: t_startf, t_stopf, t_barrierf, t_adj_detailf ! _EXTERNAL
  use parallel_mod,   only: abortmp, parallel_t, iam
  use time_mod,       only: Timelevel_t
  use prim_advance_caar_mod, only: distribute_flux_at_corners

  implicit none
  private
  save

  public :: advance_hypervis_dp, advance_hypervis_lf

  contains


  subroutine advance_hypervis_dp(edge3,elem,hvcoord,hybrid,deriv,nt,nets,nete,dt2,eta_ave_w)
  !
  !  take one timestep of:
  !          u(:,:,:,np) = u(:,:,:,np) +  dt2*nu*laplacian**order ( u )
  !          T(:,:,:,np) = T(:,:,:,np) +  dt2*nu_s*laplacian**order ( T )
  !
  !
  !  For correct scaling, dt2 should be the same 'dt2' used in the leapfrog advace
  !
  !
  use dimensions_mod, only : np, np, nlev, nc, ntrac, max_corner_elem
  use control_mod, only : nu, nu_div, nu_s, hypervis_order, hypervis_subcycle, nu_p, nu_top, psurf_vis, swest
  use hybvcoord_mod, only : hvcoord_t
  use element_mod, only : element_t
  use derivative_mod, only : derivative_t, laplace_sphere_wk, vlaplace_sphere_wk
  use derivative_mod, only : subcell_Laplace_fluxes, subcell_dss_fluxes
  use edge_mod, only : edgevpack, edgevunpack, edgeDGVunpack
  use edgetype_mod, only : EdgeBuffer_t, EdgeDescriptor_t
  use bndry_mod, only : bndry_exchangev
  use viscosity_mod, only : biharmonic_wk_dp3d
  use physical_constants, only: Cp
  use derivative_mod, only : subcell_Laplace_fluxes
  implicit none

  type (hybrid_t)      , intent(in) :: hybrid
  type (element_t)     , intent(inout), target :: elem(:)
  type (EdgeBuffer_t)  , intent(inout) :: edge3
  type (derivative_t)  , intent(in) :: deriv
  type (hvcoord_t), intent(in)      :: hvcoord

  real (kind=real_kind) :: dt2
  integer :: nets,nete

  ! local
  real (kind=real_kind) :: eta_ave_w  ! weighting for mean flux terms
  real (kind=real_kind) :: dpdn0, nu_scale_top
  integer :: k,kptr,i,j,ie,ic,nt
  real (kind=real_kind), dimension(np,np)      :: dpdn
  real (kind=real_kind), dimension(np,np,2,nlev,nets:nete)      :: vtens
  real (kind=real_kind), dimension(np,np,nlev,nets:nete)        :: ttens
  real (kind=real_kind), dimension(np,np,nlev,nets:nete)        :: dptens
  real (kind=real_kind), dimension(0:np+1,0:np+1,nlev)          :: corners
  real (kind=real_kind), dimension(2,2,2)                       :: cflux
  real (kind=real_kind), dimension(nc,nc,4,nlev,nets:nete)      :: dpflux
  real (kind=real_kind), dimension(np,np,nlev) :: p
  type (EdgeDescriptor_t)                                       :: desc


! NOTE: PGI compiler bug: when using spheremp, rspheremp and ps as pointers to elem(ie)% members,
  !       data is incorrect (offset by a few numbers actually)
  !       removed for now.
  !       real (kind=real_kind), dimension(:,:), pointer :: spheremp,rspheremp
  !       real (kind=real_kind), dimension(:,:,:), pointer   :: ps

  real (kind=real_kind), dimension(np,np) :: lap_t,lap_dp
  real (kind=real_kind), dimension(np,np,2) :: lap_v
  real (kind=real_kind) :: v1,v2,dt,heating

  real (kind=real_kind)                     :: temp      (np,np,nlev)
  real (kind=real_kind)                     :: laplace_fluxes(nc,nc,4)



  if (nu_s == 0 .and. nu == 0 .and. nu_p==0 ) return;
!JMD  call t_barrierf('sync_advance_hypervis', hybrid%par%comm)
!pw   call t_adj_detailf(+1)
  call t_startf('advance_hypervis_dp')


  dt=dt2/hypervis_subcycle
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  regular viscosity
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if (hypervis_order == 1) then
     if (nu_p>0) call abortmp( 'ERROR: hypervis_order == 1 not coded for nu_p>0')
     do ic=1,hypervis_subcycle
        do ie=nets,nete

#if (defined COLUMN_OPENMP)
! Not sure about deriv here
!$omp parallel do private(k,lap_t,lap_v,i,j)
#endif
           do k=1,nlev
              lap_t=laplace_sphere_wk(elem(ie)%state%T(:,:,k,nt),deriv,elem(ie),var_coef=.false.)
              lap_v=vlaplace_sphere_wk(elem(ie)%state%v(:,:,:,k,nt),deriv,elem(ie),var_coef=.false.)
              ! advace in time.  (note: DSS commutes with time stepping, so we
              ! can time advance and then DSS.  this has the advantage of
              ! not letting any discontinuties accumulate in p,v via roundoff
              do j=1,np
                 do i=1,np
                    elem(ie)%state%T(i,j,k,nt)=elem(ie)%state%T(i,j,k,nt)*elem(ie)%spheremp(i,j)  +  dt*nu_s*lap_t(i,j)
                    elem(ie)%state%v(i,j,1,k,nt)=elem(ie)%state%v(i,j,1,k,nt)*elem(ie)%spheremp(i,j) + dt*nu*lap_v(i,j,1)
                    elem(ie)%state%v(i,j,2,k,nt)=elem(ie)%state%v(i,j,2,k,nt)*elem(ie)%spheremp(i,j) + dt*nu*lap_v(i,j,2)
                 enddo
              enddo
           enddo

           kptr=0
           call edgeVpack(edge3, elem(ie)%state%T(:,:,:,nt),nlev,kptr,ie)
           kptr=nlev
           call edgeVpack(edge3,elem(ie)%state%v(:,:,:,:,nt),2*nlev,kptr,ie)
        enddo

        call t_startf('ahdp_bexchV1')
        call bndry_exchangeV(hybrid,edge3)
        call t_stopf('ahdp_bexchV1')

        do ie=nets,nete

           kptr=0
           call edgeVunpack(edge3, elem(ie)%state%T(:,:,:,nt), nlev, kptr, ie)
           kptr=nlev
           call edgeVunpack(edge3, elem(ie)%state%v(:,:,:,:,nt), 2*nlev, kptr, ie)

           ! apply inverse mass matrix
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j)
#endif
           do k=1,nlev
              do j=1,np
                 do i=1,np
                    elem(ie)%state%T(i,j,k,nt)=elem(ie)%rspheremp(i,j)*elem(ie)%state%T(i,j,k,nt)
                    elem(ie)%state%v(i,j,1,k,nt)=elem(ie)%rspheremp(i,j)*elem(ie)%state%v(i,j,1,k,nt)
                    elem(ie)%state%v(i,j,2,k,nt)=elem(ie)%rspheremp(i,j)*elem(ie)%state%v(i,j,2,k,nt)
                 enddo
              enddo
           enddo
        enddo
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif
     enddo  ! subcycle
  endif


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  hyper viscosity
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! nu_p=0:
!   scale T dissipaton by dp  (conserve IE, dissipate T^2)
! nu_p>0
!   dont scale:  T equation IE dissipation matches (to truncation error)
!                IE dissipation from continuity equation
!                (1 deg: to about 0.1 W/m^2)
!



  if (hypervis_order == 2) then
     do ic=1,hypervis_subcycle
        call biharmonic_wk_dp3d(elem,dptens,dpflux,ttens,vtens,deriv,edge3,hybrid,nt,nets,nete)
        do ie=nets,nete

           ! comptue mean flux
           if (nu_p>0) then
              elem(ie)%derived%dpdiss_ave(:,:,:)=elem(ie)%derived%dpdiss_ave(:,:,:)+&
                   eta_ave_w*elem(ie)%state%dp3d(:,:,:,nt)/hypervis_subcycle
              elem(ie)%derived%dpdiss_biharmonic(:,:,:)=elem(ie)%derived%dpdiss_biharmonic(:,:,:)+&
                   eta_ave_w*dptens(:,:,:,ie)/hypervis_subcycle
           endif
#if (defined COLUMN_OPENMP)
!XXX parallel do private(k,i,j,lap_t,lap_dp,lap_v,nu_scale_top,utens_tmp,vtens_tmp,ttens_tmp,dptens_tmp,laplace_fluxes)
!$omp parallel do private(k,lap_t,lap_dp,lap_v,nu_scale_top,laplace_fluxes)
#endif
           do k=1,nlev
              ! advace in time.
              ! note: DSS commutes with time stepping, so we can time advance and then DSS.
              ! note: weak operators alreayd have mass matrix "included"

              ! add regular diffusion in top 3 layers:
              if (nu_top>0 .and. k<=3) then
                 lap_t=laplace_sphere_wk(elem(ie)%state%T(:,:,k,nt),deriv,elem(ie),var_coef=.false.)
                 lap_dp=laplace_sphere_wk(elem(ie)%state%dp3d(:,:,k,nt),deriv,elem(ie),var_coef=.false.)
                 lap_v=vlaplace_sphere_wk(elem(ie)%state%v(:,:,:,k,nt),deriv,elem(ie),var_coef=.false.)
              endif
              nu_scale_top = 1
              if (k==1) nu_scale_top=4
              if (k==2) nu_scale_top=2


              ! biharmonic terms need a negative sign:
              if (nu_top>0 .and. k<=3) then
                 vtens(:,:,:,k,ie)=(-nu*vtens(:,:,:,k,ie) + nu_scale_top*nu_top*lap_v(:,:,:))
                 ttens(:,:,k,ie)  =(-nu_s*ttens(:,:,k,ie) + nu_scale_top*nu_top*lap_t(:,:) )
                 dptens(:,:,k,ie) =(-nu_p*dptens(:,:,k,ie) + nu_scale_top*nu_top*lap_dp(:,:) )
              else
                 vtens(:,:,:,k,ie)=-nu*vtens(:,:,:,k,ie) 
                 ttens(:,:,k,ie)  =-nu_s*ttens(:,:,k,ie) 
                 dptens(:,:,k,ie) =-nu_p*dptens(:,:,k,ie) 
              endif

              if (nu_p==0) then
                 ! nu_p==0 is only for certain regression tests, so perfromance is not an issue
                 ! normalize so as to conserve IE
                 ! scale by 1/rho (normalized to be O(1))
                 ! dp/dn = O(ps0)*O(delta_eta) = O(ps0)/O(nlev)
                 dpdn(:,:) = ( hvcoord%hyai(k+1) - hvcoord%hyai(k))*hvcoord%ps0 + &
                      ( hvcoord%hybi(k+1) - hvcoord%hybi(k))*elem(ie)%state%ps_v(:,:,nt)
                 dpdn0 = ( hvcoord%hyai(k+1) - hvcoord%hyai(k) )*hvcoord%ps0 + &
                      ( hvcoord%hybi(k+1) - hvcoord%hybi(k) )*hvcoord%ps0
                 ttens(:,:,k,ie) = ttens(:,:,k,ie) * dpdn0/dpdn(:,:)
                 dptens(:,:,k,ie) = 0
              endif            

              if (0<ntrac) then
                elem(ie)%sub_elem_mass_flux(:,:,:,k) = elem(ie)%sub_elem_mass_flux(:,:,:,k) - &
                                              eta_ave_w*nu_p*dpflux(:,:,:,k,ie)/hypervis_subcycle
                if (nu_top>0 .and. k<=3) then
                  laplace_fluxes=subcell_Laplace_fluxes(elem(ie)%state%dp3d(:,:,k,nt),deriv,elem(ie),np,nc)
                  elem(ie)%sub_elem_mass_flux(:,:,:,k) = elem(ie)%sub_elem_mass_flux(:,:,:,k) + &
                                           eta_ave_w*nu_scale_top*nu_top*laplace_fluxes/hypervis_subcycle
                endif
              endif

              ! NOTE: we will DSS all tendicies, EXCEPT for dp3d, where we DSS the new state
              elem(ie)%state%dp3d(:,:,k,nt) = elem(ie)%state%dp3d(:,:,k,nt)*elem(ie)%spheremp(:,:)&
                   + dt*dptens(:,:,k,ie)

           enddo


           kptr=0
           call edgeVpack(edge3, ttens(:,:,:,ie),nlev,kptr,ie)
           kptr=nlev
           call edgeVpack(edge3,vtens(:,:,:,:,ie),2*nlev,kptr,ie)
           kptr=3*nlev
           call edgeVpack(edge3,elem(ie)%state%dp3d(:,:,:,nt),nlev,kptr,ie)
        enddo

        call t_startf('ahdp_bexchV2')
        call bndry_exchangeV(hybrid,edge3)
        call t_stopf('ahdp_bexchV2')

        do ie=nets,nete

           kptr=0
           call edgeVunpack(edge3, ttens(:,:,:,ie), nlev, kptr, ie)
           kptr=nlev
           call edgeVunpack(edge3, vtens(:,:,:,:,ie), 2*nlev, kptr, ie)
           kptr=3*nlev
           if (0<ntrac) then
             do k=1,nlev
               temp(:,:,k) = elem(ie)%state%dp3d(:,:,k,nt) / elem(ie)%spheremp  ! STATE before DSS
             enddo
             corners = 0.0d0
             corners(1:np,1:np,:) = elem(ie)%state%dp3d(:,:,:,nt) ! fill in interior data of STATE*mass
           endif
           call edgeVunpack(edge3, elem(ie)%state%dp3d(:,:,:,nt), nlev, kptr, ie)



           if (0<ntrac) then
             kptr=3*nlev
             desc = elem(ie)%desc

             call edgeDGVunpack(edge3, corners, nlev, kptr, ie)
             corners = corners/dt

             do k=1,nlev
               temp(:,:,k) =  elem(ie)%rspheremp(:,:)*elem(ie)%state%dp3d(:,:,k,nt) - temp(:,:,k)
               temp(:,:,k) =  temp(:,:,k)/dt

               call distribute_flux_at_corners(cflux, corners(:,:,k), desc%getmapP)

               cflux(1,1,:)   = elem(ie)%rspheremp(1,  1) * cflux(1,1,:)
               cflux(2,1,:)   = elem(ie)%rspheremp(np, 1) * cflux(2,1,:)
               cflux(1,2,:)   = elem(ie)%rspheremp(1, np) * cflux(1,2,:)
               cflux(2,2,:)   = elem(ie)%rspheremp(np,np) * cflux(2,2,:)

               elem(ie)%sub_elem_mass_flux(:,:,:,k) = elem(ie)%sub_elem_mass_flux(:,:,:,k) + &
                 eta_ave_w*subcell_dss_fluxes(temp(:,:,k), np, nc, elem(ie)%metdet,cflux)/hypervis_subcycle
             end do
           endif



           ! apply inverse mass matrix, accumulate tendencies
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k)
#endif
           do k=1,nlev
              vtens(:,:,1,k,ie)=dt*vtens(:,:,1,k,ie)*elem(ie)%rspheremp(:,:)
              vtens(:,:,2,k,ie)=dt*vtens(:,:,2,k,ie)*elem(ie)%rspheremp(:,:)
              ttens(:,:,k,ie)=dt*ttens(:,:,k,ie)*elem(ie)%rspheremp(:,:)

              elem(ie)%state%dp3d(:,:,k,nt)=elem(ie)%state%dp3d(:,:,k,nt)*elem(ie)%rspheremp(:,:)
           enddo

           ! apply hypervis to u -> u+utens:
           ! E0 = dpdn * .5*u dot u + dpdn * T  + dpdn*PHIS
           ! E1 = dpdn * .5*(u+utens) dot (u+utens) + dpdn * (T-X) + dpdn*PHIS
           ! E1-E0:   dpdn (u dot utens) + dpdn .5 utens dot utens   - dpdn X
           !      X = (u dot utens) + .5 utens dot utens
           !  alt:  (u+utens) dot utens
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,v1,v2,heating)
#endif
           do k=1,nlev
              do j=1,np
                 do i=1,np
                    ! update v first (gives better results than updating v after heating)
                    elem(ie)%state%v(i,j,:,k,nt)=elem(ie)%state%v(i,j,:,k,nt) + &
                         vtens(i,j,:,k,ie)

                    v1=elem(ie)%state%v(i,j,1,k,nt)
                    v2=elem(ie)%state%v(i,j,2,k,nt)
                    heating = (vtens(i,j,1,k,ie)*v1  + vtens(i,j,2,k,ie)*v2 )
                    elem(ie)%state%T(i,j,k,nt)=elem(ie)%state%T(i,j,k,nt) &
                         +ttens(i,j,k,ie)-heating/cp
                    !elem(ie)%state%dp3d(i,j,k,nt)=elem(ie)%state%dp3d(i,j,k,nt) + &
                    !     dptens(i,j,k,ie)
                 enddo
              enddo
           enddo
        enddo
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif
     enddo
  endif

  call t_stopf('advance_hypervis_dp')
!pw  call t_adj_detailf(-1)

  end subroutine advance_hypervis_dp

  subroutine advance_hypervis_lf(edge3,elem,hvcoord,hybrid,deriv,nm1,n0,nt,nets,nete,dt2)
  !
  !  take one timestep of:
  !          u(:,:,:,np) = u(:,:,:,np) +  dt2*nu*laplacian**order ( u )
  !          T(:,:,:,np) = T(:,:,:,np) +  dt2*nu_s*laplacian**order ( T )
  !
  !
  !  For correct scaling, dt2 should be the same 'dt2' used in the leapfrog advace
  !
  !
  use dimensions_mod, only : np, np, nlev
  use control_mod, only : nu, nu_div, nu_s, hypervis_order, hypervis_subcycle, nu_p, nu_top, psurf_vis
  use hybvcoord_mod, only : hvcoord_t
  use element_mod, only : element_t
  use derivative_mod, only : derivative_t, laplace_sphere_wk, vlaplace_sphere_wk
  use edge_mod, only : edgevpack, edgevunpack
  use edgetype_mod, only : EdgeBuffer_t
  use bndry_mod, only : bndry_exchangev
  use viscosity_mod, only : biharmonic_wk
  use physical_constants, only: Cp
  implicit none

  type (hybrid_t)      , intent(in) :: hybrid
  type (element_t)     , intent(inout), target :: elem(:)
  type (EdgeBuffer_t)  , intent(inout) :: edge3
  type (derivative_t)  , intent(in) :: deriv
  type (hvcoord_t), intent(in)      :: hvcoord

  real (kind=real_kind) :: dt2
  integer :: nets,nete

  ! local
  real (kind=real_kind) :: nu_scale, dpdn,dpdn0, nu_scale_top
  integer :: k,kptr,i,j,ie,ic,n0,nt,nm1
  real (kind=real_kind), dimension(np,np,2,nlev,nets:nete)      :: vtens
  real (kind=real_kind), dimension(np,np,nlev,nets:nete)        :: ptens
  real (kind=real_kind), dimension(np,np,nets:nete) :: pstens
  real (kind=real_kind), dimension(np,np,nlev) :: p
  real (kind=real_kind), dimension(np,np) :: dXdp


! NOTE: PGI compiler bug: when using spheremp, rspheremp and ps as pointers to elem(ie)% members,
  !       data is incorrect (offset by a few numbers actually)
  !       removed for now.
  !       real (kind=real_kind), dimension(:,:), pointer :: spheremp,rspheremp
  !       real (kind=real_kind), dimension(:,:,:), pointer   :: ps

  real (kind=real_kind), dimension(np,np) :: lap_p
  real (kind=real_kind), dimension(np,np,2) :: lap_v
  real (kind=real_kind) :: v1,v2,dt,heating,utens_tmp,vtens_tmp,ptens_tmp


  if (nu_s == 0 .and. nu == 0 .and. nu_p==0 ) return;
!JMD  call t_barrierf('sync_advance_hypervis_lf', hybrid%par%comm)
!pw   call t_adj_detailf(+1)
  call t_startf('advance_hypervis_lf')

! for non-leapfrog,nt=n0=nmt
!
!  nm1 = tl%nm1   ! heating term uses U,V at average of nt and nm1 levels
!  n0 = tl%n0     ! timelevel used for ps scaling.  use n0 for leapfrog.
!  nt = tl%np1    ! apply viscosity to this timelevel  (np1)


  dt=dt2/hypervis_subcycle
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  regular viscosity
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if (hypervis_order == 1) then
     if (nu_p>0) stop 'ERROR: hypervis_order == 1 not coded for nu_p>0'
     do ic=1,hypervis_subcycle
        do ie=nets,nete

#if (defined COLUMN_OPENMP)
! Not sure about deriv here
!$omp parallel do private(k,lap_p,lap_v,i,j)
#endif
           do k=1,nlev
              lap_p=laplace_sphere_wk(elem(ie)%state%T(:,:,k,nt),deriv,elem(ie),var_coef=.false.)
              lap_v=vlaplace_sphere_wk(elem(ie)%state%v(:,:,:,k,nt),deriv,elem(ie),var_coef=.false.)
              ! advace in time.  (note: DSS commutes with time stepping, so we
              ! can time advance and then DSS.  this has the advantage of
              ! not letting any discontinuties accumulate in p,v via roundoff
              do j=1,np
                 do i=1,np
                    elem(ie)%state%T(i,j,k,nt)=elem(ie)%state%T(i,j,k,nt)*elem(ie)%spheremp(i,j)  +  dt*nu_s*lap_p(i,j)
                    elem(ie)%state%v(i,j,1,k,nt)=elem(ie)%state%v(i,j,1,k,nt)*elem(ie)%spheremp(i,j) + dt*nu*lap_v(i,j,1)
                    elem(ie)%state%v(i,j,2,k,nt)=elem(ie)%state%v(i,j,2,k,nt)*elem(ie)%spheremp(i,j) + dt*nu*lap_v(i,j,2)
                 enddo
              enddo
           enddo

           kptr=0
           call edgeVpack(edge3, elem(ie)%state%T(:,:,:,nt),nlev,kptr,ie)
           kptr=nlev
           call edgeVpack(edge3,elem(ie)%state%v(:,:,:,:,nt),2*nlev,kptr,ie)
        enddo

        call t_startf('ahlf_bexchV1')
        call bndry_exchangeV(hybrid,edge3)
        call t_stopf('ahlf_bexchV1')

        do ie=nets,nete

           kptr=0
           call edgeVunpack(edge3, elem(ie)%state%T(:,:,:,nt), nlev, kptr, ie)
           kptr=nlev
           call edgeVunpack(edge3, elem(ie)%state%v(:,:,:,:,nt), 2*nlev, kptr, ie)

           ! apply inverse mass matrix
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j)
#endif
           do k=1,nlev
              do j=1,np
                 do i=1,np
                    elem(ie)%state%T(i,j,k,nt)=elem(ie)%rspheremp(i,j)*elem(ie)%state%T(i,j,k,nt)
                    elem(ie)%state%v(i,j,1,k,nt)=elem(ie)%rspheremp(i,j)*elem(ie)%state%v(i,j,1,k,nt)
                    elem(ie)%state%v(i,j,2,k,nt)=elem(ie)%rspheremp(i,j)*elem(ie)%state%v(i,j,2,k,nt)
                 enddo
              enddo
           enddo
        enddo
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif
     enddo  ! subcycle
  endif


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  hyper viscosity
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if (hypervis_order == 2) then
     do ic=1,hypervis_subcycle
        call biharmonic_wk(elem,pstens,ptens,vtens,deriv,edge3,hybrid,nt,nets,nete)
        do ie=nets,nete

           nu_scale=1
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,lap_p,lap_v,nu_scale_top,dpdn,dpdn0,nu_scale,utens_tmp,vtens_tmp,ptens_tmp)
#endif
           do k=1,nlev
              ! advace in time.
              ! note: DSS commutes with time stepping, so we can time advance and then DSS.
              ! note: weak operators alreayd have mass matrix "included"

              ! add regular diffusion in top 3 layers:
              if (nu_top>0 .and. k<=3) then
                 lap_p=laplace_sphere_wk(elem(ie)%state%T(:,:,k,nt),deriv,elem(ie),var_coef=.false.)
                 lap_v=vlaplace_sphere_wk(elem(ie)%state%v(:,:,:,k,nt),deriv,elem(ie),var_coef=.false.)
              endif
              nu_scale_top = 1
              if (k==1) nu_scale_top=4
              if (k==2) nu_scale_top=2

              do j=1,np
                 do i=1,np
                    if (psurf_vis==0) then
                       ! normalize so as to conserve IE  (not needed when using p-surface viscosity)
                       ! scale velosity by 1/rho (normalized to be O(1))
                       ! dp/dn = O(ps0)*O(delta_eta) = O(ps0)/O(nlev)
                       dpdn = ( hvcoord%hyai(k+1) - hvcoord%hyai(k) )*hvcoord%ps0 + &
                            ( hvcoord%hybi(k+1) - hvcoord%hybi(k) )*elem(ie)%state%ps_v(i,j,n0)  ! nt ?
                       dpdn0 = ( hvcoord%hyai(k+1) - hvcoord%hyai(k) )*hvcoord%ps0 + &
                            ( hvcoord%hybi(k+1) - hvcoord%hybi(k) )*hvcoord%ps0
                       nu_scale = dpdn0/dpdn
                    endif

                    ! biharmonic terms need a negative sign:
                    if (nu_top>0 .and. k<=3) then
                       utens_tmp=(-nu*vtens(i,j,1,k,ie) + nu_scale_top*nu_top*lap_v(i,j,1))
                       vtens_tmp=(-nu*vtens(i,j,2,k,ie) + nu_scale_top*nu_top*lap_v(i,j,2))
                       ptens_tmp=nu_scale*(-nu_s*ptens(i,j,k,ie) + nu_scale_top*nu_top*lap_p(i,j) )
                    else
                       utens_tmp=-nu*vtens(i,j,1,k,ie)
                       vtens_tmp=-nu*vtens(i,j,2,k,ie)
                       ptens_tmp=-nu_scale*nu_s*ptens(i,j,k,ie)
                    endif

                    ptens(i,j,k,ie) = ptens_tmp
                    vtens(i,j,1,k,ie)=utens_tmp
                    vtens(i,j,2,k,ie)=vtens_tmp
                 enddo
              enddo
           enddo

           pstens(:,:,ie)  =  -nu_p*pstens(:,:,ie)
           kptr=0
           call edgeVpack(edge3, ptens(:,:,:,ie),nlev,kptr,ie)
           kptr=nlev
           call edgeVpack(edge3,vtens(:,:,:,:,ie),2*nlev,kptr,ie)
           kptr=3*nlev
           call edgeVpack(edge3,pstens(:,:,ie),1,kptr,ie)
        enddo

        call t_startf('ahlf_bexchV2')
        call bndry_exchangeV(hybrid,edge3)
        call t_stopf('ahlf_bexchV2')

        do ie=nets,nete

           kptr=0
           call edgeVunpack(edge3, ptens(:,:,:,ie), nlev, kptr, ie)
           kptr=nlev
           call edgeVunpack(edge3, vtens(:,:,:,:,ie), 2*nlev, kptr, ie)
           kptr=3*nlev
           call edgeVunpack(edge3, pstens(:,:,ie), 1, kptr, ie)

           if (psurf_vis == 1 ) then
              ! apply p-surface correction
              do k=1,nlev
                 p(:,:,k)   = hvcoord%hyam(k)*hvcoord%ps0 + hvcoord%hybm(k)*elem(ie)%state%ps_v(:,:,nt)
              enddo
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,dXdp)
#endif
              do k=1,nlev
                 if (k.eq.1) then
                    ! no correction needed
                 else if (k.eq.nlev) then
                    ! one-sided difference
                    dXdp = (elem(ie)%state%T(:,:,k,nt) - elem(ie)%state%T(:,:,k-1,nt)) / &
                        (p(:,:,k)-p(:,:,k-1))
                    ptens(:,:,k,ie) = ptens(:,:,k,ie) - dXdp(:,:)*hvcoord%hybm(k)*pstens(:,:,ie)
                 else
                    dXdp = (elem(ie)%state%T(:,:,k+1,nt) - elem(ie)%state%T(:,:,k-1,nt)) / &
                         (p(:,:,k+1)-p(:,:,k-1))
                    ptens(:,:,k,ie) = ptens(:,:,k,ie) - dXdp(:,:)*hvcoord%hybm(k)*pstens(:,:,ie)
                 endif
              enddo
           endif


           ! apply inverse mass matrix, accumulate tendencies
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,v1,v2,heating)
#endif
           do k=1,nlev
              do j=1,np
                 do i=1,np

                    elem(ie)%state%v(i,j,1,k,nt)=elem(ie)%state%v(i,j,1,k,nt) + &
                         dt*elem(ie)%rspheremp(i,j)*vtens(i,j,1,k,ie)
                    elem(ie)%state%v(i,j,2,k,nt)=elem(ie)%state%v(i,j,2,k,nt) +  &
                         dt*elem(ie)%rspheremp(i,j)*vtens(i,j,2,k,ie)

                    ! better E conservation if we use v after adding in vtens:
                    v1=.5*(elem(ie)%state%v(i,j,1,k,nt)+elem(ie)%state%v(i,j,1,k,nm1))
                    v2=.5*(elem(ie)%state%v(i,j,2,k,nt)+elem(ie)%state%v(i,j,2,k,nm1))
                    heating = (vtens(i,j,1,k,ie)*v1  + vtens(i,j,2,k,ie)*v2 )

                    elem(ie)%state%T(i,j,k,nt)=elem(ie)%state%T(i,j,k,nt)     + &
                         dt*elem(ie)%rspheremp(i,j)*(cp*ptens(i,j,k,ie) - heating)/cp

                 enddo
              enddo
           enddo
           elem(ie)%state%ps_v(:,:,nt)=elem(ie)%state%ps_v(:,:,nt) + dt*elem(ie)%rspheremp(:,:)*pstens(:,:,ie)
        enddo
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif
     enddo
  endif

  call t_stopf('advance_hypervis_lf')
!pw  call t_adj_detailf(-1)

  end subroutine advance_hypervis_lf


end module prim_advance_hypervis_mod
