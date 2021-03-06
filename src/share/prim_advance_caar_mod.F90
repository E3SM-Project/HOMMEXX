
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!#define _DBG_ print *,"File:",__FILE__," at ",__LINE__
!#define _DBG_ !DBG
!
!
module prim_advance_caar_mod
  use edgetype_mod,   only: EdgeBuffer_t
  use control_mod,    only: qsplit,rsplit
  use derivative_mod, only: derivative_t, vorticity, divergence, gradient, gradient_wk
  use dimensions_mod, only: np, nlev, nlevp, nvar, nc, nelemd
  use element_mod,    only: element_t
  use perf_mod,       only: t_startf, t_stopf
  use prim_smooth_mod, only: edge3p1
  implicit none
  private
  save

  public :: distribute_flux_at_corners, compute_and_apply_rhs

contains

  !
  ! phl notes: output is stored in first argument. Advances from 2nd argument using tendencies evaluated at 3rd rgument:
  ! phl: for offline winds use time at 3rd argument (same as rhs currently)
  !
  subroutine compute_and_apply_rhs(np1,nm1,n0,qn0,dt2,elem,hvcoord,hybrid,&
                                   deriv,nets,nete,compute_diagnostics,eta_ave_w)
  ! ===================================
  ! compute the RHS, accumulate into u(np1) and apply DSS
  !
  !           u(np1) = u(nm1) + dt2*DSS[ RHS(u(n0)) ]
  !
  ! This subroutine is normally called to compute a leapfrog timestep
  ! but by adjusting np1,nm1,n0 and dt2, many other timesteps can be
  ! accomodated.  For example, setting nm1=np1=n0 this routine will
  ! take a forward euler step, overwriting the input with the output.
  !
  !    qn0 = timelevel used to access Qdp() in order to compute virtual Temperature
  !          qn0=-1 for the dry case
  !
  ! if  dt2<0, then the DSS'd RHS is returned in timelevel np1
  !
  ! Combining the RHS and DSS pack operation in one routine
  ! allows us to fuse these two loops for more cache reuse
  !
  ! Combining the dt advance and DSS unpack operation in one routine
  ! allows us to fuse these two loops for more cache reuse
  !
  ! note: for prescribed velocity case, velocity will be computed at
  ! "real_time", which should be the time of timelevel n0.
  !
  !
  ! ===================================

    use kinds,          only : real_kind
    use bndry_mod,      only : bndry_exchangev
    use derivative_mod, only : derivative_t, subcell_dss_fluxes
    use dimensions_mod, only : nlev
    use edge_mod,       only : edgevpack, edgevunpack, edgedgvunpack
    use edgetype_mod,   only : edgedescriptor_t
    use element_mod,    only : element_t, elem_state_temp, elem_state_v, elem_state_dp3d
    use hybvcoord_mod,  only : hvcoord_t
    use hybrid_mod,     only : hybrid_t
    use caar_pre_exchange_driver_mod, only: caar_pre_exchange_monolithic_f90

    implicit none

    type (hvcoord_t)      , intent(in)    :: hvcoord
    type (element_t)      , intent(inout) :: elem(:)
    type (hybrid_t)       , intent(in)    :: hybrid
    type (derivative_t)   , intent(in)    :: deriv
    integer               , intent(in)    :: np1, nm1, n0, qn0, nets, nete
    real (kind=real_kind) , intent(in)    :: dt2
    logical               , intent(in)    :: compute_diagnostics
    real (kind=real_kind) , intent(in)    :: eta_ave_w  ! weighting for eta_dot_dpdn mean flux

    type (EdgeDescriptor_t)                                 :: desc
    real (kind=real_kind)   , dimension(np,np,nlev)         :: stashdp3d
    real (kind=real_kind)   , dimension(0:np+1,0:np+1,nlev) :: corners
    real (kind=real_kind)   , dimension(2,2,2)              :: cflux
    real (kind=real_kind)   , dimension(nc,nc,4)            :: tempflux
    real (kind=real_kind)   , dimension(np,np)              :: tempdp3d
    integer                                                 :: ie, k, kptr

!JMD  call t_barrierf('sync_compute_and_apply_rhs', hybrid%par%comm)

!pw call t_adj_detailf(+1)
    call t_startf('caar compute')

    call caar_pre_exchange_monolithic_f90 (nm1,n0,np1,qn0,dt2,elem,hvcoord,hybrid,&
                                           deriv,nets,nete,compute_diagnostics,eta_ave_w)
    call t_stopf('caar compute')

    call t_startf('caar_bexchV')
    do ie=nets,nete
       ! =========================================================
       !
       ! Pack T, v and dp3d tendencies into comm buffer
       !
       ! =========================================================
       kptr=0
       call edgeVpack(edge3p1, elem(ie)%state%T(:,:,:,np1),nlev,kptr,ie)

       kptr=kptr+nlev
       call edgeVpack(edge3p1, elem(ie)%state%v(:,:,:,:,np1),2*nlev,kptr,ie)

       kptr=kptr+2*nlev
       call edgeVpack(edge3p1, elem(ie)%state%dp3d(:,:,:,np1),nlev,kptr, ie)
    end do

    ! =============================================================
    ! Insert communications here: for shared memory, just a single
    ! sync is required
    ! =============================================================

    call bndry_exchangeV(hybrid,edge3p1)

    do ie=nets,nete
       ! ===========================================================
       ! Unpack the edges for vgrad_T and v tendencies...
       ! ===========================================================
       kptr=0
       call edgeVunpack(edge3p1, elem(ie)%state%T(:,:,:,np1), nlev, kptr, ie)

       kptr=kptr+nlev
       call edgeVunpack(edge3p1, elem(ie)%state%v(:,:,:,:,np1), 2*nlev, kptr, ie)

       kptr=kptr+2*nlev
       call edgeVunpack(edge3p1, elem(ie)%state%dp3d(:,:,:,np1),nlev,kptr,ie)

       ! ====================================================
       ! Scale tendencies by inverse mass matrix
       ! ====================================================

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k)
#endif
       do k=1,nlev
          elem(ie)%state%T(:,:,k,np1)   = elem(ie)%rspheremp(:,:)*elem(ie)%state%T(:,:,k,np1)
          elem(ie)%state%v(:,:,1,k,np1) = elem(ie)%rspheremp(:,:)*elem(ie)%state%v(:,:,1,k,np1)
          elem(ie)%state%v(:,:,2,k,np1) = elem(ie)%rspheremp(:,:)*elem(ie)%state%v(:,:,2,k,np1)
          elem(ie)%state%dp3d(:,:,k,np1)= elem(ie)%rspheremp(:,:)*elem(ie)%state%dp3d(:,:,k,np1)
       enddo
    end do
    call t_stopf('caar_bexchV')

#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif
!pw  call t_adj_detailf(-1)

  end subroutine compute_and_apply_rhs

  subroutine distribute_flux_at_corners(cflux, corners, getmapP)
    use kinds,          only : int_kind, real_kind
    use dimensions_mod, only : np, max_corner_elem
    use control_mod,    only : swest
    implicit none

    real   (kind=real_kind), intent(out)  :: cflux(2,2,2)
    real   (kind=real_kind), intent(in)   :: corners(0:np+1,0:np+1)
    integer(kind=int_kind),  intent(in)   :: getmapP(:)

    cflux = 0.0d0
    if (getmapP(swest+0*max_corner_elem) /= -1) then
      cflux(1,1,1) =                (corners(0,1) - corners(1,1))
      cflux(1,1,1) = cflux(1,1,1) + (corners(0,0) - corners(1,1)) / 2.0d0
      cflux(1,1,1) = cflux(1,1,1) + (corners(0,1) - corners(1,0)) / 2.0d0

      cflux(1,1,2) =                (corners(1,0) - corners(1,1))
      cflux(1,1,2) = cflux(1,1,2) + (corners(0,0) - corners(1,1)) / 2.0d0
      cflux(1,1,2) = cflux(1,1,2) + (corners(1,0) - corners(0,1)) / 2.0d0
    else
      cflux(1,1,1) =                (corners(0,1) - corners(1,1))
      cflux(1,1,2) =                (corners(1,0) - corners(1,1))
    endif

    if (getmapP(swest+1*max_corner_elem) /= -1) then
      cflux(2,1,1) =                (corners(np+1,1) - corners(np,1))
      cflux(2,1,1) = cflux(2,1,1) + (corners(np+1,0) - corners(np,1)) / 2.0d0
      cflux(2,1,1) = cflux(2,1,1) + (corners(np+1,1) - corners(np,0)) / 2.0d0

      cflux(2,1,2) =                (corners(np  ,0) - corners(np,  1))
      cflux(2,1,2) = cflux(2,1,2) + (corners(np+1,0) - corners(np,  1)) / 2.0d0
      cflux(2,1,2) = cflux(2,1,2) + (corners(np  ,0) - corners(np+1,1)) / 2.0d0
    else
      cflux(2,1,1) =                (corners(np+1,1) - corners(np,1))
      cflux(2,1,2) =                (corners(np  ,0) - corners(np,1))
    endif

    if (getmapP(swest+2*max_corner_elem) /= -1) then
      cflux(1,2,1) =                (corners(0,np  ) - corners(1,np  ))
      cflux(1,2,1) = cflux(1,2,1) + (corners(0,np+1) - corners(1,np  )) / 2.0d0
      cflux(1,2,1) = cflux(1,2,1) + (corners(0,np  ) - corners(1,np+1)) / 2.0d0

      cflux(1,2,2) =                (corners(1,np+1) - corners(1,np  ))
      cflux(1,2,2) = cflux(1,2,2) + (corners(0,np+1) - corners(1,np  )) / 2.0d0
      cflux(1,2,2) = cflux(1,2,2) + (corners(1,np+1) - corners(0,np  )) / 2.0d0
    else
      cflux(1,2,1) =                (corners(0,np  ) - corners(1,np  ))
      cflux(1,2,2) =                (corners(1,np+1) - corners(1,np  ))
    endif

    if (getmapP(swest+3*max_corner_elem) /= -1) then
      cflux(2,2,1) =                (corners(np+1,np  ) - corners(np,np  ))
      cflux(2,2,1) = cflux(2,2,1) + (corners(np+1,np+1) - corners(np,np  )) / 2.0d0
      cflux(2,2,1) = cflux(2,2,1) + (corners(np+1,np  ) - corners(np,np+1)) / 2.0d0

      cflux(2,2,2) =                (corners(np  ,np+1) - corners(np,np  ))
      cflux(2,2,2) = cflux(2,2,2) + (corners(np+1,np+1) - corners(np,np  )) / 2.0d0
      cflux(2,2,2) = cflux(2,2,2) + (corners(np  ,np+1) - corners(np+1,np)) / 2.0d0
    else
      cflux(2,2,1) =                (corners(np+1,np  ) - corners(np,np  ))
      cflux(2,2,2) =                (corners(np  ,np+1) - corners(np,np  ))
    endif
  end subroutine distribute_flux_at_corners
end module prim_advance_caar_mod
