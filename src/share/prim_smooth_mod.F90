#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!#define _DBG_ print *,"File:",__FILE__," at ",__LINE__
!#define _DBG_ !DBG
!
!
module prim_smooth_mod
  use edgetype_mod,   only: EdgeBuffer_t

  implicit none
  private
  save

  type (EdgeBuffer_t) :: edge3p1
  public :: edge3p1

  public :: smooth_phis

contains

  subroutine smooth_phis(phis,elem,hybrid,deriv,nets,nete,minf,numcycle)
    use kinds,          only: real_kind
    use dimensions_mod, only : np, nlev
    use control_mod, only : smooth_phis_nudt, hypervis_scaling
    use hybrid_mod, only : hybrid_t
    use edge_mod, only : edgevpack, edgevunpack, edgevunpackmax, edgevunpackmin
    use edgetype_mod, only : EdgeBuffer_t
    use bndry_mod, only : bndry_exchangev
    use perf_mod,         only: t_startf, t_stopf
    use element_mod, only : element_t
    use derivative_mod, only : derivative_t , laplace_sphere_wk
    use time_mod, only : TimeLevel_t
    implicit none

    integer :: nets,nete
    real (kind=real_kind), dimension(np,np,nets:nete), intent(inout)   :: phis
    type (hybrid_t)      , intent(in) :: hybrid
    type (element_t)     , intent(inout), target :: elem(:)
    type (derivative_t)  , intent(in) :: deriv
    real (kind=real_kind), intent(in)   :: minf
    integer,               intent(in) :: numcycle

    ! local
    real (kind=real_kind), dimension(np,np,nets:nete) :: pstens
    real (kind=real_kind), dimension(nets:nete) :: pmin,pmax
    real (kind=real_kind) :: mx,mn
    integer :: nt,ie,ic,i,j,order,order_max, iuse
    logical :: use_var_coef


    ! compute local element neighbor min/max
    do ie=nets,nete
       pstens(:,:,ie)=minval(phis(:,:,ie))
       call edgeVpack(edge3p1,pstens(:,:,ie),1,0,ie)
    enddo

    call t_startf('smooth_phis_bexchV1')
    call bndry_exchangeV(hybrid,edge3p1)
    call t_stopf('smooth_phis_bexchV1')

    do ie=nets,nete
       call edgeVunpackMin(edge3p1, pstens(:,:,ie), 1, 0, ie)
       pmin(ie)=minval(pstens(:,:,ie))
    enddo
    do ie=nets,nete
       pstens(:,:,ie)=maxval(phis(:,:,ie))
       call edgeVpack(edge3p1,pstens(:,:,ie),1,0,ie)
    enddo

    call t_startf('smooth_phis_bexchV2')
    call bndry_exchangeV(hybrid,edge3p1)
    call t_stopf('smooth_phis_bexchV2')

    do ie=nets,nete
       call edgeVunpackMax(edge3p1, pstens(:,:,ie), 1, 0, ie)
       pmax(ie)=maxval(pstens(:,:,ie))
    enddo

    ! order = 1   grad^2 laplacian
    ! order = 2   grad^4 (need to add a negative sign)
    ! order = 3   grad^6
    ! order = 4   grad^8 (need to add a negative sign)
    order_max = 1


    use_var_coef=.true.
    if (hypervis_scaling/=0) then
       ! for tensorHV option, we turn off the tensor except for *last* laplace operator
       use_var_coef=.false.
       if (hypervis_scaling>=3) then
          ! with a 3.2 or 4 scaling, assume hyperviscosity
          order_max = 2
       endif
    endif


    do ic=1,numcycle
       pstens=phis

       do order=1,order_max-1

          do ie=nets,nete
             pstens(:,:,ie)=laplace_sphere_wk(pstens(:,:,ie),deriv,elem(ie),var_coef=use_var_coef)
             call edgeVpack(edge3p1,pstens(:,:,ie),1,0,ie)
          enddo

          call t_startf('smooth_phis_bexchV3')
          call bndry_exchangeV(hybrid,edge3p1)
          call t_stopf('smooth_phis_bexchV3')

          do ie=nets,nete
             call edgeVunpack(edge3p1, pstens(:,:,ie), 1, 0, ie)
             pstens(:,:,ie)=pstens(:,:,ie)*elem(ie)%rspheremp(:,:)
          enddo
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
          !$OMP BARRIER
#endif
#endif
       enddo
       do ie=nets,nete
          pstens(:,:,ie)=laplace_sphere_wk(pstens(:,:,ie),deriv,elem(ie),var_coef=.true.)
       enddo
       if (mod(order_max,2)==0) pstens=-pstens

       do ie=nets,nete
          !  ps(t+1) = ps(t) + Minv * DSS * M * RHS
          !  ps(t+1) = Minv * DSS * M [ ps(t) +  RHS ]
          ! but output of biharminc_wk is of the form M*RHS.  rewrite as:
          !  ps(t+1) = Minv * DSS * M [ ps(t) +  M*RHS/M ]
          ! so we can apply limiter to ps(t) +  (M*RHS)/M
#if 1
          mn=pmin(ie)
          mx=pmax(ie)
          iuse = numcycle+1  ! always apply min/max limiter
#endif
          phis(:,:,ie)=phis(:,:,ie) + &
               smooth_phis_nudt*pstens(:,:,ie)/elem(ie)%spheremp(:,:)


          ! remove new extrema.  could use conservative reconstruction from advection
          ! but no reason to conserve mean PHI.
          if (ic < iuse) then
             do i=1,np
                do j=1,np
                   if (phis(i,j,ie)>mx) phis(i,j,ie)=mx
                   if (phis(i,j,ie)<mn) phis(i,j,ie)=mn
                enddo
             enddo
          endif


          ! user specified minimum
          do i=1,np
             do j=1,np
                if (phis(i,j,ie)<minf) phis(i,j,ie)=minf
             enddo
          enddo

          phis(:,:,ie)=phis(:,:,ie)*elem(ie)%spheremp(:,:)
          call edgeVpack(edge3p1,phis(:,:,ie),1,0,ie)

       enddo

       call t_startf('smooth_phis_bexchV4')
       call bndry_exchangeV(hybrid,edge3p1)
       call t_stopf('smooth_phis_bexchV4')

       do ie=nets,nete
          call edgeVunpack(edge3p1, phis(:,:,ie), 1, 0, ie)
          phis(:,:,ie)=phis(:,:,ie)*elem(ie)%rspheremp(:,:)
       enddo
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
       !$OMP BARRIER
#endif
#endif
    enddo
  end subroutine smooth_phis


end module prim_smooth_mod
