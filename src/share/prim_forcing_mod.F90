#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!#define _DBG_ print *,"File:",__FILE__," at ",__LINE__
!#define _DBG_ !DBG
!
!
module prim_forcing_mod
  use element_mod,    only: element_t
  use kinds      ,    only: real_kind

  implicit none
  private
  save

  public :: applyCAMforcing_dynamics, applyCAMforcing

contains

  subroutine applyCAMforcing(elem,hvcoord,np1,np1_qdp,dt_q,nets,nete)
    use dimensions_mod, only: np, nc, nlev, qsize
    use hybvcoord_mod,  only: hvcoord_t
    use control_mod,    only: moisture, tracer_grid_type
    use control_mod,    only: TRACER_GRIDTYPE_GLL
    use physical_constants, only: Cp

    implicit none
    type (element_t),       intent(inout) :: elem(:)
    real (kind=real_kind),  intent(in)    :: dt_q
    type (hvcoord_t),       intent(in)    :: hvcoord
    integer,                intent(in)    :: np1,nets,nete,np1_qdp

    ! local
    integer :: i,j,k,ie,q
    real (kind=real_kind) :: v1,dp
    real (kind=real_kind) :: beta(np,np),E0(np,np),ED(np,np),dp0m1(np,np),dpsum(np,np)
    logical :: wet

    wet = (moisture /= "dry")

    do ie=nets,nete
       ! apply forcing to Qdp
       elem(ie)%derived%FQps(:,:,1)=0
#if (defined COLUMN_OPENMP)
       !$omp parallel do private(q,k,i,j,v1)
#endif
       do q=1,qsize
          do k=1,nlev
             do j=1,np
                do i=1,np
                   v1 = dt_q*elem(ie)%derived%FQ(i,j,k,q)
                   !if (elem(ie)%state%Qdp(i,j,k,q,np1) + v1 < 0 .and. v1<0) then
                   if (elem(ie)%state%Qdp(i,j,k,q,np1_qdp) + v1 < 0 .and. v1<0) then
                      !if (elem(ie)%state%Qdp(i,j,k,q,np1) < 0 ) then
                      if (elem(ie)%state%Qdp(i,j,k,q,np1_qdp) < 0 ) then
                         v1=0  ! Q already negative, dont make it more so
                      else
                         !v1 = -elem(ie)%state%Qdp(i,j,k,q,np1)
                         v1 = -elem(ie)%state%Qdp(i,j,k,q,np1_qdp)
                      endif
                   endif
                   !elem(ie)%state%Qdp(i,j,k,q,np1) = elem(ie)%state%Qdp(i,j,k,q,np1)+v1
                   elem(ie)%state%Qdp(i,j,k,q,np1_qdp) = elem(ie)%state%Qdp(i,j,k,q,np1_qdp)+v1
                   if (q==1) then
                      elem(ie)%derived%FQps(i,j,1)=elem(ie)%derived%FQps(i,j,1)+v1/dt_q
                   endif
                enddo
             enddo
          enddo
       enddo

       if (wet .and. qsize>0) then
          ! to conserve dry mass in the precese of Q1 forcing:
          elem(ie)%state%ps_v(:,:,np1) = elem(ie)%state%ps_v(:,:,np1) + &
               dt_q*elem(ie)%derived%FQps(:,:,1)
       endif

       ! Qdp(np1) and ps_v(np1) were updated by forcing - update Q(np1)
#if (defined COLUMN_OPENMP)
       !$omp parallel do private(q,k,i,j,dp)
#endif
       do q=1,qsize
          do k=1,nlev
             do j=1,np
                do i=1,np
                   dp = ( hvcoord%hyai(k+1) - hvcoord%hyai(k) )*hvcoord%ps0 + &
                        ( hvcoord%hybi(k+1) - hvcoord%hybi(k) )*elem(ie)%state%ps_v(i,j,np1)
                   elem(ie)%state%Q(i,j,k,q) = elem(ie)%state%Qdp(i,j,k,q,np1_qdp)/dp
                enddo
             enddo
          enddo
       enddo

       elem(ie)%state%T(:,:,:,np1)   = elem(ie)%state%T(:,:,:,np1)   + dt_q*elem(ie)%derived%FT(:,:,:)
       elem(ie)%state%v(:,:,:,:,np1) = elem(ie)%state%v(:,:,:,:,np1) + dt_q*elem(ie)%derived%FM(:,:,:,:)

    enddo
  end subroutine applyCAMforcing



  subroutine applyCAMforcing_dynamics(elem,hvcoord,np1,dt_q,nets,nete)

    use dimensions_mod, only: np, nlev, qsize
    use element_mod,    only: element_t
    use hybvcoord_mod,  only: hvcoord_t

    implicit none
    type (element_t)     ,  intent(inout) :: elem(:)
    real (kind=real_kind),  intent(in)    :: dt_q
    type (hvcoord_t),       intent(in)    :: hvcoord
    integer,                intent(in)    :: np1,nets,nete

    integer :: i,j,k,ie,q
    real (kind=real_kind) :: v1,dp
    logical :: wet

    do ie=nets,nete
       elem(ie)%state%T(:,:,:,np1)  = elem(ie)%state%T(:,:,:,np1)    + dt_q*elem(ie)%derived%FT(:,:,:)
       elem(ie)%state%v(:,:,:,:,np1) = elem(ie)%state%v(:,:,:,:,np1) + dt_q*elem(ie)%derived%FM(:,:,:,:)
    enddo
  end subroutine applyCAMforcing_dynamics

end module prim_forcing_mod
