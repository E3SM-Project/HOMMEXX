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

  public :: applyCAMforcing_dynamics, applyCAMforcing, CAM_forcing_tracers, CAM_forcing_states

contains

  subroutine CAM_forcing_tracers(dt_q, ps0, qsize, np1, np1_qdp, wet, hyai, hybi, FQ, Qdp, ps_v, Q) bind(c)
    use iso_c_binding,  only: c_int, c_bool
    use element_mod,    only: timelevels
    use dimensions_mod, only: np, nlev
    use physical_constants, only: Cp

    implicit none

    real (kind=real_kind), intent(in) :: dt_q, ps0
    integer (kind=c_int), intent(in) :: qsize, np1, np1_qdp
    logical (kind=c_bool), intent(in) :: wet
    real (kind=real_kind), intent(in) :: hyai(nlev)
    real (kind=real_kind), intent(in) :: hybi(nlev)
    real (kind=real_kind), intent(in) :: FQ(np, np, nlev, qsize)
    real (kind=real_kind), intent(inout) :: Qdp(np, np, nlev, qsize, timelevels)
    real (kind=real_kind), intent(inout) :: ps_v(np, np, timelevels)
    real (kind=real_kind), intent(out) :: Q(np, np, nlev, qsize)

    integer :: i, j, k, q_idx
    real (kind=real_kind) :: v1, dp
    real (kind=real_kind) :: FQps(np, np)

    FQps(:, :)=0
#if (defined COLUMN_OPENMP)
    !$omp parallel do private(q_idx,k,i,j,v1,FQps)
#endif
    do q_idx = 1, qsize
       do k = 1, nlev
          do j = 1, np
             do i = 1, np
                v1 = dt_q * FQ(i, j, k, q_idx)
                if (Qdp(i, j, k, q_idx, np1_qdp) + v1 < 0 .and. v1 < 0) then
                   if (Qdp(i, j, k, q_idx, np1_qdp) < 0 ) then
                      v1 = 0  ! Q already negative, dont make it more so
                   else
                      v1 = -Qdp(i, j, k, q_idx, np1_qdp)
                   endif
                endif
                Qdp(i, j, k, q_idx, np1_qdp) = Qdp(i, j, k, q_idx, np1_qdp) + v1
                if (q_idx == 1) then
                   FQps(i, j) = FQps(i, j) + v1 / dt_q
                endif
             enddo
          enddo
       enddo
    enddo

    if (wet .and. qsize>0) then
       ! to conserve dry mass in the precese of Q1 forcing:
       ps_v(:, :, np1) = ps_v(:, :, np1) + dt_q * FQps(:, :)
    endif

    ! Qdp(np1) and ps_v(np1) were updated by forcing - update Q(np1)
    do q_idx = 1, qsize
       do k = 1, nlev
          do j = 1, np
             do i = 1, np
                dp = ( hyai(k+1) - hyai(k) ) * ps0 + &
                     ( hybi(k+1) - hybi(k) ) * ps_v(i, j, np1)
                Q(i, j, k, q_idx) = Qdp(i, j, k, q_idx, np1_qdp) / dp
             enddo
          enddo
       enddo
    enddo
  end subroutine CAM_Forcing_Tracers

  subroutine CAM_forcing_states(dt_q, np1, FT, FM, T, v) bind(c)
    use iso_c_binding, only: c_int
    use dimensions_mod, only: np, nlev
    use element_mod,   only: timelevels
    implicit none

    real (kind=real_kind), intent(in) :: dt_q
    integer (kind=c_int), intent(in) :: np1
    real (kind=real_kind), intent(in) :: FT(np, np, nlev)
    real (kind=real_kind), intent(in) :: FM(np, np, 2, nlev)
    real (kind=real_kind), intent(inout) :: T(np, np, nlev, timelevels)
    real (kind=real_kind), intent(inout) :: v(np, np, 2, nlev, timelevels)

    T(:,:,:,np1)   = T(:,:,:,np1)   + dt_q * FT(:,:,:)
    v(:,:,:,:,np1) = v(:,:,:,:,np1) + dt_q * FM(:,:,:,:)
  end subroutine CAM_forcing_states

  subroutine applyCAMforcing(elem,hvcoord,np1,np1_qdp,dt_q,nets,nete)
    use iso_c_binding,  only: c_bool
    use dimensions_mod, only: np, nlev, qsize
    use hybvcoord_mod,  only: hvcoord_t
    use control_mod,    only: moisture
    use physical_constants, only: Cp

    implicit none
    type (element_t),       intent(inout) :: elem(:)
    real (kind=real_kind),  intent(in)    :: dt_q
    type (hvcoord_t),       intent(in)    :: hvcoord
    integer,                intent(in)    :: np1,nets,nete,np1_qdp

    ! local
    integer :: ie
    logical (kind=c_bool) :: wet

    wet = (moisture /= "dry")

    do ie=nets,nete
       call CAM_forcing_tracers(dt_q, hvcoord%ps0, qsize, np1, np1_qdp, wet, hvcoord%hyai, hvcoord%hybi, &
            elem(ie)%derived%FQ, elem(ie)%state%Qdp,  elem(ie)%state%ps_v,  elem(ie)%state%Q)

       call CAM_forcing_states(dt_q, np1, elem(ie)%derived%FT, elem(ie)%derived%FM, &
            elem(ie)%state%T, elem(ie)%state%v)
    enddo
  end subroutine applyCAMforcing

  subroutine applyCAMforcing_dynamics(elem,hvcoord,np1,dt_q,nets,nete)
    use dimensions_mod, only: np
    use element_mod,    only: element_t
    use hybvcoord_mod,  only: hvcoord_t

    implicit none

    type (element_t)     ,  intent(inout) :: elem(:)
    real (kind=real_kind),  intent(in)    :: dt_q
    type (hvcoord_t),       intent(in)    :: hvcoord
    integer,                intent(in)    :: np1,nets,nete

    ! local
    integer :: ie

    do ie=nets,nete
       call CAM_forcing_states(dt_q, np1, elem(ie)%derived%FT, elem(ie)%derived%FM, &
            elem(ie)%state%T, elem(ie)%state%v)
    enddo
  end subroutine applyCAMforcing_dynamics

end module prim_forcing_mod
