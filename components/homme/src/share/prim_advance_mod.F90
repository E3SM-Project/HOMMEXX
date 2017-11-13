#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!#define _DBG_ print *,"File:",__FILE__," at ",__LINE__
!#define _DBG_ !DBG
!
!
module prim_advance_mod

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
  use time_mod,       only: timelevel_t
  use prim_advance_exp_mod, only: prim_advance_exp, ur_weights
  use prim_advance_caar_mod, only: edge3p1

  implicit none
  private
  save
  public :: prim_advance_si, prim_advance_init, preq_robert3,&
       applyCAMforcing_dynamics, applyCAMforcing, smooth_phis, overwrite_SEdensity

  type (EdgeBuffer_t) :: edge1
  type (EdgeBuffer_t) :: edge2

  real (kind=real_kind) :: initialized_for_dt   = 0

contains

  subroutine prim_advance_init(par, elem,integration)
    use edge_mod, only : initEdgeBuffer
    implicit none

    type (parallel_t) :: par
    type (element_t), intent(inout), target   :: elem(:)
    character(len=*)    , intent(in) :: integration
    integer :: i
    integer :: ie

    call initEdgeBuffer(par,edge3p1,elem,4*nlev)

    if(integration == 'semi_imp') then
       call initEdgeBuffer(par,edge1,elem,nlev)
       call initEdgeBuffer(par,edge2,elem,2*nlev)
    end if

    ! compute averaging weights for RK+LF (tstep_type=1) timestepping:
    allocate(ur_weights(qsplit))
    ur_weights(:)=0.0d0

    if(mod(qsplit,2).NE.0)then
       ur_weights(1)=1.0d0/qsplit
       do i=3,qsplit,2
         ur_weights(i)=2.0d0/qsplit
       enddo
    else
       do i=2,qsplit,2
         ur_weights(i)=2.0d0/qsplit
       enddo
    endif

  end subroutine prim_advance_init

subroutine prim_advance_si(elem, nets, nete, cg, blkjac, red, &
          refstate, hvcoord, deriv, flt, hybrid, tl, dt)
       use bndry_mod, only : bndry_exchangev
       use cg_mod, only : cg_t, cg_create
       use control_mod, only : filter_freq,debug_level, precon_method
       use derivative_mod, only : derivative_t, vorticity, divergence, gradient, gradient_wk
       use edge_mod, only : edgevpack, edgevunpack, initEdgeBuffer
       use edgetype_mod, only : EdgeBuffer_t
       use filter_mod, only : filter_t, preq_filter
       use hybrid_mod, only : hybrid_t
       use prim_si_ref_mod, only : ref_state_t, set_vert_struct_mat
       use reduction_mod, only : reductionbuffer_ordered_1d_t
       use solver_mod, only : pcg_solver, blkjac_t, blkjac_init
       use prim_si_mod, only : preq_vertadv, preq_omegap, preq_pressure
       use diffusion_mod, only :  prim_diffusion
       use physical_constants, only : kappa, rrearth, rgas, cp, rwater_vapor
       use physics_mod, only : virtual_temperature, virtual_specific_heat
       implicit none

       integer, intent(in)               :: nets,nete
       type (element_t), intent(inout), target :: elem(:)
       type (blkjac_t), allocatable      :: blkjac(:)

       type (cg_t)                       :: cg

       type (ReductionBuffer_ordered_1d_t), intent(inout) :: red

       type (ref_state_t), intent(in), target :: refstate
       type (hvcoord_t), intent(in)      :: hvcoord
       type (derivative_t), intent(in)   :: deriv
       type (filter_t), intent(in)       :: flt
       type (hybrid_t), intent(in)       :: hybrid
       type (TimeLevel_t), intent(in)    :: tl
       real(kind=real_kind), intent(in)  :: dt
       real(kind=real_kind)              :: time_adv
#ifndef _CRAYFTN
       ! ==========================
       ! Local variables...
       ! ==========================

       real(kind=real_kind)                           :: ps0
       real(kind=real_kind)                           :: psref

       real(kind=real_kind), dimension(np,np)         :: ps
       real(kind=real_kind), dimension(np,np)         :: rps
       real(kind=real_kind), dimension(np,np,nlev)    :: rpmid
       real(kind=real_kind), dimension(np,np,nlev)    :: omegap
       real(kind=real_kind), dimension(np,np,nlev)    :: rpdel

       real(kind=real_kind) :: pintref(nlevp)
       real(kind=real_kind) :: pdelref(nlev)
       real(kind=real_kind) :: pmidref(nlev)
       real(kind=real_kind) :: rpdelref(nlev)
       real(kind=real_kind) :: rpmidref(nlev)

       real(kind=real_kind) :: pint(np,np,nlevp)
       real(kind=real_kind) :: pdel(np,np,nlev)
       real(kind=real_kind) :: pmid(np,np,nlev)

       real(kind=real_kind), dimension(np,np,nlevp) :: eta_dot_dp_deta
       real(kind=real_kind), dimension(np,np,nlev)  :: vgrad_ps

       real(kind=real_kind), dimension(np,np,nlev)   :: T_vadv
       real(kind=real_kind), dimension(np,np,2,nlev) :: v_vadv

       real(kind=real_kind), dimension(np,np)      :: HT
       real(kind=real_kind), dimension(np,np)      :: HrefT
       real(kind=real_kind), dimension(np,np)      :: HrefTm1

       real(kind=real_kind), dimension(np,np)      :: Gref0
       real(kind=real_kind), dimension(np,np)      :: Grefm1
       real(kind=real_kind), dimension(np,np)      :: E
       real(kind=real_kind), dimension(np,np)      :: Phi
       real(kind=real_kind), dimension(np,np)      :: dGref

       real(kind=real_kind), dimension(np,np,2)    :: vco
       real(kind=real_kind), dimension(np,np,2)    :: gradT
       real(kind=real_kind), dimension(np,np,2)    :: grad_Phi

       real(kind=real_kind), dimension(:,:), pointer  :: Emat
       real(kind=real_kind), dimension(:,:), pointer  :: Emat_inv
       real(kind=real_kind), dimension(:,:), pointer  :: Amat
       real(kind=real_kind), dimension(:,:), pointer  :: Amat_inv
       real(kind=real_kind), dimension(:), pointer    :: Lambda

       real(kind=real_kind), dimension(:), pointer    :: Tref
       real(kind=real_kind), dimension(:), pointer    :: RTref
       real(kind=real_kind), dimension(:), pointer    :: Pvec
       real(kind=real_kind), dimension(:,:), pointer  :: Href
       real(kind=real_kind), dimension(:,:), pointer  :: Tmat

       real(kind=real_kind) :: Vscript(np,np,2,nlev,nets:nete)
       real(kind=real_kind) :: Tscript(np,np,nlev,nets:nete)
       real(kind=real_kind) :: Pscript(np,np,nets:nete)
       real(kind=real_kind) :: Vtemp(np,np,2,nlev,nets:nete)

       real(kind=real_kind), dimension(np,np)      :: HrefTscript
       real(kind=real_kind), dimension(np,np)      :: suml
       real(kind=real_kind), dimension(np,np,2)    :: gVscript
       real(kind=real_kind), dimension(np,np,nlev) :: div_Vscript

       real(kind=real_kind) :: B(np,np,nlev,nets:nete)
       real(kind=real_kind) :: C(np,np,nlev,nets:nete)
       real(kind=real_kind) :: D(np,np,nlev,nets:nete)

       real(kind=real_kind) :: Gamma_ref(np,np,nlev,nets:nete)

       real(kind=real_kind) :: Gref(np,np,nlev,nets:nete)
       real(kind=real_kind) :: grad_dGref(np,np,2,nlev)
       real(kind=real_kind) :: grad_Gref(np,np,2,nlev)

       real(kind=real_kind) :: div(np,np)
       real(kind=real_kind) :: gv(np,np,2)

       real(kind=real_kind) :: dt2
       real(kind=real_kind) :: rpsref
       real(kind=real_kind) :: rdt
       real(kind=real_kind) :: hkk, hkl
       real(kind=real_kind) :: ddiv

       real(kind=real_kind) :: vgradT
       real(kind=real_kind) :: hybfac
       real(kind=real_kind) :: Crkk
       real(kind=real_kind) :: v1,v2
       real(kind=real_kind) :: term

       real(kind=real_kind) :: Vs1,Vs2
       real(kind=real_kind) :: glnps1, glnps2
       real(kind=real_kind) :: gGr1,gGr2

       real (kind=real_kind),allocatable :: solver_wts(:,:)  ! solver weights array for nonstaggered grid

       integer              :: nm1,n0,np1,nfilt
       integer              :: nstep
       integer              :: i,j,k,l,ie,kptr

!JMD       call t_barrierf('sync_prim_advance_si', hybrid%par%comm)
!pw    call t_adj_detailf(+1)
       call t_startf('prim_advance_si')

       nm1   = tl%nm1
       n0    = tl%n0
       np1   = tl%np1
       nstep = tl%nstep


       if ( dt /= initialized_for_dt ) then
          if(hybrid%par%masterproc) print *,'Initializing semi-implicit matricies for dt=',dt

#if (defined HORIZ_OPENMP)
          !$OMP MASTER
#endif
          call set_vert_struct_mat(dt, refstate, hvcoord, hybrid%masterthread)
#if (defined HORIZ_OPENMP)
          !$OMP END MASTER
#endif

          allocate(solver_wts(np*np,nete-nets+1))
          do ie=nets,nete
             kptr=1
             do j=1,np
                do i=1,np

                   ! so this code is BFB  with old code.  should change to simpler formula below
                   solver_wts(kptr,ie-nets+1) = 1d0/nint(1d0/(elem(ie)%mp(i,j)*elem(ie)%rmp(i,j)))
                   !solver_wts(kptr,ie-nets+1) = elem(ie)%mp(i,j)*elem(ie)%rmp(i,j)

                   kptr=kptr+1
                end do
             end do
          end do
          call cg_create(cg, np*np, nlev, nete-nets+1, hybrid, debug_level, solver_wts)
          deallocate(solver_wts)
          if (precon_method == "block_jacobi") then
             if (.not. allocated(blkjac)) then
                allocate(blkjac(nets:nete))
             endif
             call blkjac_init(elem, deriv,refstate%Lambda,nets,nete,blkjac)
          end if
          initialized_for_dt = dt
       endif


       nfilt = tl%nm1     ! time level at which filter is applied (time level n)
       dt2   = 2.0_real_kind*dt
       rdt   = 1.0_real_kind/dt

       ps0      = hvcoord%ps0
       psref    = refstate%psr

       Emat     => refstate%Emat
       Emat_inv => refstate%Emat_inv
       Amat     => refstate%Amat
       Amat_inv => refstate%Amat_inv
       Lambda   => refstate%Lambda

       RTref    => refstate%RTref
       Tref     => refstate%Tref
       Href     => refstate%Href
       Tmat     => refstate%Tmat
       Pvec     => refstate%Pvec

       ! ============================================================
       ! If the time is right, apply a filter to the state variables
       ! ============================================================

       if (nstep > 0 .and. filter_freq > 0 .and. MODULO(nstep,filter_freq) == 0 ) then
          call preq_filter(elem, edge3p1, flt, cg%hybrid, nfilt, nets, nete)
       end if

       ! ================================================
       ! boundary exchange grad_lnps
       ! ================================================

       do ie = nets, nete

          elem(ie)%derived%grad_lnps(:,:,:) = gradient(elem(ie)%state%lnps(:,:,n0),deriv)*rrearth

          do k=1,nlevp
             pintref(k)  = hvcoord%hyai(k)*ps0 + hvcoord%hybi(k)*psref
          end do

          do k=1,nlev
             pmidref(k)  = hvcoord%hyam(k)*ps0 + hvcoord%hybm(k)*psref
             pdelref(k)  = pintref(k+1) - pintref(k)
             rpmidref(k) = 1.0_real_kind/pmidref(k)
             rpdelref(k) = 1.0_real_kind/pdelref(k)
          end do

          rpsref   = 1.0_real_kind/psref

          ps(:,:) = EXP(elem(ie)%state%lnps(:,:,n0))
          rps(:,:) = 1.0_real_kind/ps(:,:)

          call preq_pressure(ps0,ps,hvcoord%hyai,hvcoord%hybi,hvcoord%hyam,hvcoord%hybm,pint,pmid,pdel)

          rpmid = 1.0_real_kind/pmid
          rpdel = 1.0_real_kind/pdel

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,v1,v2)
#endif
          do k=1,nlev
             do j=1,np
                do i=1,np
                   v1     = elem(ie)%state%v(i,j,1,k,n0)
                   v2     = elem(ie)%state%v(i,j,2,k,n0)

                   ! Contravariant velocities

                   vco(i,j,1) = elem(ie)%Dinv(i,j,1,1)*v1 + elem(ie)%Dinv(i,j,1,2)*v2
                   vco(i,j,2) = elem(ie)%Dinv(i,j,2,1)*v1 + elem(ie)%Dinv(i,j,2,2)*v2

                   vgrad_ps(i,j,k) = ps(i,j)*(vco(i,j,1)*elem(ie)%derived%grad_lnps(i,j,1) + &
                        vco(i,j,2)*elem(ie)%derived%grad_lnps(i,j,2))

                end do
             end do
          end do

          call preq_omegap(elem(ie)%derived%div(:,:,:,n0),vgrad_ps,pdel,rpmid, &
               hvcoord%hybm,hvcoord%hybd,elem(ie)%derived%omega_p)

          Pscript(:,:,ie)        = 0.0_real_kind
          eta_dot_dp_deta(:,:,1) = 0.0_real_kind

          do k=1,nlev
             do j=1,np
                do i=1,np
                   eta_dot_dp_deta(i,j,k+1) = eta_dot_dp_deta(i,j,k) + &
                        vgrad_ps(i,j,k)*hvcoord%hybd(k) + elem(ie)%derived%div(i,j,k,n0)*pdel(i,j,k)
                   ddiv = elem(ie)%derived%div(i,j,k,n0) - 0.5_real_kind*elem(ie)%derived%div(i,j,k,nm1)
                   Pscript(i,j,ie) = Pscript(i,j,ie) + ddiv*pdelref(k)
                end do
             end do
          end do

          do j=1,np
             do i=1,np
                Pscript(i,j,ie) = elem(ie)%state%lnps(i,j,nm1) + &
                     dt2*( rpsref*Pscript(i,j,ie) - rps(i,j)*eta_dot_dp_deta(i,j,nlev+1) )
             end do
          end do

          do k=1,nlev-1
             do j=1,np
                do i=1,np
                   eta_dot_dp_deta(i,j,k+1) = hvcoord%hybi(k+1)*eta_dot_dp_deta(i,j,nlev+1) - &
                        eta_dot_dp_deta(i,j,k+1)
                end do
             end do
          end do

          eta_dot_dp_deta(:,:,nlev+1) = 0.0_real_kind

          call preq_vertadv(elem(ie)%state%T(:,:,:,n0),elem(ie)%state%v(:,:,:,:,n0), &
               eta_dot_dp_deta,rpdel,T_vadv,v_vadv)

          suml(:,:) = 0.0_real_kind

          do k=1,nlev

             gradT(:,:,:) = gradient(elem(ie)%state%T(:,:,k,n0),deriv)*rrearth
             Crkk       = 0.5_real_kind

             do j=1,np
                do i=1,np
                   term = Crkk*(elem(ie)%derived%div(i,j,k,n0) - &
                        0.5_real_kind*elem(ie)%derived%div(i,j,k,nm1))*pdelref(k)
                   suml(i,j)  = suml(i,j) + term

                   v1     = elem(ie)%state%v(i,j,1,k,n0)
                   v2     = elem(ie)%state%v(i,j,2,k,n0)

                   ! Contravariant velocities

                   vco(i,j,1) = elem(ie)%Dinv(i,j,1,1)*v1 + elem(ie)%Dinv(i,j,1,2)*v2
                   vco(i,j,2) = elem(ie)%Dinv(i,j,2,1)*v1 + elem(ie)%Dinv(i,j,2,2)*v2

                   vgradT = vco(i,j,1)*gradT(i,j,1) + vco(i,j,2)*gradT(i,j,2)

                   Tscript(i,j,k,ie) = elem(ie)%state%T(i,j,k,nm1) &
                        + dt2*(- vgradT - T_vadv(i,j,k)           &
                        + kappa*(elem(ie)%state%T(i,j,k,n0)*elem(ie)%derived%omega_p(i,j,k) &
                        + Tref(k)*rpmidref(k)*suml(i,j)))
                   suml(i,j)  = suml(i,j) + term
                end do
             end do
          end do

          HrefT(:,:)   = 0.0_real_kind
          HrefTm1(:,:) = 0.0_real_kind
          HT(:,:)      = 0.0_real_kind

          do k=nlev,1,-1

             do j=1,np
                do i=1,np
                   hkl = rpmidref(k)*pdelref(k)
                   hkk = hkl*0.5_real_kind
                   Gref0(i,j)   = HrefT(i,j) + Rgas*hkk*elem(ie)%state%T(i,j,k,n0)
                   HrefT(i,j)   = HrefT(i,j) + Rgas*hkl*elem(ie)%state%T(i,j,k,n0)
                   Grefm1(i,j)  = HrefTm1(i,j) + Rgas*hkk*elem(ie)%state%T(i,j,k,nm1)
                   HrefTm1(i,j) = HrefTm1(i,j) + Rgas*hkl*elem(ie)%state%T(i,j,k,nm1)
                   hkl = rpmid(i,j,k)*pdel(i,j,k)
                   hkk = hkl*0.5_real_kind
                   Phi(i,j) = HT(i,j) + Rgas*hkk*elem(ie)%state%T(i,j,k,n0)
                   HT(i,j)  = HT(i,j) + Rgas*hkl*elem(ie)%state%T(i,j,k,n0)
                end do
             end do

             do j=1,np
                do i=1,np
                   v1     = elem(ie)%state%v(i,j,1,k,n0)
                   v2     = elem(ie)%state%v(i,j,2,k,n0)

                   ! covariant velocity

                   vco(i,j,1) = elem(ie)%D(i,j,1,1)*v1 + elem(ie)%D(i,j,2,1)*v2
                   vco(i,j,2) = elem(ie)%D(i,j,1,2)*v1 + elem(ie)%D(i,j,2,2)*v2

                   E(i,j) = 0.5_real_kind*( v1*v1 + v2*v2 )

                   Gref0(i,j)  =  Gref0(i,j)  + elem(ie)%state%phis(i,j) + RTref(k)*elem(ie)%state%lnps(i,j,n0)
                   Grefm1(i,j) =  Grefm1(i,j) + elem(ie)%state%phis(i,j) + RTref(k)*elem(ie)%state%lnps(i,j,nm1)

                   Phi(i,j)    =  Phi(i,j) + E(i,j) + elem(ie)%state%phis(i,j)
                   dGref(i,j)  =  -(Gref0(i,j)  - 0.5_real_kind*Grefm1(i,j))
                end do
             end do

             elem(ie)%derived%zeta(:,:,k) = vorticity(vco,deriv)*rrearth
             grad_Phi(:,:,:)     = gradient(Phi,deriv)*rrearth
             grad_dGref(:,:,:,k) = gradient_wk(dGref,deriv)*rrearth

             do j=1,np
                do i=1,np

                   elem(ie)%derived%zeta(i,j,k) = elem(ie)%rmetdet(i,j)*elem(ie)%derived%zeta(i,j,k)
                   hybfac =  hvcoord%hybm(k)*(ps(i,j)*rpmid(i,j,k))

                   glnps1 = elem(ie)%Dinv(i,j,1,1)*elem(ie)%derived%grad_lnps(i,j,1) + &
                        elem(ie)%Dinv(i,j,2,1)*elem(ie)%derived%grad_lnps(i,j,2)
                   glnps2 = elem(ie)%Dinv(i,j,1,2)*elem(ie)%derived%grad_lnps(i,j,1) + &
                        elem(ie)%Dinv(i,j,2,2)*elem(ie)%derived%grad_lnps(i,j,2)

                   v1 = elem(ie)%Dinv(i,j,1,1)*grad_Phi(i,j,1) + elem(ie)%Dinv(i,j,2,1)*grad_Phi(i,j,2)
                   v2 = elem(ie)%Dinv(i,j,1,2)*grad_Phi(i,j,1) + elem(ie)%Dinv(i,j,2,2)*grad_Phi(i,j,2)

                   Vscript(i,j,1,k,ie) = - v_vadv(i,j,1,k) &
                        + elem(ie)%state%v(i,j,2,k,n0) * (elem(ie)%fcor(i,j) + elem(ie)%derived%zeta(i,j,k)) &
                        - v1 - Rgas*hybfac*elem(ie)%state%T(i,j,k,n0)*glnps1

                   Vscript(i,j,2,k,ie) = - v_vadv(i,j,2,k) &
                        - elem(ie)%state%v(i,j,1,k,n0) * (elem(ie)%fcor(i,j) + elem(ie)%derived%zeta(i,j,k)) &
                        - v2 - Rgas*hybfac*elem(ie)%state%T(i,j,k,n0)*glnps2

                end do
             end do

          end do

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,Vs1,Vs2)
#endif
          do k=1,nlev
             do j=1,np
                do i=1,np
                   Vs1 = elem(ie)%Dinv(i,j,1,1)*grad_dGref(i,j,1,k) + elem(ie)%Dinv(i,j,2,1)*grad_dGref(i,j,2,k)
                   Vs2 = elem(ie)%Dinv(i,j,1,2)*grad_dGref(i,j,1,k) + elem(ie)%Dinv(i,j,2,2)*grad_dGref(i,j,2,k)

                   Vscript(i,j,1,k,ie) = elem(ie)%mp(i,j)*Vscript(i,j,1,k,ie) + Vs1
                   Vscript(i,j,2,k,ie) = elem(ie)%mp(i,j)*Vscript(i,j,2,k,ie) + Vs2

                   Vscript(i,j,1,k,ie) = elem(ie)%mp(i,j)*elem(ie)%state%v(i,j,1,k,nm1) + dt2*Vscript(i,j,1,k,ie)
                   Vscript(i,j,2,k,ie) = elem(ie)%mp(i,j)*elem(ie)%state%v(i,j,2,k,nm1) + dt2*Vscript(i,j,2,k,ie)
                end do
             end do

          end do

          HrefTscript(:,:) = 0.0_real_kind

          do k=nlev,1,-1

             do j=1,np
                do i=1,np
                   hkl = rpmidref(k)*pdelref(k)
                   hkk = hkl*0.5_real_kind
                   B(i,j,k,ie)      = HrefTscript(i,j) + Rgas*hkk*Tscript(i,j,k,ie)
                   B(i,j,k,ie)      = B(i,j,k,ie) +  elem(ie)%state%phis(i,j) + RTref(k)*Pscript(i,j,ie)
                   HrefTscript(i,j) = HrefTscript(i,j) + Rgas*hkl*Tscript(i,j,k,ie)
                end do
             end do

          end do

          kptr=0
          call edgeVpack(edge2, Vscript(1,1,1,1,ie),2*nlev,kptr,ie)

       end do

       call t_startf('pasi_bexchV1')
       call bndry_exchangeV(cg%hybrid,edge2)
       call t_stopf('pasi_bexchV1')

       do ie = nets, nete

          kptr=0
          call edgeVunpack(edge2, Vscript(1,1,1,1,ie), 2*nlev, kptr, ie)
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif
#if (defined COLUMN_OPENMP)
! Not sure about deriv here.
!$omp parallel do private(k,i,j)
#endif
          do k=1,nlev
             do j=1,np
                do i=1,np
                   Vscript(i,j,1,k,ie) = elem(ie)%rmp(i,j)*Vscript(i,j,1,k,ie)
                   Vscript(i,j,2,k,ie) = elem(ie)%rmp(i,j)*Vscript(i,j,2,k,ie)
                end do
             end do

             do j=1,np
                do i=1,np

                   ! Contravariant Vscript

                   gVscript(i,j,1) = elem(ie)%Dinv(i,j,1,1)*Vscript(i,j,1,k,ie) + &
                        elem(ie)%Dinv(i,j,1,2)*Vscript(i,j,2,k,ie)
                   gVscript(i,j,2) = elem(ie)%Dinv(i,j,2,1)*Vscript(i,j,1,k,ie) + &
                        elem(ie)%Dinv(i,j,2,2)*Vscript(i,j,2,k,ie)

                   gVscript(i,j,1) = elem(ie)%metdet(i,j)*gVscript(i,j,1)
                   gVscript(i,j,2) = elem(ie)%metdet(i,j)*gVscript(i,j,2)

                end do
             end do

             div_Vscript(:,:,k) = divergence(gVscript,deriv)*rrearth

          end do

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,l)
#endif
          do k=1,nlev

             do j=1,np
                do i=1,np
                   C(i,j,k,ie) = elem(ie)%metdet(i,j)*B(i,j,k,ie)
                end do
             end do

             do l=1,nlev
                do j=1,np
                   do i=1,np
                      C(i,j,k,ie) = C(i,j,k,ie) - dt*Amat(l,k)*div_Vscript(i,j,l)
                   end do
                end do
             end do

          end do

          ! ===============================================================
          !  Weight C (the RHS of the helmholtz problem) by the mass matrix
          ! ===============================================================

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j)
#endif
          do k=1,nlev
             do j=1,np
                do i=1,np
                   C(i,j,k,ie) = elem(ie)%mp(i,j)*C(i,j,k,ie)
                end do
             end do
          end do

          ! ===================================
          ! Pack C into the edge1 buffer
          ! ===================================

          kptr=0
          call edgeVpack(edge1,C(1,1,1,ie),nlev,kptr,ie)

       end do

       ! ==================================
       ! boundary exchange C
       ! ==================================

       call t_startf('pasi_bexchV2')
       call bndry_exchangeV(cg%hybrid,edge1)
       call t_stopf('pasi_bexchV2')

       do ie=nets,nete

          ! ===================================
          ! Unpack C from the edge1 buffer
          ! ===================================

          kptr=0
          call edgeVunpack(edge1, C(1,1,1,ie), nlev, kptr, ie)

          ! ===============================================
          ! Complete global assembly by normalizing by rmp
          ! ===============================================

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j)
#endif
          do k=1,nlev

             do j=1,np
                do i=1,np
                   D(i,j,k,ie) = elem(ie)%rmp(i,j)*C(i,j,k,ie)
                end do
             end do

          end do

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,l)
#endif
          do k=1,nlev

             do j=1,np
                do i=1,np
                   C(i,j,k,ie) = 0.0_real_kind
                end do
             end do

             do l=1,nlev
                do j=1,np
                   do i=1,np
                      C(i,j,k,ie) = C(i,j,k,ie) + Emat_inv(l,k)*D(i,j,l,ie)
                   end do
                end do
             end do

          end do

       end do
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif
       ! ==========================================
       ! solve for Gamma_ref, given C as RHS input
       ! ==========================================

       Gamma_ref = pcg_solver(elem, &
            C,          &
            cg,         &
            red,        &
            edge1,      &
            edge2,      &
            Lambda,     &
            deriv,      &
            nets,       &
            nete,       &
            blkjac)

       ! ================================================================
       ! Backsubstitute Gamma_ref into semi-implicit system of equations
       ! to find prognostic variables at time level n+1
       ! ================================================================

       kptr=0
       do ie = nets, nete

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,l)
#endif
          do k=1,nlev

             do j=1,np
                do i=1,np
                   Gref(i,j,k,ie) = 0.0_real_kind
                end do
             end do

             do l=1,nlev
                do j=1,np
                   do i=1,np
                      Gref(i,j,k,ie) = Gref(i,j,k,ie) + Emat(l,k)*Gamma_ref(i,j,l,ie)
                   end do
                end do
             end do

             do j=1,np
                do i=1,np
                   B(i,j,k,ie) = elem(ie)%mp(i,j) * dt * (B(i,j,k,ie) - Gref(i,j,k,ie))
                end do
             end do

          end do

          call edgeVpack(edge1,B(:,:,:,ie),nlev,kptr,ie)

       end do

       call t_startf('pasi_bexchV3')
       call bndry_exchangeV(cg%hybrid,edge1)
       call t_stopf('pasi_bexchV3')

       do ie = nets, nete

          kptr=0
          call edgeVunpack(edge1, B(:,:,:,ie), nlev, kptr, ie)
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j)
#endif
          do k=1,nlev

             do j=1,np
                do i=1,np
                   B(i,j,k,ie) = elem(ie)%rmp(i,j)*B(i,j,k,ie)
                end do
             end do

          end do

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,l)
#endif
          do k=1,nlev

             do j=1,np
                do i=1,np
                   D(i,j,k,ie) = 0.0_real_kind
                end do
             end do

             do l=1,nlev
                do j=1,np
                   do i=1,np
                      D(i,j,k,ie) = D(i,j,k,ie) + Emat_inv(l,k)*B(i,j,l,ie)
                   end do
                end do
             end do

          end do

#if 1
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,l)
#endif
          do k=1,nlev

             do j=1,np
                do i=1,np
                   elem(ie)%derived%div(i,j,k,np1) = 0.0_real_kind
                end do
             end do

             do l=1,nlev
                do j=1,np
                   do i=1,np
                      elem(ie)%derived%div(i,j,k,np1) = elem(ie)%derived%div(i,j,k,np1) + Emat(l,k)*D(i,j,l,ie)/Lambda(l)
                   end do
                end do
             end do

          end do
#endif

          do k=1,nlev

             grad_Gref(:,:,:,k)=gradient_wk(Gref(:,:,k,ie),deriv)*rrearth

             do j=1,np
                do i=1,np
                   gGr1 = grad_Gref(i,j,1,k)
                   gGr2 = grad_Gref(i,j,2,k)
                   elem(ie)%state%v(i,j,1,k,np1) = elem(ie)%Dinv(i,j,1,1)*gGr1 + elem(ie)%Dinv(i,j,2,1)*gGr2
                   elem(ie)%state%v(i,j,2,k,np1) = elem(ie)%Dinv(i,j,1,2)*gGr1 + elem(ie)%Dinv(i,j,2,2)*gGr2
                   Vtemp(i,j,1,k,ie) = elem(ie)%state%v(i,j,1,k,np1)
                   Vtemp(i,j,2,k,ie) = elem(ie)%state%v(i,j,2,k,np1)
                end do
             end do

             do j=1,np
                do i=1,np
                   Pscript(i,j,ie) = Pscript(i,j,ie) - dt*Pvec(k)*elem(ie)%derived%div(i,j,k,np1)
                end do
             end do


             do l=1,nlev
                do j=1,np
                   do i=1,np
                      Tscript(i,j,k,ie) = Tscript(i,j,k,ie) - dt*Tmat(l,k)*elem(ie)%derived%div(i,j,l,np1)
                   end do
                end do
             end do

          end do

          do j=1,np
             do i=1,np
                Pscript(i,j,ie) = elem(ie)%mp(i,j)*Pscript(i,j,ie)
             end do
          end do
          do k=1,nlev
             do j=1,np
                do i=1,np
                   Tscript(i,j,k,ie) = elem(ie)%mp(i,j)*Tscript(i,j,k,ie)
                end do
             end do
          end do

          ! ===============================================
          ! Pack v at time level n+1 into the edge3p1 buffer
          ! ===============================================

          kptr=0
!          call edgeVpack(edge3p1, elem(ie)%state%v(:,:,:,:,np1),2*nlev,kptr,ie)
          call edgeVpack(edge3p1, Vtemp(:,:,:,:,ie),2*nlev,kptr,ie)

          kptr=2*nlev
          call edgeVpack(edge3p1, Tscript(:,:,:,ie),nlev,kptr,ie)

          kptr=3*nlev
          call edgeVpack(edge3p1, Pscript(:,:,ie),1,kptr,ie)

       end do

       ! ======================================
       ! boundary exchange v at time level n+1
       ! ======================================

       call t_startf('pasi_bexchV4')
       call bndry_exchangeV(cg%hybrid,edge3p1)
       call t_stopf('pasi_bexchV4')

!KGEN START(prim_advance_si_bug1)
       do ie=nets,nete

          ! ===================================
          ! Unpack v from the edge2 buffer
          ! ===================================

          kptr=0
          call edgeVunpack(edge3p1, Vtemp(:,:,:,:,ie), 2*nlev, kptr, ie)
!JMD          call edgeVunpack(edge3p1, elem(ie)%state%v(:,:,:,:,np1), 2*nlev, kptr, ie)
!JMD          elem(ie)%state%v(:,:,:,:,np1) = Vtemp(:,:,:,:,ie)


          kptr=2*nlev
          call edgeVunpack(edge3p1, Tscript(:,:,:,ie), nlev, kptr, ie)

          kptr=3*nlev
          call edgeVunpack(edge3p1, Pscript(:,:,ie), 1, kptr, ie)

          ! ==========================================================
          ! Complete global assembly by normalizing velocity by rmp
          ! Vscript = Vscript - dt*grad(Gref)
          ! ==========================================================
          if(iam==1) then
!BUG  There appears to be a bug in the Intel 15.0.1 compiler that generates
!BUG  incorrect code for this loop if the following print * statement is removed.
             print *,'IAM: ',iam, ' prim_advance_si: after SUM(v(np1)) ',sum(elem(ie)%state%v(:,:,:,:,np1))
          endif

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j)
#endif
          do k=1,nlev

             do j=1,np
                do i=1,np
!                   elem(ie)%state%v(i,j,1,k,np1) = Vscript(i,j,1,k,ie) + dt*elem(ie)%rmp(i,j)*elem(ie)%state%v(i,j,1,k,np1)
!                   elem(ie)%state%v(i,j,2,k,np1) = Vscript(i,j,2,k,ie) + dt*elem(ie)%rmp(i,j)*elem(ie)%state%v(i,j,2,k,np1)
                   elem(ie)%state%v(i,j,1,k,np1) = Vscript(i,j,1,k,ie) + dt*elem(ie)%rmp(i,j)*Vtemp(i,j,1,k,ie)
                   elem(ie)%state%v(i,j,2,k,np1) = Vscript(i,j,2,k,ie) + dt*elem(ie)%rmp(i,j)*Vtemp(i,j,2,k,ie)
                   elem(ie)%state%T(i,j,k,np1)   = elem(ie)%rmp(i,j)*Tscript(i,j,k,ie)
                end do
             end do

          end do

          do j=1,np
             do i=1,np
                elem(ie)%state%lnps(i,j,np1) = elem(ie)%rmp(i,j)*Pscript(i,j,ie)
             end do
          end do

       end do
!KGEN END(prim_advance_si_bug1)
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif

       call prim_diffusion(elem, nets,nete,np1,deriv,dt2,cg%hybrid)
       call t_stopf('prim_advance_si')
!pw    call t_adj_detailf(-1)
#endif
  end subroutine prim_advance_si


  subroutine preq_robert3(nm1,n0,np1,elem,hvcoord,nets,nete)
  use dimensions_mod, only : np, nlev, qsize
  use hybvcoord_mod, only : hvcoord_t
  use element_mod, only : element_t
  use time_mod, only: smooth
  use control_mod, only : integration

  implicit none
  integer              , intent(in) :: nm1,n0,np1,nets,nete
  type (hvcoord_t), intent(in)      :: hvcoord
  type (element_t)     , intent(inout) :: elem(:)


  integer :: i,j,k,ie,q
  real (kind=real_kind) :: dp
  logical :: filter_ps = .false.
  if (integration == "explicit") filter_ps = .true.

!pw  call t_adj_detailf(+1)
  call t_startf('preq_robert')
  do ie=nets,nete
     if (filter_ps) then
        elem(ie)%state%ps_v(:,:,n0) = elem(ie)%state%ps_v(:,:,n0) + smooth*(elem(ie)%state%ps_v(:,:,nm1) &
             - 2.0D0*elem(ie)%state%ps_v(:,:,n0)   + elem(ie)%state%ps_v(:,:,np1))
        elem(ie)%state%lnps(:,:,n0) = LOG(elem(ie)%state%ps_v(:,:,n0))
     else
        elem(ie)%state%lnps(:,:,n0) = elem(ie)%state%lnps(:,:,n0) + smooth*(elem(ie)%state%lnps(:,:,nm1) &
             - 2.0D0*elem(ie)%state%lnps(:,:,n0)   + elem(ie)%state%lnps(:,:,np1))
        elem(ie)%state%ps_v(:,:,n0) = EXP(elem(ie)%state%lnps(:,:,n0))
     endif

     elem(ie)%state%T(:,:,:,n0) = elem(ie)%state%T(:,:,:,n0) + smooth*(elem(ie)%state%T(:,:,:,nm1) &
          - 2.0D0*elem(ie)%state%T(:,:,:,n0)   + elem(ie)%state%T(:,:,:,np1))
     elem(ie)%state%v(:,:,:,:,n0) = elem(ie)%state%v(:,:,:,:,n0) + smooth*(elem(ie)%state%v(:,:,:,:,nm1) &
          - 2.0D0*elem(ie)%state%v(:,:,:,:,n0) + elem(ie)%state%v(:,:,:,:,np1))

  end do
  call t_stopf('preq_robert')
!pw  call t_adj_detailf(-1)

  end subroutine preq_robert3




  subroutine applyCAMforcing(elem,fvm,hvcoord,np1,np1_qdp,dt_q,nets,nete)

  use dimensions_mod, only: np, nc, nlev, qsize, ntrac
  use control_mod,    only: moisture, tracer_grid_type
  use control_mod,    only: TRACER_GRIDTYPE_GLL, TRACER_GRIDTYPE_FVM
  use physical_constants, only: Cp
  use fvm_control_volume_mod, only : fvm_struct, n0_fvm

  implicit none
  type (element_t),       intent(inout) :: elem(:)
  type(fvm_struct),       intent(inout) :: fvm(:)
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
     !
     ! even when running fvm tracers we need to updates forcing on ps and qv on GLL grid
     !
     ! for fvm tracer qsize is usually 1 (qv)
     !
     do q=1,qsize
        do k=1,nlev
           do j=1,np
              do i=1,np
                 v1 = dt_q*elem(ie)%derived%FQ(i,j,k,q,1)
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
     ! Repeat for the fvm tracers
     do q = 1, ntrac
        do k = 1, nlev
           do j = 1, nc
              do i = 1, nc
                 v1 = fvm(ie)%fc(i,j,k,q)
                 if (fvm(ie)%c(i,j,k,q,n0_fvm) + v1 < 0 .and. v1<0) then
                    if (fvm(ie)%c(i,j,k,q,n0_fvm) < 0 ) then
                       v1 = 0  ! C already negative, dont make it more so
                    else
                       v1 = -fvm(ie)%c(i,j,k,q,n0_fvm)
                    end if
                 end if
                 fvm(ie)%c(i,j,k,q,np1_qdp) = fvm(ie)%c(i,j,k,q,n0_fvm) + v1
                 !                    if (q == 1) then
                 !!XXgoldyXX: Should update the pressure forcing here??!!??
                 !                    elem(ie)%derived%FQps(i,j,1)=elem(ie)%derived%FQps(i,j,1)+v1/dt_q
                 !                  end if
              end do
           end do
        end do
     end do

     if (wet .and. qsize>0) then
        ! to conserve dry mass in the precese of Q1 forcing:
        elem(ie)%state%ps_v(:,:,np1) = elem(ie)%state%ps_v(:,:,np1) + &
             dt_q*elem(ie)%derived%FQps(:,:,1)
     endif

#if 0
     ! disabled - energy fixers will be moving into CAM physics
     ! energy fixer for FQps term
     ! dp1 = dp0 + d(FQps)
     ! dp0-dp1 = -d(FQps)
     ! E0-E1 = sum( dp0*ED) - sum( dp1*ED) = sum( dp0-dp1) * ED )
     ! compute E0-E1
     E0=0
     do k=1,nlev
        ED(:,:) = ( 0.5d0* &
             (elem(ie)%state%v(:,:,1,k,np1)**2 + elem(ie)%state%v(:,:,2,k,np1)**2)&
             + cp*elem(ie)%state%T(:,:,k,np1)  &
             + elem(ie)%state%phis(:,:) )

        dp0m1(:,:) = -dt_q*( hvcoord%hybi(k+1) - hvcoord%hybi(k) )*elem(ie)%derived%FQps(:,:,1)

        E0(:,:) = E0(:,:) + dp0m1(:,:)*ED(:,:)
     enddo
     ! energy fixer:
     ! Tnew = T + beta
     ! cp*dp*beta  = E0-E1   beta = (E0-E1)/(cp*sum(dp))

     dpsum(:,:) = ( hvcoord%hyai(nlev+1) - hvcoord%hyai(1) )*hvcoord%ps0 + &
          ( hvcoord%hybi(nlev+1) - hvcoord%hybi(1) )*elem(ie)%state%ps_v(:,:,np1)

     beta(:,:)=E0(:,:)/(dpsum(:,:)*cp)
     do k=1,nlev
        elem(ie)%state%T(:,:,k,np1)=elem(ie)%state%T(:,:,k,np1)+beta(:,:)
     enddo
#endif

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

     elem(ie)%state%T(:,:,:,np1)   = elem(ie)%state%T(:,:,:,np1)   + dt_q*elem(ie)%derived%FT(:,:,:,1)
     elem(ie)%state%v(:,:,:,:,np1) = elem(ie)%state%v(:,:,:,:,np1) + dt_q*elem(ie)%derived%FM(:,:,:,:,1)

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
      elem(ie)%state%T(:,:,:,np1)  = elem(ie)%state%T(:,:,:,np1)    + dt_q*elem(ie)%derived%FT(:,:,:,1)
      elem(ie)%state%v(:,:,:,:,np1) = elem(ie)%state%v(:,:,:,:,np1) + dt_q*elem(ie)%derived%FM(:,:,:,:,1)
    enddo
  end subroutine applyCAMforcing_dynamics

  subroutine smooth_phis(phis,elem,hybrid,deriv,nets,nete,minf,numcycle)
  use dimensions_mod, only : np, np, nlev
  use control_mod, only : smooth_phis_nudt, hypervis_scaling
  use hybrid_mod, only : hybrid_t
  use edge_mod, only : edgevpack, edgevunpack, edgevunpackmax, edgevunpackmin
  use edgetype_mod, only : EdgeBuffer_t
  use bndry_mod, only : bndry_exchangev
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

  subroutine overwrite_SEdensity(elem, fvm, dt_q, hybrid,nets,nete, np1)

    use fvm_reconstruction_mod, only: reconstruction
    use fvm_filter_mod, only: monotonic_gradient_cart, recons_val_cart
    use dimensions_mod, only : np, nlev, nc,nhe
    use hybrid_mod, only : hybrid_t
    use edge_mod, only : edgevpack, edgevunpack, edgevunpackmax, edgevunpackmin
    use bndry_mod, only : bndry_exchangev
    use element_mod, only : element_t
    use derivative_mod, only : derivative_t , laplace_sphere_wk
    use time_mod, only : TimeLevel_t
    use fvm_control_volume_mod, only : fvm_struct

    type(element_t) , intent(inout) :: elem(:)
    type(fvm_struct), intent(inout) :: fvm(:)
    type(hybrid_t),   intent(in)    :: hybrid ! distributed parallel structure (shared)
    integer,          intent(in)    :: nets   ! starting thread element number (private)
    integer,          intent(in)    :: nete   ! ending thread element number   (private)
    integer,          intent(in)    :: np1
    integer :: ie, k

    real (kind=real_kind)             :: xp,yp, tmpval, dt_q
    integer                           :: i, j,ix, jy, starti,endi,tmpi

    real (kind=real_kind), dimension(5,1-nhe:nc+nhe,1-nhe:nc+nhe)      :: recons

    if ((nc .ne. 4) .or. (np .ne. 4)) then
      if(hybrid%masterthread) then
        print *,"You are in OVERWRITE SE AIR DENSITY MODE"
        print *,"This only works for nc=4 and np=4"
        print *,"Write a new search algorithm or pay $10000!"
      endif
      stop
    endif
#if defined(_FVM)
    do ie=nets,nete
      call reconstruction(fvm(ie)%psc, fvm(ie),recons)
      call monotonic_gradient_cart(fvm(ie)%psc, fvm(ie),recons, elem(ie)%desc)
      do j=1,np
        do i=1,np
          xp=tan(elem(ie)%cartp(i,j)%x)
          yp=tan(elem(ie)%cartp(i,j)%y)
          ix=i
          jy=j
          ! Search index along "x"  (bisection method)
!           starti = 1
!           endi = nc+1
!           do
!              if  ((endi-starti) <=  1)  exit
!              tmpi = (endi + starti)/2
!              if (xp  >  fvm%acartx(tmpi)) then
!                 starti = tmpi
!              else
!                 endi = tmpi
!              endif
!           enddo
!           ix = starti
!
!         ! Search index along "y"
!           starti = 1
!           endi = nc+1
!           do
!              if  ((endi-starti) <=  1)  exit
!              tmpi = (endi + starti)/2
!              if (yp  >  fvm%acarty(tmpi)) then
!                 starti = tmpi
!              else
!                 endi = tmpi
!              endif
!           enddo
!           jy = starti

          call recons_val_cart(fvm(ie)%psc, xp,yp,fvm(ie)%spherecentroid,recons,ix,jy,tmpval)
          elem(ie)%state%ps_v(i,j,np1)= elem(ie)%state%ps_v(i,j,np1) +&
               dt_q*(tmpval - elem(ie)%state%ps_v(i,j,np1) )/(7*24*60*60)
        end do
      end do
      elem(ie)%state%ps_v(:,:,np1)=elem(ie)%state%ps_v(:,:,np1)*elem(ie)%spheremp(:,:)
     call edgeVpack(edge3p1,elem(ie)%state%ps_v(:,:,np1),1,0,ie)
  enddo

  call t_startf('overwr_SEdens_bexchV')
  call bndry_exchangeV(hybrid,edge3p1)
  call t_stopf('overwr_SEdens_bexchV')

  do ie=nets,nete
     call edgeVunpack(edge3p1, elem(ie)%state%ps_v(:,:,np1), 1, 0, ie)
     elem(ie)%state%ps_v(:,:,np1)=elem(ie)%state%ps_v(:,:,np1)*elem(ie)%rspheremp(:,:)
  enddo
#endif
  end subroutine overwrite_SEdensity

end module prim_advance_mod
