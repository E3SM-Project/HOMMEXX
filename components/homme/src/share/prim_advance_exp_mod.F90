
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!#define _DBG_ print *,"File:",__FILE__," at ",__LINE__
!#define _DBG_ !DBG
!
!
module prim_advance_exp_mod

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

  implicit none
  private
  save

  real (kind=real_kind), allocatable :: ur_weights(:)

  public :: ur_weights, prim_advance_exp

  contains

  subroutine prim_advance_exp(elem, deriv, hvcoord, hybrid,dt, tl,  nets, nete, compute_diagnostics)

    use bndry_mod,      only: bndry_exchangev
    use control_mod,    only: prescribed_wind, qsplit, tstep_type, rsplit, qsplit, moisture, integration
    use edge_mod,       only: edgevpack, edgevunpack, initEdgeBuffer
    use edgetype_mod,   only: EdgeBuffer_t
    use reduction_mod,  only: reductionbuffer_ordered_1d_t
    use time_mod,       only: timelevel_qdp, tevolve
    use diffusion_mod,  only: prim_diffusion
    use prim_advance_caar_mod, only: compute_and_apply_rhs, edge3p1
    use prim_advance_hypervis_mod, only: advance_hypervis_dp, advance_hypervis_lf

#ifdef USE_KOKKOS_KERNELS
    use iso_c_binding,  only: c_ptr, c_loc
    use element_mod,    only: elem_state_v, elem_state_temp, elem_state_dp3d
    use element_mod,    only: elem_derived_phi, elem_derived_pecnd
    use element_mod,    only: elem_derived_omega_p, elem_derived_vn0
    use element_mod,    only: elem_derived_eta_dot_dpdn, elem_state_Qdp
#endif

#ifdef TRILINOS
    use prim_derived_type_mod ,only : derived_type, initialize
    use, intrinsic :: iso_c_binding
#endif

#ifdef CAM
    use control_mod,    only: prescribed_vertwind
#else
    use asp_tests,      only: asp_advection_vertical
#endif

    implicit none

    type (element_t),      intent(inout), target :: elem(:)
    type (derivative_t),   intent(in)            :: deriv
    type (hvcoord_t),                     target :: hvcoord
    type (hybrid_t),       intent(in)            :: hybrid
    real (kind=real_kind), intent(in)            :: dt
    type (TimeLevel_t)   , intent(in)            :: tl
    integer              , intent(in)            :: nets
    integer              , intent(in)            :: nete
    logical,               intent(in)            :: compute_diagnostics

    real (kind=real_kind) ::  dt2, time, dt_vis, x, eta_ave_w
    real (kind=real_kind) ::  tempdp3d(np,np)
    real (kind=real_kind) ::  tempmass(nc,nc)
    real (kind=real_kind) ::  tempflux(nc,nc,4)
    !real (kind=real_kind) ::  dn

    integer :: ie,nm1,n0,np1,nstep,method,qsplit_stage,k, qn0
    integer :: n,i,j,lx,lenx

#ifdef USE_KOKKOS_KERNELS
    type (c_ptr) :: elem_state_v_ptr, elem_state_t_ptr, elem_state_dp3d_ptr
    type (c_ptr) :: elem_derived_phi_ptr, elem_derived_pecnd_ptr
    type (c_ptr) :: elem_derived_omega_p_ptr, elem_derived_vn0_ptr
    type (c_ptr) :: elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr
    type (c_ptr) :: hvcoord_a_ptr, hvcoord_b_ptr
#endif

#ifdef TRILINOS
    real (c_double) ,allocatable, dimension(:) :: xstate(:)

    ! state_object is a derived data type passed thru noxinit as a pointer
    type(derived_type) ,target         :: state_object
    type(derived_type) ,pointer        :: fptr=>NULL()
    type(c_ptr)                        :: c_ptr_to_object
    type(derived_type) ,target         :: pre_object
    type(derived_type) ,pointer         :: pptr=>NULL()
    type(c_ptr)                        :: c_ptr_to_pre
    type(derived_type) ,target         :: jac_object
    type(derived_type) ,pointer         :: jptr=>NULL()
    type(c_ptr)                        :: c_ptr_to_jac

    integer(c_int) :: ierr = 0
#endif

    interface
#ifdef USE_KOKKOS_KERNELS
      subroutine init_control_caar_c (nets,nete,nelemd,qn0,ps0, &
                                      rsplit,hybrid_a_ptr,hybrid_b_ptr) bind(c)
        use iso_c_binding , only : c_ptr, c_int, c_double
        !
        ! Inputs
        !
        integer (kind=c_int),  intent(in) :: qn0,nets,nete,nelemd,rsplit
        real (kind=c_double),  intent(in) :: ps0
        type (c_ptr),          intent(in) :: hybrid_a_ptr
        type (c_ptr),          intent(in) :: hybrid_b_ptr
      end subroutine init_control_caar_c

      subroutine caar_pull_data_c (elem_state_v_ptr, elem_state_t_ptr, elem_state_dp3d_ptr, &
                                   elem_derived_phi_ptr, elem_derived_pecnd_ptr,            &
                                   elem_derived_omega_p_ptr, elem_derived_vn0_ptr,          &
                                   elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr) bind(c)
        use iso_c_binding , only : c_ptr
        !
        ! Inputs
        !
        type (c_ptr), intent(in) :: elem_state_v_ptr, elem_state_t_ptr, elem_state_dp3d_ptr
        type (c_ptr), intent(in) :: elem_derived_phi_ptr, elem_derived_pecnd_ptr
        type (c_ptr), intent(in) :: elem_derived_omega_p_ptr, elem_derived_vn0_ptr
        type (c_ptr), intent(in) :: elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr
      end subroutine caar_pull_data_c

      subroutine caar_push_results_c (elem_state_v_ptr, elem_state_t_ptr, elem_state_dp3d_ptr, &
                                      elem_derived_phi_ptr, elem_derived_pecnd_ptr,            &
                                      elem_derived_omega_p_ptr, elem_derived_vn0_ptr,          &
                                      elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr) bind(c)
        use iso_c_binding , only : c_ptr
        !
        ! Inputs
        !
        type (c_ptr), intent(in) :: elem_state_v_ptr, elem_state_t_ptr, elem_state_dp3d_ptr
        type (c_ptr), intent(in) :: elem_derived_phi_ptr, elem_derived_pecnd_ptr
        type (c_ptr), intent(in) :: elem_derived_omega_p_ptr, elem_derived_vn0_ptr
        type (c_ptr), intent(in) :: elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr
      end subroutine caar_push_results_c

      subroutine u3_5stage_timestep_c(nm1,n0,np1,dt,eta_ave_w,compute_diagnostics) bind(c)
        use iso_c_binding , only : c_int, c_double
        !
        ! Inputs
        !
        integer (kind=c_int),  intent(in) :: np1,nm1,n0
        logical,               intent(in) :: compute_diagnostics
        real (kind=c_double),  intent(in) :: dt, eta_ave_w
      end subroutine u3_5stage_timestep_c

      subroutine pull_hypervis_data_c(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr) bind(c)
        use iso_c_binding , only : c_ptr
        !
        ! Inputs
        !
        type (c_ptr), intent(in) :: elem_state_t_ptr, elem_state_dp3d_ptr, elem_state_v_ptr
      end subroutine pull_hypervis_data_c

      subroutine advance_hypervis_dp_c(np1,nets,nete,dt,eta_ave_w) bind(c)
        use iso_c_binding, only : c_int, c_double
        !
        ! Inputs
        !
        integer (kind=c_int),  intent(in) :: np1,nets,nete
        real (kind=c_double),  intent(in) :: dt, eta_ave_w
      end subroutine advance_hypervis_dp_c

      subroutine push_hypervis_results_c(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr) bind(c)
        use iso_c_binding , only : c_ptr
        !
        ! Inputs
        !
        type (c_ptr), intent(in) :: elem_state_t_ptr, elem_state_dp3d_ptr, elem_state_v_ptr
      end subroutine push_hypervis_results_c
#endif

#ifdef TRILINOS
      subroutine noxsolve(vectorSize,vector,v_container,p_container,j_container,ierr) &
        bind(C,name='noxsolve')
        use ,intrinsic :: iso_c_binding
        integer(c_int)                :: vectorSize
        real(c_double)  ,dimension(*) :: vector
        type(c_ptr)                   :: v_container
        type(c_ptr)                   :: p_container  !precon ptr
        type(c_ptr)                   :: j_container  !analytic jacobian ptr
        integer(c_int)                :: ierr         !error flag
      end subroutine noxsolve
#endif

    end interface

    call t_startf('prim_advance_exp')
    nm1   = tl%nm1
    n0    = tl%n0
    np1   = tl%np1
    nstep = tl%nstep

    ! get timelevel for accessing tracer mass Qdp() to compute virtual temperature

    qn0 = -1    ! -1 = disabled (assume dry dynamics)
    if ( moisture /= "dry") then
      call TimeLevel_Qdp(tl, qsplit, qn0)  ! compute current Qdp() timelevel
    endif

    ! get time integration method for this timestep

    method    = tstep_type  ! set default integration method
    eta_ave_w = 1d0/qsplit  ! set default vertical flux weight

    if(tstep_type==0)then
      ! 0: use leapfrog, but RK2 on first step
      if (nstep==0) method=1

    else if (tstep_type==1) then
      ! 1: use leapfrog, but RK2 on first qsplit stage
      method = 0
      qsplit_stage = mod(nstep,qsplit)    ! get qsplit stage
      if (qsplit_stage==0) method=1       ! use RK2 on first stage
      eta_ave_w=ur_weights(qsplit_stage+1)! RK2 + LF scheme has tricky weights
    endif

#ifndef CAM
    ! if "prescribed wind" set dynamics explicitly and skip time-integration
    if (prescribed_wind ==1 ) then
      call set_prescribed_wind(elem,deriv,hybrid,hvcoord,dt,tl,nets,nete,eta_ave_w)
      call t_stopf('prim_advance_exp')
      return
    endif
#endif

    ! integration = "explicit"
    !
    !   tstep_type=0  pure leapfrog except for very first timestep   CFL=1
    !                    typically requires qsplit=4 or 5
    !   tstep_type=1  RK2 followed by qsplit-1 leapfrog steps        CFL=close to qsplit
    !                    typically requires qsplit=4 or 5
    !   tstep_type=2  RK2-SSP 3 stage (as used by tracers)           CFL=.58
    !                    optimal in terms of SSP CFL, but not        CFLSSP=2
    !                    optimal in terms of CFL
    !                    typically requires qsplit=3
    !                    but if windspeed > 340m/s, could use this
    !                    with qsplit=1
    !   tstep_type=3  classic RK3                                    CFL=1.73 (sqrt(3))
    !
    !   tstep_type=4  Kinnmark&Gray RK4 4 stage                      CFL=sqrt(8)=2.8
    !                 should we replace by standard RK4 (CFL=sqrt(8))?
    !                 (K&G 1st order method has CFL=3)
    !   tstep_type=5  Kinnmark&Gray RK3 5 stage 3rd order            CFL=3.87  (sqrt(15))
    !                 From Paul Ullrich.  3rd order for nonlinear terms also
    !                 K&G method is only 3rd order for linear
    !                 optimal: for windspeeds ~120m/s,gravity: 340m/2
    !                 run with qsplit=1
    !                 (K&G 2nd order method has CFL=4. tiny CFL improvement not worth 2nd order)
    !
    ! integration = "full_imp"
    !
    !   tstep_type=1  Backward Euler or BDF2 implicit dynamics
    !

    ! ==================================
    ! Take timestep
    ! ==================================

    dt_vis = dt
    if (method==0) then
      ! regular LF step
      dt2 = 2*dt
      call t_startf("LF_timestep")
      call compute_and_apply_rhs(np1,nm1,n0,qn0,dt2,elem,hvcoord,hybrid,&
           deriv,nets,nete,compute_diagnostics,eta_ave_w)
      call t_stopf("LF_timestep")
      dt_vis = dt2  ! dt to use for time-split dissipation
    else if (method==1) then
      ! RK2
      ! forward euler to u(dt/2) = u(0) + (dt/2) RHS(0)  (store in u(np1))
      call t_startf("RK2_timestep")
      call compute_and_apply_rhs(np1,n0,n0,qn0,dt/2,elem,hvcoord,hybrid,&
           deriv,nets,nete,compute_diagnostics,0d0)
      ! leapfrog:  u(dt) = u(0) + dt RHS(dt/2)     (store in u(np1))
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,eta_ave_w)
      call t_stopf("RK2_timestep")
    else if (method==2) then
      ! RK2-SSP 3 stage.  matches tracer scheme. optimal SSP CFL, but
      ! not optimal for regular CFL
      ! u1 = u0 + dt/2 RHS(u0)
      call t_startf("RK2-SSP3_timestep")
      call compute_and_apply_rhs(np1,n0,n0,qn0,dt/2,elem,hvcoord,hybrid,&
           deriv,nets,nete,compute_diagnostics,eta_ave_w/3)
      ! u2 = u1 + dt/2 RHS(u1)
      call compute_and_apply_rhs(np1,np1,np1,qn0,dt/2,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,eta_ave_w/3)
      ! u3 = u2 + dt/2 RHS(u2)
      call compute_and_apply_rhs(np1,np1,np1,qn0,dt/2,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,eta_ave_w/3)
      ! unew = u/3 +2*u3/3  = u + 1/3 (RHS(u) + RHS(u1) + RHS(u2))
      do ie=nets,nete
        elem(ie)%state%v(:,:,:,:,np1)= elem(ie)%state%v(:,:,:,:,n0)/3 &
             + 2*elem(ie)%state%v(:,:,:,:,np1)/3
        elem(ie)%state%T(:,:,:,np1)= elem(ie)%state%T(:,:,:,n0)/3 &
             + 2*elem(ie)%state%T(:,:,:,np1)/3
        elem(ie)%state%dp3d(:,:,:,np1)= elem(ie)%state%dp3d(:,:,:,n0)/3 &
             + 2*elem(ie)%state%dp3d(:,:,:,np1)/3
      enddo
      call t_stopf("RK2-SSP3_timestep")
    else if (method==3) then
      ! classic RK3  CFL=sqrt(3)
      ! u1 = u0 + dt/3 RHS(u0)
      call t_startf("RK3_timestep")
      call compute_and_apply_rhs(np1,n0,n0,qn0,dt/3,elem,hvcoord,hybrid,&
           deriv,nets,nete,compute_diagnostics,0d0)
      ! u2 = u0 + dt/2 RHS(u1)
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt/2,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,0d0)
      ! u3 = u0 + dt RHS(u2)
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,eta_ave_w)
      call t_stopf("RK3_timestep")
    else if (method==4) then
      ! KG 4th order 4 stage:   CFL=sqrt(8)
      ! low storage version of classic RK4
      ! u1 = u0 + dt/4 RHS(u0)
      call t_startf("RK4_timestep")
      call compute_and_apply_rhs(np1,n0,n0,qn0,dt/4,elem,hvcoord,hybrid,&
           deriv,nets,nete,compute_diagnostics,0d0)
      ! u2 = u0 + dt/3 RHS(u1)
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt/3,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,0d0)
      ! u3 = u0 + dt/2 RHS(u2)
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt/2,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,0d0)
      ! u4 = u0 + dt RHS(u3)
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,eta_ave_w)
      call t_stopf("RK4_timestep")
    else if (method==5) then
#if 0
      ! KG 3nd order 5 stage:   CFL=sqrt( 4^2 -1) = 3.87
      ! but nonlinearly only 2nd order
      ! u1 = u0 + dt/5 RHS(u0)
      call t_startf("KG3-5stage_timestep")
      call compute_and_apply_rhs(np1,n0,n0,qn0,dt/5,elem,hvcoord,hybrid,&
           deriv,nets,nete,compute_diagnostics,0d0)
      ! u2 = u0 + dt/5 RHS(u1)
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt/5,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,0d0)
      ! u3 = u0 + dt/3 RHS(u2)
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt/3,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,0d0)
      ! u4 = u0 + dt/2 RHS(u3)
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt/2,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,0d0)
      ! u5 = u0 + dt RHS(u4)
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,eta_ave_w)
      call t_stopf("KG3-5stage_timestep")
#else
#ifdef USE_KOKKOS_KERNELS
      call t_startf("caar_overhead")
      hvcoord_a_ptr             = c_loc(hvcoord%hyai)
      hvcoord_b_ptr             = c_loc(hvcoord%hybi)
      ! In F, elem range is [nets,nete]. In C, elem range is [nets,nete).
      ! Also, F has index base 1, C has index base 0.
      call init_control_caar_c(nets-1,nete,nelemd,qn0-1,hvcoord%ps0,rsplit,hvcoord_a_ptr,hvcoord_b_ptr)

      elem_state_v_ptr              = c_loc(elem_state_v)
      elem_state_t_ptr              = c_loc(elem_state_temp)
      elem_state_dp3d_ptr           = c_loc(elem_state_dp3d)
      elem_derived_phi_ptr          = c_loc(elem_derived_phi)
      elem_derived_pecnd_ptr        = c_loc(elem_derived_pecnd)
      elem_derived_omega_p_ptr      = c_loc(elem_derived_omega_p)
      elem_derived_vn0_ptr          = c_loc(elem_derived_vn0)
      elem_derived_eta_dot_dpdn_ptr = c_loc(elem_derived_eta_dot_dpdn)
      elem_state_Qdp_ptr            = c_loc(elem_state_Qdp)
      call caar_pull_data_c (elem_state_v_ptr, elem_state_t_ptr, elem_state_dp3d_ptr, &
                             elem_derived_phi_ptr, elem_derived_pecnd_ptr,            &
                             elem_derived_omega_p_ptr, elem_derived_vn0_ptr,          &
                             elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr)
      call t_stopf("caar_overhead")

      call t_startf("U3-5stage_timestep")
      call u3_5stage_timestep_c(nm1-1,n0-1,np1-1,dt,eta_ave_w,compute_diagnostics)
      call t_stopf("U3-5stage_timestep")

      call t_startf("caar_overhead")
      call caar_push_results_c (elem_state_v_ptr, elem_state_t_ptr, elem_state_dp3d_ptr, &
                                elem_derived_phi_ptr, elem_derived_pecnd_ptr,            &
                                elem_derived_omega_p_ptr, elem_derived_vn0_ptr,          &
                                elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr)
      call t_stopf("caar_overhead")
#else
      ! Ullrich 3nd order 5 stage:   CFL=sqrt( 4^2 -1) = 3.87
      ! u1 = u0 + dt/5 RHS(u0)  (save u1 in timelevel nm1)
      call t_startf("U3-5stage_timestep")
      call compute_and_apply_rhs(nm1,n0,n0,qn0,dt/5,elem,hvcoord,hybrid,&
           deriv,nets,nete,compute_diagnostics,eta_ave_w/4)
      ! u2 = u0 + dt/5 RHS(u1)
      call compute_and_apply_rhs(np1,n0,nm1,qn0,dt/5,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,0d0)
      ! u3 = u0 + dt/3 RHS(u2)
      call compute_and_apply_rhs(np1,n0,np1,qn0,dt/3,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,0d0)
      ! u4 = u0 + 2dt/3 RHS(u3)
      call compute_and_apply_rhs(np1,n0,np1,qn0,2*dt/3,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,0d0)

      ! compute (5*u1/4 - u0/4) in timelevel nm1:
      do ie=nets,nete
        elem(ie)%state%v(:,:,:,:,nm1)= (5*elem(ie)%state%v(:,:,:,:,nm1) &
             - elem(ie)%state%v(:,:,:,:,n0) ) /4
        elem(ie)%state%T(:,:,:,nm1)= (5*elem(ie)%state%T(:,:,:,nm1) &
             - elem(ie)%state%T(:,:,:,n0) )/4
        elem(ie)%state%dp3d(:,:,:,nm1)= (5*elem(ie)%state%dp3d(:,:,:,nm1) &
                - elem(ie)%state%dp3d(:,:,:,n0) )/4
      enddo
      ! u5 = (5*u1/4 - u0/4) + 3dt/4 RHS(u4)
      call compute_and_apply_rhs(np1,nm1,np1,qn0,3*dt/4,elem,hvcoord,hybrid,&
           deriv,nets,nete,.false.,3*eta_ave_w/4)
      ! final method is the same as:
      ! u5 = u0 +  dt/4 RHS(u0)) + 3dt/4 RHS(u4)
      call t_stopf("U3-5stage_timestep")
#endif
#endif

    else if ((method==11).or.(method==12)) then
      ! Fully implicit JFNK method (vertically langragian not active yet)
      if (rsplit > 0) then
        call abortmp('ERROR: full_imp integration not yet coded for vert lagrangian adv option')
      end if
  !    if (hybrid%masterthread) print*, "fully implicit integration is still under development"

#ifdef TRILINOS
      call t_startf("JFNK_imp_timestep")
      lenx=(np*np*nlev*3 + np*np*1)*(nete-nets+1)  ! 3 3d vars plus 1 2d vars
      allocate(xstate(lenx))
      xstate(:) = 0d0

      call initialize(state_object, method, elem, hvcoord, compute_diagnostics, &
        qn0, eta_ave_w, hybrid, deriv, dt, tl, nets, nete)

      call initialize(pre_object, method, elem, hvcoord, compute_diagnostics, &
        qn0, eta_ave_w, hybrid, deriv, dt, tl, nets, nete)

      call initialize(jac_object, method, elem, hvcoord, compute_diagnostics, &
        qn0, eta_ave_w, hybrid, deriv, dt, tl, nets, nete)

  !      pc_elem = elem
  !      jac_elem = elem

      fptr => state_object
      c_ptr_to_object =  c_loc(fptr)
      pptr => state_object
      c_ptr_to_pre =  c_loc(pptr)
      jptr => state_object
      c_ptr_to_jac =  c_loc(jptr)

  ! create flat state vector to pass through NOX
  ! use previous time step as the first guess for the new one (because with LF time level update n0=np1)

      np1 = n0

      lx = 1
      do ie=nets,nete
        do k=1,nlev
          do j=1,np
            do i=1,np
              xstate(lx) = elem(ie)%state%v(i,j,1,k,n0)
              lx = lx+1
            end do
          end do
        end do
      end do
      do ie=nets,nete
        do k=1,nlev
          do j=1,np
            do i=1,np
              xstate(lx) = elem(ie)%state%v(i,j,2,k,n0)
              lx = lx+1
            end do
          end do
        end do
      end do
      do ie=nets,nete
        do k=1,nlev
          do j=1,np
            do i=1,np
              xstate(lx) = elem(ie)%state%T(i,j,k,n0)
              lx = lx+1
            end do
          end do
        end do
      end do
      do ie=nets,nete
        do j=1,np
          do i=1,np
            xstate(lx) = elem(ie)%state%ps_v(i,j,n0)
            lx = lx+1
          end do
        end do
      end do

  ! activate these lines to test infrastructure and still solve with explicit code
  !       ! RK2
  !       ! forward euler to u(dt/2) = u(0) + (dt/2) RHS(0)  (store in u(np1))
  !       call compute_and_apply_rhs(np1,n0,n0,qn0,dt/2,elem,hvcoord,hybrid,&
  !            deriv,nets,nete,compute_diagnostics,0d0)
  !       ! leapfrog:  u(dt) = u(0) + dt RHS(dt/2)     (store in u(np1))
  !       call compute_and_apply_rhs(np1,n0,np1,qn0,dt,elem,hvcoord,hybrid,&
  !            deriv,nets,nete,.false.,eta_ave_w)

  ! interface to use nox and loca solver libraries using JFNK, and returns xstate(n+1)
      call noxsolve(size(xstate), xstate, c_ptr_to_object, c_ptr_to_pre, c_ptr_to_jac, ierr)

      if (ierr /= 0) call abortmp('Error in noxsolve: Newton failed to converge')

      call c_f_pointer(c_ptr_to_object, fptr) ! convert C ptr to F ptr
      elem = fptr%base

      lx = 1
      do ie=nets,nete
        do k=1,nlev
          do j=1,np
            do i=1,np
              elem(ie)%state%v(i,j,1,k,np1) = xstate(lx)
              lx = lx+1
            end do
          end do
        end do
      end do
      do ie=nets,nete
        do k=1,nlev
          do j=1,np
            do i=1,np
              elem(ie)%state%v(i,j,2,k,np1) = xstate(lx)
              lx = lx+1
            end do
          end do
        end do
      end do
      do ie=nets,nete
        do k=1,nlev
          do j=1,np
            do i=1,np
              elem(ie)%state%T(i,j,k,np1) = xstate(lx)
              lx = lx+1
            end do
          end do
        end do
      end do
      do ie=nets,nete
        do j=1,np
          do i=1,np
            elem(ie)%state%ps_v(i,j,np1) = xstate(lx)
            lx = lx+1
          end do
        end do
      end do
      call t_stopf("JFNK_imp_timestep")
#endif

    else
      call abortmp('ERROR: bad choice of tstep_type')
    endif

    ! call prim_printstate(elem,tl,hybrid,hvcoord,nets,nete)

    ! ==============================================
    ! Time-split Horizontal diffusion: nu.del^2 or nu.del^4
    ! U(*) = U(t+1)  + dt2 * HYPER_DIFF_TERM(t+1)
    ! ==============================================
#ifdef ENERGY_DIAGNOSTICS
    if (compute_diagnostics) then
      do ie = nets,nete
        elem(ie)%accum%DIFF(:,:,:,:)=elem(ie)%state%v(:,:,:,:,np1)
        elem(ie)%accum%DIFFT(:,:,:)=elem(ie)%state%T(:,:,:,np1)
      enddo
    endif
#endif

    ! note:time step computes u(t+1)= u(t*) + RHS.
    ! for consistency, dt_vis = t-1 - t*, so this is timestep method dependent
    if (tstep_type==0) then
      ! leapfrog special case
      call advance_hypervis_lf(edge3p1,elem,hvcoord,hybrid,deriv,nm1,n0,np1,nets,nete,dt_vis)
    else if (method<=10) then ! not implicit
      ! forward-in-time, hypervis applied to dp3d
#ifdef USE_KOKKOS_KERNELS
      call pull_hypervis_data_c(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr)
      call advance_hypervis_dp_c(np1,nets,nete,dt_vis,eta_ave_w)
      call push_hypervis_results_c(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr)
#else
      call advance_hypervis_dp(edge3p1,elem,hvcoord,hybrid,deriv,np1,nets,nete,dt_vis,eta_ave_w)
#endif
    endif

#ifdef ENERGY_DIAGNOSTICS
    if (compute_diagnostics) then
      do ie = nets,nete
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k)
#endif
        do k=1,nlev  !  Loop index added (AAM)
          elem(ie)%accum%DIFF(:,:,:,k)=( elem(ie)%state%v(:,:,:,k,np1) -&
               elem(ie)%accum%DIFF(:,:,:,k) ) / dt_vis
          elem(ie)%accum%DIFFT(:,:,k)=( elem(ie)%state%T(:,:,k,np1) -&
               elem(ie)%accum%DIFFT(:,:,k) ) / dt_vis
        enddo
      enddo
    endif
#endif

    tevolve=tevolve+dt

    call t_stopf('prim_advance_exp')
    !pw call t_adj_detailf(-1)
  end subroutine prim_advance_exp

#ifndef CAM
  subroutine set_prescribed_wind(elem,deriv,hybrid,hv,dt,tl,nets,nete,eta_ave_w)

    use test_mod,  only: set_test_prescribed_wind
    use asp_tests, only: asp_advection_vertical

    type (element_t),      intent(inout), target  :: elem(:)
    type (derivative_t),   intent(in)             :: deriv
    type (hvcoord_t),      intent(inout)          :: hv
    type (hybrid_t),       intent(in)             :: hybrid
    real (kind=real_kind), intent(in)             :: dt
    type (TimeLevel_t)   , intent(in)             :: tl
    integer              , intent(in)             :: nets
    integer              , intent(in)             :: nete
    real (kind=real_kind), intent(in)             :: eta_ave_w

    real (kind=real_kind) :: dp(np,np)! pressure thickness, vflux
    real(kind=real_kind)  :: time
    real(kind=real_kind)  :: eta_dot_dpdn(np,np,nlevp)

    integer :: ie,k,n0,np1

    time  = tl%nstep*dt
    n0    = tl%n0
    np1   = tl%np1

    call set_test_prescribed_wind(elem,deriv,hybrid,hv,dt,tl,nets,nete)

    ! accumulate velocities and fluxes over timesteps
    ! test code only dont bother to openmp thread
    do ie = nets,nete
       eta_dot_dpdn(:,:,:)=elem(ie)%derived%eta_dot_dpdn_prescribed(:,:,:)
       ! accumulate mean fluxes for advection
       if (rsplit==0) then
          elem(ie)%derived%eta_dot_dpdn(:,:,:) = &
               elem(ie)%derived%eta_dot_dpdn(:,:,:) +eta_dot_dpdn(:,:,:)*eta_ave_w
       else
          ! lagrangian case.  mean vertical velocity = 0
          elem(ie)%derived%eta_dot_dpdn(:,:,:) = 0
          ! update position of floating levels
          do k=1,nlev
             elem(ie)%state%dp3d(:,:,k,np1) = elem(ie)%state%dp3d(:,:,k,n0)  &
                  + dt*(eta_dot_dpdn(:,:,k+1) - eta_dot_dpdn(:,:,k))
          enddo
       end if
       ! accumulate U*dp
       do k=1,nlev
          elem(ie)%derived%vn0(:,:,1,k)=elem(ie)%derived%vn0(:,:,1,k)+&
eta_ave_w*elem(ie)%state%v(:,:,1,k,n0)*elem(ie)%state%dp3d(:,:,k,tl%n0)
          elem(ie)%derived%vn0(:,:,2,k)=elem(ie)%derived%vn0(:,:,2,k)+&
eta_ave_w*elem(ie)%state%v(:,:,2,k,n0)*elem(ie)%state%dp3d(:,:,k,tl%n0)
       enddo
    end do

  end subroutine
#endif

end module prim_advance_exp_mod
