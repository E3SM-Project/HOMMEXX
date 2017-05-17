
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!#define _DBG_ print *,"File:",__FILE__," at ",__LINE__
!#define _DBG_ !DBG
!
!
module prim_advance_caar_mod

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

  type (EdgeBuffer_t) :: edge3p1

  public :: distribute_flux_at_corners, edge3p1, compute_and_apply_rhs

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
  use dimensions_mod, only : nlev, ntrac
  use edge_mod,       only : edgevpack, edgevunpack, edgedgvunpack
  use edgetype_mod,   only : edgedescriptor_t
  use element_mod,    only : element_t
  use hybvcoord_mod,  only : hvcoord_t
#ifdef CAAR_MONOLITHIC
  use caar_pre_exchange_driver_mod, only: caar_pre_exchange_monolithic
#endif

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
  call t_startf('compute_and_apply_rhs')

#ifdef CAAR_MONOLITHIC
  call caar_pre_exchange_monolithic (nm1,n0,np1,qn0,dt2,elem,hvcoord,hybrid,&
                                     deriv,nets,nete,compute_diagnostics,eta_ave_w)
#else
  call compute_and_apply_rhs_pre_exchange (nm1,n0,np1,qn0,dt2,elem,hvcoord,hybrid,&
                                           deriv,nets,nete,compute_diagnostics,eta_ave_w)
#endif

  do ie=nets,nete
     ! =========================================================
     !
     ! Pack ps(np1), T, and v tendencies into comm buffer
     !
     ! =========================================================
     kptr=0
     call edgeVpack(edge3p1, elem(ie)%state%ps_v(:,:,np1),1,kptr,ie)

     kptr=1
     call edgeVpack(edge3p1, elem(ie)%state%T(:,:,:,np1),nlev,kptr,ie)

     kptr=nlev+1
     call edgeVpack(edge3p1, elem(ie)%state%v(:,:,:,:,np1),2*nlev,kptr,ie)

     if (rsplit>0) then
        kptr=kptr+2*nlev
        call edgeVpack(edge3p1, elem(ie)%state%dp3d(:,:,:,np1),nlev,kptr, ie)
     endif
  end do

  ! =============================================================
    ! Insert communications here: for shared memory, just a single
  ! sync is required
  ! =============================================================

  call t_startf('caar_bexchV')
  call bndry_exchangeV(hybrid,edge3p1)
  call t_stopf('caar_bexchV')

  do ie=nets,nete
     ! ===========================================================
     ! Unpack the edges for vgrad_T and v tendencies...
     ! ===========================================================
     kptr=0
     call edgeVunpack(edge3p1, elem(ie)%state%ps_v(:,:,np1), 1, kptr, ie)

     kptr=1
     call edgeVunpack(edge3p1, elem(ie)%state%T(:,:,:,np1), nlev, kptr, ie)

     kptr=nlev+1
     call edgeVunpack(edge3p1, elem(ie)%state%v(:,:,:,:,np1), 2*nlev, kptr, ie)

     if (rsplit>0) then
        if (0<ntrac.and.eta_ave_w.ne.0.) then
          do k=1,nlev
             stashdp3d(:,:,k) = elem(ie)%state%dp3d(:,:,k,np1)/elem(ie)%spheremp(:,:)
          end do
        endif

        corners = 0.0d0
        corners(1:np,1:np,:) = elem(ie)%state%dp3d(:,:,:,np1)
        kptr=kptr+2*nlev
        call edgeVunpack(edge3p1, elem(ie)%state%dp3d(:,:,:,np1),nlev,kptr,ie)

        if  (0<ntrac.and.eta_ave_w.ne.0.) then
          desc = elem(ie)%desc
          call edgeDGVunpack(edge3p1, corners, nlev, kptr, ie)
          corners = corners/dt2

          do k=1,nlev
            tempdp3d = elem(ie)%rspheremp(:,:)*elem(ie)%state%dp3d(:,:,k,np1)
            tempdp3d = tempdp3d - stashdp3d(:,:,k)
            tempdp3d = tempdp3d/dt2

            call distribute_flux_at_corners(cflux, corners(:,:,k), desc%getmapP)

            cflux(1,1,:)   = elem(ie)%rspheremp(1,  1) * cflux(1,1,:)
            cflux(2,1,:)   = elem(ie)%rspheremp(np, 1) * cflux(2,1,:)
            cflux(1,2,:)   = elem(ie)%rspheremp(1, np) * cflux(1,2,:)
            cflux(2,2,:)   = elem(ie)%rspheremp(np,np) * cflux(2,2,:)

            tempflux =  eta_ave_w*subcell_dss_fluxes(tempdp3d, np, nc, elem(ie)%metdet, cflux)
            elem(ie)%sub_elem_mass_flux(:,:,:,k) = elem(ie)%sub_elem_mass_flux(:,:,:,k) + tempflux
          end do
        end if
     endif

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
     end do

     if (rsplit>0) then
        ! vertically lagrangian: complete dp3d timestep:
        do k=1,nlev
           elem(ie)%state%dp3d(:,:,k,np1)= elem(ie)%rspheremp(:,:)*elem(ie)%state%dp3d(:,:,k,np1)
        enddo
        ! when debugging: also update ps_v
        !elem(ie)%state%ps_v(:,:,np1) = elem(ie)%rspheremp(:,:)*elem(ie)%state%ps_v(:,:,np1)
     else
        ! vertically eulerian: complete ps_v timestep:
        elem(ie)%state%ps_v(:,:,np1) = elem(ie)%rspheremp(:,:)*elem(ie)%state%ps_v(:,:,np1)
     endif
  end do

#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif
  call t_stopf('compute_and_apply_rhs')
!pw  call t_adj_detailf(-1)

  end subroutine compute_and_apply_rhs

#ifndef CAAR_MONOLITHIC

#ifdef USE_KOKKOS_KERNELS
#define CAAR_COMPUTE_PRESSURE           caar_compute_pressure_c
#define CAAR_COMPUTE_VORT_AND_DIV       caar_compute_vort_and_div_c
#define CAAR_COMPUTE_T_V                caar_compute_T_v_c
#define CAAR_PREQ_HYDROSTATIC           caar_preq_hydrostatic_c
#define CAAR_PREQ_OMEGA_PS              caar_preq_omega_ps_c
#define CAAR_COMPUTE_ETA_DOT_DPDN       caar_compute_eta_dot_dpdn_c
#define CAAR_COMPUTE_PHI_KINETIC_ENERGY caar_compute_phi_kinetic_energy_c
#define CAAR_UPDATE_STATES              caar_update_states_c
#else
#define CAAR_COMPUTE_PRESSURE           caar_compute_pressure_f90
#define CAAR_COMPUTE_VORT_AND_DIV       caar_compute_vort_and_div_f90
#define CAAR_COMPUTE_T_V                caar_compute_T_v_f90
#define CAAR_PREQ_HYDROSTATIC           caar_preq_hydrostatic_f90
#define CAAR_PREQ_OMEGA_PS              caar_preq_omega_ps_f90
#define CAAR_COMPUTE_ETA_DOT_DPDN       caar_compute_eta_dot_dpdn_f90
#define CAAR_COMPUTE_PHI_KINETIC_ENERGY caar_compute_phi_kinetic_energy_f90
#define CAAR_UPDATE_STATES              caar_update_states_f90
#endif

  subroutine compute_and_apply_rhs_pre_exchange(nm1,n0,np1,qn0,dt2,elem,hvcoord,hybrid,&
                                                deriv,nets,nete,compute_diagnostics,eta_ave_w)

  use kinds,                  only : real_kind
  use control_mod,            only : use_cpstar
  use derivative_mod,         only : derivative_t, gradient_sphere, divergence_sphere, vorticity_sphere
  use dimensions_mod,         only : np, nc, nlev, ntrac, nelemd, qsize_d
  use element_mod,            only : element_t, elem_state_dp3d, elem_state_v, elem_state_Temp,   &
                                     elem_derived_vn0, elem_sub_elem_mass_flux, elem_state_Qdp,   &
                                     timelevels, &
                                     elem_D, elem_Dinv, elem_metdet, elem_rmetdet, elem_spheremp, &
                                     elem_state_phis, elem_derived_phi, elem_derived_pecnd, elem_state_ps_v,&
                                     elem_derived_omega_p, elem_derived_eta_dot_dpdn, elem_fcor
  use hybvcoord_mod,          only : hvcoord_t
  use physical_constants,     only : Rgas, kappa
  use physics_mod,            only : virtual_specific_heat, virtual_temperature
  use prim_si_mod,            only : preq_vertadv, preq_omega_ps, preq_hydrostatic
  use caar_subroutines_mod,   only : caar_compute_pressure_f90, caar_compute_vort_and_div_f90, caar_compute_T_v_f90, &
                                     caar_preq_hydrostatic_f90, caar_preq_omega_ps_f90, caar_compute_eta_dot_dpdn_f90,  &
                                     caar_compute_phi_kinetic_energy_f90, caar_energy_diagnostics_f90, &
                                     caar_update_states_f90, caar_flip_f90_array, caar_flip_f90_tensor2d, caar_flip_f90_Qdp_array
  use iso_c_binding,          only : c_ptr, c_loc

  implicit none

#ifdef USE_KOKKOS_KERNELS
  interface
    subroutine caar_compute_pressure_c(nets, nete, nelemd, n0, hyai_ps0, p_ptr, dp_ptr) bind(c)
      use kinds,         only : real_kind
      use iso_c_binding, only : c_int, c_ptr
      !
      ! Inputs
      !
      integer (kind=c_int),  intent(in) :: nets, nete, nelemd, n0
      type (c_ptr),          intent(in) :: p_ptr, dp_ptr
      real (kind=real_kind), intent(in) :: hyai_ps0
    end subroutine caar_compute_pressure_c

    subroutine caar_compute_vort_and_div_c(nets, nete, nelemd, n0, eta_ave_w,        &
                                           D_ptr, Dinv_ptr, metdet_ptr, dummy_c_ptr, &
                                           p_ptr, dp_ptr, grad_p_ptr, vgrad_p_ptr,   &
                                           elem_state_v_ptr, elem_derived_vn0_ptr,   &
                                           vdp_ptr, div_vdp_ptr, vort_ptr) bind(c)
      use kinds,         only : real_kind
      use iso_c_binding, only : c_int, c_ptr
      !
      ! Inputs
      !
      integer (kind=c_int),  intent(in) :: nets, nete, nelemd, n0
      type (c_ptr),          intent(in) :: D_ptr, Dinv_ptr, metdet_ptr, dummy_c_ptr
      type (c_ptr),          intent(in) :: p_ptr, dp_ptr, grad_p_ptr, vgrad_p_ptr
      type (c_ptr),          intent(in) :: elem_state_v_ptr, elem_derived_vn0_ptr
      type (c_ptr),          intent(in) :: vdp_ptr, div_vdp_ptr, vort_ptr
      real (kind=real_kind), intent(in) :: eta_ave_w  ! weighting for eta_dot_dpdn mean flux
    end subroutine caar_compute_vort_and_div_c

    subroutine caar_compute_T_v_c(nets, nete,nelemd, n0, qn0, use_cpstar,    &
                                  T_v_ptr, kappa_star_ptr, elem_state_dp_ptr, &
                                  elem_state_T_ptr, elem_state_Qdp_ptr) bind(c)
      use iso_c_binding, only : c_int, c_ptr
      !
      ! Inputs
      !
      integer (kind=c_int), intent(in) :: nets, nete, nelemd, n0, qn0, use_cpstar
      type (c_ptr),         intent(in) :: T_v_ptr, kappa_star_ptr, elem_state_dp_ptr
      type (c_ptr),         intent(in) :: elem_state_T_ptr, elem_state_Qdp_ptr
    end subroutine caar_compute_T_v_c

    subroutine caar_preq_hydrostatic_c(nets, nete, nelemd, n0, phi_ptr, phis_ptr, &
                                       T_v_ptr, p_ptr, dp_ptr) bind(c)
      use iso_c_binding,      only : c_int, c_ptr
      !
      ! Inputs
      !
      integer (kind=c_int), intent(in) :: nets, nete, nelemd, n0
      type (c_ptr), intent(in) :: phi_ptr, phis_ptr, T_v_ptr, p_ptr, dp_ptr
    end subroutine caar_preq_hydrostatic_c

    subroutine caar_preq_omega_ps_c(nets, nete, nelemd, div_vdp_ptr, &
                                    vgrad_p_ptr, p_ptr, omega_p_ptr) bind(c)
      use iso_c_binding,      only : c_int, c_ptr
      !
      ! Inputs
      !
      integer (kind=c_int), intent(in) :: nets, nete, nelemd
      type (c_ptr), intent(in) :: div_vdp_ptr, vgrad_p_ptr, p_ptr, omega_p_ptr
    end subroutine caar_preq_omega_ps_c

    subroutine caar_compute_eta_dot_dpdn_c(nets, nete, nelemd, eta_ave_w, &
                                           omega_p_ptr, eta_dot_dpdn_ptr, &
                                           T_vadv_ptr, v_vadv_ptr,        &
                                           elem_derived_eta_dot_dpdn_ptr, &
                                           elem_derived_omega_p_ptr) bind(c)
      use kinds,         only : real_kind
      use iso_c_binding, only : c_int, c_ptr
      !
      ! Inputs
      !
      integer (kind=c_int), intent(in) :: nets, nete, nelemd
      type (c_ptr), intent(in) :: eta_dot_dpdn_ptr, T_vadv_ptr, v_vadv_ptr, omega_p_ptr
      type (c_ptr), intent(in) :: elem_derived_eta_dot_dpdn_ptr, elem_derived_omega_p_ptr
      real (kind=real_kind), intent(in) :: eta_ave_w
    end subroutine caar_compute_eta_dot_dpdn_c

    subroutine caar_compute_phi_kinetic_energy_c (nets, nete, nelemd, n0,                       &
                                                  p_ptr, grad_p_ptr, vort_ptr,                  &
                                                  v_vadv_ptr, T_vadv_ptr,                       &
                                                  elem_Dinv_ptr, elem_fcor_ptr,                 &
                                                  elem_state_T_ptr,elem_state_v_ptr,            &
                                                  elem_derived_phi_ptr, elem_derived_pecnd_ptr, &
                                                  kappa_star_ptr, T_v_ptr, omega_p_ptr,         &
                                                  ttens_ptr, vtens1_ptr, vtens2_ptr) bind (c)
      use iso_c_binding,      only : c_int, c_ptr, c_f_pointer
      !
      ! Inputs
      !
      integer (kind=c_int), intent(in) :: nets, nete, nelemd, n0
      type (c_ptr), intent(in) :: T_v_ptr, p_ptr, grad_p_ptr, vort_ptr
      type (c_ptr), intent(in) :: elem_state_T_ptr, elem_state_v_ptr, elem_derived_phi_ptr
      type (c_ptr), intent(in) :: elem_Dinv_ptr, elem_fcor_ptr, v_vadv_ptr, T_vadv_ptr
      type (c_ptr), intent(in) :: elem_derived_pecnd_ptr, kappa_star_ptr
      type (c_ptr), intent(in) :: vtens1_ptr, vtens2_ptr, ttens_ptr, omega_p_ptr
    end subroutine caar_compute_phi_kinetic_energy_c

!    subroutine caar_energy_diagnostics_c() bind(c)
!    end subroutine caar_energy_diagnostics_c

    subroutine caar_update_states_c (nets, nete, nelemd, nm1, np1, dt2, rsplit, ntrac, eta_ave_w, &
                                     vdp_ptr, div_vdp_ptr, eta_dot_dpdn_ptr, sdot_sum_ptr,        &
                                     ttens_ptr, vtens1_ptr, vtens2_ptr,                           &
                                     elem_Dinv_ptr, elem_metdet_ptr, elem_spheremp_ptr,           &
                                     elem_state_v_ptr, elem_state_T_ptr, elem_state_dp3d_ptr,     &
                                     elem_sub_elem_mass_flux_ptr, elem_state_ps_v_ptr) bind(c)
      use iso_c_binding , only : c_int, c_ptr
      use kinds         , only : real_kind
      !
      ! Inputs
      !
      integer (kind=c_int),  intent(in) :: nets, nete, nelemd, nm1, np1, rsplit, ntrac
      real (kind=real_kind), intent(in) :: dt2, eta_ave_w
      type (c_ptr), intent(in) :: vdp_ptr, div_vdp_ptr, eta_dot_dpdn_ptr, sdot_sum_ptr
      type (c_ptr), intent(in) :: ttens_ptr, vtens1_ptr, vtens2_ptr
      type (c_ptr), intent(in) :: elem_Dinv_ptr, elem_metdet_ptr, elem_spheremp_ptr
      type (c_ptr), intent(in) :: elem_state_v_ptr, elem_state_ps_v_ptr, elem_state_T_ptr, elem_state_dp3d_ptr
      type (c_ptr), intent(in) :: elem_sub_elem_mass_flux_ptr
    end subroutine caar_update_states_c
  end interface
#endif


  type (hvcoord_t)     , intent(in)    :: hvcoord
  type (hybrid_t)      , intent(in)    :: hybrid
  type (element_t)     , intent(inout) :: elem(:)
  integer              , intent(in)    :: np1,nm1,n0,qn0,nets,nete
  real (kind=real_kind), intent(in)    :: dt2
  logical              , intent(in)    :: compute_diagnostics
  real (kind=real_kind), intent(in)    :: eta_ave_w  ! weighting for eta_dot_dpdn mean flux
  type (derivative_t)  , intent(in), target  :: deriv
  !
  ! locals
  !
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: p             ! pressure
  real (kind=real_kind), dimension(:,:,:,:,:), allocatable, target :: grad_p
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: kappa_star
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: omega_p
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: T_v
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: vgrad_p       ! v dot grad(p)
  real (kind=real_kind), dimension(:,:,:,:,:), allocatable, target :: vdp
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: div_vdp
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: vort          ! vorticity
  real (kind=real_kind), dimension(:,:,:,:,:), allocatable, target :: v_vadv        ! velocity vertical advection
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: T_vadv        ! temperature vertical advection
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: vtens1
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: vtens2
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: ttens
  real (kind=real_kind), dimension(:,:,:,:),   allocatable, target :: eta_dot_dpdn  ! half level vertical velocity on p-grid
  real (kind=real_kind), dimension(:,:,:),     allocatable, target :: sdot_sum

#ifdef USE_KOKKOS_KERNELS
  ! arrays used to flip from left layout to right layout
  real (kind=real_kind), dimension(:), allocatable, target :: elem_D_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_Dinv_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_metdet_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_rmetdet_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_fcor_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_spheremp_c
  real (kind=real_kind), dimension(:), allocatable, target :: p_c                 ! pressure
  real (kind=real_kind), dimension(:), allocatable, target :: grad_p_c
  real (kind=real_kind), dimension(:), allocatable, target :: vgrad_p_c
  real (kind=real_kind), dimension(:), allocatable, target :: vdp_c
  real (kind=real_kind), dimension(:), allocatable, target :: div_vdp_c
  real (kind=real_kind), dimension(:), allocatable, target :: vort_c
  real (kind=real_kind), dimension(:), allocatable, target :: kappa_star_c
  real (kind=real_kind), dimension(:), allocatable, target :: T_v_c
  real (kind=real_kind), dimension(:), allocatable, target :: omega_p_c
  real (kind=real_kind), dimension(:), allocatable, target :: eta_dot_dpdn_c
  real (kind=real_kind), dimension(:), allocatable, target :: T_vadv_c
  real (kind=real_kind), dimension(:), allocatable, target :: v_vadv_c
  real (kind=real_kind), dimension(:), allocatable, target :: ttens_c
  real (kind=real_kind), dimension(:), allocatable, target :: vtens1_c
  real (kind=real_kind), dimension(:), allocatable, target :: vtens2_c
  real (kind=real_kind), dimension(:), allocatable, target :: sdot_sum_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_state_dp3d_c   ! delta pressure
  real (kind=real_kind), dimension(:), allocatable, target :: elem_state_v_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_state_Temp_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_state_Qdp_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_state_phis_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_state_ps_v_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_derived_vn0_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_derived_phi_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_derived_pecnd_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_derived_eta_dot_dpdn_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_derived_omega_p_c
  real (kind=real_kind), dimension(:), allocatable, target :: elem_sub_elem_mass_flux_c
#endif

  type (c_ptr) :: elem_D_ptr, elem_Dinv_ptr, elem_metdet_ptr
  type (c_ptr) :: elem_rmetdet_ptr, elem_spheremp_ptr, elem_fcor_ptr
  type (c_ptr) :: p_ptr, grad_p_ptr, vgrad_p_ptr, vdp_ptr, div_vdp_ptr, vort_ptr
  type (c_ptr) :: vtens1_ptr, vtens2_ptr, ttens_ptr, T_v_ptr, v_vadv_ptr, T_vadv_ptr
  type (c_ptr) :: kappa_star_ptr, elem_state_Qdp_ptr
  type (c_ptr) :: elem_state_phis_ptr, elem_derived_phi_ptr, elem_state_v_ptr
  type (c_ptr) :: elem_state_T_ptr, elem_state_dp3d_ptr, elem_sub_elem_mass_flux_ptr
  type (c_ptr) :: elem_derived_eta_dot_dpdn_ptr, elem_derived_omega_p_ptr
  type (c_ptr) :: elem_derived_vn0_ptr, omega_p_ptr, eta_dot_dpdn_ptr, sdot_sum_ptr
  type (c_ptr) :: elem_derived_pecnd_ptr, elem_state_ps_v_ptr

  ! ---------------------------------------------------------- !
  !                        Subroutine body                     !
  !----------------------------------------------------------- !

  ! Allocate temporaries
  allocate(p             (np,np,nlev,nelemd)  )
  allocate(grad_p        (np,np,2,nlev,nelemd))
  allocate(kappa_star    (np,np,nlev,nelemd)  )
  allocate(omega_p       (np,np,nlev,nelemd)  )
  allocate(T_v           (np,np,nlev,nelemd)  )
  allocate(vgrad_p       (np,np,nlev,nelemd)  )
  allocate(vdp           (np,np,2,nlev,nelemd))
  allocate(div_vdp       (np,np,nlev,nelemd)  )
  allocate(vort          (np,np,nlev,nelemd)  )
  allocate(v_vadv        (np,np,2,nlev,nelemd))
  allocate(T_vadv        (np,np,nlev,nelemd)  )
  allocate(vtens1        (np,np,nlev,nelemd)  )
  allocate(vtens2        (np,np,nlev,nelemd)  )
  allocate(ttens         (np,np,nlev,nelemd)  )
  allocate(eta_dot_dpdn  (np,np,nlev+1,nelemd))
  allocate(sdot_sum      (np,np,nelemd)       )

#ifdef USE_KOKKOS_KERNELS
  ! Allocate c-ordering arrays
  allocate(elem_D_c                    (nelemd*2*2*np*np))
  allocate(elem_Dinv_c                 (nelemd*2*2*np*np))
  allocate(elem_metdet_c               (nelemd*np*np))
  allocate(elem_rmetdet_c              (nelemd*np*np))
  allocate(elem_fcor_c                 (nelemd*np*np))
  allocate(elem_spheremp_c             (nelemd*np*np))
  allocate(p_c                         (nelemd*nlev*np*np))
  allocate(grad_p_c                    (nelemd*nlev*2*np*np))
  allocate(vgrad_p_c                   (nelemd*nlev*np*np))
  allocate(vdp_c                       (nelemd*nlev*2*np*np))
  allocate(div_vdp_c                   (nelemd*nlev*np*np))
  allocate(vort_c                      (nelemd*nlev*np*np))
  allocate(kappa_star_c                (nelemd*nlev*np*np))
  allocate(T_v_c                       (nelemd*nlev*np*np))
  allocate(omega_p_c                   (nelemd*nlev*np*np))
  allocate(eta_dot_dpdn_c              (nelemd*nlevp*np*np))
  allocate(T_vadv_c                    (nelemd*nlev*np*np))
  allocate(v_vadv_c                    (nelemd*nlev*2*np*np))
  allocate(ttens_c                     (nelemd*nlev*np*np))
  allocate(vtens1_c                    (nelemd*nlev*np*np))
  allocate(vtens2_c                    (nelemd*nlev*np*np))
  allocate(sdot_sum_c                  (nelemd*np*np))
  allocate(elem_state_dp3d_c           (nelemd*timelevels*nlev*np*np))
  allocate(elem_state_v_c              (nelemd*timelevels*nlev*2*np*np))
  allocate(elem_state_Temp_c           (nelemd*timelevels*nlev*np*np))
  allocate(elem_state_Qdp_c            (nelemd*qsize_d*2*nlev*np*np))
  allocate(elem_state_phis_c           (nelemd*np*np))
  allocate(elem_state_ps_v_c           (nelemd*timelevels*np*np))
  allocate(elem_derived_vn0_c          (nelemd*nlev*2*np*np))
  allocate(elem_derived_phi_c          (nelemd*nlev*np*np))
  allocate(elem_derived_pecnd_c        (nelemd*nlev*np*np))
  allocate(elem_derived_eta_dot_dpdn_c (nelemd*nlevp*np*np))
  allocate(elem_derived_omega_p_c      (nelemd*nlev*np*np))
  allocate(elem_sub_elem_mass_flux_c   (nelemd*nlev*4*nc*nc))

  ! Flip f90 input arrays into cxx arrays
  ! We will have to flip back states at the end though
  call caar_flip_f90_tensor2d (elem_D, elem_D_c)
  call caar_flip_f90_tensor2d (elem_Dinv, elem_Dinv_c)
  call caar_flip_f90_array (elem_metdet, elem_metdet_c, .TRUE.)
  call caar_flip_f90_array (elem_rmetdet, elem_rmetdet_c, .TRUE.)
  call caar_flip_f90_array (elem_fcor, elem_fcor_c, .TRUE.)
  call caar_flip_f90_array (elem_spheremp, elem_spheremp_c, .TRUE.)
  call caar_flip_f90_array (elem_state_dp3d,elem_state_dp3d_c,.TRUE.)
  call caar_flip_f90_array (elem_state_v, elem_state_v_c, .TRUE.)
  call caar_flip_f90_array (elem_state_Temp, elem_state_Temp_c, .TRUE.)
  call caar_flip_f90_Qdp_array (elem_state_Qdp, elem_state_Qdp_c)
  call caar_flip_f90_array (elem_derived_vn0, elem_derived_vn0_c, .TRUE.)
  call caar_flip_f90_array (elem_derived_phi, elem_derived_phi_c, .TRUE.)
  call caar_flip_f90_array (elem_derived_pecnd, elem_derived_pecnd_c, .TRUE.)
  call caar_flip_f90_array (elem_state_phis, elem_state_phis_c, .TRUE.)
  call caar_flip_f90_array (elem_derived_eta_dot_dpdn, elem_derived_eta_dot_dpdn_c, .TRUE. )
  call caar_flip_f90_array (elem_derived_omega_p, elem_derived_omega_p_c, .TRUE. )
#endif

  ! Create the pointers. Based on the build type, we point to f90 or cxx arrays

#ifdef USE_KOKKOS_KERNELS
  elem_D_ptr                    = c_loc(elem_D_c)
  elem_Dinv_ptr                 = c_loc(elem_Dinv_c)
  elem_metdet_ptr               = c_loc(elem_metdet_c)
  elem_rmetdet_ptr              = c_loc(elem_rmetdet_c)
  elem_fcor_ptr                 = c_loc(elem_fcor_c)
  elem_spheremp_ptr             = c_loc(elem_spheremp_c)
  p_ptr                         = c_loc(p_c)
  grad_p_ptr                    = c_loc(grad_p_c)
  vgrad_p_ptr                   = c_loc(vgrad_p_c)
  vdp_ptr                       = c_loc(vdp_c)
  div_vdp_ptr                   = c_loc(div_vdp_c)
  vort_ptr                      = c_loc(vort_c)
  kappa_star_ptr                = c_loc(kappa_star_c)
  T_v_ptr                       = c_loc(T_v_c)
  omega_p_ptr                   = c_loc(omega_p_c)
  eta_dot_dpdn_ptr              = c_loc(eta_dot_dpdn_c)
  T_vadv_ptr                    = c_loc(T_vadv_c)
  v_vadv_ptr                    = c_loc(v_vadv_c)
  ttens_ptr                     = c_loc(ttens_c)
  vtens1_ptr                    = c_loc(vtens1_c)
  vtens2_ptr                    = c_loc(vtens2_c)
  sdot_sum_ptr                  = c_loc(sdot_sum)
  elem_state_dp3d_ptr           = c_loc(elem_state_dp3d_c)
  elem_state_v_ptr              = c_loc(elem_state_v_c)
  elem_state_T_ptr              = c_loc(elem_state_Temp_c)
  elem_state_Qdp_ptr            = c_loc(elem_state_Qdp_c)
  elem_state_phis_ptr           = c_loc(elem_state_phis_c)
  elem_state_ps_v_ptr           = c_loc(elem_state_ps_v_c)
  elem_derived_vn0_ptr          = c_loc(elem_derived_vn0_c)
  elem_derived_phi_ptr          = c_loc(elem_derived_phi_c)
  elem_derived_pecnd_ptr        = c_loc(elem_derived_pecnd_c)
  elem_derived_eta_dot_dpdn_ptr = c_loc(elem_derived_eta_dot_dpdn_c)
  elem_derived_omega_p_ptr      = c_loc(elem_derived_omega_p_c)
  elem_sub_elem_mass_flux_ptr   = c_loc(elem_sub_elem_mass_flux_c)
#else
  elem_D_ptr                    = c_loc(elem_D)
  elem_Dinv_ptr                 = c_loc(elem_Dinv)
  elem_metdet_ptr               = c_loc(elem_metdet)
  elem_rmetdet_ptr              = c_loc(elem_rmetdet)
  elem_fcor_ptr                 = c_loc(elem_fcor)
  elem_spheremp_ptr             = c_loc(elem_spheremp)
  p_ptr                         = c_loc(p)
  grad_p_ptr                    = c_loc(grad_p)
  vgrad_p_ptr                   = c_loc(vgrad_p)
  vdp_ptr                       = c_loc(vdp)
  div_vdp_ptr                   = c_loc(div_vdp)
  vort_ptr                      = c_loc(vort)
  kappa_star_ptr                = c_loc(kappa_star)
  T_v_ptr                       = c_loc(T_v)
  omega_p_ptr                   = c_loc(omega_p)
  eta_dot_dpdn_ptr              = c_loc(eta_dot_dpdn)
  T_vadv_ptr                    = c_loc(T_vadv)
  v_vadv_ptr                    = c_loc(v_vadv)
  ttens_ptr                     = c_loc(ttens)
  vtens1_ptr                    = c_loc(vtens1)
  vtens2_ptr                    = c_loc(vtens2)
  sdot_sum_ptr                  = c_loc(sdot_sum)
  elem_state_dp3d_ptr           = c_loc(elem_state_dp3d)
  elem_state_v_ptr              = c_loc(elem_state_v)
  elem_state_T_ptr              = c_loc(elem_state_Temp)
  elem_state_Qdp_ptr            = c_loc(elem_state_Qdp)
  elem_state_phis_ptr           = c_loc(elem_state_phis)
  elem_state_ps_v_ptr           = c_loc(elem_state_ps_v)
  elem_derived_vn0_ptr          = c_loc(elem_derived_vn0)
  elem_derived_phi_ptr          = c_loc(elem_derived_phi)
  elem_derived_pecnd_ptr        = c_loc(elem_derived_pecnd)
  elem_derived_eta_dot_dpdn_ptr = c_loc(elem_derived_eta_dot_dpdn)
  elem_derived_omega_p_ptr      = c_loc(elem_derived_omega_p)
  elem_sub_elem_mass_flux_ptr   = c_loc(elem_sub_elem_mass_flux)
#endif

  ! ================================
  ! Compute pressure
  ! ================================

  call t_startf ("caar_compute_pressure")
  call CAAR_COMPUTE_PRESSURE(nets,nete,nelemd,n0,hvcoord%hyai(1)*hvcoord%ps0, p_ptr, elem_state_dp3d_ptr)
  call t_stopf ("caar_compute_pressure")

  ! ======================================
  ! Compute vorticity and divergence
  ! ======================================

  call t_startf ("caar_compute_vort_and_div")
  call CAAR_COMPUTE_VORT_AND_DIV (nets, nete, nelemd, n0, eta_ave_w,                   &
                                  elem_D_ptr, elem_Dinv_ptr,                           &
                                  elem_metdet_ptr, elem_rmetdet_ptr,                   &
                                  p_ptr, elem_state_dp3d_ptr, grad_p_ptr,              &
                                  vgrad_p_ptr, elem_state_v_ptr, elem_derived_vn0_ptr, &
                                  vdp_ptr, div_vdp_ptr, vort_ptr)
  call t_stopf ("caar_compute_vort_and_div")

  ! =====================================
  ! Compute T_v
  ! =======================================

  call t_startf ("caar_compute_T_v")
  call CAAR_COMPUTE_T_V(nets, nete, nelemd, n0, qn0, use_cpstar, T_v_ptr, kappa_star_ptr, &
                        elem_state_dp3d_ptr, elem_state_T_ptr, elem_state_Qdp_ptr)
  call t_stopf ("caar_compute_T_v")

  ! ====================================================
  ! Compute Hydrostatic equation, modeled after CCM-3
  ! ====================================================

  call t_startf ("caar_preq_hydrostatic")
  call CAAR_PREQ_HYDROSTATIC(nets, nete, nelemd, n0, elem_derived_phi_ptr, elem_state_phis_ptr, &
                             T_v_ptr, p_ptr, elem_state_dp3d_ptr)
  call t_stopf ("caar_preq_hydrostatic")

  ! ====================================================
  ! Compute omega_p according to CCM-3
  ! ====================================================

  call t_startf ("caar_preq_omega_ps")
  call CAAR_PREQ_OMEGA_PS(nets, nete, nelemd, div_vdp_ptr, vgrad_p_ptr, p_ptr, omega_p_ptr)
  call t_stopf ("caar_preq_omega_ps")

  call t_startf ("caar_compute_eta_dot_dpdn")
  call CAAR_COMPUTE_ETA_DOT_DPDN(nets, nete, nelemd, eta_ave_w, &
                                 omega_p_ptr, eta_dot_dpdn_ptr, &
                                 T_vadv_ptr, v_vadv_ptr,        &
                                 elem_derived_eta_dot_dpdn_ptr, &
                                 elem_derived_omega_p_ptr)
  call t_stopf ("caar_compute_eta_dot_dpdn")

  ! ===========================================
  ! Compute phi and kinetic energy (vertloop)
  ! ===========================================

  call t_startf ("caar_compute_phi_kinetic_energy")
  call CAAR_COMPUTE_PHI_KINETIC_ENERGY(nets, nete, nelemd, n0,                       &
                                       p_ptr, grad_p_ptr, vort_ptr,                  &
                                       v_vadv_ptr, T_vadv_ptr,                       &
                                       elem_Dinv_ptr, elem_fcor_ptr,                 &
                                       elem_state_T_ptr,elem_state_v_ptr,            &
                                       elem_derived_phi_ptr, elem_derived_pecnd_ptr, &
                                       kappa_star_ptr, T_v_ptr, omega_p_ptr,         &
                                       ttens_ptr, vtens1_ptr, vtens2_ptr)
  call t_stopf ("caar_compute_phi_kinetic_energy")

!#ifdef ENERGY_DIAGNOSTICS
!  call CAAR_ENERGY_DIAGNOSTICS()
!#endif
!

  ! ===========================================
  ! Update states at np1
  ! ===========================================

  call CAAR_UPDATE_STATES(nets, nete, nelemd, nm1, np1, dt2, rsplit, ntrac, eta_ave_w,  &
                          vdp_ptr, div_vdp_ptr, eta_dot_dpdn_ptr, sdot_sum_ptr,         &
                          ttens_ptr, vtens1_ptr, vtens2_ptr,                            &
                          elem_Dinv_ptr, elem_metdet_ptr, elem_spheremp_ptr,            &
                          elem_state_v_ptr, elem_state_T_ptr, elem_state_dp3d_ptr,      &
                          elem_sub_elem_mass_flux_ptr, elem_state_ps_v_ptr)

#ifdef USE_KOKKOS_KERNELS
  ! Flip back output arrays to pass the results back to fortran
  call caar_flip_f90_array (elem_derived_vn0, elem_derived_vn0_c, .FALSE.)
  call caar_flip_f90_array (elem_derived_phi, elem_derived_phi_c, .FALSE. )
  call caar_flip_f90_array (elem_derived_eta_dot_dpdn, elem_derived_eta_dot_dpdn_c, .FALSE. )
  call caar_flip_f90_array (elem_derived_omega_p, elem_derived_omega_p_c, .FALSE. )
  call caar_flip_f90_array (elem_state_Temp, elem_state_Temp_c, .FALSE. )
  call caar_flip_f90_array (elem_state_v, elem_state_v_c, .FALSE. )
  call caar_flip_f90_array (elem_state_dp3d, elem_state_dp3d_c, .FALSE. )
  call caar_flip_f90_array (elem_state_ps_v, elem_state_ps_v_c, .FALSE. )
  call caar_flip_f90_array (elem_sub_elem_mass_flux, elem_sub_elem_mass_flux_c, .FALSE. )
#endif

  end subroutine compute_and_apply_rhs_pre_exchange
! ifdef CAAR_MONOLITHIC
#endif

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
