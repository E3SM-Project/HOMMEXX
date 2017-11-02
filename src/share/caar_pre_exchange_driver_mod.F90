#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

module caar_pre_exchange_driver_mod
  use utils_mod, only : FrobeniusNorm

  implicit none

  interface
    subroutine init_control_caar_c (nets,nete,nelemd,nm1,n0,np1,qn0,dt2,ps0, &
                               compute_diagnostics,eta_ave_w,hybrid_a_ptr) bind(c)
      use kinds         , only : real_kind
      use iso_c_binding , only : c_ptr, c_int, c_bool
      !
      ! Inputs
      !
      integer (kind=c_int),  intent(in) :: np1,nm1,n0,qn0,nets,nete,nelemd
      logical,               intent(in) :: compute_diagnostics
      real (kind=real_kind), intent(in) :: dt2, ps0, eta_ave_w
      type (c_ptr),          intent(in) :: hybrid_a_ptr
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
    subroutine caar_pre_exchange_monolithic_c () bind(c)
    end subroutine caar_pre_exchange_monolithic_c
  end interface

contains

  subroutine caar_pre_exchange_monolithic(nm1,n0,np1,qn0,dt2,elem,hvcoord,hybrid,&
                                          deriv,nets,nete,compute_diagnostics,eta_ave_w)
    use kinds          , only : real_kind
    use dimensions_mod , only : np, nc, nlev, ntrac
    use element_mod    , only : element_t
    use derivative_mod , only : derivative_t
    use control_mod    , only : moisture, qsplit, use_cpstar, rsplit, swest
    use hybvcoord_mod  , only : hvcoord_t
    use hybrid_mod     , only : hybrid_t
    use perf_mod       , only : t_startf, t_stopf

#ifdef USE_KOKKOS_KERNELS
    use dimensions_mod , only : nelemd
    use element_mod    , only : elem_state_v, elem_state_temp, elem_state_dp3d, &
                                elem_derived_phi, elem_derived_pecnd,           &
                                elem_derived_omega_p, elem_derived_vn0,         &
                                elem_derived_eta_dot_dpdn, elem_state_Qdp
    use iso_c_binding  , only : c_ptr, c_loc

    implicit none

    type (c_ptr) :: elem_state_v_ptr, elem_state_t_ptr, elem_state_dp3d_ptr
    type (c_ptr) :: elem_derived_phi_ptr, elem_derived_pecnd_ptr
    type (c_ptr) :: elem_derived_omega_p_ptr, elem_derived_vn0_ptr
    type (c_ptr) :: elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr
    type (c_ptr) :: hvcoord_a_ptr
#else
    implicit none
#endif

    !
    ! Inputs
    !
    integer, intent(in) :: np1,nm1,n0,qn0,nets,nete
    real*8,  intent(in) :: dt2
    logical, intent(in) :: compute_diagnostics

    type (hvcoord_t)     , intent(in)   , target :: hvcoord
    type (hybrid_t)      , intent(in)            :: hybrid
    type (element_t)     , intent(inout), target :: elem(:)
    type (derivative_t)  , intent(in)            :: deriv
    real (kind=real_kind), intent(in)            :: eta_ave_w  ! weighting for eta_dot_dpdn mean flux

#ifdef USE_KOKKOS_KERNELS
    call t_startf("caar_overhead")

    hvcoord_a_ptr             = c_loc(hvcoord%hyai)
    call init_control_caar_c(nets,nete,nelemd,nm1,n0,np1,qn0,dt2,hvcoord%ps0,compute_diagnostics,eta_ave_w,hvcoord_a_ptr)

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
#endif
    !og disabling this for now to keep # of timers low
    call t_startf("caar_pre_exchange")
#ifdef USE_KOKKOS_KERNELS
    call caar_pre_exchange_monolithic_c ()
#else
    call caar_pre_exchange_monolithic_f90(nm1,n0,np1,qn0,dt2,elem,hvcoord,hybrid,&
                                          deriv,nets,nete,compute_diagnostics,eta_ave_w)
#endif
    !og disabling this for now to keep # of timers low
    call t_stopf("caar_pre_exchange")

#ifdef USE_KOKKOS_KERNELS
    call t_startf("caar_overhead")
    call caar_push_results_c (elem_state_v_ptr, elem_state_t_ptr, elem_state_dp3d_ptr, &
                                         elem_derived_phi_ptr, elem_derived_pecnd_ptr,            &
                                         elem_derived_omega_p_ptr, elem_derived_vn0_ptr,          &
                                         elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr)
    call t_stopf("caar_overhead")
#endif

  end subroutine caar_pre_exchange_monolithic

  ! An interface to enable access from C/C++
  subroutine caar_compute_energy_grad_c_int(dvv, Dinv, pecnd, phi, v, tvirt, press, press_grad, vtemp) bind(c)
    use kinds, only : real_kind
    use dimensions_mod, only : np
    use physical_constants, only : Rgas
    use derivative_mod, only : derivative_t

    real (kind=real_kind), intent(in) :: dvv(np, np) ! (np, np)
    real (kind=real_kind), intent(in) :: Dinv(np, np, 2, 2) ! (np, np, 2, 2)
    real (kind=real_kind), intent(in) :: pecnd(np, np) ! (np, np)
    real (kind=real_kind), intent(in) :: phi(np, np) ! (np, np)
    real (kind=real_kind), intent(in) :: v(np, np, 2) ! (np, np, 2)
    real (kind=real_kind), intent(in) :: tvirt(np, np) ! (np, np)
    real (kind=real_kind), intent(in) :: press(np, np) ! (np, np)
    real (kind=real_kind), intent(in) :: press_grad(np, np, 2) ! (np, np, 2)
    real (kind=real_kind), intent(inout) :: vtemp(np, np, 2) ! (np, np, 2)

    ! locals
    integer :: i, j
    type (derivative_t) :: deriv

    deriv%dvv = dvv

    call caar_compute_energy_grad(deriv, Dinv, pecnd, phi, v, vtemp)

    do i=1,np
      do j=1,np
        vtemp(i,j,1) = vtemp(i,j,1) + Rgas*(tvirt(i,j)/press(i,j))*press_grad(i,j,1)
        vtemp(i,j,2) = vtemp(i,j,2) + Rgas*(tvirt(i,j)/press(i,j))*press_grad(i,j,2)
      enddo
    enddo
  end subroutine caar_compute_energy_grad_c_int

  subroutine caar_compute_energy_grad(deriv, Dinv, pecnd, phi, v, vtemp)
    use kinds, only : real_kind
    use dimensions_mod, only : np
    use derivative_mod, only : derivative_t, gradient_sphere
    use physical_constants, only : Rgas
    type (derivative_t), intent(in) :: deriv
    real (kind=real_kind), intent(in) :: Dinv(np, np, 2, 2)
    real (kind=real_kind), intent(in) :: pecnd(np, np)
    real (kind=real_kind), intent(in) :: phi(np, np)
    real (kind=real_kind), intent(in) :: v(np, np, 2)
    real (kind=real_kind), intent(out) :: vtemp(np, np, 2)

    integer :: h, i, j
    real (kind=real_kind), dimension(np,np) :: Ephi, gpterm
    real (kind=real_kind) :: v1, v2, E
    do j = 1, np
      do i = 1, np
        v1 = v(i, j, 1)
        v2 = v(i, j, 2)
        E = 0.5D0 * (v1 * v1 + v2 * v2)
        Ephi(i, j)=E + (phi(i, j) + pecnd(i, j))
      end do
    end do
    vtemp = gradient_sphere(Ephi, deriv, Dinv)
  end subroutine caar_compute_energy_grad

  subroutine caar_compute_dp3d_np1_c_int(np1, nm1, dt2, spheremp, divdp, eta_dot_dpdn, dp3d) bind(c)
    use kinds, only : real_kind
    use element_mod, only : timelevels
    use dimensions_mod, only : np, nlev

    integer, value, intent(in) :: np1, nm1
    real (kind=real_kind), intent(in) :: dt2
    ! Note: These array dims are explicitly specified so they can be passed directly from C
    real (kind=real_kind), intent(in) :: spheremp(np, np)
    real (kind=real_kind), intent(in) :: divdp(np, np, nlev)
    real (kind=real_kind), intent(in) :: eta_dot_dpdn(np, np, nlev)
    real (kind=real_kind), intent(out) :: dp3d(np, np, nlev, timelevels)

    ! locals
    integer :: i, j, k

    do k = 1, nlev
      do j = 1, np
        do i = 1, np
          dp3d(i, j, k, np1) = &
               spheremp(i, j) * (dp3d(i, j, k, nm1) - &
               dt2 * (divdp(i, j, k) + eta_dot_dpdn(i, j, k + 1) - eta_dot_dpdn(i, j, k)))
        end do
      end do
    end do
  end subroutine caar_compute_dp3d_np1_c_int

  ! computes derived_vn0, vdp and divdp
  subroutine caar_compute_divdp_c_int(eta_ave_w, velocity, dp3d, dinv, metdet, &
                                      dvv, derived_vn0, vdp, divdp) bind(c)
    use kinds, only : real_kind
    use element_mod, only : element_t
    use derivative_mod, only : derivative_t
    use dimensions_mod, only : np, nlev

    real (kind=real_kind), value, intent(in) :: eta_ave_w
    real (kind=real_kind), intent(in) :: velocity(np, np, 2)
    real (kind=real_kind), intent(in) :: dp3d(np, np)
    real (kind=real_kind), intent(in) :: dinv(np, np, 2, 2)
    real (kind=real_kind), intent(in) :: metdet(np, np)
    real (kind=real_kind), intent(in) :: dvv(np, np)

    real (kind=real_kind), intent(out) :: derived_vn0(np, np, 2)
    real (kind=real_kind), intent(out) :: vdp(np, np, 2)
    real (kind=real_kind), intent(out) :: divdp(np, np)

    type(element_t) :: elem
    type(derivative_t) :: deriv

    deriv%dvv = dvv

#ifdef HOMME_USE_FLAT_ARRAYS
    ! This code is only used for unit testing, so allocates are okay
    allocate(elem%Dinv(np, np, 2, 2))
    allocate(elem%metdet(np, np))
#endif

    elem%Dinv = dinv
    elem%metdet = metdet

    call caar_compute_divdp(elem, deriv, eta_ave_w, dp3d, &
                            velocity, derived_vn0, vdp, divdp)

#ifdef HOMME_USE_FLAT_ARRAYS
    deallocate(elem%Dinv)
    deallocate(elem%metdet)
#endif
  end subroutine caar_compute_divdp_c_int

  subroutine caar_compute_pressure_c_int(hyai, ps0, dp, pressure) bind(c)
    use kinds, only : real_kind
    use dimensions_mod, only : np, nlev

    implicit none

    real (kind=real_kind), intent(in) :: hyai
    real (kind=real_kind), intent(in) :: ps0
    real (kind=real_kind), intent(in) :: dp(np, np, nlev)
    real (kind=real_kind), intent(out) :: pressure(np, np, nlev)

    integer :: i, j, k
    do j = 1, np
      do i = 1, np
        pressure(i, j, 1) = hyai * ps0 + dp(i, j, 1) / 2
      end do
    end do
    do k = 2, nlev
      do j = 1, np
        do i = 1, np
          pressure(i, j, k) = pressure(i, j, k - 1) + &
               dp(i, j, k - 1) / 2 + dp(i, j, k) / 2
        end do
      end do
    end do
  end subroutine caar_compute_pressure_c_int

  ! computes derived_vn0, vdp and divdp
  subroutine caar_compute_divdp(elem, deriv, eta_ave_w, dp3d, &
                                velocity, derived_vn0, vdp, divdp)
    use kinds, only : real_kind
    use element_mod, only : element_t
    use derivative_mod, only : derivative_t, divergence_sphere
    use dimensions_mod, only : np, nlev

    type(element_t), intent(in) :: elem
    type(derivative_t), intent(in) :: deriv

    real (kind=real_kind), intent(in) :: eta_ave_w
    real (kind=real_kind), intent(in) :: dp3d(np, np)
    real (kind=real_kind), intent(in) :: velocity(np, np, 2)

    real (kind=real_kind), intent(inout) :: derived_vn0(np, np, 2)
    real (kind=real_kind), intent(out) :: vdp(np, np, 2)
    real (kind=real_kind), intent(out) :: divdp(np, np)

    ! locals
    integer :: i, j, k
    do k = 1, 2
      do j = 1, np
        do i = 1, np
           vdp(i, j, k) = velocity(i, j, k) * dp3d(i, j)
           ! ================================
           ! Accumulate mean Vel_rho flux in vn0
           ! ================================
           derived_vn0(i, j, k) = derived_vn0(i, j, k) + eta_ave_w * vdp(i, j, k)
        end do
      end do
    end do
    divdp = divergence_sphere(vdp, deriv, elem)
  end subroutine caar_compute_divdp

  subroutine caar_compute_temperature_no_tracers_c_int(T, T_v) bind(c)
    use kinds, only : real_kind
    use dimensions_mod, only : np, nlev

    implicit none

    real (kind=real_kind), intent(in) :: T(np, np, nlev)
    real (kind=real_kind), intent(out) :: T_v(np, np, nlev)

    ! locals
    integer :: i, j, k

    do k=1,nlev
      do j=1,np
        do i=1,np
          T_v(i, j, k) = T(i, j, k)
        end do
      end do
    end do
  end subroutine caar_compute_temperature_no_tracers_c_int

  subroutine caar_compute_temperature_tracers_c_int(Qdp, dp, T, T_v) bind(c)
    use kinds, only : real_kind
    use dimensions_mod, only : np, nlev
    use physics_mod, only : virtual_temperature

    implicit none

    real (kind=real_kind), intent(in) :: Qdp(np, np, nlev)
    real (kind=real_kind), intent(in) :: dp(np, np, nlev)
    real (kind=real_kind), intent(in) :: T(np, np, nlev)
    real (kind=real_kind), intent(out) :: T_v(np, np, nlev)

    ! locals
    real (kind=real_kind) :: Qt
    integer :: i, j, k

    do k=1,nlev
      do j=1,np
        do i=1,np
          Qt = Qdp(i, j, k) / dp(i, j, k)
          T_v(i, j, k) = Virtual_Temperature(T(i, j, k), Qt)
        end do
      end do
    end do
  end subroutine caar_compute_temperature_tracers_c_int

  subroutine caar_compute_omega_p_c_int(eta_ave_w, omega_p_source, omega_p_dest) bind(c)
    use kinds, only : real_kind
    use dimensions_mod, only : np, nlev

    implicit none

    real (kind=real_kind), value, intent(in) :: eta_ave_w
    real (kind=real_kind), intent(in) :: omega_p_source(np, np, nlev)
    real (kind=real_kind), intent(inout) :: omega_p_dest(np, np, nlev)

    ! locals
    integer :: i, j, k

    do k=1,nlev
      do j=1,np
        do i=1,np
          omega_p_dest(i, j, k) = &
               omega_p_dest(i, j, k) + eta_ave_w * omega_p_source(i, j, k)
        end do
      end do
    end do
  end subroutine caar_compute_omega_p_c_int

  subroutine caar_compute_temperature_c_int(dt, spheremp, dinv, dvv, velocity, t_virt, omega_p, &
                                            t_vadv, t_previous, t_current, t_future) bind(c)
    use kinds, only : real_kind
    use dimensions_mod, only : np
    use derivative_mod, only : derivative_t

    implicit none

    real (kind=real_kind), value, intent(in) :: dt
    real (kind=real_kind), intent(in) :: spheremp(np, np)
    real (kind=real_kind), intent(in) :: dinv(np, np, 2, 2)
    real (kind=real_kind), intent(in) :: dvv(np, np)
    real (kind=real_kind), intent(in) :: velocity(np, np, 2)
    real (kind=real_kind), intent(in) :: t_virt(np, np)
    real (kind=real_kind), intent(in) :: omega_p(np, np)
    real (kind=real_kind), intent(in) :: t_vadv(np, np)
    real (kind=real_kind), intent(in) :: t_previous(np, np)
    real (kind=real_kind), intent(in) :: t_current(np, np)
    real (kind=real_kind), intent(out) :: t_future(np, np)

    ! locals

    type (derivative_t) :: deriv

    deriv%dvv = dvv

    call caar_compute_temperature(dt, deriv, spheremp, dinv, velocity, t_virt, &
                                  omega_p, t_vadv, t_previous, t_current, t_future)

  end subroutine caar_compute_temperature_c_int

  subroutine caar_compute_temperature(dt, deriv, spheremp, dinv, velocity, t_virt, omega_p, &
                                      t_vadv, t_previous, t_current, t_future)
    use kinds, only : real_kind
    use dimensions_mod, only : np
    use derivative_mod, only : derivative_t, gradient_sphere
    use physical_constants, only : kappa

    implicit none

    real (kind=real_kind), value, intent(in) :: dt
    type (derivative_t), intent(in) :: deriv
    real (kind=real_kind), intent(in) :: spheremp(np, np)
    real (kind=real_kind), intent(in) :: dinv(np, np, 2, 2)
    real (kind=real_kind), intent(in) :: velocity(np, np, 2)
    real (kind=real_kind), intent(in) :: t_virt(np, np)
    real (kind=real_kind), intent(in) :: omega_p(np, np)
    real (kind=real_kind), intent(in) :: t_vadv(np, np)
    real (kind=real_kind), intent(in) :: t_previous(np, np)
    real (kind=real_kind), intent(in) :: t_current(np, np)
    real (kind=real_kind), intent(out) :: t_future(np, np)

    ! locals

    real (kind=real_kind) :: vtemp(np, np, 2)
    real (kind=real_kind) :: ttens, vgrad_t

    integer :: i, j

    ! ================================================
    ! compute gradp term (ps/p)*(dp/dps)*T
    ! ================================================
    vtemp(:,:,:) = gradient_sphere(t_current, deriv, dinv)
    do j=1,np
      do i=1,np
        vgrad_t =  velocity(i, j, 1) * vtemp(i, j, 1) + &
             velocity(i, j, 2) * vtemp(i, j, 2)
        ! Note: This ignores the case where use_cpstar = 1
        ! In this case, we need to use Rgas / VirtualSpecificHeat(Qt)
        ttens = -t_vadv(i, j) - vgrad_t + kappa * t_virt(i, j) * omega_p(i, j)
        t_future(i, j) = spheremp(i, j) * (t_previous(i, j) + dt * ttens)
      end do
    end do
  end subroutine caar_compute_temperature

  subroutine caar_pre_exchange_monolithic_f90(nm1,n0,np1,qn0,dt2,elem,hvcoord,hybrid,&
                                              deriv,nets,nete,compute_diagnostics,eta_ave_w)
    use kinds, only : real_kind
    use dimensions_mod, only : np, nc, nlev, ntrac, max_corner_elem
    use element_mod, only : element_t,PrintElem
    use derivative_mod, only : derivative_t, divergence_sphere, gradient_sphere, vorticity_sphere
    use derivative_mod, only : subcell_div_fluxes, subcell_dss_fluxes
    use edge_mod, only : edgevpack, edgevunpack, edgeDGVunpack
    use edgetype_mod, only : edgedescriptor_t
    use bndry_mod, only : bndry_exchangev
    use control_mod, only : moisture, qsplit, use_cpstar, rsplit, swest
    use hybvcoord_mod, only : hvcoord_t
    use hybrid_mod,     only: hybrid_t

    use physical_constants, only : cp, cpwater_vapor, Rgas, kappa
    use physics_mod, only : virtual_specific_heat
    use prim_si_mod, only : preq_vertadv, preq_omega_ps, preq_hydrostatic
#if ( defined CAM )
    use control_mod, only: se_met_nudge_u, se_met_nudge_p, se_met_nudge_t, se_met_tevolve
#endif

    use time_mod, only : tevolve

    implicit none

    !
    ! Inputs
    !
    integer, intent(in) :: np1,nm1,n0,qn0,nets,nete
    real*8, intent(in) :: dt2
    logical, intent(in)  :: compute_diagnostics

    type (hvcoord_t)     , intent(in) :: hvcoord
    type (hybrid_t)      , intent(in) :: hybrid
    type (element_t)     , intent(inout), target :: elem(:)
    type (derivative_t)  , intent(in) :: deriv
    real (kind=real_kind), intent(in) :: eta_ave_w  ! weighting for eta_dot_dpdn mean flux
    !
    ! Locals
    !
    real (kind=real_kind), pointer, dimension(:,:,:)   :: phi
    real (kind=real_kind), pointer, dimension(:,:,:)   :: dp

    real (kind=real_kind), dimension(np,np,nlev)   :: omega_p
    real (kind=real_kind), dimension(np,np,nlev)   :: T_v
    real (kind=real_kind), dimension(np,np,nlev)   :: divdp
    real (kind=real_kind), dimension(np,np,nlev+1)   :: eta_dot_dpdn  ! half level vertical velocity on p-grid
    real (kind=real_kind), dimension(np,np)      :: sdot_sum   ! temporary field
    real (kind=real_kind), dimension(np,np,2)    :: vtemp     ! generic gradient storage
    real (kind=real_kind), dimension(np,np,2,nlev):: vdp       !
    real (kind=real_kind), dimension(np,np,2     ):: v         !
    real (kind=real_kind), dimension(np,np,2,nlev) :: grad_p
    real (kind=real_kind), dimension(np,np,2,nlev) :: grad_p_m_pmet  ! gradient(p- p_met)
    real (kind=real_kind), dimension(np,np,nlev)   :: vort       ! vorticity
    real (kind=real_kind), dimension(np,np,nlev)   :: p          ! pressure
    real (kind=real_kind), dimension(np,np,nlev)   :: rdp        ! inverse of delta pressure
    real (kind=real_kind), dimension(np,np,nlev)   :: T_vadv     ! temperature vertical advection
    real (kind=real_kind), dimension(np,np,nlev)   :: vgrad_p    ! v.grad(p)
    real (kind=real_kind), dimension(np,np,nlev+1) :: ph               ! half level pressures on p-grid
    real (kind=real_kind), dimension(np,np,2,nlev) :: v_vadv   ! velocity vertical advection
    real (kind=real_kind), dimension(0:np+1,0:np+1,nlev)          :: corners
    real (kind=real_kind), dimension(2,2,2)                         :: cflux
    real (kind=real_kind) ::  kappa_star(np,np,nlev)
    real (kind=real_kind) ::  vtens1(np,np,nlev)
    real (kind=real_kind) ::  vtens2(np,np,nlev)
    type (EdgeDescriptor_t)                                       :: desc

    real (kind=real_kind) ::  cp2,cp_ratio,E,de,Qt,v1,v2
    real (kind=real_kind) ::  glnps1,glnps2,gpterm
    integer :: h,i,j,k,kptr,ie
    real (kind=real_kind) :: u_m_umet, v_m_vmet, t_m_tmet

    do ie=nets,nete
      !ps => elem(ie)%state%ps_v(:,:,n0)
      phi => elem(ie)%derived%phi(:,:,:)
      dp  => elem(ie)%state%dp3d(:,:,:,n0)
! dont thread this because of k-1 dependence:
      p(:,:,1)=hvcoord%hyai(1)*hvcoord%ps0 + dp(:,:,1)/2
      do k=2,nlev
         p(:,:,k)=p(:,:,k-1) + dp(:,:,k-1)/2 + dp(:,:,k)/2
      enddo

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,v1,v2,vtemp)
#endif
      do k=1,nlev
         grad_p(:,:,:,k) = gradient_sphere(p(:,:,k),deriv,elem(ie)%Dinv)
         rdp(:,:,k) = 1.0D0/dp(:,:,k)

        ! ============================
        ! compute vgrad_lnps
        ! ============================

!og: why is this here? there is i,j loop in the routine.
!og: computes vdp = v*dp, derived_vn0=+eta_ave_w*vdp (why +=?),
!divdp=div_sphere(vdp)
        call caar_compute_divdp(elem(ie), deriv, eta_ave_w, elem(ie)%state%dp3d(:, :, k, n0), &
                                elem(ie)%state%v(:, :, :, k, n0), elem(ie)%derived%vn0(:, :, :, k), &
                                vdp(:, :, :, k), divdp(:, :, k))
        do j=1,np
          do i=1,np
            v1 = elem(ie)%state%v(i,j,1,k,n0)
            v2 = elem(ie)%state%v(i,j,2,k,n0)
            vgrad_p(i,j,k) = (v1*grad_p(i,j,1,k) + v2*grad_p(i,j,2,k))
          end do
        end do

#if ( defined CAM )
        ! ============================
        ! compute grad(P-P_met)
        ! ============================
        if (se_met_nudge_p.gt.0.D0) then
          grad_p_m_pmet(:,:,:,k) = &
               grad_p(:,:,:,k) - &
               hvcoord%hybm(k)* &
               gradient_sphere(elem(ie)%derived%ps_met(:,:)+tevolve*elem(ie)%derived%dpsdt_met(:,:), &
                                deriv,elem(ie)%Dinv)
        endif
#endif

        ! =========================================
        !
        ! Compute relative vorticity and divergence
        !
        ! =========================================
        vort(:,:,k)=vorticity_sphere(elem(ie)%state%v(:,:,:,k,n0),deriv,elem(ie))
      enddo

      ! compute T_v for timelevel n0
      !if ( moisture /= "dry") then
      if (qn0 == -1 ) then
        call caar_compute_temperature_no_tracers_c_int(elem(ie)%state%T(:, :, :, n0), T_v)
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j)
#endif
        do k = 1, nlev
          do i = 1, np
            do j = 1, np
              kappa_star(i,j,k) = kappa
            end do
          end do
        end do
      else
        call caar_compute_temperature_tracers_c_int(elem(ie)%state%Qdp(:, :, :, 1, qn0), dp, elem(ie)%state%T(:, :, :, n0), T_v)
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,Qt)
#endif
        do k=1,nlev
          do j=1,np
            do i=1,np
              if (use_cpstar==1) then
                 kappa_star(i,j,k) =  Rgas/Virtual_Specific_Heat(Qt)
              else
                 kappa_star(i,j,k) = kappa
              endif
            end do
          end do
        end do
      end if

      ! ====================================================
      ! Compute Hydrostatic equation, modeld after CCM-3
      ! ====================================================
      !call geopotential_t(p,dp,T_v,Rgas,phi)
      call preq_hydrostatic(phi,elem(ie)%state%phis,T_v,p,dp)

      ! ====================================================
      ! Compute omega_p according to CCM-3
      ! ====================================================
      call preq_omega_ps(omega_p,hvcoord,p,vgrad_p,divdp)

      ! ==================================================
      ! zero partial sum for accumulating sum
      !    (div(v_k) + v_k.grad(lnps))*dsigma_k = div( v dp )
      ! used by eta_dot_dpdn and lnps tendency
      ! ==================================================
      sdot_sum=0


      ! ==================================================
      ! Compute eta_dot_dpdn
      ! save sdot_sum as this is the -RHS of ps_v equation
      ! ==================================================
      if (rsplit>0) then
         ! VERTICALLY LAGRANGIAN:   no vertical motion
         eta_dot_dpdn=0
         T_vadv=0
         v_vadv=0
      else
         do k=1,nlev
            ! ==================================================
            ! add this term to PS equation so we exactly conserve dry mass
            ! ==================================================
            sdot_sum(:,:) = sdot_sum(:,:) + divdp(:,:,k)
            eta_dot_dpdn(:,:,k+1) = sdot_sum(:,:)
         end do

         ! ===========================================================
         ! at this point, eta_dot_dpdn contains integral_etatop^eta[ divdp ]
         ! compute at interfaces:
         !    eta_dot_dpdn = -dp/dt - integral_etatop^eta[ divdp ]
         ! for reference: at mid layers we have:
         !    omega = v grad p  - integral_etatop^eta[ divdp ]
         ! ===========================================================
         do k=1,nlev-1
            eta_dot_dpdn(:,:,k+1) = hvcoord%hybi(k+1)*sdot_sum(:,:) -eta_dot_dpdn(:,:,k+1)
         end do

         eta_dot_dpdn(:,:,1     ) = 0.0D0
         eta_dot_dpdn(:,:,nlev+1) = 0.0D0

         ! ===========================================================
         ! Compute vertical advection of T and v from eq. CCM2 (3.b.1)
         ! ==============================================
         call preq_vertadv(elem(ie)%state%T(:,:,:,n0),elem(ie)%state%v(:,:,:,:,n0), &
              eta_dot_dpdn,rdp,T_vadv,v_vadv)
      endif

      ! ================================
      ! accumulate mean vertical flux:
      ! ================================
      call caar_compute_omega_p_c_int(eta_ave_w, omega_p, elem(ie)%derived%omega_p)
#if (defined COLUMN_OPENMP)
       !$omp parallel do private(k)
#endif
      do k=1,nlev  !  Loop index added (AAM)
         elem(ie)%derived%eta_dot_dpdn(:,:,k) = &
              elem(ie)%derived%eta_dot_dpdn(:,:,k) + eta_ave_w*eta_dot_dpdn(:,:,k)
      enddo
      elem(ie)%derived%eta_dot_dpdn(:,:,nlev+1) = &
           elem(ie)%derived%eta_dot_dpdn(:,:,nlev+1) + eta_ave_w*eta_dot_dpdn(:,:,nlev+1)





      ! ==============================================
      ! Compute phi + kinetic energy term: 10*nv*nv Flops
      ! ==============================================

#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,v1,v2,E,vtemp,gpterm,glnps1,glnps2,u_m_umet,v_m_vmet,t_m_tmet)
#endif
      vertloop: do k=1,nlev
         call caar_compute_temperature(dt2, deriv, elem(ie)%spheremp, elem(ie)%Dinv, &
                                       elem(ie)%state%v(:, :, :, k, n0), T_v(:, :, k), &
                                       omega_p(:, :, k), T_vadv(:, :, k), &
                                       elem(ie)%state%T(:, :, k, nm1), &
                                       elem(ie)%state%T(:, :, k, n0), &
                                       elem(ie)%state%T(:, :, k, np1))
         ! vtemp = grad ( E + PHI )
         call caar_compute_energy_grad(deriv, elem(ie)%Dinv, elem(ie)%derived%pecnd(:,:,k), phi(:,:,k), elem(ie)%state%v(:,:,:,k,n0), vtemp)

         do j=1,np
            do i=1,np
               gpterm = T_v(i,j,k)/p(i,j,k)
               glnps1 = Rgas*gpterm*grad_p(i,j,1,k)
               glnps2 = Rgas*gpterm*grad_p(i,j,2,k)

               v1     = elem(ie)%state%v(i,j,1,k,n0)
               v2     = elem(ie)%state%v(i,j,2,k,n0)

               vtens1(i,j,k) =  - v_vadv(i,j,1,k)                           &
                    + v2*(elem(ie)%fcor(i,j) + vort(i,j,k))        &
                    - (vtemp(i,j,1) + glnps1)
               vtens2(i,j,k) =   - v_vadv(i,j,2,k)                            &
                    - v1*(elem(ie)%fcor(i,j) + vort(i,j,k))        &
                    - (vtemp(i,j,2) + glnps2)
            end do
         end do
      end do vertloop
#ifdef ENERGY_DIAGNOSTICS
      ! =========================================================
      !
      ! diagnostics
      ! recomputes some gradients that were not saved above
      ! uses:  sdot_sum(), eta_dot_dpdn(), grad_ps()
      ! grad_phi(), dp(), p(), T_vadv(), v_vadv(), divdp()
      ! =========================================================

      ! =========================================================
      ! (AAM) - This section has accumulations over vertical levels.
      !   Be careful if implementing OpenMP
      ! =========================================================

      if (compute_diagnostics) then
        elem(ie)%accum%KEhorz1=0
        elem(ie)%accum%KEhorz2=0
        elem(ie)%accum%IEhorz1=0
        elem(ie)%accum%IEhorz2=0
        elem(ie)%accum%IEhorz1_wet=0
        elem(ie)%accum%IEhorz2_wet=0
        elem(ie)%accum%KEvert1=0
        elem(ie)%accum%KEvert2=0
        elem(ie)%accum%IEvert1=0
        elem(ie)%accum%IEvert2=0
        elem(ie)%accum%IEvert1_wet=0
        elem(ie)%accum%IEvert2_wet=0
        elem(ie)%accum%T1=0
        elem(ie)%accum%T2=0
        elem(ie)%accum%T2_s=0
        elem(ie)%accum%S1=0
        elem(ie)%accum%S1_wet=0
        elem(ie)%accum%S2=0

        do j=1,np
          do i=1,np
             elem(ie)%accum%S2(i,j) = elem(ie)%accum%S2(i,j) - &
                  sdot_sum(i,j)*elem(ie)%state%phis(i,j)
          enddo
        enddo

        do k=1,nlev
          ! vtemp = grad_E(:,:,k)
          do j=1,np
            do i=1,np
              v1     = elem(ie)%state%v(i,j,1,k,n0)
              v2     = elem(ie)%state%v(i,j,2,k,n0)
              Ephi(i,j)=0.5D0*( v1*v1 + v2*v2 )
            enddo
          enddo
          vtemp = gradient_sphere(Ephi,deriv,elem(ie)%Dinv)
          do j=1,np
            do i=1,np
              ! dp/dn u dot grad(E)
              v1     = elem(ie)%state%v(i,j,1,k,n0)
              v2     = elem(ie)%state%v(i,j,2,k,n0)
              elem(ie)%accum%KEhorz2(i,j) = elem(ie)%accum%KEhorz2(i,j) + &
                   (v1*vtemp(i,j,1)  + v2*vtemp(i,j,2))*dp(i,j,k)
              ! E div( u dp/dn )
              elem(ie)%accum%KEhorz1(i,j) = elem(ie)%accum%KEhorz1(i,j) + Ephi(i,j)*divdp(i,j,k)

              ! Cp T div( u dp/dn)   ! dry horizontal advection component
              elem(ie)%accum%IEhorz1(i,j) = elem(ie)%accum%IEhorz1(i,j) + Cp*elem(ie)%state%T(i,j,k,n0)*divdp(i,j,k)


            enddo
          enddo


          ! vtemp = grad_phi(:,:,k)
          vtemp = gradient_sphere(phi(:,:,k),deriv,elem(ie)%Dinv)
          do j=1,np
            do i=1,np
              v1     = elem(ie)%state%v(i,j,1,k,n0)
              v2     = elem(ie)%state%v(i,j,2,k,n0)
              E = 0.5D0*( v1*v1 + v2*v2 )
              ! NOTE:  Cp_star = Cp + (Cpv-Cp)*q
              ! advection terms can thus be broken into two components: dry and wet
              ! dry components cancel exactly
              ! wet components should cancel exactly
              !
              ! some diagnostics
              ! e = eta_dot_dpdn()
              de =  eta_dot_dpdn(i,j,k+1)-eta_dot_dpdn(i,j,k)
              ! Cp T de/dn, integral dn:
              elem(ie)%accum%IEvert1(i,j)=elem(ie)%accum%IEvert1(i,j) + Cp*elem(ie)%state%T(i,j,k,n0)*de
              ! E de/dn
              elem(ie)%accum%KEvert1(i,j)=elem(ie)%accum%KEvert1(i,j) + E*de
              ! Cp T_vadv dp/dn
              elem(ie)%accum%IEvert2(i,j)=elem(ie)%accum%IEvert2(i,j) + Cp*T_vadv(i,j,k)*dp(i,j,k)
              ! dp/dn V dot V_vadv
              elem(ie)%accum%KEvert2(i,j)=elem(ie)%accum%KEvert2(i,j) + (v1*v_vadv(i,j,1,k) + v2*v_vadv(i,j,2,k)) *dp(i,j,k)

              ! IEvert1_wet():  (Cpv-Cp) T Qdp_vadv  (Q equation)
              ! IEvert2_wet():  (Cpv-Cp) Qdp T_vadv   T equation
              if (use_cpstar==1) then
              elem(ie)%accum%IEvert2_wet(i,j)=elem(ie)%accum%IEvert2_wet(i,j) +&
                   (Cpwater_vapor-Cp)*elem(ie)%state%Q(i,j,k,1)*T_vadv(i,j,k)*dp(i,j,k)
              endif

              gpterm = T_v(i,j,k)/p(i,j,k)
              elem(ie)%accum%T1(i,j) = elem(ie)%accum%T1(i,j) - &
                   Rgas*gpterm*(grad_p(i,j,1,k)*v1 + grad_p(i,j,2,k)*v2)*dp(i,j,k)

              elem(ie)%accum%T2(i,j) = elem(ie)%accum%T2(i,j) - &
                   (vtemp(i,j,1)*v1 + vtemp(i,j,2)*v2)*dp(i,j,k)

              ! S1 = < Cp_star dp/dn , RT omega_p/cp_star >
              elem(ie)%accum%S1(i,j) = elem(ie)%accum%S1(i,j) + &
                   Rgas*T_v(i,j,k)*omega_p(i,j,k)*dp(i,j,k)

              ! cp_star = cp + cp2
              if (use_cpstar==1) then
              cp2 = (Cpwater_vapor-Cp)*elem(ie)%state%Q(i,j,k,1)
              cp_ratio = cp2/(cp+cp2)
              elem(ie)%accum%S1_wet(i,j) = elem(ie)%accum%S1_wet(i,j) + &
                   cp_ratio*(Rgas*T_v(i,j,k)*omega_p(i,j,k)*dp(i,j,k))
              endif

              elem(ie)%accum%CONV(i,j,:,k)=-Rgas*gpterm*grad_p(i,j,:,k)-vtemp(i,j,:)
            enddo
          enddo

          vtemp(:,:,:) = gradient_sphere(elem(ie)%state%phis(:,:),deriv,elem(ie)%Dinv)
          do j=1,np
            do i=1,np
              v1     = elem(ie)%state%v(i,j,1,k,n0)
              v2     = elem(ie)%state%v(i,j,2,k,n0)
              elem(ie)%accum%T2_s(i,j) = elem(ie)%accum%T2_s(i,j) - &
                   (vtemp(i,j,1)*v1 + vtemp(i,j,2)*v2)*dp(i,j,k)
            enddo
          enddo

          vtemp(:,:,:)   = gradient_sphere(elem(ie)%state%T(:,:,k,n0),deriv,elem(ie)%Dinv)
          do j=1,np
            do i=1,np
              v1     = elem(ie)%state%v(i,j,1,k,n0)
              v2     = elem(ie)%state%v(i,j,2,k,n0)

              ! Cp dp/dn u dot gradT
              elem(ie)%accum%IEhorz2(i,j) = elem(ie)%accum%IEhorz2(i,j) + &
                   Cp*(v1*vtemp(i,j,1) + v2*vtemp(i,j,2))*dp(i,j,k)

              if (use_cpstar==1) then
              elem(ie)%accum%IEhorz2_wet(i,j) = elem(ie)%accum%IEhorz2_wet(i,j) + &
                   (Cpwater_vapor-Cp)*elem(ie)%state%Q(i,j,k,1)*&
                   (v1*vtemp(i,j,1) + v2*vtemp(i,j,2))*dp(i,j,k)
              endif

            enddo
          enddo
        enddo
      endif
#endif
!     ! =========================================================
!     ! local element timestep, store in np1.
!     ! note that we allow np1=n0 or nm1
!     ! apply mass matrix
!     ! =========================================================

      call caar_compute_dp3d_np1_c_int(np1, nm1, dt2, elem(ie)%spheremp, &
           divdp, eta_dot_dpdn, elem(ie)%state%dp3d)


#if (defined COLUMN_OPENMP)
!$omp parallel do private(k)
#endif

      do k=1,nlev
        elem(ie)%state%v(:,:,1,k,np1) = elem(ie)%spheremp(:,:)*( elem(ie)%state%v(:,:,1,k,nm1) + dt2*vtens1(:,:,k) )
        elem(ie)%state%v(:,:,2,k,np1) = elem(ie)%spheremp(:,:)*( elem(ie)%state%v(:,:,2,k,nm1) + dt2*vtens2(:,:,k) )
      enddo

    enddo

  end subroutine caar_pre_exchange_monolithic_f90


end module caar_pre_exchange_driver_mod
