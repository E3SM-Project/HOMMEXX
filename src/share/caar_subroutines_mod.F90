#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

module caar_subroutines_mod
  use kinds,        only : real_kind

  implicit none

  interface caar_flip_f90_array
    module procedure caar_flip_f90_array_2D
    module procedure caar_flip_f90_array_3D
    module procedure caar_flip_f90_array_4D
    module procedure caar_flip_f90_array_5D
    module procedure caar_flip_f90_array_6D
  end interface caar_flip_f90_array

  public :: caar_flip_f90_tensor2d
  public :: caar_flip_f90_array

contains
  subroutine caar_compute_pressure_f90(nets, nete, nelemd, n0, hyai_ps0, p_ptr, dp_ptr) bind(c)
    use iso_c_binding,  only : c_int, c_ptr, c_f_pointer
    use control_mod,    only : rsplit
    use dimensions_mod, only : nlev, np
    use element_mod,    only : timelevels
    !
    ! Inputs
    !
    integer (kind=c_int),  intent(in) :: nets, nete, nelemd, n0
    type (c_ptr),          intent(in) :: p_ptr, dp_ptr
    real (kind=real_kind), intent(in) :: hyai_ps0
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:,:,:),   pointer :: p
    real (kind=real_kind), dimension(:,:,:,:,:), pointer :: dp
    integer :: ie, k

    call c_f_pointer(p_ptr,  p,  [np, np, nlev, nelemd])
    call c_f_pointer(dp_ptr, dp, [np, np, nlev, timelevels, nelemd])

    do ie=nets,nete


       ! ============================
       ! compute p and delta p
       ! ============================
       ! dont thread this because of k-1 dependence:
       p(:,:,1,ie)=hyai_ps0 + dp(:,:,1,n0,ie)/2
       do k=2,nlev
          p(:,:,k,ie)=p(:,:,k-1,ie) + dp(:,:,k-1,n0,ie)/2 + dp(:,:,k,n0,ie)/2
       enddo
    end do
  end subroutine caar_compute_pressure_f90

  subroutine caar_compute_vort_and_div_f90(nets, nete, nelemd, n0, eta_ave_w, dvv_ptr,  &
                                           D_ptr, Dinv_ptr, metdet_ptr, rmetdet_ptr,    &
                                           p_ptr, dp_ptr, grad_p_ptr, vgrad_p_ptr,      &
                                           elem_state_v_ptr, elem_derived_vn0_ptr,      &
                                           vdp_ptr, div_vdp_ptr, vort_ptr) bind(c)
    use iso_c_binding,  only : c_int, c_ptr, c_f_pointer
    use dimensions_mod, only : np, nlev
    use derivative_mod, only : derivative_t, gradient_sphere, vorticity_sphere, divergence_sphere
    use element_mod,    only : element_t, timelevels
    !
    ! Inputs
    !
    integer (kind=c_int),  intent(in) :: nets, nete, nelemd, n0
    type (c_ptr),          intent(in) :: dvv_ptr, D_ptr, Dinv_ptr, metdet_ptr, rmetdet_ptr
    type (c_ptr),          intent(in) :: p_ptr, dp_ptr, grad_p_ptr, vgrad_p_ptr
    type (c_ptr),          intent(in) :: elem_state_v_ptr, elem_derived_vn0_ptr
    type (c_ptr),          intent(in) :: vdp_ptr, div_vdp_ptr, vort_ptr
    real (kind=real_kind), intent(in) :: eta_ave_w  ! weighting for eta_dot_dpdn mean flux
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:),         pointer :: dvv
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: D
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: Dinv
    real (kind=real_kind), dimension(:,:,:),       pointer :: metdet
    real (kind=real_kind), dimension(:,:,:),       pointer :: rmetdet
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: p
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: grad_p
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: dp
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: vdp
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: div_vdp
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: vort
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: vgrad_p
    real (kind=real_kind), dimension(:,:,:,:,:,:), pointer :: elem_state_v
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: elem_derived_vn0
    integer               :: ie, k, i, j
    real (kind=real_kind) :: v1, v2
    type (derivative_t)   :: deriv
    type (element_t)      :: elem

    ! cast the pointers
    call c_f_pointer(dvv_ptr,              dvv,              [np, np])
    call c_f_pointer(D_ptr,                D,                [np, np, 2, 2, nelemd])
    call c_f_pointer(Dinv_ptr,             Dinv,             [np, np, 2, 2, nelemd])
    call c_f_pointer(metdet_ptr,           metdet,           [np, np, nelemd])
    call c_f_pointer(rmetdet_ptr,          rmetdet,          [np, np, nelemd])
    call c_f_pointer(p_ptr,                p,                [np, np, nlev, nelemd])
    call c_f_pointer(dp_ptr,               dp,               [np, np, nlev, timelevels, nelemd])
    call c_f_pointer(grad_p_ptr,           grad_p,           [np, np, 2, nlev, nelemd])
    call c_f_pointer(vdp_ptr,              vdp,              [np, np, 2, nlev, nelemd])
    call c_f_pointer(div_vdp_ptr,          div_vdp,          [np, np, nlev, nelemd])
    call c_f_pointer(vort_ptr,             vort,             [np, np, nlev, nelemd])
    call c_f_pointer(vgrad_p_ptr,          vgrad_p,          [np, np, nlev, nelemd])
    call c_f_pointer(elem_state_v_ptr,     elem_state_v,     [np, np, 2, nlev, timelevels, nelemd])
    call c_f_pointer(elem_derived_vn0_ptr, elem_derived_vn0, [np, np, 2, nlev, nelemd])

    deriv%dvv = dvv

    do ie=nets,nete
#ifdef HOMME_USE_FLAT_ARRAYS
      elem%D       => D(:,:,:,:,ie)
      elem%Dinv    => Dinv(:,:,:,:,ie)
      elem%metdet  => metdet(:,:,ie)
      elem%rmetdet => rmetdet(:,:,ie)
#else
      elem%D       = D(:,:,:,:,ie)
      elem%Dinv    = Dinv(:,:,:,:,ie)
      elem%metdet  = metdet(:,:,ie)
      elem%rmetdet = rmetdet(:,:,ie)
#endif

#if (defined COLUMN_OPENMP)
      !$omp parallel do private(k,i,j,v1,v2)
#endif
      do k=1,nlev
        grad_p(:,:,:,k,ie) = gradient_sphere(p(:,:,k,ie),deriv,elem%Dinv)

        ! ============================
        ! compute vgrad_lnps
        ! ============================
        do j=1,np
           do i=1,np
              v1 = elem_state_v(i,j,1,k,n0,ie)
              v2 = elem_state_v(i,j,2,k,n0,ie)
              vgrad_p(i,j,k,ie) = (v1*grad_p(i,j,1,k,ie) + v2*grad_p(i,j,2,k,ie))
              vdp(i,j,1,k,ie) = v1*dp(i,j,k,n0,ie)
              vdp(i,j,2,k,ie) = v2*dp(i,j,k,n0,ie)
           end do
        end do

!  #if ( defined CAM )
!          ! ============================
!          ! compute grad(P-P_met)
!          ! ============================
!          if (se_met_nudge_p.gt.0.D0) then
!             grad_p_m_pmet(:,:,:,k) = &
!                  grad_p(:,:,:,k) - &
!                  hvcoord%hybm(k)* &
!                  gradient_sphere( elem(ie)%derived%ps_met(:,:)+tevolve*elem(ie)%derived%dpsdt_met(:,:), &
!                                   deriv,elem(ie)%Dinv)
!          endif
!  #endif

        ! ================================
        ! Accumulate mean Vel_rho flux in vn0
        ! ================================
        elem_derived_vn0(:,:,:,k,ie)=elem_derived_vn0(:,:,:,k,ie)+eta_ave_w*vdp(:,:,:,k,ie)

        ! =========================================
        !
        ! Compute relative vorticity and divergence
        !
        ! =========================================
        div_vdp(:,:,k,ie) = divergence_sphere(vdp(:,:,:,k,ie),deriv,elem)
        vort(:,:,k,ie)  = vorticity_sphere(elem_state_v(:,:,:,k,n0,ie),deriv,elem)
      end do
    end do
  end subroutine caar_compute_vort_and_div_f90

  subroutine caar_compute_T_v_f90(nets, nete, nelemd, n0, qn0, use_cpstar,    &
                                  T_v_ptr, kappa_star_ptr, elem_state_dp_ptr, &
                                  elem_state_T_ptr, elem_state_Qdp_ptr) bind(c)
    use iso_c_binding,      only : c_int, c_ptr, c_f_pointer
    use dimensions_mod,     only : nlev, np, qsize
    use element_mod,        only : timelevels
    use physical_constants, only : Rgas, kappa
    use physics_mod,        only : virtual_specific_heat, virtual_temperature
    !
    ! Inputs
    !
    integer (kind=c_int), intent(in) :: nets, nete, nelemd, n0, qn0, use_cpstar
    type (c_ptr),         intent(in) :: T_v_ptr, kappa_star_ptr, elem_state_dp_ptr
    type (c_ptr),         intent(in) :: elem_state_T_ptr, elem_state_Qdp_ptr
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: kappa_star
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: dp
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: T_v
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: elem_state_Temp
    real (kind=real_kind), dimension(:,:,:,:,:,:), pointer :: elem_state_Qdp
    integer :: ie, k, i, j
    real (kind=real_kind) :: Qt

    ! Cast the pointers
    call c_f_pointer(elem_state_dp_ptr,  dp,              [np, np, nlev, timelevels, nelemd])
    call c_f_pointer(T_v_ptr,            T_v,             [np, np, nlev, nelemd])
    call c_f_pointer(kappa_star_ptr,     kappa_star,      [np, np, nlev, nelemd])
    call c_f_pointer(elem_state_T_ptr,   elem_state_Temp, [np, np, nlev, timelevels, nelemd])
    call c_f_pointer(elem_state_Qdp_ptr, elem_state_Qdp,  [np, np, nlev, qsize, 2, nelemd])

    ! compute T_v for timelevel n0
    if (qn0 == -1 ) then
      do ie=nets, nete
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j)
#endif
        do k=1,nlev
          do j=1,np
            do i=1,np
              T_v(i,j,k,ie) = elem_state_Temp(i,j,k,n0,ie)
              kappa_star(i,j,k,ie) = kappa
            end do
          end do
        end do
      end do
    else
      do ie=nets, nete
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,i,j,Qt)
#endif
        do k=1,nlev
          do j=1,np
            do i=1,np
              Qt = elem_state_Qdp(i,j,k,1,qn0,ie)/dp(i,j,k,n0,ie)
              T_v(i,j,k,ie) = Virtual_Temperature(elem_state_Temp(i,j,k,n0,ie),Qt)
              if (use_cpstar==1) then
                kappa_star(i,j,k,ie) =  Rgas/Virtual_Specific_Heat(Qt)
              else
                kappa_star(i,j,k,ie) = kappa
              endif
            end do
          end do
        end do
      end do
    end if

  end subroutine caar_compute_T_v_f90

  subroutine caar_preq_hydrostatic_f90(nets, nete, n0, phi_ptr, phis_ptr, &
                                       T_v_ptr, p_ptr, dp_ptr) bind(c)
    use iso_c_binding,      only : c_int, c_ptr, c_f_pointer
    use dimensions_mod,     only : np, nlev, nelemd
    use element_mod,        only : timelevels
    use physical_constants, only : rgas
    !
    ! Inputs
    !
    integer (kind=c_int), intent(in) :: nets, nete, n0
    type (c_ptr), intent(in) :: phi_ptr, phis_ptr, T_v_ptr, p_ptr, dp_ptr
    !
    ! Locals
    !
    integer :: ie, k, i, j
    real(kind=real_kind), dimension(:,:,:,:),   pointer :: phi
    real(kind=real_kind), dimension(:,:,:),     pointer :: phis
    real(kind=real_kind), dimension(:,:,:,:),   pointer :: T_v
    real(kind=real_kind), dimension(:,:,:,:),   pointer :: p
    real(kind=real_kind), dimension(:,:,:,:,:), pointer :: dp
    real(kind=real_kind) Hkk,Hkl                              ! diagonal term of energy conversion matrix
    real(kind=real_kind), dimension(np,np,nlev) :: phii       ! Geopotential at interfaces

    ! Cast the pointers
    call c_f_pointer(phi_ptr,  phi,  [np, np, nlev, nelemd])
    call c_f_pointer(phis_ptr, phis, [np, np, nelemd])
    call c_f_pointer(T_v_ptr,  T_v,  [np, np, nlev, nelemd])
    call c_f_pointer(p_ptr,    p,    [np, np, nlev, nelemd])
    call c_f_pointer(dp_ptr,   dp,   [np, np, nlev, timelevels, nelemd])

    do ie=nets, nete
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,j,i,hkk,hkl)
#endif
      do j=1,np   !   Loop inversion (AAM)
        do i=1,np
          hkk = dp(i,j,nlev,n0,ie)*0.5d0/p(i,j,nlev,ie)
          hkl = 2*hkk
          phii(i,j,nlev)  = Rgas*T_v(i,j,nlev,ie)*hkl
          phi(i,j,nlev,ie) = phis(i,j,ie) + Rgas*T_v(i,j,nlev,ie)*hkk
        end do

        do k=nlev-1,2,-1
          do i=1,np
            ! hkk = dp*ckk
            hkk = dp(i,j,k,n0,ie)*0.5d0/p(i,j,k,ie)
            hkl = 2*hkk
            phii(i,j,k) = phii(i,j,k+1) + Rgas*T_v(i,j,k,ie)*hkl
            phi(i,j,k,ie)  = phis(i,j,ie) + phii(i,j,k+1) + Rgas*T_v(i,j,k,ie)*hkk
          end do
        end do

        do i=1,np
          ! hkk = dp*ckk
          hkk = 0.5d0*dp(i,j,1,n0,ie)/p(i,j,1,ie)
          phi(i,j,1,ie) = phis(i,j,ie) + phii(i,j,2) + Rgas*T_v(i,j,1,ie)*hkk
        end do
      end do
    end do

  end subroutine caar_preq_hydrostatic_f90

  subroutine caar_preq_omega_ps_f90(nets, nete, div_vdp_ptr, &
                                    vgrad_p_ptr, p_ptr, omega_p_ptr) bind(c)
    use iso_c_binding,  only : c_int, c_ptr, c_f_pointer
    use dimensions_mod, only : np, nlev, nelemd
    !
    ! Inputs
    !
    integer (kind=c_int), intent(in) :: nets, nete
    type (c_ptr), intent(in) :: div_vdp_ptr, vgrad_p_ptr, p_ptr, omega_p_ptr
    !
    ! Locals
    !
    integer :: ie, k, i, j
    real (kind=real_kind), dimension(:,:,:,:), pointer :: div_vdp  ! divergence
    real (kind=real_kind), dimension(:,:,:,:), pointer :: vgrad_p  ! v.grad(p)
    real (kind=real_kind), dimension(:,:,:,:), pointer :: p        ! layer thicknesses (pressure)
    real (kind=real_kind), dimension(:,:,:,:), pointer :: omega_p  ! vertical pressure velocity
    real (kind=real_kind) term             ! one half of basic term in omega/p summation
    real (kind=real_kind) Ckk,Ckl          ! diagonal term of energy conversion matrix
    real (kind=real_kind) suml(np,np)      ! partial sum over l = (1, k-1)

    ! Cast the pointers
    call c_f_pointer (div_vdp_ptr, div_vdp, [np, np, nlev, nelemd])
    call c_f_pointer (vgrad_p_ptr, vgrad_p, [np, np, nlev, nelemd])
    call c_f_pointer (p_ptr,       p,       [np, np, nlev, nelemd])
    call c_f_pointer (omega_p_ptr, omega_p, [np, np, nlev, nelemd])

    do ie=nets, nete
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,j,i,ckk,term,ckl)
#endif
      do j=1,np   !   Loop inversion (AAM)
        do i=1,np
           ckk = 0.5d0/p(i,j,1,ie)
           term = div_vdp(i,j,1,ie)
           omega_p(i,j,1,ie) = vgrad_p(i,j,1,ie)/p(i,j,1,ie)
           omega_p(i,j,1,ie) = omega_p(i,j,1,ie) - ckk*term
           suml(i,j) = term
        end do

        do k=2,nlev-1
          do i=1,np
            ckk = 0.5d0/p(i,j,k,ie)
            ckl = 2*ckk
            term = div_vdp(i,j,k,ie)
            omega_p(i,j,k,ie) = vgrad_p(i,j,k,ie)/p(i,j,k,ie)
            omega_p(i,j,k,ie) = omega_p(i,j,k,ie) - ckl*suml(i,j) - ckk*term
            suml(i,j) = suml(i,j) + term
          end do
        end do

        do i=1,np
          ckk = 0.5d0/p(i,j,nlev,ie)
          ckl = 2*ckk
          term = div_vdp(i,j,nlev,ie)
          omega_p(i,j,nlev,ie) = vgrad_p(i,j,nlev,ie)/p(i,j,nlev,ie)
          omega_p(i,j,nlev,ie) = omega_p(i,j,nlev,ie) - ckl*suml(i,j) - ckk*term
        end do
      end do
    end do
  end subroutine caar_preq_omega_ps_f90

  subroutine caar_compute_eta_dot_dpdn_f90(nets, nete, eta_dot_dpdn_ptr, T_vadv_ptr, v_vadv_ptr,    &
                                           elem_derived_eta_dot_dpdn_ptr, elem_derived_omega_p_ptr, &
                                           omega_p_ptr, eta_ave_w) bind(c)
    use iso_c_binding,  only : c_int, c_ptr, c_f_pointer
    use dimensions_mod, only : np, nlev, nelemd
    !
    ! Inputs
    !
    integer (kind=c_int), intent(in) :: nets, nete
    type (c_ptr), intent(in) :: eta_dot_dpdn_ptr, T_vadv_ptr, v_vadv_ptr, omega_p_ptr
    type (c_ptr), intent(in) :: elem_derived_eta_dot_dpdn_ptr, elem_derived_omega_p_ptr
    real (kind=real_kind), intent(in) :: eta_ave_w
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:,:,:),   pointer :: eta_dot_dpdn
    real (kind=real_kind), dimension(:,:,:,:),   pointer :: T_vadv
    real (kind=real_kind), dimension(:,:,:,:,:), pointer :: v_vadv
    real (kind=real_kind), dimension(:,:,:,:),   pointer :: omega_p
    real (kind=real_kind), dimension(:,:,:,:),   pointer :: elem_derived_eta_dot_dpdn
    real (kind=real_kind), dimension(:,:,:,:),   pointer :: elem_derived_omega_p
    integer :: ie, k

    ! Cast the pointers
    call c_f_pointer (eta_dot_dpdn_ptr,              eta_dot_dpdn,              [np,np,nlev+1,nelemd])
    call c_f_pointer (T_vadv_ptr,                    T_vadv,                    [np,np,nlev,nelemd])
    call c_f_pointer (v_vadv_ptr,                    v_vadv,                    [np,np,2,nlev,nelemd])
    call c_f_pointer (omega_p_ptr,                   omega_p,                   [np,np,nlev,nelemd])
    call c_f_pointer (elem_derived_eta_dot_dpdn_ptr, elem_derived_eta_dot_dpdn, [np,np,nlev+1,nelemd])
    call c_f_pointer (elem_derived_omega_p_ptr,      elem_derived_omega_p,      [np,np,nlev,nelemd])

    do ie=nets,nete
      ! ==================================================
      ! Compute eta_dot_dpdn
      ! save sdot_sum as this is the -RHS of ps_v equation
      ! ==================================================
!      if (rsplit>0) then
         ! VERTICALLY LAGRANGIAN:   no vertical motion
         eta_dot_dpdn=0
         T_vadv=0
         v_vadv=0
!      else
!         do k=1,nlev
!            ! ==================================================
!            ! add this term to PS equation so we exactly conserve dry mass
!            ! ==================================================
!            sdot_sum(:,:) = sdot_sum(:,:) + divdp(:,:,k)
!            eta_dot_dpdn(:,:,k+1) = sdot_sum(:,:)
!         end do
!
!
!         ! ===========================================================
!         ! at this point, eta_dot_dpdn contains integral_etatop^eta[ divdp ]
!         ! compute at interfaces:
!         !    eta_dot_dpdn = -dp/dt - integral_etatop^eta[ divdp ]
!         ! for reference: at mid layers we have:
!         !    omega = v grad p  - integral_etatop^eta[ divdp ]
!         ! ===========================================================
!         do k=1,nlev-1
!            eta_dot_dpdn(:,:,k+1) = hvcoord%hybi(k+1)*sdot_sum(:,:) - eta_dot_dpdn(:,:,k+1)
!         end do
!
!         eta_dot_dpdn(:,:,1     ) = 0.0D0
!         eta_dot_dpdn(:,:,nlev+1) = 0.0D0
!
!         ! ===========================================================
!         ! Compute vertical advection of T and v from eq. CCM2 (3.b.1)
!         ! ==============================================
!         call preq_vertadv(elem(ie)%state%T(:,:,:,n0),&
!                           elem(ie)%state%v(:,:,:,:,n0), &
!                           eta_dot_dpdn,rdp,T_vadv,v_vadv)
!      endif
      ! ================================
      ! accumulate mean vertical flux:
      ! ================================
#if (defined COLUMN_OPENMP)
      !$omp parallel do private(k)
#endif
      do k=1,nlev  !  Loop index added (AAM)
         elem_derived_eta_dot_dpdn(:,:,k,ie) = &
              elem_derived_eta_dot_dpdn(:,:,k,ie) + eta_ave_w*eta_dot_dpdn(:,:,k,ie)
         elem_derived_omega_p(:,:,k,ie) = &
              elem_derived_omega_p(:,:,k,ie) + eta_ave_w*omega_p(:,:,k,ie)
      enddo
      elem_derived_eta_dot_dpdn(:,:,nlev+1,ie) = &
            elem_derived_eta_dot_dpdn(:,:,nlev+1,ie) + eta_ave_w*eta_dot_dpdn(:,:,nlev+1,ie)
    end do
  end subroutine caar_compute_eta_dot_dpdn_f90

  subroutine caar_compute_phi_kinetic_energy_f90(nets, nete, n0, dvv_ptr, p_ptr, grad_p_ptr, &
                                                 v_vadv_ptr, T_vadv_ptr, elem_Dinv_ptr,      &
                                                 elem_fcor_ptr, vort_ptr, elem_state_T_ptr,  &
                                                 elem_state_v_ptr, elem_derived_phi_ptr,     &
                                                 elem_derived_pecnd_ptr,                     &
                                                 kappa_star_ptr, T_v_ptr, omega_p_ptr,       &
                                                 ttens_ptr, vtens1_ptr, vtens2_ptr) bind(c)
    use iso_c_binding,      only : c_int, c_ptr, c_f_pointer
    use derivative_mod,     only : derivative_t, gradient_sphere
    use dimensions_mod,     only : np, nlev, nelemd
    use element_mod,        only : timelevels
    use physical_constants, only : Rgas
    !
    ! Inputs
    !
    integer (kind=c_int), intent(in) :: nets, nete, n0
    type (c_ptr), intent(in) :: dvv_ptr, T_v_ptr, p_ptr, grad_p_ptr, v_vadv_ptr, T_vadv_ptr
    type (c_ptr), intent(in) :: elem_state_T_ptr, elem_state_v_ptr, elem_derived_phi_ptr
    type (c_ptr), intent(in) :: elem_Dinv_ptr, elem_fcor_ptr, vort_ptr
    type (c_ptr), intent(in) :: elem_derived_pecnd_ptr, kappa_star_ptr
    type (c_ptr), intent(in) :: vtens1_ptr, vtens2_ptr, ttens_ptr, omega_p_ptr
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:),         pointer :: dvv
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: p
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: grad_p
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: vort
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: v_vadv
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: T_vadv
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: kappa_star
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: T_v
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: omega_p
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: vtens1
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: vtens2
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: ttens
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: elem_Dinv
    real (kind=real_kind), dimension(:,:,:),       pointer :: elem_fcor
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: elem_state_Temp
    real (kind=real_kind), dimension(:,:,:,:,:,:), pointer :: elem_state_v
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: elem_derived_phi
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: elem_derived_pecnd
    integer :: ie, k, i, j
    real (kind=real_kind) :: v1, v2, gpterm, glnps1, glnps2, E
    real (kind=real_kind), dimension(np, np, 2)    :: vtemp
    real (kind=real_kind), dimension(np, np)       :: vgrad_T
    real (kind=real_kind), dimension(np, np)       :: Ephi
    type (derivative_t) :: deriv


    ! Cast the pointers
    call c_f_pointer (dvv_ptr,                dvv,                [np,np])
    call c_f_pointer (T_v_ptr,                T_v,                [np,np,nlev,nelemd])
    call c_f_pointer (p_ptr,                  p,                  [np,np,nlev,nelemd])
    call c_f_pointer (grad_p_ptr,             grad_p,             [np,np,2,nlev,nelemd])
    call c_f_pointer (vort_ptr,               vort,               [np,np,nlev,nelemd])
    call c_f_pointer (kappa_star_ptr,         kappa_star,         [np, np, nlev, nelemd])
    call c_f_pointer (omega_p_ptr,            omega_p,            [np, np, nlev, nelemd])
    call c_f_pointer (t_vadv_ptr,             T_vadv,             [np,np,nlev,nelemd])
    call c_f_pointer (v_vadv_ptr,             v_vadv,             [np,np,2,nlev,nelemd])
    call c_f_pointer (vtens1_ptr,             vtens1,             [np,np,nlev,nelemd])
    call c_f_pointer (vtens2_ptr,             vtens2,             [np,np,nlev,nelemd])
    call c_f_pointer (ttens_ptr,              ttens,              [np,np,nlev,nelemd])
    call c_f_pointer (elem_state_T_ptr,       elem_state_Temp,    [np,np,nlev,timelevels,nelemd])
    call c_f_pointer (elem_Dinv_ptr,          elem_Dinv,          [np,np,2,2,nelemd])
    call c_f_pointer (elem_fcor_ptr,          elem_fcor,          [np,np,nelemd])
    call c_f_pointer (elem_state_v_ptr,       elem_state_v,       [np,np,2,nlev,timelevels,nelemd])
    call c_f_pointer (elem_derived_phi_ptr,   elem_derived_phi,   [np,np,nlev,nelemd])
    call c_f_pointer (elem_derived_pecnd_ptr, elem_derived_pecnd, [np,np,nlev,nelemd])

    ! Setup the derivative
    deriv%dvv = dvv

    do ie=nets,nete
      ! ==============================================
      ! Compute phi + kinetic energy term: 10*nv*nv Flops
      ! ==============================================
#if (defined COLUMN_OPENMP)
      !$omp parallel do private(k,i,j,v1,v2,E,Ephi,vtemp,vgrad_T,gpterm,glnps1,glnps2)
#endif
      vertloop: do k=1,nlev
         do j=1,np
            do i=1,np
               v1     = elem_state_v(i,j,1,k,n0,ie)
               v2     = elem_state_v(i,j,2,k,n0,ie)
               E = 0.5D0*( v1*v1 + v2*v2 )
               Ephi(i,j)=E+elem_derived_phi(i,j,k,ie)+elem_derived_pecnd(i,j,k,ie)
            end do
         end do
         ! ================================================
         ! compute gradp term (ps/p)*(dp/dps)*T
         ! ================================================
         vtemp(:,:,:) = gradient_sphere(elem_state_Temp(:,:,k,n0,ie),deriv,elem_Dinv(:,:,:,:,ie))
         do j=1,np
            do i=1,np
               v1     = elem_state_v(i,j,1,k,n0,ie)
               v2     = elem_state_v(i,j,2,k,n0,ie)
               vgrad_T(i,j) =  v1*vtemp(i,j,1) + v2*vtemp(i,j,2)
            end do
         end do

         ! vtemp = grad ( E + PHI )
         vtemp = gradient_sphere(Ephi(:,:),deriv,elem_Dinv(:,:,:,:,ie))

         do j=1,np
           do i=1,np
              gpterm = T_v(i,j,k,ie)/p(i,j,k,ie)
              glnps1 = Rgas*gpterm*grad_p(i,j,1,k,ie)
              glnps2 = Rgas*gpterm*grad_p(i,j,2,k,ie)

              v1     = elem_state_v(i,j,1,k,n0,ie)
              v2     = elem_state_v(i,j,2,k,n0,ie)

              vtens1(i,j,k,ie) = - v_vadv(i,j,1,k,ie)                           &
                   + v2*(elem_fcor(i,j,ie) + vort(i,j,k,ie))        &
                   - vtemp(i,j,1) - glnps1
            !
            ! phl: add forcing term to zonal wind u
            !
            vtens2(i,j,k,ie) = - v_vadv(i,j,2,k,ie)                            &
                 - v1*(elem_fcor(i,j,ie) + vort(i,j,k,ie))        &
                 - vtemp(i,j,2) - glnps2
            !
            ! phl: add forcing term to meridional wind v
            !
            ttens(i,j,k,ie) = - T_vadv(i,j,k,ie) - vgrad_T(i,j) + kappa_star(i,j,k,ie)*T_v(i,j,k,ie)*omega_p(i,j,k,ie)
            !
            ! phl: add forcing term to T
            !

          end do
        end do
      end do vertloop
    end do
  end subroutine caar_compute_phi_kinetic_energy_f90

  subroutine caar_update_states_f90(nets, nete, nm1, np1, dt2, &
                                    vdp_ptr, div_vdp_ptr, vtens1_ptr, vtens2_ptr, ttens_ptr, &
                                    elem_Dinv_ptr, elem_metdet_ptr, elem_spheremp_ptr,       &
                                    elem_state_v_ptr, elem_state_T_ptr, elem_state_dp3d_ptr, &
                                    elem_sub_elem_mass_flux_ptr, eta_dot_dpdn_ptr, eta_ave_w,&
                                    elem_state_ps_v_ptr, sdot_sum_ptr) bind(c)
    use iso_c_binding,  only : c_int, c_ptr, c_f_pointer
    use control_mod,    only : rsplit
    use derivative_mod, only : subcell_div_fluxes
    use dimensions_mod, only : nlev, ntrac, nelemd, np, nc
    use element_mod,    only : timelevels
    !
    ! Inputs
    !
    integer (kind=c_int),  intent(in) :: nets, nete, nm1, np1
    real (kind=real_kind), intent(in) :: dt2, eta_ave_w
    type (c_ptr), intent(in) :: vdp_ptr, div_vdp_ptr, vtens1_ptr, vtens2_ptr, ttens_ptr
    type (c_ptr), intent(in) :: elem_Dinv_ptr, elem_metdet_ptr, elem_spheremp_ptr
    type (c_ptr), intent(in) :: elem_state_v_ptr, elem_state_ps_v_ptr, elem_state_T_ptr, elem_state_dp3d_ptr
    type (c_ptr), intent(in) :: elem_sub_elem_mass_flux_ptr, eta_dot_dpdn_ptr, sdot_sum_ptr
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: elem_Dinv
    real (kind=real_kind), dimension(:,:,:),       pointer :: elem_metdet
    real (kind=real_kind), dimension(:,:,:),       pointer :: elem_spheremp
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: elem_state_ps_v
    real (kind=real_kind), dimension(:,:,:,:,:,:), pointer :: elem_state_v
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: elem_state_T
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: elem_state_dp3d
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: elem_sub_elem_mass_flux
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: vdp
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: div_vdp
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: eta_dot_dpdn
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: vtens1
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: vtens2
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: ttens
    real (kind=real_kind), dimension(:,:,:),       pointer :: sdot_sum
    real (kind=real_kind), dimension(np, np, 2)            :: v
    real (kind=real_kind), dimension(nc, nc, 4)            :: tempflux
    integer :: ie, k

    ! Cast the pointers
    call c_f_pointer(vdp_ptr,                     vdp,                     [np, np, 2, nlev, nelemd])
    call c_f_pointer(div_vdp_ptr,                 div_vdp,                 [np, np, nlev, nelemd])
    call c_f_pointer(vtens1_ptr,                  vtens1,                  [np, np, nlev, nelemd])
    call c_f_pointer(vtens2_ptr,                  vtens2,                  [np, np, nlev, nelemd])
    call c_f_pointer(ttens_ptr,                   ttens,                   [np, np, nlev, nelemd])
    call c_f_pointer(elem_Dinv_ptr,               elem_Dinv,               [np, np, 2, 2, nelemd])
    call c_f_pointer(elem_metdet_ptr,             elem_metdet,             [np, np, nelemd])
    call c_f_pointer(elem_spheremp_ptr,           elem_spheremp,           [np, np, nelemd])
    call c_f_pointer(elem_state_ps_v_ptr,         elem_state_ps_v,         [np, np, timelevels, nelemd])
    call c_f_pointer(elem_state_v_ptr,            elem_state_v,            [np, np, 2, nlev, timelevels, nelemd])
    call c_f_pointer(elem_state_T_ptr,            elem_state_T,            [np, np, nlev, timelevels, nelemd])
    call c_f_pointer(elem_state_dp3d_ptr,         elem_state_dp3d,         [np, np, nlev, timelevels, nelemd])
    call c_f_pointer(elem_sub_elem_mass_flux_ptr, elem_sub_elem_mass_flux, [nc, nc, 4, nlev, nelemd])
    call c_f_pointer(eta_dot_dpdn_ptr,            eta_dot_dpdn,            [np, np, nlev+1, nelemd])
    call c_f_pointer(sdot_sum_ptr,                sdot_sum,                [np, np, nelemd])

    do ie=nets,nete
      ! =========================================================
      ! local element timestep, store in np1.
      ! note that we allow np1=n0 or nm1
      ! apply mass matrix
      ! =========================================================
!     if (dt2<0) then
!        ! calling program just wanted DSS'd RHS, skip time advance
!#if (defined COLUMN_OPENMP)
!!$omp parallel do private(k,tempflux)
!#endif
!        do k=1,nlev
!           elem(ie)%state%v(:,:,1,k,np1) = elem(ie)%spheremp(:,:)*vtens1(:,:,k)
!           elem(ie)%state%v(:,:,2,k,np1) = elem(ie)%spheremp(:,:)*vtens2(:,:,k)
!           elem(ie)%state%T(:,:,k,np1) = elem(ie)%spheremp(:,:)*ttens(:,:,k)
!           if (rsplit>0) &
!              elem(ie)%state%dp3d(:,:,k,np1) = -elem(ie)%spheremp(:,:)*&
!              (divdp(:,:,k) + eta_dot_dpdn(:,:,k+1)-eta_dot_dpdn(:,:,k))
!           if (0<rsplit.and.0<ntrac.and.eta_ave_w.ne.0.) then
!              v(:,:,1) =  elem(ie)%Dinv(:,:,1,1)*vdp(:,:,1,k) + elem(ie)%Dinv(:,:,1,2)*vdp(:,:,2,k)
!              v(:,:,2) =  elem(ie)%Dinv(:,:,2,1)*vdp(:,:,1,k) + elem(ie)%Dinv(:,:,2,2)*vdp(:,:,2,k)
!              tempflux =  eta_ave_w*subcell_div_fluxes(v, np, nc, elem(ie)%metdet)
!              elem(ie)%sub_elem_mass_flux(:,:,:,k) = elem(ie)%sub_elem_mass_flux(:,:,:,k) - tempflux
!           end if
!        enddo
!        elem(ie)%state%ps_v(:,:,np1) = -elem(ie)%spheremp(:,:)*sdot_sum
!      else
#if (defined COLUMN_OPENMP)
!$omp parallel do private(k,tempflux)
#endif
        do k=1,nlev
          elem_state_v(:,:,1,k,np1,ie) = elem_spheremp(:,:,ie)*( elem_state_v(:,:,1,k,nm1,ie) + dt2*vtens1(:,:,k,ie) )
          elem_state_v(:,:,2,k,np1,ie) = elem_spheremp(:,:,ie)*( elem_state_v(:,:,2,k,nm1,ie) + dt2*vtens2(:,:,k,ie) )
          elem_state_T(:,:,k,np1,ie) = elem_spheremp(:,:,ie) * (elem_state_T(:,:,k,nm1,ie) + dt2*ttens(:,:,k,ie))
          elem_state_dp3d(:,:,k,np1,ie) = elem_spheremp(:,:,ie) * (elem_state_dp3d(:,:,k,nm1,ie) - &
                                       dt2 * (div_vdp(:,:,k,ie) + eta_dot_dpdn(:,:,k+1,ie)-eta_dot_dpdn(:,:,k,ie)))

          if (rsplit>0.and.0<ntrac.and.eta_ave_w.ne.0.) then
            v(:,:,1) = elem_Dinv(:,:,1,1,ie)*vdp(:,:,1,k,ie) + elem_Dinv(:,:,1,2,ie)*vdp(:,:,2,k,ie)
            v(:,:,2) = elem_Dinv(:,:,2,1,ie)*vdp(:,:,1,k,ie) + elem_Dinv(:,:,2,2,ie)*vdp(:,:,2,k,ie)
            tempflux = eta_ave_w*subcell_div_fluxes(v, np, nc,elem_metdet(:,:,ie))
            elem_sub_elem_mass_flux(:,:,:,k,ie) = elem_sub_elem_mass_flux(:,:,:,k,ie) - tempflux
          end if
        enddo
        elem_state_ps_v(:,:,np1,ie) = elem_spheremp(:,:,ie)*(elem_state_ps_v(:,:,nm1,ie) - dt2*sdot_sum(:,:,ie) )
!      endif
    end do
  end subroutine caar_update_states_f90

  subroutine caar_energy_diagnostics_f90() bind(c)
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

!     if (compute_diagnostics) then
!        elem(ie)%accum%KEhorz1=0
!        elem(ie)%accum%KEhorz2=0
!        elem(ie)%accum%IEhorz1=0
!        elem(ie)%accum%IEhorz2=0
!        elem(ie)%accum%IEhorz1_wet=0
!        elem(ie)%accum%IEhorz2_wet=0
!        elem(ie)%accum%KEvert1=0
!        elem(ie)%accum%KEvert2=0
!        elem(ie)%accum%IEvert1=0
!        elem(ie)%accum%IEvert2=0
!        elem(ie)%accum%IEvert1_wet=0
!        elem(ie)%accum%IEvert2_wet=0
!        elem(ie)%accum%T1=0
!        elem(ie)%accum%T2=0
!        elem(ie)%accum%T2_s=0
!        elem(ie)%accum%S1=0
!        elem(ie)%accum%S1_wet=0
!        elem(ie)%accum%S2=0
!
!        do j=1,np
!           do i=1,np
!              elem(ie)%accum%S2(i,j) = elem(ie)%accum%S2(i,j) - &
!                   sdot_sum(i,j)*elem(ie)%state%phis(i,j)
!           enddo
!        enddo
!        do k=1,nlev
!           ! vtemp = grad_E(:,:,k)
!           do j=1,np
!              do i=1,np
!                 v1     = elem(ie)%state%v(i,j,1,k,n0)
!                 v2     = elem(ie)%state%v(i,j,2,k,n0)
!                 Ephi(i,j)=0.5D0*( v1*v1 + v2*v2 )
!              enddo
!           enddo
!           vtemp = gradient_sphere(Ephi,deriv,elem(ie)%Dinv)
!           do j=1,np
!              do i=1,np
!                 ! dp/dn u dot grad(E)
!                 v1     = elem(ie)%state%v(i,j,1,k,n0)
!                 v2     = elem(ie)%state%v(i,j,2,k,n0)
!                 elem(ie)%accum%KEhorz2(i,j) = elem(ie)%accum%KEhorz2(i,j) + &
!                      (v1*vtemp(i,j,1)  + v2*vtemp(i,j,2))*dp(i,j,k)
!                 ! E div( u dp/dn )
!                 elem(ie)%accum%KEhorz1(i,j) = elem(ie)%accum%KEhorz1(i,j) +
!Ephi(i,j)*divdp(i,j,k)
!
!                 ! Cp T div( u dp/dn)   ! dry horizontal advection component
!                 elem(ie)%accum%IEhorz1(i,j) = elem(ie)%accum%IEhorz1(i,j) +
!Cp*elem(ie)%state%T(i,j,k,n0)*divdp(i,j,k)
!
!
!              enddo
!           enddo
!           ! vtemp = grad_phi(:,:,k)
!           vtemp =
!gradient_sphere(elem(ie)%derived%phi(:,:,k),deriv,elem(ie)%Dinv)
!           do j=1,np
!              do i=1,np
!                 v1     = elem(ie)%state%v(i,j,1,k,n0)
!                 v2     = elem(ie)%state%v(i,j,2,k,n0)
!                 E = 0.5D0*( v1*v1 + v2*v2 )
!                 ! NOTE:  Cp_star = Cp + (Cpv-Cp)*q
!                 ! advection terms can thus be broken into two components: dry
!and wet
!                 ! dry components cancel exactly
!                 ! wet components should cancel exactly
!                 !
!                 ! some diagnostics
!                 ! e = eta_dot_dpdn()
!                 de =  eta_dot_dpdn(i,j,k+1)-eta_dot_dpdn(i,j,k)
!                 ! Cp T de/dn, integral dn:
!                 elem(ie)%accum%IEvert1(i,j)=elem(ie)%accum%IEvert1(i,j) +
!Cp*elem(ie)%state%T(i,j,k,n0)*de
!                 ! E de/dn
!                 elem(ie)%accum%KEvert1(i,j)=elem(ie)%accum%KEvert1(i,j) + E*de
!                 ! Cp T_vadv dp/dn
!                 elem(ie)%accum%IEvert2(i,j)=elem(ie)%accum%IEvert2(i,j) +
!Cp*T_vadv(i,j,k)*dp(i,j,k)
!                 ! dp/dn V dot V_vadv
!                 elem(ie)%accum%KEvert2(i,j)=elem(ie)%accum%KEvert2(i,j) +
!(v1*v_vadv(i,j,1,k) + v2*v_vadv(i,j,2,k)) *dp(i,j,k)
!
!                 ! IEvert1_wet():  (Cpv-Cp) T Qdp_vadv  (Q equation)
!                 ! IEvert2_wet():  (Cpv-Cp) Qdp T_vadv   T equation
!                 if (use_cpstar==1) then
!                 elem(ie)%accum%IEvert2_wet(i,j)=elem(ie)%accum%IEvert2_wet(i,j)
!+&
!                      (Cpwater_vapor-Cp)*elem(ie)%state%Q(i,j,k,1)*T_vadv(i,j,k)*dp(i,j,k)
!                 endif
!
!                 gpterm = T_v(i,j,k)/p(i,j,k)
!                 elem(ie)%accum%T1(i,j) = elem(ie)%accum%T1(i,j) - &
!                      Rgas*gpterm*(grad_p(i,j,1,k)*v1 +
!grad_p(i,j,2,k)*v2)*dp(i,j,k)
!
!                 elem(ie)%accum%T2(i,j) = elem(ie)%accum%T2(i,j) - &
!                      (vtemp(i,j,1)*v1 + vtemp(i,j,2)*v2)*dp(i,j,k)
!
!                 ! S1 = < Cp_star dp/dn , RT omega_p/cp_star >
!                 elem(ie)%accum%S1(i,j) = elem(ie)%accum%S1(i,j) + &
!                      Rgas*T_v(i,j,k)*omega_p(i,j,k)*dp(i,j,k)
!
!                 ! cp_star = cp + cp2
!                 if (use_cpstar==1) then
!                 cp2 = (Cpwater_vapor-Cp)*elem(ie)%state%Q(i,j,k,1)
!                 cp_ratio = cp2/(cp+cp2)
!                 elem(ie)%accum%S1_wet(i,j) = elem(ie)%accum%S1_wet(i,j) + &
!                      cp_ratio*(Rgas*T_v(i,j,k)*omega_p(i,j,k)*dp(i,j,k))
!                 endif
!
!                 elem(ie)%accum%CONV(i,j,:,k)=-Rgas*gpterm*grad_p(i,j,:,k)-vtemp(i,j,:)
!              enddo
!           enddo
!
!           vtemp(:,:,:) =
!gradient_sphere(elem(ie)%state%phis(:,:),deriv,elem(ie)%Dinv)
!           do j=1,np
!              do i=1,np
!                 v1     = elem(ie)%state%v(i,j,1,k,n0)
!                 v2     = elem(ie)%state%v(i,j,2,k,n0)
!                 elem(ie)%accum%T2_s(i,j) = elem(ie)%accum%T2_s(i,j) - &
!                      (vtemp(i,j,1)*v1 + vtemp(i,j,2)*v2)*dp(i,j,k)
!              enddo
!           enddo
!
!           vtemp(:,:,:)   =
!gradient_sphere(elem(ie)%state%T(:,:,k,n0),deriv,elem(ie)%Dinv)
!           do j=1,np
!              do i=1,np
!                 v1     = elem(ie)%state%v(i,j,1,k,n0)
!                 v2     = elem(ie)%state%v(i,j,2,k,n0)
!
!                 ! Cp dp/dn u dot gradT
!                 elem(ie)%accum%IEhorz2(i,j) = elem(ie)%accum%IEhorz2(i,j) + &
!                      Cp*(v1*vtemp(i,j,1) + v2*vtemp(i,j,2))*dp(i,j,k)
!
!                 if (use_cpstar==1) then
!                 elem(ie)%accum%IEhorz2_wet(i,j) =
!elem(ie)%accum%IEhorz2_wet(i,j) + &
!                      (Cpwater_vapor-Cp)*elem(ie)%state%Q(i,j,k,1)*&
!                      (v1*vtemp(i,j,1) + v2*vtemp(i,j,2))*dp(i,j,k)
!                 endif
!
!              enddo
!           enddo
!
!        enddo
!     endif
  end subroutine caar_energy_diagnostics_f90

  subroutine caar_flip_f90_array_2D (f90_array, c_1d_array, to_c)
    !
    ! Inputs
    !
    real (kind=real_kind), dimension(:,:), intent(inout) :: f90_array
    real (kind=real_kind), dimension(:),   intent(inout) :: c_1d_array
    logical, intent(in) :: to_c
    !
    ! Locals
    !
    integer, dimension(2) :: dims
    integer :: i, j, iter

    dims = SHAPE(f90_array)

    iter = 1
    if (to_c) then
      do i=1,dims(1)
        do j=1,dims(2)
          c_1d_array (iter) = f90_array(i,j)
          iter = iter + 1
        end do
      end do
    else
      do i=1,dims(1)
        do j=1,dims(2)
          f90_array(i,j) = c_1d_array (iter)
          iter = iter + 1
        end do
      end do
    end if
  end subroutine caar_flip_f90_array_2D

  subroutine caar_flip_f90_array_3D (f90_array, c_1d_array, to_c)
    !
    ! Inputs
    !
    real (kind=real_kind), dimension(:,:,:), intent(inout) :: f90_array
    real (kind=real_kind), dimension(:),     intent(inout) :: c_1d_array
    logical, intent(in) :: to_c
    !
    ! Locals
    !
    integer, dimension(3) :: dims
    integer :: i, j, k, iter

    dims = SHAPE(f90_array)

    iter = 1
    if (to_c) then
      do k=1,dims(3)
        do i=1,dims(1)
          do j=1,dims(2)
            c_1d_array (iter) = f90_array(i,j,k)
            iter = iter + 1
          end do
        end do
      end do
    else
      do k=1,dims(3)
        do i=1,dims(1)
          do j=1,dims(2)
            f90_array(i,j,k) = c_1d_array (iter)
            iter = iter + 1
          end do
        end do
      end do
    end if
  end subroutine caar_flip_f90_array_3D

  subroutine caar_flip_f90_array_4D (f90_array, c_1d_array, to_c)
    !
    ! Inputs
    !
    real (kind=real_kind), dimension(:,:,:,:), intent(inout) :: f90_array
    real (kind=real_kind), dimension(:),       intent(inout) :: c_1d_array
    logical, intent(in) :: to_c
    !
    ! Locals
    !
    integer, dimension(4) :: dims
    integer :: i, j, k, l, iter

    dims = SHAPE(f90_array)

    iter = 1
    if (to_c) then
      do l=1,dims(4)
        do k=1,dims(3)
          do i=1,dims(1)
            do j=1,dims(2)
              c_1d_array (iter) = f90_array(i,j,k,l)
              iter = iter + 1
            end do
          end do
        end do
      end do
    else
      do l=1,dims(4)
        do k=1,dims(3)
          do i=1,dims(1)
            do j=1,dims(2)
              f90_array(i,j,k,l) = c_1d_array (iter)
              iter = iter + 1
            end do
          end do
        end do
      end do
    end if
  end subroutine caar_flip_f90_array_4D

  subroutine caar_flip_f90_array_5D (f90_array, c_1d_array, to_c)
    !
    ! Inputs
    !
    real (kind=real_kind), dimension(:,:,:,:,:), intent(inout) :: f90_array
    real (kind=real_kind), dimension(:),         intent(inout) :: c_1d_array
    logical, intent(in) :: to_c
    !
    ! Locals
    !
    integer, dimension(5) :: dims
    integer :: i, j, k, l, m, iter

    dims = SHAPE(f90_array)

    iter = 1
    if (to_c) then
      do m=1,dims(5)
        do l=1,dims(4)
          do k=1,dims(3)
            do i=1,dims(1)
              do j=1,dims(2)
                c_1d_array (iter) = f90_array(i,j,k,l,m)
                iter = iter + 1
              end do
            end do
          end do
        end do
      end do
    else
      do m=1,dims(5)
        do l=1,dims(4)
          do k=1,dims(3)
            do i=1,dims(1)
              do j=1,dims(2)
                f90_array(i,j,k,l,m) = c_1d_array (iter)
                iter = iter + 1
              end do
            end do
          end do
        end do
      end do
    end if
  end subroutine caar_flip_f90_array_5D

  subroutine caar_flip_f90_array_6D (f90_array, c_1d_array, to_c)
    !
    ! Inputs
    !
    real (kind=real_kind), dimension(:,:,:,:,:,:), intent(inout) :: f90_array
    real (kind=real_kind), dimension(:),           intent(inout) :: c_1d_array
    logical, intent(in) :: to_c
    !
    ! Locals
    !
    integer, dimension(6) :: dims
    integer :: i, j, k, l, m, n, iter

    dims = SHAPE(f90_array)

    iter = 1
    if (to_c) then
      do n=1,dims(6)
        do m=1,dims(5)
          do l=1,dims(4)
            do k=1,dims(3)
              do i=1,dims(1)
                do j=1,dims(2)
                  c_1d_array (iter) = f90_array(i,j,k,l,m,n)
                  iter = iter + 1
                end do
              end do
            end do
          end do
        end do
      end do
    else
      do n=1,dims(6)
        do m=1,dims(5)
          do l=1,dims(4)
            do k=1,dims(3)
              do i=1,dims(1)
                do j=1,dims(2)
                  f90_array(i,j,k,l,m,n) = c_1d_array (iter)
                  iter = iter + 1
                end do
              end do
            end do
          end do
        end do
      end do
    end if
  end subroutine caar_flip_f90_array_6D

  subroutine caar_flip_f90_tensor2d (f90_tensor, c_1d_tensor)
    !
    ! Inputs
    !
    real (kind=real_kind), dimension(:,:,:,:,:), intent(inout) :: f90_tensor
    real (kind=real_kind), dimension(:),         intent(inout) :: c_1d_tensor
    !
    ! Locals
    !
    integer, dimension(5) :: dims
    integer :: i, j, k, l, m, iter

    dims = SHAPE(f90_tensor)

    iter = 1
    do m=1,dims(5)
      do l=1,dims(3)
        do k=1,dims(4)
          do i=1,dims(1)
            do j=1,dims(2)
              c_1d_tensor (iter) = f90_tensor(i,j,l,k,m)
              iter = iter + 1
            end do
          end do
        end do
      end do
    end do
  end subroutine caar_flip_f90_tensor2d

  subroutine caar_flip_f90_Qdp_array (f90_Qdp, c_1d_Qdp)
    !
    ! Inputs
    !
    real (kind=real_kind), dimension(:,:,:,:,:,:), intent(inout) :: f90_Qdp
    real (kind=real_kind), dimension(:),           intent(inout) :: c_1d_Qdp
    !
    ! Locals
    !
    integer, dimension(6) :: dims
    integer :: igp, jgp, k, qdim, tl, ie, iter

    dims = SHAPE(f90_Qdp)

    iter = 1
    do ie=1,dims(6)
      do qdim=1,dims(4)
        do tl=1,dims(5)
          do k=1,dims(3)
            do igp=1,dims(1)
              do jgp=1,dims(2)
                c_1d_Qdp (iter) = f90_Qdp(igp,jgp,k,qdim,tl,ie)
                iter = iter + 1
              end do
            end do
          end do
        end do
      end do
    end do
  end subroutine caar_flip_f90_Qdp_array

end module caar_subroutines_mod
