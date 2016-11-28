#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

module advance_rk_unit_test

  use derivative_mod, only: derivative_t
  use element_mod,    only: element_t
  use kinds,          only: real_kind

  implicit none

  interface
    subroutine init_physical_constants_c (rearth, g, omega, Rgas, Cp, p0, MVDAIR,&
                                          Rwater_vapor, Cpwater_vapor, kappa,&
                                          Rd_on_Rv, Cpd_on_Cpv, rrearth, Lc, pi) bind(c)
      use iso_c_binding, only: c_double

      real (kind=c_double), intent(in) :: rearth, g, omega, Rgas, Cp, p0, MVDAIR
      real (kind=c_double), intent(in) :: Rwater_vapor, Cpwater_vapor, kappa
      real (kind=c_double), intent(in) :: Rd_on_Rv, Cpd_on_Cpv, rrearth, Lc, pi
    end subroutine init_physical_constants_c

    subroutine init_derivative_c (Dvv, Dvv_diag, Dvv_twt, Mvv_twt,&
                                  Mfvm, Cfvm, legdg) bind(c)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(in) :: Dvv, Dvv_diag, Dvv_twt, Mvv_twt
      type(c_ptr), intent(in) :: Mfvm, Cfvm, legdg
    end subroutine init_derivative_c

    subroutine init_pointers_pool_c (elem_state_ps_cptr, elem_state_p_cptr, elem_state_v_cptr, metdet_cptr,&
                                     rmetdet_cptr, metinv_cptr, mp_cptr,vec_sphere2cart_cptr, spheremp_cptr,&
                                     rspheremp_cptr, hypervisc_cptr, tensor_visc_cptr, D_cptr, Dinv_cptr) bind (c)
      use iso_c_binding, only: c_ptr
      type (c_ptr), intent(in) :: elem_state_ps_cptr, elem_state_p_cptr, elem_state_v_cptr, metdet_cptr
      type (c_ptr), intent(in) :: rmetdet_cptr, metinv_cptr, mp_cptr, vec_sphere2cart_cptr, spheremp_cptr
      type (c_ptr), intent(in) :: rspheremp_cptr, hypervisc_cptr, tensor_visc_cptr, D_cptr, Dinv_cptr
    end subroutine init_pointers_pool_c

    subroutine init_control_parameters_c (hypervisc_scaling, hypervisc_power) bind(c)
      use iso_c_binding, only: c_int, c_double
      integer (kind=c_int), intent(in) :: hypervisc_scaling
      real (kind=c_double), intent(in) :: hypervisc_power
    end subroutine init_control_parameters_c

  end interface

  type (derivative_t), target, private          :: deriv
  type (element_t), allocatable, target, public :: elem(:)
  integer                                       :: first_time=1

contains

  subroutine init_derivative_f90() bind(c)
    use iso_c_binding, only: c_ptr, c_loc

    type(c_ptr) :: Dvv,Dvv_diag,Dvv_twt,Mvv_twt,Mfvm,Cfvm,legdg

    call random_number(deriv%Dvv)
    call random_number(deriv%Dvv_diag)
    call random_number(deriv%Dvv_twt)
    call random_number(deriv%Mvv_twt)
    call random_number(deriv%Mfvm)
    call random_number(deriv%Cfvm)
    call random_number(deriv%legdg)

    Dvv       = c_loc(deriv%Dvv)
    Dvv_diag  = c_loc(deriv%Dvv_diag)
    Dvv_twt   = c_loc(deriv%Dvv_twt)
    Mvv_twt   = c_loc(deriv%Mvv_twt)
    Mfvm      = c_loc(deriv%Mfvm)
    Cfvm      = c_loc(deriv%Cfvm)
    legdg     = c_loc(deriv%legdg)

    call init_derivative_c(Dvv, Dvv_diag, Dvv_twt, Mvv_twt, Mfvm, Cfvm, legdg);
  end subroutine init_derivative_f90

  subroutine init_physical_constants_f90() bind(c)
    use kinds,              only: real_kind
    use physical_constants, only: rearth, rrearth

    real (kind=real_kind) :: g, omega, Rgas, Cp, p0, MWDAIR, Rwater_vapor,&
                             Cpwater_vapor, kappa, Rd_on_Rv, Cpd_on_Cpv, Lc, pi

    ! Only these two are used in the kokkos kernels
    call random_number (rearth)
    call random_number (rrearth)

    call init_physical_constants_c(rearth,g, omega, Rgas, Cp, p0, MWDAIR, Rwater_vapor,&
                                   Cpwater_vapor, kappa, Rd_on_Rv, Cpd_on_Cpv, rrearth, Lc, pi)
  end subroutine init_physical_constants_f90

  subroutine pick_random_control_parameters_f90() bind(c)
    use iso_c_binding, only: c_int
    use kinds,         only: real_kind
    use control_mod,   only: hypervis_power, hypervis_scaling

    ! Only these two are used in the kokkos kernels
    call random_number (hypervis_power)
    call random_number (hypervis_scaling)
    hypervis_scaling = REAL(FLOOR(5*hypervis_scaling),real_kind)

    call init_control_parameters_c(INT(hypervis_scaling,c_int), hypervis_power)
  end subroutine pick_random_control_parameters_f90

  subroutine init_elem_f90 (nelems) bind(c)
    use iso_c_binding,  only: c_int, c_ptr, c_loc
    use kinds,          only: real_kind
    use dimensions_mod, only: np, nlev, nelemd
    use element_mod,    only: elem_state_ps, elem_state_p, elem_state_v,&
                              elem_metdet, elem_rmetdet, elem_metinv,&
                              elem_mp, elem_vec_sphere2cart, elem_spheremp,&
                              elem_rspheremp, elem_hyperviscosity,&
                              elem_tensorVisc, elem_D, elem_Dinv,&
                              setup_element_pointers_sw

    integer(kind=c_int), intent(in) :: nelems

    type (c_ptr) :: elem_state_ps_cptr, elem_state_p_cptr, elem_state_v_cptr, metdet_cptr
    type (c_ptr) :: rmetdet_cptr, metinv_cptr, mp_cptr, vec_sphere2cart_cptr, spheremp_cptr
    type (c_ptr) :: rspheremp_cptr, hypervisc_cptr, tensor_visc_cptr, D_cptr, Dinv_cptr

    ! Since many tests use the module in the same execution, variables my
    ! already be allocated
    if (.not. allocated(elem)) then
      nelemd = nelems
      first_time = 0
      allocate(elem(nelems))
      call setup_element_pointers_sw(elem)

      elem_state_ps_cptr    = c_loc(elem_state_ps)
      elem_state_p_cptr     = c_loc(elem_state_p)
      elem_state_v_cptr     = c_loc(elem_state_v)

      metdet_cptr           = c_loc(elem_metdet)
      rmetdet_cptr          = c_loc(elem_rmetdet)
      metinv_cptr           = c_loc(elem_metinv)
      mp_cptr               = c_loc(elem_mp)
      vec_sphere2cart_cptr  = c_loc(elem_vec_sphere2cart)
      spheremp_cptr         = c_loc(elem_spheremp)

      rspheremp_cptr        = c_loc(elem_rspheremp)
      hypervisc_cptr        = c_loc(elem_hyperviscosity)
      tensor_visc_cptr      = c_loc(elem_tensorVisc)
      D_cptr                = c_loc(elem_D)
      Dinv_cptr             = c_loc(elem_Dinv)

      call init_pointers_pool_c (elem_state_ps_cptr, elem_state_p_cptr, elem_state_v_cptr, metdet_cptr,&
                                 rmetdet_cptr, metinv_cptr, mp_cptr, vec_sphere2cart_cptr, spheremp_cptr,&
                                 rspheremp_cptr, hypervisc_cptr, tensor_visc_cptr, D_cptr, Dinv_cptr)
    endif
  end subroutine init_elem_f90

  subroutine test_laplace_sphere_wk_f90 (nets, nete, nelems, var_coef_c, input_cptr, output_cptr) bind(c)
    use iso_c_binding,  only: c_int, c_bool, c_double, c_ptr, c_f_pointer
    use kinds,          only: real_kind
    use dimensions_mod, only: np, nlev
    use derivative_mod, only: laplace_sphere_wk

    integer(kind=c_int),   intent(in)    :: nets, nete, nelems
    logical(kind=c_bool),  intent(in)    :: var_coef_c
    type(c_ptr),           intent(in)    :: input_cptr
    type(c_ptr),           intent(inout) :: output_cptr

    real (kind=real_kind), pointer :: input(:,:,:,:), output(:,:,:,:)
    logical :: var_coef
    integer :: ie, k

    var_coef = LOGICAL(var_coef_c,4)

    call c_f_pointer(input_cptr,  input,  [np, np, nlev, nelems])
    call c_f_pointer(output_cptr, output, [np, np, nlev, nelems])

    do ie=nets,nete
      do k=1,nlev
        output(:,:,k,ie) = laplace_sphere_wk (input(:,:,k,ie), deriv, elem(ie), var_coef)
      enddo
    enddo
  end subroutine test_laplace_sphere_wk_f90

  subroutine test_vlaplace_sphere_wk_f90 (nets, nete, nelems, var_coef_c, nu_ratio, input_cptr, output_cptr) bind(c)
    use iso_c_binding,  only: c_int, c_bool, c_double, c_ptr, c_f_pointer
    use kinds,          only: real_kind
    use dimensions_mod, only: np, nlev
    use derivative_mod, only: vlaplace_sphere_wk

    integer(kind=c_int),   intent(in)    :: nets, nete, nelems
    logical(kind=c_bool),  intent(in)    :: var_coef_c
    real (kind=c_double),  intent(in)    :: nu_ratio
    type(c_ptr),           intent(in)    :: input_cptr
    type(c_ptr),           intent(inout) :: output_cptr

    real (kind=real_kind), pointer :: input(:,:,:,:,:), output(:,:,:,:,:)
    logical :: var_coef
    integer :: ie, ip, k

    var_coef = LOGICAL(var_coef_c,4)

    call c_f_pointer(input_cptr,  input,  [np, np, 2, nlev, nelems])
    call c_f_pointer(output_cptr, output, [np, np, 2, nlev, nelems])

    do ie=nets,nete
      do k=1,nlev
        output(:,:,:,k,ie) = vlaplace_sphere_wk (input(:,:,:,k,ie), deriv, elem(ie), var_coef, nu_ratio)
      enddo
    enddo
  end subroutine test_vlaplace_sphere_wk_f90

  subroutine test_lapl_pre_bndry_ex_f90 (nets, nete, nelems, n0, var_coef, nu_ratio, ptens_cptr, vtens_cptr) bind(c)
    use iso_c_binding,  only: c_int, c_bool, c_double, c_ptr, c_f_pointer
    use kinds,          only: real_kind
    use dimensions_mod, only: np, nlev
    use advance_mod,    only: loop_lapl_pre_bndry_ex_f90

    integer(kind=c_int),   intent(in)    :: nets, nete, nelems, n0
    logical(kind=c_bool),  intent(in)    :: var_coef
    real (kind=c_double),  intent(in)    :: nu_ratio
    type(c_ptr),           intent(inout) :: vtens_cptr, ptens_cptr

    real (kind=real_kind), pointer :: ptens(:,:,:,:), vtens(:,:,:,:,:)
    logical :: var_coef_f90

    var_coef_f90 = LOGICAL(var_coef,4)

    call c_f_pointer(ptens_cptr, ptens, [np, np, nlev, nelems])
    call c_f_pointer(vtens_cptr, vtens, [np, np, 2, nlev, nelems])

    call loop_lapl_pre_bndry_ex_f90 (nets, nete, n0, elem, deriv, var_coef_f90, nu_ratio, ptens, vtens)
  end subroutine test_lapl_pre_bndry_ex_f90

  subroutine test_lapl_post_bndry_ex_f90 (nets, nete, nelems, nu_ratio, ptens_cptr, vtens_cptr) bind(c)
    use iso_c_binding,  only: c_int, c_bool, c_double, c_ptr, c_f_pointer
    use kinds,          only: real_kind
    use dimensions_mod, only: np, nlev
    use advance_mod,    only: loop_lapl_post_bndry_ex_f90

    integer(kind=c_int),   intent(in)    :: nets, nete, nelems
    real (kind=c_double),  intent(in)    :: nu_ratio
    type(c_ptr),           intent(inout) :: vtens_cptr, ptens_cptr

    real (kind=real_kind), pointer :: ptens(:,:,:,:), vtens(:,:,:,:,:)

    call c_f_pointer(ptens_cptr, ptens, [np, np, nlev, nelems])
    call c_f_pointer(vtens_cptr, vtens, [np, np, 2, nlev, nelems])

    call loop_lapl_post_bndry_ex_f90 (nets, nete, elem, deriv, nu_ratio, ptens, vtens)
  end subroutine test_lapl_post_bndry_ex_f90

  subroutine cleanup_testing_f90 () bind(c)
    deallocate(elem)
  end subroutine cleanup_testing_f90

end module
