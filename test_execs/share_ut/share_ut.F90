module share_ut_mod
  implicit none

contains

  subroutine matrix_matrix_f90 (Aptr, Bptr, Cptr, trA, trB) bind(c)
    use kinds          , only : real_kind
    use dimensions_mod , only : np
    use iso_c_binding  , only : c_ptr, c_int, c_f_pointer
    !
    ! Inputs
    !
    type (c_ptr),         intent(in) :: Aptr, Bptr, Cptr
    integer (kind=c_int), intent(in) :: trA, trB
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:), pointer :: A, B, C

    call c_f_pointer(Aptr, A, [np,np])
    call c_f_pointer(Bptr, B, [np,np])
    call c_f_pointer(Cptr, C, [np,np])

    if (trA==1) then
      if (trB==1) then
        C = MATMUL(TRANSPOSE(A),TRANSPOSE(B))
      else
        C = MATMUL(TRANSPOSE(A),B)
      endif
    else
      if (trB==1) then
        C = MATMUL(A,TRANSPOSE(B))
      else
        C = MATMUL(A,B)
      endif
    endif
  end subroutine matrix_matrix_f90

  subroutine init_deriv_arrays_f90 (dvv_ptr, integ_mat_ptr, bd_interp_mat_ptr) bind(c)
    use derivative_mod_base , only : integration_matrix, boundary_interp_matrix, &
                                     derivative_t, derivinit,                    &
                                     allocate_subcell_integration_matrix
    use dimensions_mod      , only : np, nc
    use kinds               , only : real_kind
    use iso_c_binding       , only : c_ptr, c_f_pointer
    !
    ! Inputs
    !
    type (c_ptr), intent(in) :: dvv_ptr, integ_mat_ptr, bd_interp_mat_ptr
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:),   pointer :: dvv
    real (kind=real_kind), dimension(:,:),   pointer :: integ_mat
    real (kind=real_kind), dimension(:,:,:), pointer :: bd_interp_mat
    type (derivative_t) :: deriv

    call c_f_pointer (dvv_ptr,           dvv,           [np, np])
    call c_f_pointer (integ_mat_ptr,     integ_mat,     [nc, np])
    call c_f_pointer (bd_interp_mat_ptr, bd_interp_mat, [nc, 2, np])

    call allocate_subcell_integration_matrix (np, nc)
    call derivinit (deriv)

    dvv = deriv%dvv
    integ_mat = integration_matrix
    bd_interp_mat = boundary_interp_matrix

  end subroutine init_deriv_arrays_f90

  subroutine subcell_div_fluxes_f90 (u_ptr, metdet_ptr, flux_ptr) bind(c)
    use kinds               , only : real_kind
    use dimensions_mod      , only : np, nc
    use derivative_mod_base , only : subcell_div_fluxes
    use iso_c_binding       , only : c_ptr, c_int, c_f_pointer
    !
    ! Inputs
    !
    type (c_ptr),         intent(in) :: u_ptr, metdet_ptr, flux_ptr
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:),   pointer :: metdet
    real (kind=real_kind), dimension(:,:,:), pointer :: u, flux

    call c_f_pointer(u_ptr,      u,      [np, np, 2])
    call c_f_pointer(metdet_ptr, metdet, [np, np])
    call c_f_pointer(flux_ptr,   flux,   [nc, nc, 4])

    flux = subcell_div_fluxes(u,np,nc,metdet)

  end subroutine subcell_div_fluxes_f90

end module share_ut_mod
