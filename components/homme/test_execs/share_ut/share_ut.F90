module share_ut_mod
  use derivative_mod_base, only : derivative_t
  use element_mod        , only : element_t
  use kinds              , only : real_kind
  use dimensions_mod     , only : np

  implicit none

  type(derivative_t) :: deriv
  type(element_t)    :: elem

contains

  subroutine init_deriv_f90 (dvv_ptr) bind(c)
    use derivative_mod_base , only : derivinit
    use iso_c_binding       , only : c_ptr, c_f_pointer
    !
    ! Inputs
    !
    type (c_ptr), intent(in) :: dvv_ptr
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:),   pointer :: dvv

    call c_f_pointer (dvv_ptr,           dvv,           [np, np])

    call derivinit (deriv)

    dvv = deriv%dvv

  end subroutine init_deriv_f90

  subroutine gradient_sphere_f90 (scalar_ptr, DInv_ptr, vector_ptr, nelems) bind(c)
    use derivative_mod_base , only : gradient_sphere
    use iso_c_binding       , only : c_ptr, c_int, c_f_pointer
    !
    ! Inputs
    !
    integer (kind=c_int), intent(in) :: nelems
    type (c_ptr)        , intent(in) :: scalar_ptr, DInv_ptr, vector_ptr
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:,:),     pointer :: scalar
    real (kind=real_kind), dimension(:,:,:,:),   pointer :: vector
    real (kind=real_kind), dimension(:,:,:,:,:), pointer :: DInv
    integer :: ie

    call c_f_pointer(scalar_ptr, scalar, [np, np, nelems])
    call c_f_pointer(DInv_ptr,   DInv,   [np, np, 2, 2, nelems])
    call c_f_pointer(vector_ptr, vector, [np, np, 2, nelems])

    do ie=1,nelems
      vector(:,:,:,ie) = gradient_sphere(scalar(:,:,ie),deriv,DInv(:,:,:,:,ie))
    enddo
  end subroutine gradient_sphere_f90

  subroutine divergence_sphere_f90 (vector_ptr, DInv_ptr, metdet_ptr, div_ptr, nelems) bind(c)
    use derivative_mod_base , only : divergence_sphere
    use iso_c_binding       , only : c_ptr, c_int, c_f_pointer
    !
    ! Inputs
    !
    type (c_ptr)        , intent(in) :: vector_ptr, DInv_ptr, metdet_ptr, div_ptr
    integer (kind=c_int), intent(in) :: nelems
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:,:,:),   pointer :: vector
    real (kind=real_kind), dimension(:,:,:,:,:), pointer :: DInv
    real (kind=real_kind), dimension(:,:,:),     pointer :: metdet
    real (kind=real_kind), dimension(:,:,:),     pointer :: div
    integer :: ie

    call c_f_pointer(DInv_ptr,   DInv,   [np, np, 2, 2, nelems])
    call c_f_pointer(vector_ptr, vector, [np, np, 2, nelems])
    call c_f_pointer(metdet_ptr, metdet, [np, np, nelems])
    call c_f_pointer(div_ptr,    div,    [np, np, nelems])

    do ie=1,nelems
      elem%DInv   = DInv(:,:,:,:,ie)
      elem%metdet = metdet(:,:,ie)

      div(:,:,ie) = divergence_sphere(vector(:,:,:,ie),deriv,elem)
    enddo
  end subroutine divergence_sphere_f90

  subroutine vorticity_sphere_f90 (vector_ptr, D_ptr, metdet_ptr, vort_ptr, nelems) bind(c)
    use derivative_mod_base , only : vorticity_sphere
    use iso_c_binding       , only : c_ptr, c_int, c_f_pointer
    !
    ! Inputs
    !
    type (c_ptr)        , intent(in) :: vector_ptr, D_ptr, metdet_ptr, vort_ptr
    integer (kind=c_int), intent(in) :: nelems
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:,:,:),   pointer :: vector
    real (kind=real_kind), dimension(:,:,:,:,:), pointer :: D
    real (kind=real_kind), dimension(:,:,:),     pointer :: metdet
    real (kind=real_kind), dimension(:,:,:),     pointer :: vort
    integer :: ie

    call c_f_pointer(D_ptr,      D,      [np, np, 2, 2, nelems])
    call c_f_pointer(vector_ptr, vector, [np, np, 2, nelems])
    call c_f_pointer(metdet_ptr, metdet, [np, np, nelems])
    call c_f_pointer(vort_ptr,   vort,   [np, np, nelems])

    do ie=1,nelems
      elem%D      = D(:,:,:,:,ie)
      elem%metdet = metdet(:,:,ie)

      vort(:,:,ie) = vorticity_sphere(vector(:,:,:,ie),deriv,elem)
    enddo
  end subroutine vorticity_sphere_f90

end module share_ut_mod
