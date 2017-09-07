module share_ut_mod
  use derivative_mod_base, only : derivative_t
  use kinds              , only : real_kind
  use dimensions_mod     , only : np

  implicit none

contains

  subroutine init_deriv_f90 (dvv) bind(c)
    use derivative_mod_base , only : derivinit
    !
    ! Inputs
    !
    real (kind=real_kind), intent(inout) :: dvv(np, np)
    
    !
    ! Locals
    !

    type (derivative_t) :: deriv

    call derivinit (deriv)
    dvv = deriv%dvv
  end subroutine init_deriv_f90

end module share_ut_mod
