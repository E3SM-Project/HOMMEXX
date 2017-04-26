#ifndef HOMMEXX_DERIVATIVE_HPP
#define HOMMEXX_DERIVATIVE_HPP

#include "Dimensions.hpp"
#include "Types.hpp"

namespace Homme
{

class Derivative
{
public:

  Derivative ();

  void init (F90Ptr& dvv, F90Ptr& integration_matrix, F90Ptr& boundary_interp_matrix);

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> get_dvv () const { return m_dvv_exec; }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NC][NP]> get_integration_matrix () const { return m_integ_mat_exec; }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[2][NC][NP]> get_bd_interpolation_matrix () const { return m_bd_interp_mat_exec; }

private:

  ExecViewManaged<Real[NP][NP]> m_dvv_exec;
  ExecViewManaged<Real[NC][NP]> m_integ_mat_exec;
  ExecViewManaged<Real[2][NC][NP]> m_bd_interp_mat_exec;
};

Derivative& get_derivative ();

void subcell_div_fluxes (const Kokkos::TeamPolicy<ExecSpace>::member_type& team_member,
                         const ExecViewUnmanaged<const Real[2][NP][NP]>    u,
                         const ExecViewUnmanaged<const Real[NP][NP]>       metdet,
                         ExecViewUnmanaged<Real[4][NC][NC]>                flux);

extern "C"
{
void init_derivative_c (F90Ptr& dvv, F90Ptr& integration_matrix, F90Ptr& boundary_interp_matrix);
} // extern "C"

} // namespace Homme

#endif // HOMMEXX_DERIVATIVE_HPP
