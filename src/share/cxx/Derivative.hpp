#ifndef HOMMEXX_DERIVATIVE_HPP
#define HOMMEXX_DERIVATIVE_HPP

#include "Dimensions.hpp"
#include "Types.hpp"

#include <random>

namespace Homme {

class Derivative {
public:
  Derivative();

  void init(CF90Ptr &dvv);
  void init(CF90Ptr &dvv, CF90Ptr &integration_matrix,
            CF90Ptr &boundary_interp_matrix);

  void random_init(std::mt19937_64 &engine);

  void dvv(Real *dvv);

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> get_dvv() const { return m_dvv_exec; }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NC][NP]> get_integration_matrix() const {
    return m_integ_mat_exec;
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[2][NC][NP]> get_bd_interpolation_matrix() const {
    return m_bd_interp_mat_exec;
  }

private:
  ExecViewManaged<Real[NP][NP]> m_dvv_exec;
  ExecViewManaged<Real[NC][NP]> m_integ_mat_exec;
  ExecViewManaged<Real[2][NC][NP]> m_bd_interp_mat_exec;
};

Derivative &get_derivative();

// I put this function here since in F90 it is inside derivative_mod_base.
// But honestly, I'm not sure it belongs here.
void subcell_div_fluxes(
    const Kokkos::TeamPolicy<ExecSpace>::member_type &team_member,
    const ExecViewUnmanaged<const Real[2][NP][NP]> u,
    const ExecViewUnmanaged<const Real[NP][NP]> metdet,
    ExecViewUnmanaged<Real[4][NC][NC]> flux);

} // namespace Homme

#endif // HOMMEXX_DERIVATIVE_HPP
