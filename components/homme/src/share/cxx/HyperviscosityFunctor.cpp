#include "Context.hpp"
#include "HyperviscosityFunctor.hpp"
#include "BoundaryExchange.hpp"

namespace Homme
{

HyperviscosityFunctor::HyperviscosityFunctor (const Control& m_data, const Elements& elements, const Derivative& deriv)
 : m_data     (m_data)
 , m_elements (elements)
 , m_deriv    (deriv)
{
  // Nothing to be done here
}

void HyperviscosityFunctor::compute_t_v_laplace (const int itl, const bool var_coeff, const Real nu_ratio, const Real hypervis_scaling)
{
  // Store the time level and nu_ratio
  m_itl = itl;
  m_nu_ratio = nu_ratio;

  if (var_coeff) {
    if (hypervis_scaling>0) {
      auto policy_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplaceTensor_T_Cartesian_V>(m_data.num_elems);
      Kokkos::parallel_for(policy_laplace, *this);
    } else {
      auto policy_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplaceTensor_T_Contra_V>(m_data.num_elems);
      Kokkos::parallel_for(policy_laplace, *this);
    }
  } else {
    auto policy_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplaceSimple_T_Contra_V>(m_data.num_elems);
    Kokkos::parallel_for(policy_laplace, *this);
  }

  // Get be structure
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange("HyperviscosityFunctor::laplace_t_v");

  // Sanity check
  assert (be.is_registration_completed());

  // Exchange
  be.exchange(m_data.nets, m_data.nete);

  // Apply inverse mass matrix
  Kokkos::RangePolicy<ExecSpace,TagApplyInvMass_T_V> policy_mass(0, m_data.num_elems*NP*NP*NUM_LEV);
  Kokkos::parallel_for(policy_mass, *this);
}

void HyperviscosityFunctor::compute_t_v_dp3d_laplace (const int itl, const bool var_coeff, const Real nu_ratio, const Real hypervis_scaling)
{
  // Store the time level and nu_ratio
  m_itl = itl;
  m_nu_ratio = nu_ratio;

  if (var_coeff) {
    if (hypervis_scaling>0) {
      auto policy_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplaceTensor_T_DP3D_Cartesian_V>(m_data.num_elems);
      Kokkos::parallel_for(policy_laplace, *this);
    } else {
      auto policy_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplaceTensor_T_DP3D_Contra_V>(m_data.num_elems);
      Kokkos::parallel_for(policy_laplace, *this);
    }
  } else {
    auto policy_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplaceSimple_T_DP3D_Contra_V>(m_data.num_elems);
    Kokkos::parallel_for(policy_laplace, *this);
  }

  // Get be structure
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange("HyperviscosityFunctor::laplace_t_v");

  // Sanity check
  assert (be.is_registration_completed());

  // Exchange
  be.exchange(m_data.nets, m_data.nete);

  // Apply inverse mass matrix
  Kokkos::RangePolicy<ExecSpace,TagApplyInvMass_T_DP3D_V> policy_mass(0, m_data.num_elems*NP*NP*NUM_LEV);
  Kokkos::parallel_for(policy_mass, *this);
}

} // namespace Homme
