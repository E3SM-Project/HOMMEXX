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

void HyperviscosityFunctor::run (const int hypervis_subcycle) const
{
  for (int icycle=0; icycle<hypervis_subcycle; ++icycle) {
    biharmonic_wk_dp3d ();
    // dispatch parallel_for for first kernel

    // Boundary Echange
    std::string be_name = "HyperviscosityFunctor";
    BoundaryExchange& be = *Context::singleton().get_boundary_exchange(be_name);
    assert (be.is_registration_completed());

    // Exchange
    be.exchange(m_data.nets, m_data.nete);

    // dispatch parallel_for for second kernel
  }
}

void HyperviscosityFunctor::biharmonic_wk_dp3d() const
{
  // Extract requested time level, and stuff it into members (will be overwritten with laplacian)
  Kokkos::RangePolicy<ExecSpace,TagFetchStates> policy_fetch(0, m_data.num_elems*NP*NP*NUM_LEV);
  Kokkos::parallel_for(policy_fetch, *this);

  // Compute first laplacian
  auto policy_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplace>(m_data.num_elems);
  Kokkos::parallel_for(policy_laplace, *this);

  // Get be structure
  std::string be_name = "HyperviscosityFunctor";
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange(be_name);
  assert (be.is_registration_completed());

  // Exchange
  be.exchange(m_data.nets, m_data.nete);

  // Apply inverse mass matrix
  Kokkos::RangePolicy<ExecSpace,TagApplyInvMass> policy_mass(0, m_data.num_elems*NP*NP*NUM_LEV);
  Kokkos::parallel_for(policy_mass, *this);

  // TODO: update m_data.nu_ratio if nu_div!=nu
  // Compute second laplacian
  Kokkos::parallel_for(policy_laplace, *this);
}

} // namespace Homme
