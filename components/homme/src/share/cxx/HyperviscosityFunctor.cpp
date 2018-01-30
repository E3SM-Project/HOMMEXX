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
  // TODO: so far in our tests, we use the same nu_ratio in the 1st and 2nd laplacian, and it is always 1.0.
  //       If there are other test cases, load this from parameters, and differentiate between 1st and 2nd sweep.
  //       If not, remove it alltogether (also from SphereOperators)
  m_nu_ratio = 1.0;
}

void HyperviscosityFunctor::biharmonic_wk_dp3d(const int itl)
{
  // Store the time level and nu_ratio
  m_itl = itl;

  // Extract requested time level, and stuff it into members (will be overwritten with laplacian)
  Kokkos::RangePolicy<ExecSpace,TagFetchStates> policy_fetch(0, m_data.num_elems*NP*NP*NUM_LEV);
  Kokkos::parallel_for(policy_fetch, *this);

  // Compute first laplacian
  auto policy_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplace>(m_data.num_elems);
  Kokkos::parallel_for(policy_laplace, *this);

  // Get be structure
  std::string be_name = "HyperviscosityFunctor:biharmonic_wk_dp3d";
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange(be_name);
  if (!be.is_registration_completed()) {
    std::shared_ptr<BuffersManager> buffers_manager = Context::singleton().get_buffers_manager(MPI_EXCHANGE);
    be.set_buffers_manager(buffers_manager);

    // Set the views of this time level into this time level's boundary exchange
    be.set_num_fields(0,0,4);
    be.register_field(m_vtens,2,0);
    be.register_field(m_ttens);
    be.register_field(m_dptens);
    be.registration_completed();
  }

  // Exchange
  be.exchange(m_data.nets, m_data.nete);

  // Apply inverse mass matrix
  Kokkos::RangePolicy<ExecSpace,TagApplyInvMass> policy_mass(0, m_data.num_elems*NP*NP*NUM_LEV);
  Kokkos::parallel_for(policy_mass, *this);

  // Compute second laplacian
  Kokkos::parallel_for(policy_laplace, *this);
}

} // namespace Homme
