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
  if (m_data.nu_top>0) {
    m_nu_scale_top = ExecViewManaged<Scalar[NUM_LEV]>("nu_scale_top");
    ExecViewManaged<Scalar[NUM_LEV]>::HostMirror h_nu_scale_top;
    h_nu_scale_top = Kokkos::create_mirror_view(m_nu_scale_top);

    constexpr int NUM_BIHARMONIC_PHYSICAL_LEVELS = 3;
    const Real lev_nu_scale_top[NUM_BIHARMONIC_PHYSICAL_LEVELS] = { 4.0, 2.0, 1.0 };
    for (int phys_lev=0; phys_lev<NUM_BIHARMONIC_PHYSICAL_LEVELS; ++phys_lev) {
      const int ilev = phys_lev / VECTOR_SIZE;
      const int ivec = phys_lev % VECTOR_SIZE;
      h_nu_scale_top(ilev)[ivec] = lev_nu_scale_top[phys_lev]*m_data.nu_top;
    }
    Kokkos::deep_copy(m_nu_scale_top, h_nu_scale_top);
  }
}

void HyperviscosityFunctor::run (const int hypervis_subcycle)
{
  m_hypervis_subcycle = hypervis_subcycle;
  Kokkos::RangePolicy<ExecSpace,TagUpdateStates> policy_update_states(0, m_data.num_elems*NP*NP*NUM_LEV);
  auto policy_pre_exchange =
      Homme::get_default_team_policy<ExecSpace, TagHyperPreExchange>(
          m_data.num_elems);
  for (int icycle = 0; icycle < hypervis_subcycle; ++icycle) {
    biharmonic_wk_dp3d ();
    // dispatch parallel_for for first kernel
    Kokkos::parallel_for(policy_pre_exchange, *this);
    Kokkos::fence();

    // Boundary Echange
    std::string be_name = "HyperviscosityFunctor";
    BoundaryExchange& be = *Context::singleton().get_boundary_exchange(be_name);
    assert (be.is_registration_completed());

    // Exchange
    be.exchange(m_data.nets, m_data.nete);

    // Update states
    Kokkos::parallel_for(policy_update_states, *this);
    Kokkos::fence();
  }
}

void HyperviscosityFunctor::biharmonic_wk_dp3d() const
{
  // For the first laplacian we use a differnt kernel, which uses directly the states
  // at timelevel np1 as inputs. This way we avoid copying the states to *tens buffers.
  auto policy_first_laplace = Homme::get_default_team_policy<ExecSpace,TagFirstLaplace>(m_data.num_elems);
  Kokkos::parallel_for(policy_first_laplace, *this);
  Kokkos::fence();

  // Get be structure
  std::string be_name = "HyperviscosityFunctor";
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange(be_name);
  assert (be.is_registration_completed());

  // Exchange
  be.exchange(m_data.nets, m_data.nete);

  // Apply inverse mass matrix
  Kokkos::RangePolicy<ExecSpace,TagApplyInvMass> policy_mass(0, m_data.num_elems*NP*NP*NUM_LEV);
  Kokkos::parallel_for(policy_mass, *this);
  Kokkos::fence();

  // TODO: update m_data.nu_ratio if nu_div!=nu
  // Compute second laplacian
  auto policy_second_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplace>(m_data.num_elems);
  Kokkos::parallel_for(policy_second_laplace, *this);
  Kokkos::fence();
}

} // namespace Homme
