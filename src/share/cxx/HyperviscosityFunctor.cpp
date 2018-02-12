#include "Context.hpp"
#include "HyperviscosityFunctor.hpp"
#include "BoundaryExchange.hpp"

namespace Homme
{

HyperviscosityFunctor::HyperviscosityFunctor (const SimulationParams& params, const Elements& elements, const Derivative& deriv)
 : m_elements (elements)
 , m_deriv    (deriv)
{
  // Sanity check
  assert(params.params_set);

  m_data.nu_top = params.nu_top;
  m_data.nu = params.nu;
  m_data.nu_s = params.nu_s;
  m_data.nu_p = params.nu_p;
  m_data.nu_ratio = 1.0;
  m_data.hypervis_subcycle = params.hypervis_subcycle;

  if (m_data.nu_top>0) {
    m_nu_scale_top = ExecViewManaged<Scalar[NUM_LEV]>("nu_scale_top");
    ExecViewManaged<Scalar[NUM_LEV]>::HostMirror h_nu_scale_top;
    h_nu_scale_top = Kokkos::create_mirror_view(m_nu_scale_top);

    constexpr int NUM_BIHARMONIC_PHYSICAL_LEVELS = 3;
    Kokkos::Array<Real,NUM_BIHARMONIC_PHYSICAL_LEVELS> lev_nu_scale_top = { 4.0, 2.0, 1.0 };
    for (int phys_lev=0; phys_lev<NUM_BIHARMONIC_PHYSICAL_LEVELS; ++phys_lev) {
      const int ilev = phys_lev / VECTOR_SIZE;
      const int ivec = phys_lev % VECTOR_SIZE;
      h_nu_scale_top(ilev)[ivec] = lev_nu_scale_top[phys_lev]*m_data.nu_top;
    }
    Kokkos::deep_copy(m_nu_scale_top, h_nu_scale_top);
  }
}

void HyperviscosityFunctor::run (const int np1, const Real dt, const Real eta_ave_w)
{
  m_data.np1 = np1;
  m_data.dt = dt/m_data.hypervis_subcycle;
  m_data.eta_ave_w = eta_ave_w;

  Kokkos::RangePolicy<ExecSpace,TagUpdateStates> policy_update_states(0, m_elements.num_elems()*NP*NP*NUM_LEV);
  auto policy_pre_exchange =
      Homme::get_default_team_policy<ExecSpace, TagHyperPreExchange>(
          m_elements.num_elems());
  for (int icycle = 0; icycle < m_data.hypervis_subcycle; ++icycle) {
    biharmonic_wk_dp3d ();
    // dispatch parallel_for for first kernel
    Kokkos::parallel_for(policy_pre_exchange, *this);
    Kokkos::fence();

    // Boundary Echange
    std::string be_name = "HyperviscosityFunctor";
    BoundaryExchange& be = *Context::singleton().get_boundary_exchange(be_name);
    assert (be.is_registration_completed());

    // Exchange
    be.exchange();

    // Update states
    Kokkos::parallel_for(policy_update_states, *this);
    Kokkos::fence();
  }
}

void HyperviscosityFunctor::biharmonic_wk_dp3d() const
{
  // For the first laplacian we use a differnt kernel, which uses directly the states
  // at timelevel np1 as inputs. This way we avoid copying the states to *tens buffers.
  auto policy_first_laplace = Homme::get_default_team_policy<ExecSpace,TagFirstLaplace>(m_elements.num_elems());
  Kokkos::parallel_for(policy_first_laplace, *this);
  Kokkos::fence();

  // Get be structure
  std::string be_name = "HyperviscosityFunctor";
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange(be_name);
  assert (be.is_registration_completed());

  // Exchange
  be.exchange(m_elements.m_rspheremp);

  // TODO: update m_data.nu_ratio if nu_div!=nu
  // Compute second laplacian
  auto policy_second_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplace>(m_elements.num_elems());
  Kokkos::parallel_for(policy_second_laplace, *this);
  Kokkos::fence();
}

} // namespace Homme
