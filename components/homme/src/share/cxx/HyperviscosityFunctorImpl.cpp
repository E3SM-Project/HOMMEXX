#include "Context.hpp"
#include "HyperviscosityFunctorImpl.hpp"
#include "BoundaryExchange.hpp"
#include "profiling.hpp"

namespace Homme
{

HyperviscosityFunctorImpl::HyperviscosityFunctorImpl (const SimulationParams& params, const Elements& elements, const Derivative& deriv)
 : m_elements   (elements)
 , m_deriv      (deriv)
 , m_data       (params.hypervis_subcycle,1.0,params.nu_top,params.nu,params.nu_p,params.nu_s)
 , m_sphere_ops (Context::singleton().get_sphere_operators())
{
  // Sanity check
  assert(params.params_set);

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

size_t HyperviscosityFunctorImpl::buffers_size () const {
  const int num_scalar_buffers = 4;
  const int num_vector_buffers = 2;

  const int scalar_buffer_size = m_elements.num_elems()*NP*NP*NUM_LEV*VECTOR_SIZE;
  const int vector_buffer_size = m_elements.num_elems()*2*NP*NP*NUM_LEV*VECTOR_SIZE;

  return  num_scalar_buffers*scalar_buffer_size +
          num_vector_buffers*vector_buffer_size;
}

void HyperviscosityFunctorImpl::init_buffers (Real* raw_buffer, const size_t buffer_size)
{
  const int scalar_buffer_size = m_elements.num_elems()*NP*NP*NUM_LEV*VECTOR_SIZE;
  const int vector_buffer_size = m_elements.num_elems()*2*NP*NP*NUM_LEV*VECTOR_SIZE;

  const int ne = m_elements.num_elems();

  Real* start = raw_buffer;

  auto ptr = [](Real* raw_buffer) { return reinterpret_cast<Scalar*>(raw_buffer); };

  // TODO: rearrange views order to maximize caching
  m_buffers.dptens = ScalarViewUnmanaged(ptr(raw_buffer),ne);
  raw_buffer += scalar_buffer_size;
  m_buffers.ttens = ScalarViewUnmanaged(ptr(raw_buffer),ne);
  raw_buffer += scalar_buffer_size;
  m_buffers.vtens = VectorViewUnmanaged(ptr(raw_buffer),ne);
  raw_buffer += vector_buffer_size;

  m_buffers.laplace_dp = ScalarViewUnmanaged(ptr(raw_buffer),ne);
  raw_buffer += scalar_buffer_size;
  m_buffers.laplace_t = ScalarViewUnmanaged(ptr(raw_buffer),ne);
  raw_buffer += scalar_buffer_size;
  m_buffers.laplace_v = VectorViewUnmanaged(ptr(raw_buffer),ne);
  raw_buffer += vector_buffer_size;

  // Sanity check
  size_t used_size = static_cast<size_t>(std::distance(start,raw_buffer));
  assert(used_size <= buffer_size);
  (void)used_size;    // Suppresses a warning in debug build
  (void)buffer_size;  // Suppresses a warning in debug build
}

void HyperviscosityFunctorImpl::init_boundary_exchanges () {
  m_be = std::make_shared<BoundaryExchange>();
  auto& be = *m_be;
  auto bm_exchange = Context::singleton().get_buffers_manager(MPI_EXCHANGE);
  be.set_buffers_manager(bm_exchange);
  be.set_num_fields(0, 0, 4);
  be.register_field(m_buffers.vtens, 2, 0);
  be.register_field(m_buffers.ttens);
  be.register_field(m_buffers.dptens);
  be.registration_completed();
}

void HyperviscosityFunctorImpl::run (const int np1, const Real dt, const Real eta_ave_w)
{
  m_data.np1 = np1;
  m_data.dt = dt/m_data.hypervis_subcycle;
  m_data.eta_ave_w = eta_ave_w;

  Kokkos::RangePolicy<ExecSpace,TagUpdateStates> policy_update_states(0, m_elements.num_elems()*NP*NP*NUM_LEV);
  auto policy_pre_exchange =
      Homme::get_default_team_policy<ExecSpace, TagHyperPreExchange>(
          m_elements.num_elems());
  for (int icycle = 0; icycle < m_data.hypervis_subcycle; ++icycle) {
    GPTLstart("hvf-bhwk");
    biharmonic_wk_dp3d ();
    GPTLstop("hvf-bhwk");
    // dispatch parallel_for for first kernel
    Kokkos::parallel_for(policy_pre_exchange, *this);
    Kokkos::fence();

    // Exchange
    assert (m_be->is_registration_completed());
    GPTLstart("hvf-bexch");
    m_be->exchange();
    GPTLstop("hvf-bexch");

    // Update states
    Kokkos::parallel_for(policy_update_states, *this);
    Kokkos::fence();
  }
}

void HyperviscosityFunctorImpl::biharmonic_wk_dp3d() const
{
  // For the first laplacian we use a differnt kernel, which uses directly the states
  // at timelevel np1 as inputs. This way we avoid copying the states to *tens buffers.
  auto policy_first_laplace = Homme::get_default_team_policy<ExecSpace,TagFirstLaplace>(m_elements.num_elems());
  Kokkos::parallel_for(policy_first_laplace, *this);
  Kokkos::fence();

  // Exchange
  assert (m_be->is_registration_completed());
  GPTLstart("hvf-bexch");
  m_be->exchange(m_elements.m_rspheremp);
  GPTLstop("hvf-bexch");

  // TODO: update m_data.nu_ratio if nu_div!=nu
  // Compute second laplacian
  auto policy_second_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplace>(m_elements.num_elems());
  Kokkos::parallel_for(policy_second_laplace, *this);
  Kokkos::fence();
}

} // namespace Homme
