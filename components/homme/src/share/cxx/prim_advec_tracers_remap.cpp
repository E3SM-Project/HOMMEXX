#include "Context.hpp"
#include "Control.hpp"
#include "Derivative.hpp"
#include "Elements.hpp"
#include "KernelVariables.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"
#include "TimeLevel.hpp"

namespace Homme
{

void euler_step_c (const int np1_qdp, const int n0_qdp, const Real dt,
                   const Control::DSSOption::Enum dss_opt, const int rhs_multiplier);
void qdp_time_avg_c();
void advance_hypervis_scalar_c();
void prim_advec_tracers_remap_ALE_c(const Real dt);
void prim_advec_tracers_remap_RK2_c(const Real dt);
void prim_advec_tracers_remap_c(const Real dt);

// ----------- IMPLEMENTATION ---------- //

void prim_advec_tracers_remap_c(const Real dt) {
  SimulationParams& params = Context::singleton().get_simulation_params();

  if (params.use_semi_lagrangian_transport) {
    prim_advec_tracers_remap_ALE_c(dt);
  } else {
    prim_advec_tracers_remap_RK2_c(dt);
  }
}

void prim_advec_tracers_remap_ALE_c(const Real /*dt*/)
{
  Errors::runtime_abort("'prim_advec_tracers_remap_ALE' functionality not yet available in C++ build.\n",
                        Errors::functionality_not_yet_implemented);
}

void prim_advec_tracers_remap_RK2_c(const Real dt)
{
  // Get control and simulation params
  Control&          data   = Context::singleton().get_control();
  SimulationParams& params = Context::singleton().get_simulation_params();
  assert(params.params_set);

  // Get time info and update tracers time levels
  TimeLevel& tl = Context::singleton().get_time_level();
  tl.update_tracers_levels(params.qsplit);

  // Get the elements and derivative structure
  Elements& elements = Context::singleton().get_elements();
  Derivative& deriv  = Context::singleton().get_derivative();

  // Precompute divdp
  auto divdp      = elements.m_derived_divdp;
  auto divdp_proj = elements.m_derived_divdp_proj;
  auto policy = Homme::get_default_team_policy<ExecSpace>(data.num_elems);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team) {
    KernelVariables kv(team);
    divergence_sphere(
        kv, elements.m_dinv, elements.m_metdet, deriv.get_dvv(),
        Homme::subview(elements.m_derived_vn0, kv.ie),
        elements.buffers.div_buf, Homme::subview(divdp, kv.ie));
  });
  Kokkos::deep_copy(divdp_proj,divdp);

  // Euler steps
  int rhs_multiplier;
  Control::DSSOption::Enum dss_opt;

  // Euler step 1
  rhs_multiplier = 0;
  dss_opt = Control::DSSOption::div_vdp_ave;
  euler_step_c (tl.np1_qdp, tl.n0_qdp, dt/2, dss_opt, rhs_multiplier);

  // Euler step 2
  rhs_multiplier = 1;
  dss_opt = Control::DSSOption::eta;
  euler_step_c (tl.np1_qdp, tl.np1_qdp, dt/2, dss_opt, rhs_multiplier);

  // Euler step 3
  rhs_multiplier = 2;
  dss_opt = Control::DSSOption::omega;
  euler_step_c (tl.np1_qdp, tl.np1_qdp, dt/2, dss_opt, rhs_multiplier);

  // to finish the 2D advection step, we need to average the t and t+2 results to get a second order estimate for t+1.
  qdp_time_avg_c();

  if (params.limiter_option!=8) {
std::cout << "lim opt: " << params.limiter_option << "\n";
    advance_hypervis_scalar_c();
  }
}

void euler_step_c (const int np1_qdp, const int n0_qdp, const Real dt,
                   const Control::DSSOption::Enum dss_opt, const int rhs_multiplier)
{
  Errors::runtime_abort("'euler_step_c' functionality not yet available in C++ build.\n",
                        Errors::functionality_not_yet_implemented);
}

void qdp_time_avg_c()
{
  Errors::runtime_abort("'qdp_time_avg_c' functionality not yet available in C++ build.\n",
                        Errors::functionality_not_yet_implemented);
}

void advance_hypervis_scalar_c()
{
  Errors::runtime_abort("'advance_hypervis_scalar_c' functionality not yet available in C++ build.\n",
                        Errors::functionality_not_yet_implemented);
}

} // namespace Homme

