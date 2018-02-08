#include "Context.hpp"
#include "Control.hpp"
#include "Derivative.hpp"
#include "Elements.hpp"
#include "EulerStepFunctor.hpp"
#include "KernelVariables.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"
#include "TimeLevel.hpp"

namespace Homme
{

void prim_advec_tracers_remap_RK2 (const Real dt);
void prim_advec_tracers_remap (const Real dt);

// ----------- IMPLEMENTATION ---------- //

void prim_advec_tracers_remap (const Real dt) {
  SimulationParams& params = Context::singleton().get_simulation_params();

  if (params.use_semi_lagrangian_transport) {
    Errors::runtime_abort("[prim_advec_tracers_remap]", Errors::err_not_implemented);
  } else {
    prim_advec_tracers_remap_RK2(dt);
  }
}

void prim_advec_tracers_remap_RK2 (const Real dt)
{
  // Get control and simulation params
  Control&          data   = Context::singleton().get_control();
  SimulationParams& params = Context::singleton().get_simulation_params();
  assert(params.params_set);

  // Get time info and update tracers time levels
  TimeLevel& tl = Context::singleton().get_time_level();
  tl.update_tracers_levels(params.qsplit);

  // Create the ESF
  data.rhs_viss = 0;
  EulerStepFunctor esf(data);

  // Precompute divdp
  esf.precompute_divdp();
  Kokkos::fence();

  // Euler steps
  Control::DSSOption::Enum DSSopt;
  Real rhs_multiplier;

  // Euler step 1
  rhs_multiplier = 0.0;
  DSSopt = Control::DSSOption::div_vdp_ave;
  esf.euler_step(tl.np1_qdp,tl.n0_qdp,dt/2.0,rhs_multiplier,DSSopt);

  // Euler step 2
  rhs_multiplier = 1.0;
  DSSopt = Control::DSSOption::eta;
  esf.euler_step(tl.np1_qdp,tl.np1_qdp,dt/2.0,rhs_multiplier,DSSopt);

  // Euler step 3
  rhs_multiplier = 2.0;
  DSSopt = Control::DSSOption::omega;
  esf.euler_step(tl.np1_qdp,tl.np1_qdp,dt/2.0,rhs_multiplier,DSSopt);

  // to finish the 2D advection step, we need to average the t and t+2 results to get a second order estimate for t+1.
  esf.qdp_time_avg(tl.n0_qdp,tl.np1_qdp);
  Kokkos::fence();

  if (params.limiter_option!=8) {
    Errors::runtime_abort("[prim_advec_tracers_remap_RK2]", Errors::err_not_implemented);
    // call advance_hypervis_scalar(edgeadv,elem,hvcoord,hybrid,deriv,tl%np1,np1_qdp,nets,nete,dt)
  }
}

} // namespace Homme

