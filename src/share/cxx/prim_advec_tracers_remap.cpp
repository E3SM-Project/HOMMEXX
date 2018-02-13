#include "Context.hpp"
#include "EulerStepFunctor.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "profiling.hpp"

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
  GPTLstart("tl-at prim_advec_tracers_remap_RK2");
  // Get control and simulation params
  SimulationParams& params = Context::singleton().get_simulation_params();
  assert(params.params_set);

  // Get time info and update tracers time levels
  TimeLevel& tl = Context::singleton().get_time_level();
  tl.update_tracers_levels(params.qsplit);

  // Create the ESF
  EulerStepFunctor esf(params);

  // Precompute divdp
  GPTLstart("tl-at precompute_divdp");
  esf.precompute_divdp();
  Kokkos::fence();
  GPTLstop("tl-at precompute_divdp");

  // Euler steps
  DSSOption DSSopt;
  Real rhs_multiplier;

  // Euler step 1
  GPTLstart("tl-at esf-0");
  rhs_multiplier = 0.0;
  DSSopt = DSSOption::DIV_VDP_AVE;
  esf.euler_step(tl.np1_qdp,tl.n0_qdp,dt/2.0,rhs_multiplier,DSSopt);
  GPTLstop("tl-at esf-0");

  // Euler step 2
  GPTLstart("tl-at esf-1");
  rhs_multiplier = 1.0;
  DSSopt = DSSOption::ETA;
  esf.euler_step(tl.np1_qdp,tl.np1_qdp,dt/2.0,rhs_multiplier,DSSopt);
  GPTLstop("tl-at esf-1");

  // Euler step 3
  GPTLstart("tl-at esf-2");
  rhs_multiplier = 2.0;
  DSSopt = DSSOption::OMEGA;
  esf.euler_step(tl.np1_qdp,tl.np1_qdp,dt/2.0,rhs_multiplier,DSSopt);
  GPTLstop("tl-at esf-2");

  // to finish the 2D advection step, we need to average the t and t+2 results to get a second order estimate for t+1.
  GPTLstart("tl-at qdp_time_avg");
  esf.qdp_time_avg(tl.n0_qdp,tl.np1_qdp);
  Kokkos::fence();
  GPTLstop("tl-at qdp_time_avg");

  if (params.limiter_option!=8) {
    Errors::runtime_abort("[prim_advec_tracers_remap_RK2]", Errors::err_not_implemented);
    // call advance_hypervis_scalar(edgeadv,elem,hvcoord,hybrid,deriv,tl%np1,np1_qdp,nets,nete,dt)
  }
  GPTLstop("tl-at prim_advec_tracers_remap_RK2");
}

} // namespace Homme

