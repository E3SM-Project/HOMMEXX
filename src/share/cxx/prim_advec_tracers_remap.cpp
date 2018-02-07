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
  Kokkos::fence();
  Kokkos::deep_copy(divdp_proj,divdp);

  // Euler steps
  data.dt = dt/2;

  // Euler step 1
  data.n0_qdp  = tl.n0_qdp;
  data.np1_qdp = tl.np1_qdp;
  data.rhs_multiplier = 0;
  data.DSSopt = Control::DSSOption::div_vdp_ave;
  EulerStepFunctor::euler_step(data);

  // Euler step 2
  data.n0_qdp  = tl.np1_qdp;
  data.np1_qdp = tl.np1_qdp;
  data.rhs_multiplier = 1;
  data.DSSopt = Control::DSSOption::eta;
  EulerStepFunctor::euler_step(data);

  // Euler step 3
  data.n0_qdp  = tl.np1_qdp;
  data.np1_qdp = tl.np1_qdp;
  data.rhs_multiplier = 2;
  data.DSSopt = Control::DSSOption::omega;
  EulerStepFunctor::euler_step(data);

  // to finish the 2D advection step, we need to average the t and t+2 results to get a second order estimate for t+1.
  constexpr Real rkstage = 3.0;
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,data.num_elems*data.qsize*NP*NP*NUM_LEV),
                       KOKKOS_LAMBDA(const int idx) {
    const int ie   = (((idx / NUM_LEV) / NP) / NP) / data.qsize;
    const int iq   = (((idx / NUM_LEV) / NP) / NP) % data.qsize;
    const int igp  =  ((idx / NUM_LEV) / NP) % NP;
    const int jgp  =   (idx / NUM_LEV) % NP;
    const int ilev =    idx % NUM_LEV;

    elements.m_qdp(ie,tl.np1_qdp,iq,igp,jgp,ilev) =
          (elements.m_qdp(ie,tl.n0_qdp,iq,igp,jgp,ilev) +
           (rkstage-1)*elements.m_qdp(ie,tl.np1_qdp,iq,igp,jgp,ilev)) / rkstage;
  });

  if (params.limiter_option!=8) {
    Errors::runtime_abort("[prim_advec_tracers_remap_RK2]", Errors::err_not_implemented);
    // call advance_hypervis_scalar(edgeadv,elem,hvcoord,hybrid,deriv,tl%np1,np1_qdp,nets,nete,dt)
  }
}

} // namespace Homme

