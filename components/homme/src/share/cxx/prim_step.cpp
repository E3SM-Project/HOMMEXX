#include "Context.hpp"
#include "Elements.hpp"
#include "TimeLevel.hpp"
#include "SimulationParams.hpp"
#include "profiling.hpp"

namespace Homme
{

void prim_advance_exp (const int nm1, const int n0, const int np1,
                       const Real dt, const bool compute_diagnostics);
void prim_advec_tracers_remap(const Real);

void prim_step (const Real dt, const bool compute_diagnostics)
{
  GPTLstart("tl-s prim_step");
  // Get control and simulation params
  SimulationParams& params = Context::singleton().get_simulation_params();
  assert(params.params_set);

  // Get the elements structure
  Elements& elements = Context::singleton().get_elements();

  // Get the time level info
  TimeLevel& tl = Context::singleton().get_time_level();

  // ===============
  // initialize mean flux accumulation variables and save some variables at n0
  // for use by advection
  // ===============
  GPTLstart("tl-s deep_copy");
  Kokkos::deep_copy(elements.m_eta_dot_dpdn,0);
  Kokkos::deep_copy(elements.m_derived_vn0,0);
  Kokkos::deep_copy(elements.m_omega_p,0);
  if (params.nu_p>0) {
    Kokkos::deep_copy(elements.m_derived_dpdiss_ave,0);
    Kokkos::deep_copy(elements.m_derived_dpdiss_biharmonic,0);
  }
  GPTLstop("tl-s deep_copy");

  if (params.use_semi_lagrangian_transport) {
    Errors::runtime_abort("[prim_step]", Errors::err_not_implemented);
    // Set derived_star = v
  }
  GPTLstart("tl-s derived_dp");
  {
    const auto derived_dp = elements.m_derived_dp;
    const auto dp3d = elements.m_dp3d;
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace> (0,elements.num_elems()*NP*NP*NUM_LEV),
                         KOKKOS_LAMBDA(const int idx) {
      const int ie   = ((idx / NUM_LEV) / NP) / NP;
      const int igp  = ((idx / NUM_LEV) / NP) % NP;
      const int jgp  =  (idx / NUM_LEV) % NP;
      const int ilev =   idx % NUM_LEV;

      derived_dp(ie,igp,jgp,ilev) = dp3d(ie,tl.n0,igp,jgp,ilev);
    });
  }
  ExecSpace::fence();
  GPTLstop("tl-s derived_dp");

  // ===============
  // Dynamical Step
  // ===============
  GPTLstart("tl-s prim_advance_exp-loop");
  prim_advance_exp(tl.nm1,tl.n0,tl.np1,dt,compute_diagnostics);
  tl.tevolve += dt;
  for (int n=1; n<params.qsplit; ++n) {
    tl.update_dynamics_levels(UpdateType::LEAPFROG);
    prim_advance_exp(tl.nm1,tl.n0,tl.np1,dt,false);
    tl.tevolve += dt;
  }
  GPTLstop("tl-s prim_advance_exp-loop");

  // ===============
  // Tracer Advection.
  // in addition, this routine will apply the DSS to:
  //        derived%eta_dot_dpdn    =  mean vertical velocity (used for remap below)
  //        derived%omega           =
  // Tracers are always vertically lagrangian.
  // For rsplit=0:
  //   if tracer scheme needs v on lagrangian levels it has to vertically interpolate
  //   if tracer scheme needs dp3d, it needs to derive it from ps_v
  // ===============
  // Advect tracers if their count is > 0.
  // not be advected.  This will be cleaned up when the physgrid is merged into CAM trunk
  // Currently advecting all species
  GPTLstart("tl-s prim_advec_tracers_remap");
  if (params.qsize>0) {
    prim_advec_tracers_remap(dt*params.qsplit);
  }
  GPTLstop("tl-s prim_advec_tracers_remap");
  GPTLstop("tl-s prim_step");
}

} // namespace Homme
