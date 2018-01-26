#include "Context.hpp"
#include "Control.hpp"
#include "Elements.hpp"
#include "TimeLevel.hpp"
#include "SimulationParams.hpp"

namespace Homme
{

void prim_advance_exp_c(const Real, const bool);
void prim_advec_tracers_remap_c(const Real);

void prim_step_c (const Real dt, const bool compute_diagnostics)
{
  // Get control and simulation params
  Control&          data   = Context::singleton().get_control();
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
  Kokkos::deep_copy(elements.m_eta_dot_dpdn,0);
  Kokkos::deep_copy(elements.m_derived_vn0,0);
  Kokkos::deep_copy(elements.m_omega_p,0);
  if (params.nu_p>0) {
    Kokkos::deep_copy(elements.m_derived_dpdiss_ave,0);
    Kokkos::deep_copy(elements.m_derived_dpdiss_biharmonic,0);
  }

  auto policy = Homme::get_default_team_policy<ExecSpace>(data.num_elems);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team) {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV),
                           KOKKOS_LAMBDA (const int& ilev) {
        if (data.use_semi_lagrangian_transport) {
          elements.m_derived_vstar(ie,0,igp,jgp,ilev) = elements.m_u(ie,tl.n0,igp,jgp,ilev);
          elements.m_derived_vstar(ie,1,igp,jgp,ilev) = elements.m_v(ie,tl.n0,igp,jgp,ilev);
        }
        elements.m_derived_dp(ie,igp,jgp,ilev) = elements.m_dp3d(ie,tl.n0,igp,jgp,ilev);
      });
    });
  });

  // ===============
  // Dynamical Step
  // ===============
  prim_advance_exp_c(dt,compute_diagnostics);
  for (int n=1; n<params.qsplit; ++n) {
    tl.update_dynamics_levels(UpdateType::LEAPFROG);
    prim_advance_exp_c(dt,compute_diagnostics);
  }

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
  if (params.qsize>0) {
    prim_advec_tracers_remap_c(dt*params.qsplit);
  }
}

} // namespace Homme
