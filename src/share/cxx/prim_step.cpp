#include "Context.hpp"
#include "Elements.hpp"
#include "TimeLevel.hpp"
#include "SimulationParams.hpp"
#include "utilities/SubviewUtils.hpp"
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
  GPTLstart("tl-s deep_copy+derived_dp");
  {
    const auto elem_view = elements.get_elements();
    const int n0 = tl.n0;
    Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace>(elements.num_elems()),
                         KOKKOS_LAMBDA(const TeamMember& team) {
      const Element& elem = elem_view(team.league_rank());
      const auto& dp3d = Homme::subview(elem.m_dp3d,n0);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,NP*NP),
                           KOKKOS_LAMBDA(const int idx){
        const int igp = idx / NP;
        const int jgp = idx % NP;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,NUM_LEV),
                             KOKKOS_LAMBDA(const int ilev){
          elem.m_eta_dot_dpdn(igp,jgp,ilev) = 0.0;
          elem.m_derived_vn0(0,igp,jgp,ilev) = 0.0;
          elem.m_derived_vn0(1,igp,jgp,ilev) = 0.0;
          elem.m_omega_p(igp,jgp,ilev) = 0.0;
          if (params.nu_p>0) {
            elem.m_derived_dpdiss_ave(igp,jgp,ilev) = 0.0;
            elem.m_derived_dpdiss_biharmonic(igp,jgp,ilev) = 0.0;
          }
          elem.m_derived_dp(igp,jgp,ilev) = dp3d(igp,jgp,ilev);
        });
      });
    });
  }
  ExecSpace::fence();
  GPTLstop("tl-s deep_copy+derived_dp");

  if (params.use_semi_lagrangian_transport) {
    Errors::option_error("prim_step", "use_semi_lagrangian_transport",params.use_semi_lagrangian_transport);
    // Set derived_star = v
  }

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
