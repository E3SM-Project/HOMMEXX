#include "Control.hpp"
#include "Context.hpp"
#include "Elements.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"

#include "mpi/ErrorDefs.hpp"

#include <iostream>

namespace Homme
{

void apply_test_forcing_c ();

void prim_step_c (const Real, const bool);

extern "C" {

void vertical_remap_c(const Real);

void prim_run_subcycle_c (const int& nets, const int& nete, const Real& dt)
{
  // Get control and simulation params
  Control&          data   = Context::singleton().get_control();
  SimulationParams& params = Context::singleton().get_simulation_params();
  assert(params.params_set);

  // Set elements range info in the control
  data.nets = nets-1;
  data.nete = nete-1;

  // Get time info and compute dt for tracers and remap
  TimeLevel& tl = Context::singleton().get_time_level();
  const Real dt_q = dt*params.qsplit;
  Real dt_remap = dt_q;
  int nstep_end = tl.nstep + params.qsplit;
  if (params.rsplit>0) {
    dt_remap  = dt_q*params.rsplit;
    nstep_end = tl.nstep + params.qsplit*params.rsplit;
  }

  // Check if needed to compute diagnostics or energy
  bool compute_diagnostics = false;
  bool compute_energy      = params.energy_fixer;
  if (nstep_end%params.state_frequency==0 || nstep_end==tl.nstep0) {
    compute_diagnostics = true;
    compute_energy      = true;
  }

  if (params.disable_diagnostics) {
    compute_diagnostics = false;
  }

  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call prim_diag_scalars(elem,hvcoord,tl,4,.true.,nets,nete)
  }

  // Apply forcing
#ifdef CAM
  Errors::runtime_abort("CAM forcing not yet availble in C++.\n"
                        Errors::functionality_not_yet_implemented);
  // call TimeLevel_Qdp(tl, qsplit, n0_qdp)

  // if (ftype==0) then
  //   call t_startf("ApplyCAMForcing")
  //   call ApplyCAMForcing(elem, fvm, hvcoord,tl%n0,n0_qdp, dt_remap,nets,nete)
  //   call t_stopf("ApplyCAMForcing")

  // elseif (ftype==2) then
  //   call t_startf("ApplyCAMForcing_dynamics")
  //   call ApplyCAMForcing_dynamics(elem, hvcoord,tl%n0,dt_remap,nets,nete)
  //   call t_stopf("ApplyCAMForcing_dynamics")
  // endif

#else
  apply_test_forcing_c ();
#endif

  if (compute_energy) {
    Errors::runtime_abort("'compute energy' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call prim_energy_halftimes(elem,hvcoord,tl,1,.true.,nets,nete)
  }

  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call prim_diag_scalars(elem,hvcoord,tl,1,.true.,nets,nete)
  }

  // Initialize dp3d from ps
  Elements& elements = Context::singleton().get_elements();
  int n0 = tl.n0;
  auto policy = Homme::get_default_team_policy<ExecSpace>(data.num_elems);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team) {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV),
                           KOKKOS_LAMBDA (const int& ilev) {
        elements.m_dp3d(ie,n0,igp,jgp,ilev) = data.hybrid_a_delta[ilev]*data.ps0
                                            + data.hybrid_b_delta[ilev]*elements.m_ps_v(ie,n0,igp,jgp);
      });
    });
  });

  // Loop over rsplit vertically lagrangian timesteps
  prim_step_c(dt,compute_diagnostics);
  for (int r=1; r<params.rsplit; ++r) {
    tl.update_dynamics_levels(UpdateType::LEAPFROG);
    prim_step_c(dt,false);
  }

  ////////////////////////////////////////////////////////////////////////
  // apply vertical remap
  // always for tracers
  // if rsplit>0:  also remap dynamics and compute reference level ps_v
  ////////////////////////////////////////////////////////////////////////
  tl.update_tracers_levels(params.qsplit);
  vertical_remap_c(dt_remap);

  ////////////////////////////////////////////////////////////////////////
  // time step is complete.  update some diagnostic variables:
  // lnps (we should get rid of this)
  // Q    (mixing ratio)
  ////////////////////////////////////////////////////////////////////////
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team) {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      elements.m_lnps(ie,tl.np1,igp,jgp) = log(elements.m_ps_v(ie,tl.np1,igp,jgp));

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV*data.qsize),
                           KOKKOS_LAMBDA (const int& idx) {
        const int ilev = idx / data.qsize;
        const int iq   = idx % data.qsize;
        elements.m_Q(ie,iq,igp,jgp,ilev) = elements.m_qdp(ie,tl.np1_qdp,iq,igp,jgp,ilev) /
                                             ( data.hybrid_a_delta[ilev]*data.ps0 +
                                               data.hybrid_b_delta[ilev]*elements.m_ps_v(ie,tl.np1,igp,jgp));
      });
    });
  });

  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call prim_diag_scalars(elem,hvcoord,tl,2,.false.,nets,nete)
  }

  if (compute_energy) {
    // call prim_energy_halftimes(elem,hvcoord,tl,2,.false.,nets,nete)
    Errors::runtime_abort("'compute energy' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
  }

  if (params.energy_fixer) {
    Errors::runtime_abort("'energu fixer' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call prim_energy_fixer(elem,hvcoord,hybrid,tl,nets,nete,nsubstep)
  }

  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call prim_diag_scalars(elem,hvcoord,tl,3,.false.,nets,nete)
    // call prim_energy_halftimes(elem,hvcoord,tl,3,.false.,nets,nete)
  }

  // Update dynamics time levels
  tl.update_dynamics_levels(UpdateType::LEAPFROG);

  // ============================================================
  // Print some diagnostic information
  // ============================================================
  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call prim_printstate(elem, tl, hybrid,hvcoord,nets,nete, fvm)
  }
}

} // extern "C"

void apply_test_forcing_c () {
  // Get simulation params
  SimulationParams& params = Context::singleton().get_simulation_params();

  if (params.test_case==TestCase::DCMIP2012_TEST2_1 ||
      params.test_case==TestCase::DCMIP2012_TEST2_2) {
    Errors::runtime_abort("Test case not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
  }
}

} // namespace Homme
