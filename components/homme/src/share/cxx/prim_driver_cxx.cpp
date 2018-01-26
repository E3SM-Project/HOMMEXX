#include "Control.hpp"
#include "Context.hpp"

#include "mpi/ErrorDefs.hpp"

#include <iostream>

namespace Homme
{

// NOTE: This do-nothing function should do something if the test case is one of the dcmip ones
void apply_test_forcing () {}

void prim_step_c (const bool);

extern "C" {

void prim_run_subcycle ()
{
  // Get control and simulation params
  Control&          data   = Context::singleton().get_control();
  SimulationParams& params = Context::singleton().get_simulation_params();

  // Get time info and compute dt for tracers and remap
  TimeLevel& tl = Context::singleton().get_time_level();
  const Real dt_q = params.time_step*params.qsplit;
  const Real dt_remap = dt_q;
  int nstep_end = tl.nstep + param.qsplit;
  if (params.rsplit>0) {
    dt_remap  = dt_q*rsplit;
    nstep_end = dl.nstep + qsplit*rsplit;
  }

  // Check if needed to compute diagnostics or energy
  bool compute_diagonstics = false;
  bool compute_energy      = params.energy_fixer;
  if (nstep_end%params.state_frequency==0 || nstep_end==tl.nstep0) {
    compute_diagonstics = true;
    compute_energy      = true;
  }

  if (params.disable_diagnostics) {
    compute_diagnostics = false
  }

  if (compute_diagonstics) {
    Errors::runtime_abort("'compute diagonstic' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
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
  apply_test_forcing ();
#endif

  if (compute_energy) {
    Errors::runtime_abort("'compute energy' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call prim_energy_halftimes(elem,hvcoord,tl,1,.true.,nets,nete)
  }

  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagonstic' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call prim_diag_scalars(elem,hvcoord,tl,1,.true.,nets,nete)
  }

  // Initialize dp3d from ps
  Elements& elements = Context::singleton().get_elements();
  auto dp3d = elements.m_dp3d;
  auto ps_v = elements.m_ps_v;
  ExecViewUnmanaged<Scalar[NUM_LEV]> hyai    (reinterpret_cast<Scalar*>(control.hybrid_a.data()));
  ExecViewUnmanaged<Scalar[NUM_LEV]> hybi    (reinterpret_cast<Scalar*>(control.hybrid_b.data()));
  ExecViewUnmanaged<Scalar[NUM_LEV]> hyai_p1 (reinterpret_cast<Scalar*>(control.hybrid_a.data()+1));
  ExecViewUnmanaged<Scalar[NUM_LEV]> hybi_p1 (reinterpret_cast<Scalar*>(control.hybrid_b.data()+1));
  int n0 = tl.n0;
  auto policy = Homme::get_default_team_policy<ExecSpace>(data.num_elems);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team) {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           KOKKOS_LAMBDA (const int& ilev) {
        dp3d(ie,n0,igp,jgp,ilev) = (hyai_p1[ilev]-hyai[ilev])*ps0
                                 + (hybi_p1[ilev]-hybi[ilev])*ps_v(ie,n0,igp,jgp);
      });
    });
  });

  // Loop over rsplit vertically lagrangian timesteps
  prim_step_c(compute_diagnostics);
  for (int r=1; r<params.rsplit; ++r) {
    tl.update_dynamics_levels(UpdateType::LEAPFROG);
    prim_step_c(false);
  }

  ////////////////////////////////////////////////////////////////////////
  // apply vertical remap
  // always for tracers
  // if rsplit>0:  also remap dynamics and compute reference level ps_v
  ////////////////////////////////////////////////////////////////////////
  tl.update_tracers_levels(params.qsplit);
  vertical_remap_c();

  ////////////////////////////////////////////////////////////////////////
  // time step is complete.  update some diagnostic variables:
  // lnps (we should get rid of this)
  // Q    (mixing ratio)
  ////////////////////////////////////////////////////////////////////////
  auto lnps = elements.m_lnps;
  auto Q = elements.m_Q;
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team) {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      lnps(ie,tl.np1,igp,jgp) = log(ps_v(ie,tl.np1,igp,jgp));

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV*data.qsize),
                           KOKKOS_LAMBDA (const int& idx) {
        const int ilev = idx / data.qsize;
        const int iq   = idx % data.qsize;
        Q(ie,iq,igp,jgp,ilev) = qdp(ie,tl.np1_qdp,iq,igp,jgp,ilev) \
                                 ( (hyai_p1[ilev]-hyai[ilev])*ps0 +
                                   (hybi_p1[ilev]-hybi[ilev])*ps_v(ie,tl.np1,igp,jgp));
      });
    });
  });

  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagonstic' functionality not yet available in C++ build.\n",
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
    Errors::runtime_abort("'compute diagonstic' functionality not yet available in C++ build.\n",
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
    Errors::runtime_abort("'compute diagonstic' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call prim_printstate(elem, tl, hybrid,hvcoord,nets,nete, fvm)
  }

}

} // extern "C"

} // namespace Homme
