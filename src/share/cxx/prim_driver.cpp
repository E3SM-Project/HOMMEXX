#include "Control.hpp"
#include "Context.hpp"
#include "Elements.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"

#include "ErrorDefs.hpp"

#include <iostream>

namespace Homme
{

void prim_step (const Real, const bool);
void vertical_remap (const Real);
void apply_test_forcing ();

extern "C" {

void prim_run_subcycle_c (const Real& dt, int& nstep, int& nm1, int& n0, int& np1)
{
  // Get control and simulation params
  Control&          data   = Context::singleton().get_control();
  SimulationParams& params = Context::singleton().get_simulation_params();
  assert(params.params_set);

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
#ifndef NDEBUG
    std::cout << "WARNING! You need to implement something at line " << __LINE__ << " of file " << __FILE__ << "\n";
#endif
    // TODO: uncomment these lines, and implement diagnostics stuff
    //compute_diagnostics = true;
    //compute_energy      = true;
  }

  if (params.disable_diagnostics) {
    compute_diagnostics = false;
  }

  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
    // call prim_diag_scalars(elem,hvcoord,tl,4,.true.,nets,nete)
  }

  // Apply forcing
#ifdef CAM
  Errors::runtime_abort("CAM forcing not yet availble in C++.\n"
                        Errors::err_not_implemented);
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
                          Errors::err_not_implemented);
    // call prim_energy_halftimes(elem,hvcoord,tl,1,.true.,nets,nete)
  }

  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
    // call prim_diag_scalars(elem,hvcoord,tl,1,.true.,nets,nete)
  }

  // Initialize dp3d from ps
  Elements& elements = Context::singleton().get_elements();
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace> (0,data.num_elems*NP*NP*NUM_LEV),
                       KOKKOS_LAMBDA(const int idx) {
    const int ie   = ((idx / NUM_LEV) / NP) / NP;
    const int igp  = ((idx / NUM_LEV) / NP) % NP;
    const int jgp  =  (idx / NUM_LEV) % NP;
    const int ilev =   idx % NUM_LEV;

    elements.m_dp3d(ie,tl.n0,igp,jgp,ilev) = data.hybrid_ai_delta[ilev]*data.ps0
                                           + data.hybrid_bi_delta[ilev]*elements.m_ps_v(ie,tl.n0,igp,jgp);
  });

  // Loop over rsplit vertically lagrangian timesteps
  prim_step(dt,compute_diagnostics);
  for (int r=1; r<params.rsplit; ++r) {
    tl.update_dynamics_levels(UpdateType::LEAPFROG);
    prim_step(dt,false);
  }

  ////////////////////////////////////////////////////////////////////////
  // apply vertical remap
  // always for tracers
  // if rsplit>0:  also remap dynamics and compute reference level ps_v
  ////////////////////////////////////////////////////////////////////////
  tl.update_tracers_levels(params.qsplit);
  vertical_remap(dt_remap);

  ////////////////////////////////////////////////////////////////////////
  // time step is complete.  update some diagnostic variables:
  // lnps (we should get rid of this)
  // Q    (mixing ratio)
  ////////////////////////////////////////////////////////////////////////
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,data.num_elems*NP*NP),
                       KOKKOS_LAMBDA(const int idx) {
    const int ie   = (idx / NP) / NP;
    const int igp  = (idx / NP) % NP;
    const int jgp  =  idx % NP;

    elements.m_lnps(ie,tl.np1,igp,jgp) = log(elements.m_ps_v(ie,tl.np1,igp,jgp));
  });
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,data.num_elems*data.qsize*NP*NP*NUM_LEV),
                       KOKKOS_LAMBDA(const int idx) {
    const int ie   = (((idx / NUM_LEV) / NP) / NP) / data.qsize;
    const int iq   = (((idx / NUM_LEV) / NP) / NP) % data.qsize;
    const int igp  =  ((idx / NUM_LEV) / NP) % NP;
    const int jgp  =   (idx / NUM_LEV) % NP;
    const int ilev =    idx % NUM_LEV;

    elements.m_Q(ie,iq,igp,jgp,ilev) = elements.m_qdp(ie,tl.np1_qdp,iq,igp,jgp,ilev) /
                                         ( data.hybrid_ai_delta[ilev]*data.ps0 +
                                           data.hybrid_bi_delta[ilev]*elements.m_ps_v(ie,tl.np1,igp,jgp));
  });

  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
    // call prim_diag_scalars(elem,hvcoord,tl,2,.false.,nets,nete)
  }

  if (compute_energy) {
    // call prim_energy_halftimes(elem,hvcoord,tl,2,.false.,nets,nete)
    Errors::runtime_abort("'compute energy' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
  }

  if (params.energy_fixer) {
    Errors::runtime_abort("'energu fixer' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
    // call prim_energy_fixer(elem,hvcoord,hybrid,tl,nets,nete,nsubstep)
  }

  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
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
                          Errors::err_not_implemented);
    // call prim_printstate(elem, tl, hybrid,hvcoord,nets,nete, fvm)
  }

  // Update the timelevel info to pass back to fortran
  nstep = tl.nstep;
  nm1   = tl.nm1;
  n0    = tl.n0;
  np1   = tl.np1;
}

} // extern "C"

void apply_test_forcing () {
  // Get simulation params
  SimulationParams& params = Context::singleton().get_simulation_params();

  if (params.test_case==TestCase::DCMIP2012_TEST2_1 ||
      params.test_case==TestCase::DCMIP2012_TEST2_2) {
    Errors::runtime_abort("Test case not yet available in C++ build.\n",
                          Errors::err_not_implemented);
  }
}

} // namespace Homme
