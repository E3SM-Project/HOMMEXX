/*********************************************************************************
 *
 * Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 *
 * For five (5) years from  the United States Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide
 * license in this data to reproduce, prepare derivative works, and perform
 * publicly and display publicly, by or on behalf of the Government. There is
 * provision for the possible extension of the term of this license. Subsequent
 * to that period or any extension granted, the United States Government is
 * granted for itself and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable worldwide license in this data to reproduce, prepare derivative
 * works, distribute copies to the public, perform publicly and display publicly,
 * and to permit others to do so. The specific term of the license can be
 * identified by inquiry made to National Technology and Engineering Solutions of
 * Sandia, LLC or DOE.
 *
 * NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
 * ENERGY, NOR NATIONAL TECHNOLOGY AND ENGINEERING SOLUTIONS OF SANDIA, LLC, NOR
 * ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY
 * LEGAL RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
 * INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS
 * USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
 *
 * Any licensee of this software has the obligation and responsibility to abide
 * by the applicable export control laws, regulations, and general prohibitions
 * relating to the export of technical data. Failure to obtain an export control
 * license or other authority from the Government may result in criminal
 * liability under U.S. laws.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 *     - Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *     - Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimers in the documentation
 *       and/or other materials provided with the distribution.
 *     - Neither the name of Sandia Corporation,
 *       nor the names of its contributors may be used to endorse or promote
 *       products derived from this Software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ********************************************************************************/


#include "Context.hpp"
#include "Elements.hpp"
#include "Tracers.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "ErrorDefs.hpp"
#include "profiling.hpp"

#include <iostream>

namespace Homme
{

void prim_step (const Real, const bool);
void vertical_remap (const Real);
void apply_test_forcing ();

extern "C" {

void prim_run_subcycle_c (const Real& dt, int& nstep, int& nm1, int& n0, int& np1)
{
  GPTLstart("tl-sc prim_run_subcycle_c");
  // Get control and simulation params
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
HybridVCoord  Errors::runtime_abort("CAM forcing not yet availble in C++.\n"
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
  GPTLstart("tl-sc dp3d-from-ps");
  Elements& elements = Context::singleton().get_elements();
  Tracers& tracers = Context::singleton().get_tracers();
  HybridVCoord& hvcoord = Context::singleton().get_hvcoord();
  const auto hybrid_ai_delta = hvcoord.hybrid_ai_delta;
  const auto hybrid_bi_delta = hvcoord.hybrid_bi_delta;
  const auto ps0 = hvcoord.ps0;
  const auto ps_v = elements.m_ps_v;
  {
    const auto dp3d = elements.m_dp3d;
    const auto n0 = tl.n0;
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace> (0,elements.num_elems()*NP*NP*NUM_LEV),
                         KOKKOS_LAMBDA(const int idx) {
      const int ie   = ((idx / NUM_LEV) / NP) / NP;
      const int igp  = ((idx / NUM_LEV) / NP) % NP;
      const int jgp  =  (idx / NUM_LEV) % NP;
      const int ilev =   idx % NUM_LEV;

      dp3d(ie,n0,igp,jgp,ilev) = hybrid_ai_delta[ilev]*ps0
                               + hybrid_bi_delta[ilev]*ps_v(ie,n0,igp,jgp);
    });
  }
  ExecSpace::fence();
  GPTLstop("tl-sc dp3d-from-ps");

  // Loop over rsplit vertically lagrangian timesteps
  GPTLstart("tl-sc prim_step-loop");
  prim_step(dt,compute_diagnostics);
  for (int r=1; r<params.rsplit; ++r) {
    tl.update_dynamics_levels(UpdateType::LEAPFROG);
    prim_step(dt,false);
  }
  GPTLstop("tl-sc prim_step-loop");

  ////////////////////////////////////////////////////////////////////////
  // apply vertical remap
  // always for tracers
  // if rsplit>0:  also remap dynamics and compute reference level ps_v
  ////////////////////////////////////////////////////////////////////////
  tl.update_tracers_levels(params.qsplit);
  GPTLstart("tl-sc vertical_remap");
  vertical_remap(dt_remap);
  GPTLstop("tl-sc vertical_remap");

  ////////////////////////////////////////////////////////////////////////
  // time step is complete.  update some diagnostic variables:
  // lnps (we should get rid of this)
  // Q    (mixing ratio)
  ////////////////////////////////////////////////////////////////////////
  GPTLstart("tl-sc Q-from-qdp");
  {
    const auto qdp = tracers.qdp;
    const auto np1_qdp = tl.np1_qdp;
    const auto np1 = tl.np1;
    const auto qsize = params.qsize;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, elements.num_elems() * params.qsize *
                                              NP * NP * NUM_LEV),
        KOKKOS_LAMBDA(const int idx) {
          const int ie = (((idx / NUM_LEV) / NP) / NP) / qsize;
          const int iq = (((idx / NUM_LEV) / NP) / NP) % qsize;
          const int igp = ((idx / NUM_LEV) / NP) % NP;
          const int jgp = (idx / NUM_LEV) % NP;
          const int ilev = idx % NUM_LEV;

          tracers.q(ie, iq, igp, jgp, ilev) = qdp(ie, np1_qdp, iq, igp, jgp, ilev) /
                              (hybrid_ai_delta[ilev] * ps0 +
                               hybrid_bi_delta[ilev] * ps_v(ie, np1, igp, jgp));
        });
  }
  ExecSpace::fence();
  GPTLstop("tl-sc Q-from-qdp");

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

  GPTLstop("tl-sc prim_run_subcycle_c");
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
