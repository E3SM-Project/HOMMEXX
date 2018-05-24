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

#include "CaarFunctor.hpp"
#include "Context.hpp"
#include "HyperviscosityFunctor.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "ErrorDefs.hpp"
#include "mpi/BoundaryExchange.hpp"

#include "profiling.hpp"

namespace Homme
{

void u3_5stage_timestep(const int nm1, const int n0, const int np1, const int n0_qdp,
                        const Real dt, const Real eta_ave_w, const bool compute_diagnostics);

// -------------- IMPLEMENTATIONS -------------- //

void prim_advance_exp (const int nm1, const int n0, const int np1,
                       const Real dt, const bool compute_diagnostics)
{
  GPTLstart("tl-ae prim_advance_exp");
  // Get simulation params
  SimulationParams& params = Context::singleton().get_simulation_params();

  // Note: In the following, all the checks are superfluous, since we already check that
  //       the options are supported when we init the simulation params. However, this way
  //       we remind ourselves that in these cases there is some missing code to convert from Fortran

  // Get time level info, and determine the tracers time level
  TimeLevel& tl = Context::singleton().get_time_level();
  tl.n0_qdp= -1;
  if (params.moisture == MoistDry::MOIST) {
    tl.update_tracers_levels(params.qsplit);
  }

  // Set eta_ave_w
  int method = params.time_step_type;
  Real eta_ave_w = 1.0/params.qsplit;

  if (params.time_step_type==0) {
    std::string msg = "[prim_advance_exp_iter]:";
    msg += "missing some code for this time step method. ";
    msg += "The program should have errored out earlier though. Plese, investigate.";
    Errors::runtime_abort(msg,Errors::err_not_implemented);
  } else if (params.time_step_type==1) {
    std::string msg = "[prim_advance_exp_iter]:";
    msg += "missing some code for this time step method. ";
    msg += "The program should have errored out earlier though. Plese, investigate.";
    Errors::runtime_abort(msg,Errors::err_not_implemented);
  }

#ifndef CAM
  // if "prescribed wind" set dynamics explicitly and skip time-integration
  if (params.prescribed_wind) {
    Errors::runtime_abort("'prescribed wind' functionality not yet available in C++ build.\n",
                           Errors::err_not_implemented);
  }
#endif

  // Perform time-advance
  switch (method) {
    case 5:
      // Perform RK stages
      u3_5stage_timestep(nm1, n0, np1, tl.n0_qdp, dt, eta_ave_w, compute_diagnostics);
      break;
    default:
      {
        std::string msg = "[prim_advance_exp_iter]:";
        msg += "missing some code for this time step method. ";
        msg += "The program should have errored out earlier though. Plese, investigate.";
        Errors::runtime_abort(msg,Errors::err_not_implemented);
      }
  }

#ifdef ENERGY_DIAGNOSTICS
  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
  }
#endif

  if (params.time_step_type==0) {
    std::string msg = "[prim_advance_exp_iter]:";
    msg += "missing some code for this time step method. ";
    msg += "The program should have errored out earlier though. Plese, investigate.";
    Errors::runtime_abort(msg,Errors::err_not_implemented);
    // call advance_hypervis_lf(edge3p1,elem,hvcoord,hybrid,deriv,nm1,n0,np1,nets,nete,dt_vis)

  } else if (params.time_step_type<=10) {
    // Get and run the HVF
    HyperviscosityFunctor& functor = Context::singleton().get_hyperviscosity_functor();
    GPTLstart("tl-ae advance_hypervis_dp");
    functor.run(np1,dt,eta_ave_w);
    GPTLstop("tl-ae advance_hypervis_dp");
  }

#ifdef ENERGY_DIAGNOSTICS
  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
  }
#endif
  GPTLstop("tl-ae prim_advance_exp");
}

void u3_5stage_timestep(const int nm1, const int n0, const int np1, const int n0_qdp,
                        const Real dt, const Real eta_ave_w, const bool compute_diagnostics)
{
  GPTLstart("tl-ae U3-5stage_timestep");
  // Get elements structure
  Elements& elements = Context::singleton().get_elements();

  // Create the functor
  CaarFunctor& functor = Context::singleton().get_caar_functor();
  functor.set_n0_qdp(n0_qdp);

  // ===================== RK STAGES ===================== //

  // Stage 1: u1 = u0 + dt/5 RHS(u0),          t_rhs = t
  functor.run(n0,n0,nm1,dt/5.0,eta_ave_w/4.0,compute_diagnostics);

  // Stage 2: u2 = u0 + dt/5 RHS(u1),          t_rhs = t + dt/5
  functor.run(n0,nm1,np1,dt/5.0,0.0,false);

  // Stage 3: u3 = u0 + dt/3 RHS(u2),          t_rhs = t + dt/5 + dt/5
  functor.run(n0,np1,np1,dt/3.0,0.0,false);

  // Stage 4: u4 = u0 + 2dt/3 RHS(u3),         t_rhs = t + dt/5 + dt/5 + dt/3
  functor.run(n0,np1,np1,2.0*dt/3.0,0.0,false);

  // Compute (5u1-u0)/4 and store it in timelevel nm1
  {
    const auto t = elements.m_t;
    const auto v = elements.m_v;
    const auto dp3d = elements.m_dp3d;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, elements.num_elems()*NP*NP*NUM_LEV),
      KOKKOS_LAMBDA(const int it) {
         const int ie = it / (NP*NP*NUM_LEV);
         const int igp = (it / (NP*NUM_LEV)) % NP;
         const int jgp = (it / NUM_LEV) % NP;
         const int ilev = it % NUM_LEV;
         t(ie,nm1,igp,jgp,ilev) = (5.0*t(ie,nm1,igp,jgp,ilev)-t(ie,n0,igp,jgp,ilev))/4.0;
         v(ie,nm1,0,igp,jgp,ilev) = (5.0*v(ie,nm1,0,igp,jgp,ilev)-v(ie,n0,0,igp,jgp,ilev))/4.0;
         v(ie,nm1,1,igp,jgp,ilev) = (5.0*v(ie,nm1,1,igp,jgp,ilev)-v(ie,n0,1,igp,jgp,ilev))/4.0;
         dp3d(ie,nm1,igp,jgp,ilev) = (5.0*dp3d(ie,nm1,igp,jgp,ilev)-dp3d(ie,n0,igp,jgp,ilev))/4.0;
    });
  }
  ExecSpace::fence();

  // Stage 5: u5 = (5u1-u0)/4 + 3dt/4 RHS(u4), t_rhs = t + dt/5 + dt/5 + dt/3 + 2dt/3
  functor.run(nm1,np1,np1,3.0*dt/4.0,3.0*eta_ave_w/4.0,false);
  GPTLstop("tl-ae U3-5stage_timestep");
}

} // namespace Homme
