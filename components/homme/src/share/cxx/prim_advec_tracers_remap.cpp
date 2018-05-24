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
    Errors::option_error("prim_advec_tracers_remap","use_semi_lagrangian_transport",
                          params.use_semi_lagrangian_transport);
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

  // Get the ESF
  EulerStepFunctor& esf = Context::singleton().get_euler_step_functor();
  esf.reset(params);

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

  if ( ! EulerStepFunctor::is_quasi_monotone(params.limiter_option)) {
    Errors::option_error("prim_advec_tracers_remap_RK2","limiter_option",
                          params.limiter_option);
    // call advance_hypervis_scalar(edgeadv,elem,hvcoord,hybrid,deriv,tl%np1,np1_qdp,nets,nete,dt)
  }
  GPTLstop("tl-at prim_advec_tracers_remap_RK2");
}

} // namespace Homme
