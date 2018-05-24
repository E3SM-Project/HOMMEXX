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

  if (params.use_semi_lagrangian_transport) {
    Errors::option_error("prim_step", "use_semi_lagrangian_transport",params.use_semi_lagrangian_transport);
    // Set derived_star = v
  }

  // ===============
  // initialize mean flux accumulation variables and save some variables at n0
  // for use by advection
  // ===============
  GPTLstart("tl-s deep_copy+derived_dp");
  {
    const auto eta_dot_dpdn = elements.m_eta_dot_dpdn;
    const auto derived_vn0 = elements.m_derived_vn0;
    const auto omega_p = elements.m_omega_p;
    const auto derived_dpdiss_ave = elements.m_derived_dpdiss_ave;
    const auto derived_dpdiss_biharmonic = elements.m_derived_dpdiss_biharmonic;
    const auto derived_dp = elements.m_derived_dp;
    const auto dp3d = elements.m_dp3d;
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace> (0,elements.num_elems()*NP*NP*NUM_LEV),
                         KOKKOS_LAMBDA(const int idx) {
      const int ie   = ((idx / NUM_LEV) / NP) / NP;
      const int igp  = ((idx / NUM_LEV) / NP) % NP;
      const int jgp  =  (idx / NUM_LEV) % NP;
      const int ilev =   idx % NUM_LEV;
      eta_dot_dpdn(ie,igp,jgp,ilev) = 0;
      derived_vn0(ie,0,igp,jgp,ilev) = 0;
      derived_vn0(ie,1,igp,jgp,ilev) = 0;
      omega_p(ie,igp,jgp,ilev) = 0;
      if (params.nu_p>0) {
        derived_dpdiss_ave(ie,igp,jgp,ilev) = 0;
        derived_dpdiss_biharmonic(ie,igp,jgp,ilev) = 0;
      }
      derived_dp(ie,igp,jgp,ilev) = dp3d(ie,tl.n0,igp,jgp,ilev);
    });
  }
  ExecSpace::fence();
  GPTLstop("tl-s deep_copy+derived_dp");

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
