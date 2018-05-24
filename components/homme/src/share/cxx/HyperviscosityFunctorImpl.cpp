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
#include "HyperviscosityFunctorImpl.hpp"
#include "BoundaryExchange.hpp"
#include "profiling.hpp"

namespace Homme
{

HyperviscosityFunctorImpl::HyperviscosityFunctorImpl (const SimulationParams& params, const Elements& elements, const Derivative& deriv)
 : m_elements   (elements)
 , m_deriv      (deriv)
 , m_data       (params.hypervis_subcycle,1.0,params.nu_top,params.nu,params.nu_p,params.nu_s)
 , m_sphere_ops (Context::singleton().get_sphere_operators())
{
  // Sanity check
  assert(params.params_set);

  if (m_data.nu_top>0) {
    m_nu_scale_top = ExecViewManaged<Scalar[NUM_LEV]>("nu_scale_top");
    ExecViewManaged<Scalar[NUM_LEV]>::HostMirror h_nu_scale_top;
    h_nu_scale_top = Kokkos::create_mirror_view(m_nu_scale_top);

    constexpr int NUM_BIHARMONIC_PHYSICAL_LEVELS = 3;
    Kokkos::Array<Real,NUM_BIHARMONIC_PHYSICAL_LEVELS> lev_nu_scale_top = { 4.0, 2.0, 1.0 };
    for (int phys_lev=0; phys_lev<NUM_BIHARMONIC_PHYSICAL_LEVELS; ++phys_lev) {
      const int ilev = phys_lev / VECTOR_SIZE;
      const int ivec = phys_lev % VECTOR_SIZE;
      h_nu_scale_top(ilev)[ivec] = lev_nu_scale_top[phys_lev]*m_data.nu_top;
    }
    Kokkos::deep_copy(m_nu_scale_top, h_nu_scale_top);
  }

  // Allocate buffers in the sphere operators
  m_sphere_ops.allocate_buffers(Homme::get_default_team_policy<ExecSpace>(m_elements.num_elems()));
}

void HyperviscosityFunctorImpl::init_boundary_exchanges () {
  m_be = std::make_shared<BoundaryExchange>();
  auto& be = *m_be;
  auto bm_exchange = Context::singleton().get_buffers_manager(MPI_EXCHANGE);
  be.set_buffers_manager(bm_exchange);
  be.set_num_fields(0, 0, 4);
  be.register_field(m_elements.buffers.vtens, 2, 0);
  be.register_field(m_elements.buffers.ttens);
  be.register_field(m_elements.buffers.dptens);
  be.registration_completed();
}

void HyperviscosityFunctorImpl::run (const int np1, const Real dt, const Real eta_ave_w)
{
  m_data.np1 = np1;
  m_data.dt = dt/m_data.hypervis_subcycle;
  m_data.eta_ave_w = eta_ave_w;

  Kokkos::RangePolicy<ExecSpace,TagUpdateStates> policy_update_states(0, m_elements.num_elems()*NP*NP*NUM_LEV);
  const auto policy_pre_exchange =
      Homme::get_default_team_policy<ExecSpace, TagHyperPreExchange>(
          m_elements.num_elems());
  for (int icycle = 0; icycle < m_data.hypervis_subcycle; ++icycle) {
    GPTLstart("hvf-bhwk");
    biharmonic_wk_dp3d ();
    GPTLstop("hvf-bhwk");
    // dispatch parallel_for for first kernel
    Kokkos::parallel_for(policy_pre_exchange, *this);
    Kokkos::fence();

    // Exchange
    assert (m_be->is_registration_completed());
    GPTLstart("hvf-bexch");
    m_be->exchange();
    GPTLstop("hvf-bexch");

    // Update states
    Kokkos::parallel_for(policy_update_states, *this);
    Kokkos::fence();
  }
}

void HyperviscosityFunctorImpl::biharmonic_wk_dp3d() const
{
  // For the first laplacian we use a differnt kernel, which uses directly the states
  // at timelevel np1 as inputs. This way we avoid copying the states to *tens buffers.
  auto policy_first_laplace = Homme::get_default_team_policy<ExecSpace,TagFirstLaplace>(m_elements.num_elems());
  Kokkos::parallel_for(policy_first_laplace, *this);
  Kokkos::fence();

  // Exchange
  assert (m_be->is_registration_completed());
  GPTLstart("hvf-bexch");
  m_be->exchange(m_elements.m_rspheremp);
  GPTLstop("hvf-bexch");

  // TODO: update m_data.nu_ratio if nu_div!=nu
  // Compute second laplacian
  auto policy_second_laplace = Homme::get_default_team_policy<ExecSpace,TagLaplace>(m_elements.num_elems());
  Kokkos::parallel_for(policy_second_laplace, *this);
  Kokkos::fence();
}

} // namespace Homme
