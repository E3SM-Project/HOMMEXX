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
#include "CaarFunctorImpl.hpp"
#include "Context.hpp"
#include "SimulationParams.hpp"

#include "profiling.hpp"

#include <assert.h>
#include <type_traits>


namespace Homme {

CaarFunctor::CaarFunctor()
 : m_policy (Homme::get_default_team_policy<ExecSpace>(Context::singleton().get_elements().num_elems()))
{
  Elements&        elements   = Context::singleton().get_elements();
  Tracers&         tracers    = Context::singleton().get_tracers();
  Derivative&      derivative = Context::singleton().get_derivative();
  HybridVCoord&    hvcoord    = Context::singleton().get_hvcoord();
  SphereOperators& sphere_ops = Context::singleton().get_sphere_operators();
  const int        rsplit     = Context::singleton().get_simulation_params().rsplit;

  // Build functor impl
  m_caar_impl.reset(new CaarFunctorImpl(elements,tracers,derivative,hvcoord,sphere_ops,rsplit));
  m_caar_impl->m_sphere_ops.allocate_buffers(m_policy);
}

CaarFunctor::CaarFunctor(const Elements &elements, const Tracers &tracers,
                         const Derivative &derivative,
                         const HybridVCoord &hvcoord,
                         const SphereOperators &sphere_ops, const int rsplit)
    : m_policy(
          Homme::get_default_team_policy<ExecSpace>(elements.num_elems())) {
  // Build functor impl
  m_caar_impl.reset(new CaarFunctorImpl(elements, tracers, derivative, hvcoord,
                                        sphere_ops, rsplit));
  m_caar_impl->m_sphere_ops.allocate_buffers(m_policy);
}

CaarFunctor::~CaarFunctor ()
{
  // This empty destructor (where CaarFunctorImpl type is completely known)
  // is necessary for pimpl idiom to work with unique_ptr. The issue is the
  // deleter, which needs to know the size of the stored type, and which
  // would be called from the implicitly declared default destructor, which
  // would be in the header file, where CaarFunctorImpl type is incomplete.
}

void CaarFunctor::init_boundary_exchanges (const std::shared_ptr<BuffersManager>& bm_exchange) {
  assert (m_caar_impl);
  m_caar_impl->init_boundary_exchanges(bm_exchange);
}

void CaarFunctor::set_n0_qdp (const int n0_qdp)
{
  // Sanity check (should NEVER happen)
  assert (m_caar_impl);

  // Forward input to impl
  m_caar_impl->set_n0_qdp(n0_qdp);
}

void CaarFunctor::set_rk_stage_data (const int nm1, const int n0,   const int np1,
                                     const Real dt, const Real eta_ave_w,
                                     const bool compute_diagnostics)
{
  // Sanity check (should NEVER happen)
  assert (m_caar_impl);

  // Forward inputs to impl
  m_caar_impl->set_rk_stage_data(nm1,n0,np1,dt,eta_ave_w,compute_diagnostics);
}

void CaarFunctor::run ()
{
  // Sanity check (should NEVER happen)
  assert (m_caar_impl);

  // Run functor
  profiling_resume();
  GPTLstart("caar compute");
  Kokkos::parallel_for("caar loop pre-boundary exchange", m_policy, *m_caar_impl);
  ExecSpace::fence();
  GPTLstop("caar compute");
  profiling_pause();
}

void CaarFunctor::run (const int nm1, const int n0,   const int np1,
          const Real dt, const Real eta_ave_w,
          const bool compute_diagnostics)
{
  // Sanity check (should NEVER happen)
  assert (m_caar_impl);

  // Forward inputs to impl
  m_caar_impl->set_rk_stage_data(nm1,n0,np1,dt,eta_ave_w,compute_diagnostics);

  // Run functor
  profiling_resume();
  GPTLstart("caar compute");
  Kokkos::parallel_for("caar loop pre-boundary exchange", m_policy, *m_caar_impl);
  ExecSpace::fence();
  GPTLstop("caar compute");
  start_timer("caar_bexchV");
  m_caar_impl->m_bes[np1]->exchange(m_caar_impl->m_elements.m_rspheremp);
  stop_timer("caar_bexchV");
  profiling_pause();
}

} // Namespace Homme
