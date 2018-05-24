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

#include "BoundaryExchange.hpp"
#include "BuffersManager.hpp"
#include "CaarFunctor.hpp"
#include "Comm.hpp"
#include "Connectivity.hpp"
#include "Derivative.hpp"
#include "Elements.hpp"
#include "Tracers.hpp"
#include "HybridVCoord.hpp"
#include "HyperviscosityFunctor.hpp"
#include "SphereOperators.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "VerticalRemapManager.hpp"
#include "EulerStepFunctor.hpp"

namespace Homme {

Context::Context() {}

Context::~Context() {}

CaarFunctor& Context::get_caar_functor() {
  if ( ! caar_functor_) {
    caar_functor_.reset(new CaarFunctor());
  }
  return *caar_functor_;
}

Comm& Context::get_comm() {
  if ( ! comm_) {
    comm_.reset(new Comm());
    comm_->init();
  }
  return *comm_;
}

Elements& Context::get_elements() {
  if ( ! elements_) elements_.reset(new Elements());
  return *elements_;
}

Tracers& Context::get_tracers() {
  if ( ! tracers_) {
    auto &elems = get_elements();
    const auto &params = get_simulation_params();
    tracers_.reset(new Tracers(elems.num_elems(), params.qsize));
  }
  return *tracers_;
}

HybridVCoord& Context::get_hvcoord() {
  if ( ! hvcoord_) hvcoord_.reset(new HybridVCoord());
  return *hvcoord_;
}

HyperviscosityFunctor& Context::get_hyperviscosity_functor() {
  if ( ! hyperviscosity_functor_) {
    Elements& e = get_elements();
    Derivative& d = get_derivative();
    SimulationParams& p = get_simulation_params();
    hyperviscosity_functor_.reset(new HyperviscosityFunctor(p,e,d));
  }
  return *hyperviscosity_functor_;
}

Derivative& Context::get_derivative() {
  //if ( ! derivative_) derivative_ = std::make_shared<Derivative>();
  if ( ! derivative_) derivative_.reset(new Derivative());
  return *derivative_;
}

SimulationParams& Context::get_simulation_params() {
  if ( ! simulation_params_) simulation_params_.reset(new SimulationParams());
  return *simulation_params_;
}

TimeLevel& Context::get_time_level() {
  if ( ! time_level_) time_level_.reset(new TimeLevel());
  return *time_level_;
}

VerticalRemapManager& Context::get_vertical_remap_manager() {
  if ( ! vertical_remap_mgr_) vertical_remap_mgr_.reset(new VerticalRemapManager());
  return *vertical_remap_mgr_;
}

std::shared_ptr<BuffersManager> Context::get_buffers_manager(short int exchange_type) {
  if ( ! buffers_managers_) {
    buffers_managers_.reset(new BMMap());
  }

  if (!(*buffers_managers_)[exchange_type]) {
    (*buffers_managers_)[exchange_type] = std::make_shared<BuffersManager>(get_connectivity());
  }
  return (*buffers_managers_)[exchange_type];
}

std::shared_ptr<Connectivity> Context::get_connectivity() {
  if ( ! connectivity_) connectivity_.reset(new Connectivity());
  return connectivity_;
}

SphereOperators& Context::get_sphere_operators() {
  if ( ! sphere_operators_) {
    Elements&   elements   = get_elements();
    Derivative& derivative = get_derivative();
    sphere_operators_.reset(new SphereOperators(elements,derivative));
  }
  return *sphere_operators_;
}

EulerStepFunctor& Context::get_euler_step_functor() {
  if ( ! euler_step_functor_) euler_step_functor_.reset(new EulerStepFunctor());
  return *euler_step_functor_;
}

void Context::clear() {
  comm_ = nullptr;
  elements_ = nullptr;
  tracers_ = nullptr;
  derivative_ = nullptr;
  hvcoord_ = nullptr;
  hyperviscosity_functor_ = nullptr;
  connectivity_ = nullptr;
  buffers_managers_ = nullptr;
  simulation_params_ = nullptr;
  sphere_operators_ = nullptr;
  time_level_ = nullptr;
  vertical_remap_mgr_ = nullptr;
  caar_functor_ = nullptr;
  euler_step_functor_ = nullptr;
}

Context& Context::singleton() {
  static Context c;
  return c;
}

void Context::finalize_singleton() {
  singleton().clear();
}

} // namespace Homme
