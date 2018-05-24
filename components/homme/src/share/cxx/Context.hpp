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

#ifndef HOMMEXX_CONTEXT_HPP
#define HOMMEXX_CONTEXT_HPP

#include <string>
#include <map>
#include <memory>

namespace Homme {

class BoundaryExchange;
class BuffersManager;
class CaarFunctor;
class Comm;
class Connectivity;
class Derivative;
class Elements;
class Tracers;
class HybridVCoord;
class HyperviscosityFunctor;
class SimulationParams;
class SphereOperators;
class TimeLevel;
class VerticalRemapManager;
class EulerStepFunctor;

/* A Context manages resources previously treated as singletons. Context is
 * meant to have two roles. First, a Context singleton is the only singleton in
 * the program. Second, a context need not be a singleton, and each Context
 * object can have different Elements, Derivative, etc., objects. (That
 * probably isn't needed, but Context immediately supports it.)
 *
 * Finally, Context has two singleton functions: singleton(), which returns
 * Context&, and finalize_singleton(). The second is called in a unit test exe
 * main before Kokkos::finalize().
 */
class Context {
public:
  using BMMap = std::map<int,std::shared_ptr<BuffersManager>>;

private:
  // Note: using uniqe_ptr disables copy construction
  std::unique_ptr<CaarFunctor>            caar_functor_;
  std::unique_ptr<Comm>                   comm_;
  std::unique_ptr<Elements>               elements_;
  std::unique_ptr<Tracers>                tracers_;
  std::unique_ptr<Derivative>             derivative_;
  std::unique_ptr<HybridVCoord>           hvcoord_;
  std::unique_ptr<HyperviscosityFunctor>  hyperviscosity_functor_;
  std::shared_ptr<Connectivity>           connectivity_;
  std::shared_ptr<BMMap>                  buffers_managers_;
  std::unique_ptr<SimulationParams>       simulation_params_;
  std::unique_ptr<TimeLevel>              time_level_;
  std::unique_ptr<VerticalRemapManager>   vertical_remap_mgr_;
  std::unique_ptr<SphereOperators>        sphere_operators_;
  std::unique_ptr<EulerStepFunctor>       euler_step_functor_;

  // Clear the objects Context manages.
  void clear();

public:
  Context();
  virtual ~Context();

  // Getters for each managed object.
  CaarFunctor& get_caar_functor();
  Comm& get_comm();
  Elements& get_elements();
  Tracers& get_tracers();
  Derivative& get_derivative();
  HybridVCoord& get_hvcoord();
  HyperviscosityFunctor& get_hyperviscosity_functor();
  SimulationParams& get_simulation_params();
  SphereOperators& get_sphere_operators();
  TimeLevel& get_time_level();
  EulerStepFunctor& get_euler_step_functor();
  VerticalRemapManager& get_vertical_remap_manager();
  std::shared_ptr<Connectivity> get_connectivity();
  BMMap& get_buffers_managers();
  std::shared_ptr<BuffersManager> get_buffers_manager(short int exchange_type);

  // Exactly one singleton.
  static Context& singleton();

  static void finalize_singleton();
};

}

#endif // HOMMEXX_CONTEXT_HPP
