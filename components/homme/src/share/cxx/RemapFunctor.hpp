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

#ifndef HOMMEXX_REMAP_FUNCTOR_HPP
#define HOMMEXX_REMAP_FUNCTOR_HPP

#include <memory>
#include <type_traits>

#include "ErrorDefs.hpp"

#include "Elements.hpp"
#include "Tracers.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "Types.hpp"
#include "utilities/LoopsUtils.hpp"
#include "utilities/MathUtils.hpp"
#include "utilities/SubviewUtils.hpp"
#include "utilities/SyncUtils.hpp"

#include "profiling.hpp"

namespace Homme {
namespace Remap {
// All VertRemapAlg types must provide the following methods:
// compute_grids_phase, and compute_remap_phase
//
// compute_grids_phase is expected to have less parallelism available and to
// compute quantities which are independent of the tracers,
// based on the computed partitions
//
// compute_remap_phase remaps each of the tracers based on the quantities
// previously computed in compute_grids_phase.
// It is also expected to have a large amount of parallelism, specifically
// qsize * num_elems
struct VertRemapAlg {};

template <bool nonzero_rsplit> struct _RemapFunctorRSplit {
  static_assert(nonzero_rsplit == false, "The template specialization for "
                                         "_RemapFunctorRSplit seems to have "
                                         "been removed.");

  static constexpr int num_states_remap = 0;

  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> m_src_layer_thickness;
  explicit _RemapFunctorRSplit(const int &num_elems)
      : m_src_layer_thickness("Source layer thickness", num_elems) {}

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> get_source_thickness(
      const int ie, const int np1,
      ExecViewUnmanaged<const Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp3d)
      const {
    return Homme::subview(m_src_layer_thickness, ie);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> compute_source_thickness(
      KernelVariables &kv, const int &np1, const Real &dt,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness,
      ExecViewUnmanaged<const Scalar * [NP][NP][NUM_LEV]> eta_dot_dpdn,
      ExecViewUnmanaged<const Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp3d)
      const {
    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness =
        get_source_thickness(kv.ie, np1, dp3d);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
                           [&](const int &level) {
        const int ilev = level / VECTOR_SIZE;
        const int vlev = level % VECTOR_SIZE;
        const int next_ilev = (level + 1) / VECTOR_SIZE;
        const int next_vlev = (level + 1) % VECTOR_SIZE;
        const auto eta_dot_dpdn_next =
            (level + 1 < NUM_PHYSICAL_LEV
                 ? eta_dot_dpdn(kv.ie, igp, jgp, next_ilev)[next_vlev]
                 : 0);
        const Real delta_dpdn =
            eta_dot_dpdn_next - eta_dot_dpdn(kv.ie, igp, jgp, ilev)[vlev];
        src_layer_thickness(igp, jgp, ilev)[vlev] =
            tgt_layer_thickness(igp, jgp, ilev)[vlev] + dt * delta_dpdn;
      });
    });
    kv.team_barrier();
    return src_layer_thickness;
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>
  get_state(const KernelVariables &kv, const Elements &elements, int np1,
            int var) const {
    return ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>();
  }
};

template <> struct _RemapFunctorRSplit<true> {
  explicit _RemapFunctorRSplit(const int &num_elems) {}

  static constexpr int num_states_remap = 3;

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> get_source_thickness(
      const int ie, const int np1,
      ExecViewUnmanaged<const Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp3d)
      const {
    return Homme::subview(dp3d, ie, np1);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> compute_source_thickness(
      KernelVariables &kv, const int np1, const Real dt,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness,
      ExecViewUnmanaged<const Scalar * [NP][NP][NUM_LEV]> eta_dot_dpdn,
      ExecViewUnmanaged<const Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp3d)
      const {
    return get_source_thickness(kv.ie, np1, dp3d);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>
  get_state(const KernelVariables &kv, const Elements &elements, int np1,
            int var) const {
    switch (var) {
    case 0:
      return Homme::subview(elements.m_v, kv.ie, np1, 0);
    case 1:
      return Homme::subview(elements.m_v, kv.ie, np1, 1);
    default:
      assert(var == 2);
      return Homme::subview(elements.m_t, kv.ie, np1);
    }
  }
};

struct Remapper {
  virtual ~Remapper() {}
  virtual void run_remap(int np1, int n0_qdp, double dt) = 0;
};

template <bool nonzero_rsplit, template <typename...> class _RemapType,
          typename... RemapOptions>
struct RemapFunctor : public Remapper,
                      public _RemapFunctorRSplit<nonzero_rsplit> {

  using RemapType = _RemapType<RemapOptions...>;

  static_assert(std::is_base_of<VertRemapAlg, RemapType>::value,
                "RemapFunctor not given a remap algorithm to use");

  struct RemapData {
    RemapData(const int qsize_in) : qsize(qsize_in) {}
    const int qsize;
    int np1;
    int n0_qdp;
    Real dt;
  };

  RemapData m_data;
  const Elements m_elements;
  const Tracers m_tracers;
  const HybridVCoord m_hvcoord;

  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> m_tgt_layer_thickness;

  ExecViewManaged<bool *> valid_layer_thickness;
  typename decltype(valid_layer_thickness)::HostMirror host_valid_input;

  RemapType m_remap;

  explicit RemapFunctor(const int qsize, const Elements &elements,
                        const Tracers &tracers, const HybridVCoord &hvcoord)
      : _RemapFunctorRSplit<nonzero_rsplit>(elements.num_elems()),
        m_data(qsize), m_elements(elements), m_tracers(tracers),
        m_hvcoord(hvcoord),
        m_tgt_layer_thickness("Target Layer Thickness", elements.num_elems()),
        valid_layer_thickness(
            "Check for whether the surface thicknesses are positive",
            elements.num_elems()),
        host_valid_input(Kokkos::create_mirror_view(valid_layer_thickness)),
        m_remap(elements.num_elems(), this->num_to_remap()) {
    // Nothing to be done here
  }

  void input_valid_assert() {
    Kokkos::deep_copy(host_valid_input, valid_layer_thickness);
    for (int ie = 0; ie < m_elements.num_elems(); ++ie) {
      if (host_valid_input(ie) == false) {
        Errors::runtime_abort("Negative layer thickness detected, aborting!",
                              Errors::err_negative_layer_thickness);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  int num_to_remap() const { return this->num_states_remap + m_data.qsize; }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>
  get_remap_val(const KernelVariables &kv, int var) const {
    if (!nonzero_rsplit || var >= this->num_states_remap) {
      if (var >= this->num_states_remap)
        var -= this->num_states_remap;
      return Homme::subview(m_tracers.qdp, kv.ie, m_data.n0_qdp, var);
    } else {
      return this->get_state(kv, m_elements, m_data.np1, var);
    }
  }

  struct ComputeThicknessTag {};
  struct ComputeGridsTag {};
  struct ComputeRemapTag {};
  // Computes the extrinsic values of the states in the initial map
  // i.e. velocity -> momentum
  struct ComputeExtrinsicsTag {};
  // Computes the intrinsic values of the states in the final map
  // i.e. momentum -> velocity
  struct ComputeIntrinsicsTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeThicknessTag, const TeamMember &team) const {
    KernelVariables kv(team);
    compute_ps_v(kv, Homme::subview(m_elements.m_dp3d, kv.ie, m_data.np1),
                 Homme::subview(m_elements.m_ps_v, kv.ie, m_data.np1));

    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness =
        compute_target_thickness(kv);

    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness =
        this->compute_source_thickness(
            kv, m_data.np1, m_data.dt, tgt_layer_thickness,
            m_elements.m_eta_dot_dpdn, m_elements.m_dp3d);

    check_source_thickness(kv, src_layer_thickness);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeExtrinsicsTag, const TeamMember &team) const {
    KernelVariables kv(team);

    assert(this->num_states_remap > 0);
    const int den = (this->num_states_remap > 0) ? this->num_states_remap : 1;
    const int var = kv.ie % den;
    kv.ie /= den;
    assert(kv.ie < m_elements.num_elems());

    compute_extrinsic_state(
        kv, this->get_source_thickness(kv.ie, m_data.np1, m_elements.m_dp3d),
        this->get_state(kv, m_elements, m_data.np1, var));
  }

  // This functor is the only one guaranteed to be run
  // (in rare cases, no tracers and rsplit==0),
  // so it needs to be separated from the others to reduce latency on the GPU
  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeGridsTag, const TeamMember &team) const {
    KernelVariables kv(team);
    m_remap.compute_grids_phase(
        kv, this->get_source_thickness(kv.ie, m_data.np1, m_elements.m_dp3d),
        Homme::subview(m_tgt_layer_thickness, kv.ie));
  }

  // This asserts if num_to_remap() == 0
  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeRemapTag, const TeamMember &team) const {
    KernelVariables kv(team);
    assert(num_to_remap() != 0);
    const int var = kv.ie % num_to_remap();
    kv.ie /= num_to_remap();
    assert(kv.ie < m_elements.num_elems());

    auto tgt_layer_thickness = Homme::subview(m_tgt_layer_thickness, kv.ie);
    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness =
        this->get_source_thickness(kv.ie, m_data.np1, m_elements.m_dp3d);

    this->m_remap.compute_remap_phase(kv, get_remap_val(kv, var));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeIntrinsicsTag, const TeamMember &team) const {
    KernelVariables kv(team);

    assert(this->num_states_remap != 0);
    const int den = (this->num_states_remap > 0) ? this->num_states_remap : 1;
    const int var = kv.ie % den;
    kv.ie /= den;
    assert(kv.ie < m_elements.num_elems());

    auto tgt_layer_thickness = Homme::subview(m_tgt_layer_thickness, kv.ie);
    compute_intrinsic_state(kv, tgt_layer_thickness,
                            this->get_state(kv, m_elements, m_data.np1, var));
  }

  void run_remap(int np1, int n0_qdp, double dt) override {
    m_data.np1 = np1;
    m_data.n0_qdp = n0_qdp;
    m_data.dt = dt;
    run_remap();
  }

  void run_remap() {
    // This runs the remap algorithm after determining it needs to
    // It also verifies the state of the simulation is valid
    // If there's nothing to remap, it will only perform the verification
    run_functor<ComputeThicknessTag>("Remap Thickness Functor",
                                     this->m_elements.num_elems());
    if (num_to_remap() > 0) {
      // We don't want the latency of launching an empty kernel
      if (nonzero_rsplit) {
        run_functor<ComputeExtrinsicsTag>("Remap Scale States Functor",
                                          m_elements.num_elems() *
                                              this->num_states_remap);
      }
      run_functor<ComputeGridsTag>("Remap Compute Grids Functor",
                                   this->m_elements.num_elems());
      run_functor<ComputeRemapTag>("Remap Compute Remap Functor",
                                   this->m_elements.num_elems() *
                                       num_to_remap());
      if (nonzero_rsplit) {
        run_functor<ComputeIntrinsicsTag>("Remap Rescale States Functor",
                                          m_elements.num_elems() *
                                              this->num_states_remap);
      }
    }
    this->input_valid_assert();
  }

private:
  template <typename FunctorTag>
  typename std::enable_if<OnGpu<ExecSpace>::value == false,
                          Kokkos::TeamPolicy<ExecSpace, FunctorTag> >::type
  remap_team_policy(int num_exec) {
    return Homme::get_default_team_policy<ExecSpace, FunctorTag>(num_exec);
  }

  template <typename FunctorTag>
  typename std::enable_if<OnGpu<ExecSpace>::value == true,
                          Kokkos::TeamPolicy<ExecSpace, FunctorTag> >::type
  remap_team_policy(int num_exec) {
    constexpr int num_threads = 16;
    constexpr int num_vectors = 32;
    return Kokkos::TeamPolicy<ExecSpace, FunctorTag>(num_exec, num_threads,
                                                     num_vectors);
  }

  template <typename FunctorTag>
  void run_functor(const std::string functor_name, int num_exec) {
    const auto policy = remap_team_policy<FunctorTag>(num_exec);
    // Timers don't work on CUDA, so place them here
    GPTLstart(functor_name.c_str());
    profiling_resume();
    Kokkos::parallel_for("vertical remap", policy, *this);
    ExecSpace::fence();
    profiling_pause();
    GPTLstop(functor_name.c_str());
  }

  KOKKOS_INLINE_FUNCTION
  void compute_ps_v(KernelVariables &kv,
                    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> dp3d,
                    ExecViewUnmanaged<Real[NP][NP]> ps_v) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      // Using parallel_reduce to calculate this doesn't seem to work,
      // even when accumulating into a Scalar and computing the final sum in
      // serial, and is likely a Kokkos bug since ivdep shouldn't matter.
      Kokkos::single(Kokkos::PerThread(kv.team), [&]() {
        ps_v(igp, jgp) = 0.0;
        for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
          const int ilev = level / VECTOR_SIZE;
          const int vlev = level % VECTOR_SIZE;
          ps_v(igp, jgp) += dp3d(igp, jgp, ilev)[vlev];
        }
        ps_v(igp, jgp) += m_hvcoord.hybrid_ai0 * m_hvcoord.ps0;
      });
    });
    kv.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION void compute_extrinsic_state(
      KernelVariables &kv,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness,
      ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> state_remap) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &ilev) {
        state_remap(igp, jgp, ilev) *= src_layer_thickness(igp, jgp, ilev);
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_intrinsic_state(
      KernelVariables &kv,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness,
      ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> state_remap) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &ilev) {
        state_remap(igp, jgp, ilev) /= tgt_layer_thickness(igp, jgp, ilev);
      });
    });
  }

  KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]>
  compute_target_thickness(KernelVariables &kv) const {
    auto tgt_layer_thickness = Homme::subview(m_tgt_layer_thickness, kv.ie);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
                           [&](const int &ilevel) {
        const int ilev = ilevel / VECTOR_SIZE;
        const int vec_lev = ilevel % VECTOR_SIZE;
        tgt_layer_thickness(igp, jgp, ilev)[vec_lev] =
            (m_hvcoord.hybrid_ai(ilevel + 1) - m_hvcoord.hybrid_ai(ilevel)) *
                m_hvcoord.ps0 +
            (m_hvcoord.hybrid_bi(ilevel + 1) - m_hvcoord.hybrid_bi(ilevel)) *
                m_elements.m_ps_v(kv.ie, m_data.np1, igp, jgp);
      });
    });
    kv.team_barrier();
    return tgt_layer_thickness;
  }

  KOKKOS_INLINE_FUNCTION void check_source_thickness(
      KernelVariables &kv,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness)
      const {
    // Kokkos parallel reduce doesn't support bool as a reduction type, so use
    // int instead
    // Reduce starts with false (0), making that the default state
    // If there is an error, this becomes true
    int invalid = false;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(kv.team, NP * NP * NUM_PHYSICAL_LEV),
        [&](const int &loop_idx, int &is_invalid) {
          const int igp = (loop_idx / NUM_PHYSICAL_LEV) / NP;
          const int jgp = (loop_idx / NUM_PHYSICAL_LEV) % NP;
          const int level = loop_idx % NUM_PHYSICAL_LEV;
          const int ilev = level / VECTOR_SIZE;
          const int vlev = level % VECTOR_SIZE;
          is_invalid |= (src_layer_thickness(igp, jgp, ilev)[vlev] < 0.0);
        },
        invalid);
    valid_layer_thickness(kv.ie) = !invalid;
  }
};

} // namespace Remap
} // namespace Homme

#endif // HOMMEXX_REMAP_FUNCTOR_HPP
