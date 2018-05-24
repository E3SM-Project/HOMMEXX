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

#include "HybridVCoord.hpp"
#include "utilities/TestUtils.hpp"

#include <random>

namespace Homme
{

void HybridVCoord::init(const Real ps0_in,
                   CRCPtr hybrid_am_ptr,
                   CRCPtr hybrid_ai_ptr,
                   CRCPtr hybrid_bm_ptr,
                   CRCPtr hybrid_bi_ptr)
{
  ps0 = ps0_in;

  // hybrid_am = ExecViewManaged<Real[NUM_PHYSICAL_LEV]>(
  //    "Hybrid coordinates; coefficient A_midpoints");
  hybrid_ai = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid coordinates; coefficient A_interfaces");
  hybrid_bi = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid coordinates; coefficient B_interfaces");
  // hybrid_bm = ExecViewManaged<Real[NUM_PHYSICAL_LEV]>(
  //    "Hybrid coordinates; coefficient B_midpoints");

  // HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV]>
  // host_hybrid_am(hybrid_am_ptr);
  // HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV]>
  // host_hybrid_bm(hybrid_bm_ptr);
  HostViewUnmanaged<const Real[NUM_INTERFACE_LEV]> host_hybrid_ai(
      hybrid_ai_ptr);
  Kokkos::deep_copy(hybrid_ai, host_hybrid_ai);
  HostViewUnmanaged<const Real[NUM_INTERFACE_LEV]> host_hybrid_bi(
      hybrid_bi_ptr);
  Kokkos::deep_copy(hybrid_bi, host_hybrid_bi);

  // i don't think this saves us much now
  {
    // Only hybrid_ai(0) is needed.
    hybrid_ai0 = hybrid_ai_ptr[0];
  }
  // this is not in master anymore?
  //  assert(hybrid_ai_ptr != nullptr);
  //  assert(hybrid_bi_ptr != nullptr);

  compute_deltas();
}

void HybridVCoord::random_init(int seed) {
  const int min_value = std::numeric_limits<Real>::epsilon();
  const int max_value = 1.0 - min_value;
  hybrid_ai = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid a_interface coefs");
  hybrid_bi = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid b_interface coefs");

  std::mt19937_64 engine(seed);
  ps0 = 1.0;

  // hybrid_a can technically range from 0 to 1 like hybrid_b,
  // but doing so makes enforcing the monotonicity of p = a + b difficult
  // So only go to 0.25
  genRandArray(hybrid_ai, engine, std::uniform_real_distribution<Real>(
                                      min_value, max_value / 4.0));

  HostViewManaged<Real[NUM_INTERFACE_LEV]> host_hybrid_ai(
      "Host hybrid ai coefs");
  Kokkos::deep_copy(host_hybrid_ai, hybrid_ai);

  hybrid_ai0 = host_hybrid_ai(0);

  // p = a + b must be monotonically increasing
  // OG: what is this for? does a test require it?
  // (not critisizm, but i don't understand)
  const auto check_coords = [=](
      HostViewUnmanaged<Real[NUM_INTERFACE_LEV]> coords) {
    // Enforce the boundaries
    coords(0) = 0.0;
    coords(1) = 1.0;
    // Put them in order
    std::sort(coords.data(), coords.data() + coords.size());
    Real p_prev = hybrid_ai0 + coords(0);
    for (int i = 1; i < NUM_INTERFACE_LEV; ++i) {
      // Make certain they're all distinct
      if (coords(i) <=
          coords(i - 1) * (1.0 + std::numeric_limits<Real>::epsilon())) {
        return false;
      }
      // Make certain p = a + b is increasing
      Real p_cur = coords(i) + host_hybrid_ai(i);
      if (p_cur <= p_prev) {
        return false;
      }
      p_prev = p_cur;
    }
    // All good!
    return true;
  };
  genRandArray(hybrid_bi, engine,
               std::uniform_real_distribution<Real>(min_value, max_value),
               check_coords);

  compute_deltas();
}

void HybridVCoord::compute_deltas ()
{
  const auto host_hybrid_ai = Kokkos::create_mirror_view(hybrid_ai);
  const auto host_hybrid_bi = Kokkos::create_mirror_view(hybrid_bi);
  Kokkos::deep_copy(host_hybrid_ai, hybrid_ai);
  Kokkos::deep_copy(host_hybrid_bi, hybrid_bi);

  hybrid_ai_delta = ExecViewManaged<Scalar[NUM_LEV]>(
      "Difference in Hybrid a coordinates between consecutive interfaces");
  hybrid_bi_delta = ExecViewManaged<Scalar[NUM_LEV]>(
      "Difference in Hybrid b coordinates between consecutive interfaces");

  decltype(hybrid_ai_delta)::HostMirror host_hybrid_ai_delta =
      Kokkos::create_mirror_view(hybrid_ai_delta);
  decltype(hybrid_bi_delta)::HostMirror host_hybrid_bi_delta =
      Kokkos::create_mirror_view(hybrid_bi_delta);
  for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
    const int ilev = level / VECTOR_SIZE;
    const int ivec = level % VECTOR_SIZE;

    host_hybrid_ai_delta(ilev)[ivec] =
        host_hybrid_ai(level + 1) - host_hybrid_ai(level);
    host_hybrid_bi_delta(ilev)[ivec] =
        host_hybrid_bi(level + 1) - host_hybrid_bi(level);
  }
  Kokkos::deep_copy(hybrid_ai_delta, host_hybrid_ai_delta);
  Kokkos::deep_copy(hybrid_bi_delta, host_hybrid_bi_delta);
  {
    dp0 = ExecViewManaged<Scalar[NUM_LEV]>("dp0");
    const auto hdp0 = Kokkos::create_mirror_view(dp0);
    for (int ilev = 0; ilev < NUM_LEV; ++ilev) {
      // BFB way of writing it.
      hdp0(ilev) =
          host_hybrid_ai_delta(ilev) * ps0 + host_hybrid_bi_delta(ilev) * ps0;
    }
    Kokkos::deep_copy(dp0, hdp0);
  }
}

} // namespace Homme
