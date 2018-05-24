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

#ifndef KERNEL_VARIABLES_HPP
#define KERNEL_VARIABLES_HPP

#include "Types.hpp"

namespace Homme {

struct KernelVariables {
private:
  struct TeamInfo {

    template<typename ExecSpaceType>
    static
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<!std::is_same<ExecSpaceType,Hommexx_Cuda>::value &&
                            !std::is_same<ExecSpaceType,Hommexx_OpenMP>::value,
                            int
                           >::type
    get_team_idx (const int /*team_size*/, const int /*league_rank*/)
    {
      return 0;
    }

#ifdef KOKKOS_HAVE_CUDA
#ifdef __CUDA_ARCH__
    template <typename ExecSpaceType>
    static KOKKOS_INLINE_FUNCTION typename std::enable_if<
        OnGpu<ExecSpaceType>::value, int>::type
    get_team_idx(const int /*team_size*/, const int league_rank) {
      return league_rank;
    }
#else
    template <typename ExecSpaceType>
    static KOKKOS_INLINE_FUNCTION typename std::enable_if<
        OnGpu<ExecSpaceType>::value, int>::type
    get_team_idx(const int /*team_size*/, const int /*league_rank*/) {
      assert(false); // should never happen
      return -1;
    }
#endif // __CUDA_ARCH__
#endif // KOKKOS_HAVE_CUDA

#ifdef KOKKOS_HAVE_OPENMP
    template<typename ExecSpaceType>
    static
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<ExecSpaceType,Kokkos::OpenMP>::value,int>::type
    get_team_idx (const int team_size, const int /*league_rank*/)
    {
      // Kokkos puts consecutive threads into the same team.
      return omp_get_thread_num() / team_size;
    }
#endif
  };

public:

  KOKKOS_INLINE_FUNCTION
  KernelVariables(const TeamMember &team_in)
      : team(team_in)
      , ie(team_in.league_rank())
      , iq(-1)
      , team_idx(TeamInfo::get_team_idx<ExecSpace>(team_in.team_size(),team_in.league_rank()))
  {
    // Nothing to be done here
  }

  KOKKOS_INLINE_FUNCTION
  KernelVariables(const TeamMember &team_in, const int qsize)
      : team(team_in)
      , ie(team_in.league_rank() / qsize)
      , iq(team_in.league_rank() % qsize)
      , team_idx(TeamInfo::get_team_idx<ExecSpace>(team_in.team_size(),team_in.league_rank()))
  {
    // Nothing to be done here
  }

  template <typename Primitive, typename Data>
  KOKKOS_INLINE_FUNCTION Primitive *allocate_team() const {
    ScratchView<Data> view(team.team_scratch(0));
    return view.data();
  }

  template <typename Primitive, typename Data>
  KOKKOS_INLINE_FUNCTION Primitive *allocate_thread() const {
    ScratchView<Data> view(team.thread_scratch(0));
    return view.data();
  }

  KOKKOS_INLINE_FUNCTION
  static size_t shmem_size(int team_size) {
    size_t mem_size = 0 * team_size;
    return mem_size;
  }

  const TeamMember &team;

  KOKKOS_FORCEINLINE_FUNCTION void team_barrier() const {
    team.team_barrier();
  }

  int ie, iq;
  const int team_idx;
}; // KernelVariables

} // Homme

#endif // KERNEL_VARIABLES_HPP
