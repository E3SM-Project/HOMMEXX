#ifndef KERNEL_VARIABLES_HPP
#define KERNEL_VARIABLES_HPP

#include "Types.hpp"

namespace Homme {

struct KernelVariables {
  //amb Don't use scratch memory unless definitely needed. Little
  // stack-allocated arrays are usually the way to go, even on GPU.
  KOKKOS_INLINE_FUNCTION
  KernelVariables(const TeamMember &team_in)
    : team(team_in), /*scratch_mem(allocate_thread<Real, Real[NP][NP]>()),*/
        ie(team.league_rank()), ilev(-1) {} //, igp(-1), jgp(-1) {}

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
    size_t mem_size = sizeof(Real[NP][NP]) * team_size;
    return mem_size;
  }


  KOKKOS_INLINE_FUNCTION
  void team_barrier() const {
    team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  int team_rank() const {
    return team.team_rank();
  }

  KOKKOS_INLINE_FUNCTION
  int team_size() const {
    return team.team_size();
  }

  const TeamMember& team;

  // Fast memory for the kernel
  //ExecViewUnmanaged<Real[NP][NP]> scratch_mem;

  int ie, ilev;
}; // KernelVariables

} // Homme

#endif // KERNEL_VARIABLES_HPP
