#ifndef KERNEL_VARIABLES_HPP
#define KERNEL_VARIABLES_HPP

#include "Types.hpp"

namespace Homme {

struct KernelVariables {
  KOKKOS_INLINE_FUNCTION
  KernelVariables(const TeamMember &team_in)
      : team(team_in)
      , scratch_mem_1(allocate_thread<Real, Real[NP][NP]>())
      , scratch_mem_2(allocate_thread<Real, Real[NP][NP]>())
      , ie(team.league_rank()), ilev(-1)
  {
    // Nothing else to be done here
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
    size_t mem_size = 2 * sizeof(Real[NP][NP]) * team_size;
    return mem_size;
  }

  const TeamMember &team;

  // Fast memory for the kernel
  ExecViewUnmanaged<Real[NP][NP]> scratch_mem_1;
  ExecViewUnmanaged<Real[NP][NP]> scratch_mem_2;

  int ie, ilev;
}; // KernelVariables

} // Homme

#endif // KERNEL_VARIABLES_HPP
