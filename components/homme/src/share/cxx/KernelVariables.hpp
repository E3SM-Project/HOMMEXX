#ifndef KERNEL_VARIABLES_HPP
#define KERNEL_VARIABLES_HPP

#include "Types.hpp"

namespace Homme {

struct KernelVariables {
  KOKKOS_INLINE_FUNCTION
  KernelVariables(const TeamMember &team_in)
      : team(team_in), ie(team.league_rank()), iq(-1) {
  } //, igp(-1), jgp(-1) {}

  KOKKOS_INLINE_FUNCTION
  KernelVariables(const TeamMember &team_in, const int qsize)
      : team(team_in), ie(team.league_rank() / qsize), iq(team.league_rank() % qsize) {
  } //, igp(-1), jgp(-1) {}

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
}; // KernelVariables

} // Homme

#endif // KERNEL_VARIABLES_HPP
