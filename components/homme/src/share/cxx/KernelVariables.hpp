#ifndef KERNEL_VARIABLES_HPP
#define KERNEL_VARIABLES_HPP

#include "Types.hpp"

namespace Homme {

struct KernelVariables {
  KOKKOS_INLINE_FUNCTION
  KernelVariables(const TeamMember &team_in)
      : team(team_in), scratch_mem(allocate_thread<Real, Real[NP][NP]>()),
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

  const TeamMember &team;

  // Fast memory for the kernel
  ExecViewUnmanaged<Real[NP][NP]> scratch_mem;

  struct BufferViews {
    static constexpr int NUM_SCALAR_BUFFERS = Region::BufferViews::NUM_SCALAR_BUFFERS;
    static constexpr int NUM_VECTOR_BUFFERS = Region::BufferViews::NUM_VECTOR_BUFFERS;
    static constexpr int NUM_TRACER_BUFFERS = Region::BufferViews::NUM_TRACER_BUFFERS;

    ExecViewUnmanaged<Scalar [NUM_SCALAR_BUFFERS][NP][NP][NUM_LEV]>          scalars;
    ExecViewUnmanaged<Scalar [NUM_VECTOR_BUFFERS][2][NP][NP][NUM_LEV]>       vectors;
    ExecViewUnmanaged<Scalar [NUM_TRACER_BUFFERS][QSIZE_D][NP][NP][NUM_LEV]> tracers;
  } buffers;

  int ie, ilev;
}; // KernelVariables

} // Homme

#endif // KERNEL_VARIABLES_HPP
