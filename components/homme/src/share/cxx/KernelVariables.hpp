#ifndef KERNEL_VARIABLES_HPP
#define KERNEL_VARIABLES_HPP

#include "Types.hpp"

namespace Homme {

struct KernelVariables {
  KOKKOS_INLINE_FUNCTION
  KernelVariables(const TeamMember &team_in)
      : team(team_in), ie(team.league_rank()), ilev(-1), iq(-1) {
  } //, igp(-1), jgp(-1) {}

  KOKKOS_INLINE_FUNCTION
  KernelVariables(const TeamMember &team_in, const int qsize)
      : team(team_in), ie(team.league_rank() / qsize), ilev(-1), iq(team.league_rank() % qsize) {
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

  int ie, ilev, iq;
}; // KernelVariables

template <typename ExeSpace>
struct Memory {
  enum : bool { on_gpu = false };

  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION static
  Scalar* get_shmem (const KernelVariables& kv, const size_t sz = 0) {
    return nullptr;
  }

  template <typename Scalar, int N>
  class AutoArray {
    Scalar data_[N];
  public:
    KOKKOS_INLINE_FUNCTION AutoArray (Scalar*) {}
    KOKKOS_INLINE_FUNCTION Scalar& operator[] (const int& i) { return data_[i]; }
  };
};

template <>
struct Memory<Hommexx_Cuda> {
  enum : bool { on_gpu = true };

  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION static
  Scalar* get_shmem (const KernelVariables& kv, const size_t n = 0) {
    return static_cast<Scalar*>(kv.team.team_shmem().get_shmem(n*sizeof(Scalar)));
  }  

  template <typename Scalar, int N>
  class AutoArray {
    Scalar* data_;
  public:
    KOKKOS_INLINE_FUNCTION AutoArray (Scalar* data) : data_(data) {}
    KOKKOS_INLINE_FUNCTION Scalar& operator[] (const int& i) { return data_[i]; }
  };
};

} // Homme

#endif // KERNEL_VARIABLES_HPP
