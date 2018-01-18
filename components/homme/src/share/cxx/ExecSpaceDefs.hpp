#ifndef HOMMEXX_EXEC_SPACE_DEFS_HPP
#define HOMMEXX_EXEC_SPACE_DEFS_HPP

#include <cassert>

#include <Kokkos_Core.hpp>

#include "Hommexx_config.h"

namespace Homme
{

// Some in-house names for Kokkos exec spaces, which are
// always defined, possibly as alias of void
#ifdef KOKKOS_HAVE_CUDA
using Hommexx_Cuda = Kokkos::Cuda;
#else
using Hommexx_Cuda = void;
#endif

#ifdef KOKKOS_HAVE_OPENMP
using Hommexx_OpenMP = Kokkos::OpenMP;
#else
using Hommexx_OpenMP = void;
#endif

#ifdef KOKKOS_HAVE_PTHREADS
using Hommexx_Threads = Kokkos::Threads;
#else
using Hommexx_Threads = void;
#endif

#ifdef KOKKOS_HAVE_SERIAL
using Hommexx_Serial = Kokkos::Serial;
#else
using Hommexx_Serial = void;
#endif

// Selecting the execution space. If no specific request, use Kokkos default
// exec space
#if defined(HOMMEXX_CUDA_SPACE)
using ExecSpace = Hommexx_Cuda;
#elif defined(HOMMEXX_OPENMP_SPACE)
using ExecSpace = Hommexx_OpenMP;
#elif defined(HOMMEXX_THREADS_SPACE)
using ExecSpace = Hommexx_Threads;
#elif defined(HOMMEXX_SERIAL_SPACE)
using ExecSpace = Hommexx_Serial;
#elif defined(HOMMEXX_DEFAULT_SPACE)
using ExecSpace = Kokkos::DefaultExecutionSpace::execution_space;
#else
#error "No valid execution space choice"
#endif // HOMMEXX_EXEC_SPACE

static_assert (!std::is_same<ExecSpace,void>::value,
               "Error! You are trying to use an ExecutionSpace not enabled in Kokkos.\n");

// Call this instead of Kokkos::initialize.
void initialize_kokkos();

// What follows provides utilities to parameterize the parallel machine (CPU/KNL
// cores within a rank, GPU attached to a rank) optimally. The parameterization
// is a nontrivial function of available resources, number of parallel
// iterations to be performed, and kernel-specific preferences regarding team
// and vector dimensions of parallelization.
//   So far, we are able to hide the details inside a call like this:
//     Homme::get_default_team_policy<ExecSpace>(data.num_elems * data.qsize);
// thus, all that follows except the function get_default_team_policy may not
// need to be used except in the implementation of get_default_team_policy.
//   If that remains true forever, we can move all of this code to
// ExecSpaceDefs.cpp.

// Preferences to guide distribution of physical threads among team and vector
// dimensions. Default values are sensible.
struct ThreadPreferences {
  // Max number of threads a kernel can use. Default: NP*NP.
  int max_threads_usable;
  // Max number of vectors a kernel can use. Default: NUM_PHYSICAL_LEV.
  int max_vectors_usable;
  // Prefer threads to vectors? Default: true.
  bool prefer_threads;

  ThreadPreferences();
};

namespace Parallel {
// Previous (inclusive) power of 2. E.g., prevpow2(4) -> 4, prevpow2(5) -> 4.
unsigned short prevpow2(unsigned short n);

// Determine (#threads, #vectors) as a function of a pool of threads provided to
// the process and the number of parallel iterations to perform.
std::pair<int, int>
team_num_threads_vectors_from_pool(
  const int pool_size, const int num_parallel_iterations,
  const ThreadPreferences tp = ThreadPreferences());

// Determine (#threads, #vectors) as a function of a pool of warps provided to
// the process, the number of threads per warp, the maximum number of warps a
// team can use, and the number of parallel iterations to perform.
std::pair<int, int>
team_num_threads_vectors_for_gpu(
  const int num_warps_total, const int num_threads_per_warp,
  const int min_num_warps, const int max_num_warps,
  const int num_parallel_iterations,
  const ThreadPreferences tp = ThreadPreferences());
} // namespace Parallel

// Device-dependent distribution of physical threads over teams and vectors. The
// general case is for a machine with a pool of threads, like KNL and CPU.
template <typename ExecSpaceType>
struct DefaultThreadsDistribution {
  static std::pair<int, int>
  team_num_threads_vectors(const int num_parallel_iterations,
                           const ThreadPreferences tp = ThreadPreferences()) {
    return Parallel::team_num_threads_vectors_from_pool(
      ExecSpaceType::thread_pool_size(), num_parallel_iterations, tp);
  }
};

// Specialization for a GPU, where threads can't be viewed as existing simply in
// a pool.
template <>
struct DefaultThreadsDistribution<Hommexx_Cuda> {
  static std::pair<int, int>
  team_num_threads_vectors(const int num_parallel_iterations,
                           const ThreadPreferences tp = ThreadPreferences());
};

// Return a TeamPolicy using defaults that, so far, have been good for all use
// cases. Use of this function means you don't have to use
// DefaultThreadsDistribution.
template <typename ExecSpace, typename Tag=void>
Kokkos::TeamPolicy<ExecSpace, Tag>
get_default_team_policy(const int num_parallel_iterations) {
  const auto threads_vectors =
    DefaultThreadsDistribution<ExecSpace>::team_num_threads_vectors(
      num_parallel_iterations);
  auto policy = Kokkos::TeamPolicy<ExecSpace, Tag>(num_parallel_iterations,
                                                   threads_vectors.first,
                                                   threads_vectors.second);
  policy.set_chunk_size(1);
  return policy;
}

// A templated typedef for MD range policy (used in RK stages)
template<typename ExecutionSpace, int Rank>
using MDRangePolicy = Kokkos::Experimental::MDRangePolicy
                          < ExecutionSpace,
                            Kokkos::Experimental::Rank
                              < Rank,
                                Kokkos::Experimental::Iterate::Right,
                                Kokkos::Experimental::Iterate::Right
                              >,
                            Kokkos::IndexType<int>
                          >;

template <typename ExeSpace>
struct OnGpu { enum : bool { value = false }; };

template <>
struct OnGpu<Hommexx_Cuda> { enum : bool { value = true }; };

template <typename ExeSpace>
struct Memory {
  enum : bool { on_gpu = OnGpu<ExeSpace>::value };

  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION static
  Scalar* get_shmem (const typename Kokkos::TeamPolicy<ExeSpace>::member_type&,
                     const size_t sz = 0) {
    return nullptr;
  }

  template <typename Scalar, int N>
  class AutoArray {
    Scalar data_[N];
  public:
    KOKKOS_INLINE_FUNCTION AutoArray (Scalar*) {}
    KOKKOS_INLINE_FUNCTION Scalar& operator[] (const int& i) {
      assert(i >= 0);
      assert(i < N);
      return data_[i];
    }
    KOKKOS_INLINE_FUNCTION Scalar* data () { return data_; }
  };
};

template <>
struct Memory<Hommexx_Cuda> {
  enum : bool { on_gpu = OnGpu<Hommexx_Cuda>::value };

  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION static
  Scalar* get_shmem (const Kokkos::TeamPolicy<Hommexx_Cuda>::member_type& team,
                     const size_t n = 0) {
    return static_cast<Scalar*>(team.team_shmem().get_shmem(n*sizeof(Scalar)));
  }

  template <typename Scalar, int N>
  class AutoArray {
    Scalar* data_;
  public:
    KOKKOS_INLINE_FUNCTION AutoArray (Scalar* data) : data_(data) {}
    KOKKOS_INLINE_FUNCTION Scalar& operator[] (const int& i) {
      assert(i >= 0);
      assert(i < N);
      return data_[i];
    }
    KOKKOS_INLINE_FUNCTION Scalar* data () { return data_; }
  };
};

} // namespace Homme

#endif // HOMMEXX_EXEC_SPACE_DEFS_HPP
