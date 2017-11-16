#ifndef HOMMEXX_EXEC_SPACE_DEFS_HPP
#define HOMMEXX_EXEC_SPACE_DEFS_HPP

#include <Kokkos_Core.hpp>
#include <Hommexx_config.h>

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

using Hommexx_DefaultExecSpace = Kokkos::DefaultExecutionSpace::execution_space;

// Selecting the execution space. If no specific request, use Kokkos default
// exec space.
#if defined(HOMMEXX_CUDA_SPACE)
  using ExecSpace = Hommexx_Cuda;
#elif defined(HOMMEXX_OPENMP_SPACE)
  using ExecSpace = Hommexx_OpenMP;
#elif defined(HOMMEXX_THREADS_SPACE)
  using ExecSpace = Hommexx_Threads;
#elif defined(HOMMEXX_SERIAL_SPACE)
  using ExecSpace = Hommexx_Serial;
#elif defined(HOMMEXX_DEFAULT_SPACE)
  using ExecSpace = Hommexx_DefaultExecSpace;
#else
#error "No valid execution space choice"
#endif // HOMMEXX_EXEC_SPACE
static_assert (!std::is_same<ExecSpace,void>::value, "Error! You are trying to use an ExecutionSpace not enabled in Kokkos.\n");

template <typename ExecSpaceType>
struct DefaultThreadsDistribution {

  static constexpr int vectors_per_thread () { return vectors_per_thread_impl<ExecSpaceType>(); }

  static int threads_per_team(const int num_elems) { return threads_per_team_impl<ExecSpaceType>(num_elems); }

private:
  template <typename ArgExecSpace>
  static constexpr
  typename std::enable_if<!std::is_same<ArgExecSpace,Hommexx_Cuda>::value,int>::type
  vectors_per_thread_impl() { return 1; }

  template <typename ArgExecSpace>
  static constexpr
  typename std::enable_if<std::is_same<ArgExecSpace,Hommexx_Cuda>::value,int>::type
  vectors_per_thread_impl() { return 16 /*8*/; }

  template <typename ArgExecSpace>
  static
  typename std::enable_if<!std::is_same<ArgExecSpace,Hommexx_Cuda>::value,int>::type
  threads_per_team_impl(const int num_elems) {
#ifdef KOKKOS_COLUMN_THREAD_ONLY
    (void) num_elems;
    return ExecSpace::thread_pool_size();
#else
#ifdef KOKKOS_PARALLELIZE_ON_ELEMENTS
    if (max_threads_per_team<ExecSpaceType>() >= num_elems) {
      return max_threads_per_team<ExecSpaceType>() / num_elems;
    } else {
      return 1;
    }
#else
    return 1;
#endif // KOKKOS_PARALLELIZE_ON_ELEMENTS
#endif // KOKKOS_COLUMN_THREAD_ONLY
  }

  template <typename ArgExecSpace>
  static
  typename std::enable_if<std::is_same<ArgExecSpace,Hommexx_Cuda>::value,int>::type
  threads_per_team_impl(const int /*num_elems*/) {
    return max_threads_per_team<ExecSpaceType>();
  }

  template <typename ArgExecSpace>
  static
  typename std::enable_if<std::is_same<ArgExecSpace,Hommexx_Cuda>::value,int>::type
  max_threads_per_team () { return 16; }

  template <typename ArgExecSpace>
  static
  typename std::enable_if<!std::is_same<ArgExecSpace,Hommexx_Cuda>::value,int>::type
  max_threads_per_team () { return ArgExecSpace::thread_pool_size(); }
};

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

} // namespace Homme

#endif // HOMMEXX_EXEC_SPACE_DEFS_HPP
