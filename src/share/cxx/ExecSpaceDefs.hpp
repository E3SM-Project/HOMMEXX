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

// Selecting the parallel host execution space. This is the same as ExecSpace if
// ExecSpace!=Kokkos::Cuda. If ExecSpace==Kokkos::Cuda, we select the first
// available in 'OpenMP,Threads,Serial'
// This space is used for parallel operation outside the device, mostly during
// MPI-related routines. This is to avoid repeated copies to/from the device
// when multiple MPI calls are used in a short chunk of code. Instead, we do
// MPI pre/post process (such as packing/unpacking data) on the host, in the
// most parallel way possible.
// NOTE: this is relevant only if ExecSpace is Cuda, since otherwise there is
//       no 'real' difference between ExecSpace and HostSpace...
using HostExecSpace = std::conditional<!std::is_same<Hommexx_OpenMP,void>::value,
                                       Hommexx_OpenMP,
                                       std::conditional<!std::is_same<Hommexx_Threads,void>::value,
                                                        Hommexx_Threads, Hommexx_Serial>::type
                                      >::type;
static_assert (!std::is_same<HostExecSpace,void>::value, "Error! There is no Host execution space. This is highly odd. Most likely cause is that no Host execution space is enabled in Kokkos (not even Kokkos::Serial).\n");

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
    if (max_threads_per_team<ExecSpaceType> >= num_elems) {
      return max_threads_per_team<ExecSpaceType> / num_elems;
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

} // namespace Homme

#endif // HOMMEXX_EXEC_SPACE_DEFS_HPP
