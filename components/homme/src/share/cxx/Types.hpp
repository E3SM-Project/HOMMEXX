#ifndef HOMMEXX_TYPES_HPP
#define HOMMEXX_TYPES_HPP

#include <Hommexx_config.h>
#include <Kokkos_Core.hpp>

#include "Dimensions.hpp"

#include <vector/KokkosKernels_Vector.hpp>

#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif

#define __MACRO_STRING(MacroVal) #MacroVal
#define MACRO_STRING(MacroVal) __MACRO_STRING(MacroVal)

namespace Homme {

// Usual typedef for real scalar type
using Real = double;
using RCPtr = Real *const;
using CRCPtr = const Real *const;
using F90Ptr = Real *const; // Using this in a function signature emphasizes
                            // that the ordering is Fortran
using CF90Ptr = const Real *const; // Using this in a function signature
                                   // emphasizes that the ordering is Fortran

// Selecting the execution space. If no specific request, use Kokkos default
// exec space
#ifdef HOMMEXX_CUDA_SPACE
using ExecSpace = Kokkos::Cuda;

template <> struct DefaultThreadsDistribution<Kokkos::Cuda> {
  static void init() {}

  static constexpr int vectors_per_thread() { return 16; }

  static int threads_per_team(const int /*num_elems*/) {
    return Max_Threads_Per_Team;
  }

private:
  static constexpr int Max_Threads_Per_Team = 8;
};

#else

template <typename ExecSpace> struct DefaultThreadsDistribution {
  static void init() { Max_Threads_Per_Team = ExecSpace::thread_pool_size(); }

  static constexpr int vectors_per_thread() { return 1; }

  static int threads_per_team(const int num_elems) {
#ifdef KOKKOS_PARALLELIZE_ON_ELEMENTS
    if (Max_Threads_Per_Team >= num_elems)
      return Max_Threads_Per_Team / num_elems;
    else
      return 1;
#else
    return 1;
#endif
  }

private:
  static int Max_Threads_Per_Team;
};

#if defined(HOMMEXX_OPENMP_SPACE)
using ExecSpace = Kokkos::OpenMP;
#elif defined(HOMMEXX_THREADS_SPACE)
using ExecSpace = Kokkos::Threads;
#elif defined(HOMMEXX_SERIAL_SPACE)
using ExecSpace = Kokkos::Serial;
#elif defined(HOMMEXX_DEFAULT_SPACE)
using ExecSpace = Kokkos::DefaultExecutionSpace::execution_space;
#else
#error "No valid execution space choice"
#endif // HOMMEXX_EXEC_SPACE

#endif // HOMMEXX_SPACE

#if (AVX_VERSION==4 || AVX_VERSION==8)
using VectorTagType = KokkosKernels::Batched::Experimental::AVX<Real, ExecSpace>;
#else
using VectorTagType =
    KokkosKernels::Batched::Experimental::SIMD<Real, ExecSpace>;
#endif // AVX_VERSION

using VectorType =
    KokkosKernels::Batched::Experimental::VectorTag<VectorTagType, VECTOR_SIZE>;

using Scalar = KokkosKernels::Batched::Experimental::Vector<VectorType>;

template <typename ExecSpace>
int DefaultThreadsDistribution<ExecSpace>::Max_Threads_Per_Team;

// The memory spaces
using ExecMemSpace = ExecSpace::memory_space;
using ScratchMemSpace = ExecSpace::scratch_memory_space;
using HostMemSpace = Kokkos::HostSpace;

// A team member type
using TeamMember = Kokkos::TeamPolicy<ExecSpace>::member_type;

// Native language layouts
using FortranLayout = Kokkos::LayoutLeft;
using CXXLayout = Kokkos::LayoutRight;

// Short name for views
template <typename DataType, typename... Types>
using ViewType = Kokkos::View<DataType, Types...>;

// Further specializations for execution space and managed/unmanaged memory
template <typename DataType>
using ExecViewManaged = ViewType<DataType, ExecMemSpace, Kokkos::MemoryManaged>;
template <typename DataType>
using ExecViewUnmanaged =
    ViewType<DataType, ExecMemSpace, Kokkos::MemoryUnmanaged>;

// Further specializations for host space.
template <typename DataType>
using HostViewManaged = ViewType<DataType, HostMemSpace, Kokkos::MemoryManaged>;
template <typename DataType>
using HostViewUnmanaged =
    ViewType<DataType, HostMemSpace, Kokkos::MemoryUnmanaged>;

template <typename DataType>
using FortranViewManaged =
    ViewType<DataType, HostMemSpace, Kokkos::MemoryManaged, FortranLayout>;
template <typename DataType>
using FortranViewUnmanaged =
    ViewType<DataType, HostMemSpace, Kokkos::MemoryUnmanaged, FortranLayout>;

// The scratch view type: always unmanaged, and always with c pointers
template <typename DataType>
using ScratchView =
    ViewType<DataType, ScratchMemSpace, Kokkos::MemoryUnmanaged>;

// To view the fully expanded name of a complicated template type T,
// just try to access some non-existent field of MyDebug<T>. E.g.:
// MyDebug<T>::type i;
template <typename T> struct MyDebug {};

} // Homme

#endif // HOMMEXX_TYPES_HPP
