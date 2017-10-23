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

template <typename ExecSpace> struct DefaultThreadsDistribution {
  static constexpr int vectors_per_thread() { return 1; }

#ifdef KOKKOS_COLUMN_THREAD_ONLY
  static int threads_per_team(const int num_elems) {
    return ExecSpace::thread_pool_size();
  }
#else
#ifdef KOKKOS_PARALLELIZE_ON_ELEMENTS
  static int threads_per_team(const int num_elems) {
    int Max_Threads_Per_Team = ExecSpace::thread_pool_size();
    if (Max_Threads_Per_Team >= num_elems) {
      return Max_Threads_Per_Team / num_elems;
    } else {
      return 1;
    }
  }
#else
  static int threads_per_team(const int num_elems) { return 1; }
#endif // KOKKOS_PARALLELIZE_ON_ELEMENTS
#endif // KOKKOS_COLUMN_THREAD_ONLY
};

#ifdef KOKKOS_HAVE_CUDA
template <> struct DefaultThreadsDistribution<Kokkos::Cuda> {
  static constexpr int vectors_per_thread() { return 8; }

  static int threads_per_team(const int /*num_elems*/) {
    return Max_Threads_Per_Team;
  }

private:
  static constexpr int Max_Threads_Per_Team = 16;
};
#endif // KOKKOS_HAVE_CUDA

// Selecting the execution space. If no specific request, use Kokkos default
// exec space
#if defined(HOMMEXX_CUDA_SPACE)
using ExecSpace = Kokkos::Cuda;
#elif defined(HOMMEXX_OPENMP_SPACE)
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

#if (AVX_VERSION > 0)
using VectorTagType =
    KokkosKernels::Batched::Experimental::AVX<Real, ExecSpace>;
#else
using VectorTagType =
    KokkosKernels::Batched::Experimental::SIMD<Real, ExecSpace>;
#endif // AVX_VERSION

using VectorType =
    KokkosKernels::Batched::Experimental::VectorTag<VectorTagType, VECTOR_SIZE>;

using Scalar = KokkosKernels::Batched::Experimental::Vector<VectorType>;

static_assert(sizeof(Scalar) > 0, "Vector type has 0 size");
static_assert(sizeof(Scalar) == sizeof(Real[VECTOR_SIZE]), "Vector type is not correctly defined");

using MemoryManaged   = Kokkos::MemoryTraits<Kokkos::Restrict>;
using MemoryUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>;

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
template <typename DataType, typename MemorySpace, typename MemoryManagement>
using ViewType = Kokkos::View<DataType, Kokkos::LayoutRight, MemorySpace, MemoryManagement>;

// Managed/Unmanaged view
template <typename DataType, typename MemorySpace>
using ViewManaged = ViewType<DataType, MemorySpace, MemoryManaged>;
template <typename DataType, typename MemorySpace>
using ViewUnmanaged = ViewType<DataType, MemorySpace, MemoryUnmanaged>;

// Host/Device views
template <typename DataType, typename MemoryManagement>
using HostView = ViewType<DataType, HostMemSpace, MemoryManagement>;
template <typename DataType, typename MemoryManagement>
using ExecView = ViewType<DataType, ExecMemSpace, MemoryManagement>;

// Further specializations for execution space and managed/unmanaged memory
template <typename DataType>
using ExecViewManaged = ExecView<DataType, MemoryManaged>;
template <typename DataType>
using ExecViewUnmanaged = ExecView<DataType, MemoryUnmanaged>;

// Further specializations for host space.
template <typename DataType>
using HostViewManaged = HostView<DataType, MemoryManaged>;
template <typename DataType>
using HostViewUnmanaged = HostView<DataType, MemoryUnmanaged>;

// The scratch view type: always unmanaged, and always with c pointers
template <typename DataType>
using ScratchView = ViewType<DataType, ScratchMemSpace, MemoryUnmanaged>;

// To view the fully expanded name of a complicated template type T,
// just try to access some non-existent field of MyDebug<T>. E.g.:
// MyDebug<T>::type i;
template <typename T> struct MyDebug {};

} // Homme

#endif // HOMMEXX_TYPES_HPP
