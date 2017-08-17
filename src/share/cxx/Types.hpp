#ifndef HOMMEXX_TYPES_HPP
#define HOMMEXX_TYPES_HPP

#include <Hommexx_config.h>
#include <Kokkos_Core.hpp>

#include "Dimensions.hpp"

#include <vector/KokkosKernels_Vector.hpp>

#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif

// This must be above the macro below that messes with underscores
//
#ifdef USE_SACADO_MP_VECTOR
#include "Sacado.hpp"
#include "Stokhos_Sacado_Kokkos_MathFunctions.hpp"

#include "Stokhos_KokkosTraits.hpp"
#include "Stokhos_StaticFixedStorage.hpp"
#include "Stokhos_ViewStorage.hpp"

#include "Sacado_MP_ExpressionTraits.hpp"
#include "Sacado_MP_VectorTraits.hpp"
#include "Sacado_MP_Vector.hpp"
//#include "Kokkos_View_MP_Vector.hpp"
#include "Kokkos_Atomic_MP_Vector.hpp"
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
  static constexpr int vectors_per_thread() { return 16; }

  static int threads_per_team(const int /*num_elems*/) {
    return Max_Threads_Per_Team;
  }

private:
  static constexpr int Max_Threads_Per_Team = 8;
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


#ifndef USE_SACADO_MP_VECTOR

// Use Kyung-Joo Vector
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

#else // Use Phipps Vector

using MPStorageType = Stokhos::StaticFixedStorage<int, Real, VECTOR_SIZE, ExecSpace>;
using Scalar =  Sacado::MP::Vector<MPStorageType>;

#endif

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

// Managed/Unmanaged view
template <typename DataType, typename MemorySpace>
using ViewManaged = ViewType<DataType, Kokkos::LayoutRight, MemorySpace, Kokkos::MemoryManaged>;
template <typename DataType, typename MemorySpace>
using ViewUnmanaged = ViewType<DataType, Kokkos::LayoutRight, MemorySpace, Kokkos::MemoryManaged>;

// Host/Device views
template <typename DataType, typename MemoryManagement>
using HostView = ViewType<DataType, Kokkos::LayoutRight, HostMemSpace, MemoryManagement>;
template <typename DataType, typename MemoryManagement>
using ExecView = ViewType<DataType, Kokkos::LayoutRight, ExecMemSpace, MemoryManagement>;

// Further specializations for execution space and managed/unmanaged memory
template <typename DataType>
using ExecViewManaged = ExecView<DataType, Kokkos::MemoryManaged>;
template <typename DataType>
using ExecViewUnmanaged = ExecView<DataType, Kokkos::MemoryUnmanaged>;

// Further specializations for host space.
template <typename DataType>
using HostViewManaged = HostView<DataType, Kokkos::MemoryManaged>;
template <typename DataType>
using HostViewUnmanaged = HostView<DataType, Kokkos::MemoryUnmanaged>;

template <typename DataType>
using FortranViewManaged = ViewType<DataType, HostMemSpace, Kokkos::MemoryManaged, FortranLayout>;
template <typename DataType>
using FortranViewUnmanaged = ViewType<DataType, HostMemSpace, Kokkos::MemoryUnmanaged, FortranLayout>;

// The scratch view type: always unmanaged, and always with c pointers
template <typename DataType>
using ScratchView = ViewType<DataType, ScratchMemSpace, Kokkos::MemoryUnmanaged>;

// To view the fully expanded name of a complicated template type T,
// just try to access some non-existent field of MyDebug<T>. E.g.:
// MyDebug<T>::type i;
template <typename T> struct MyDebug {};

} // Homme

#endif // HOMMEXX_TYPES_HPP
