#ifndef HOMMEXX_TYPES_HPP
#define HOMMEXX_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <Hommexx_config.h>

namespace Homme {

// Usual typedef for real scalar type
using Real   = double;
using RCPtr  = Real* const;
using CRCPtr = const Real* const;
using F90Ptr  = Real* const; // Using this in a function signature emphasizes that the ordering is Fortran
using CF90Ptr = const Real* const; // Using this in a function signature emphasizes that the ordering is Fortran

// Selecting the execution space. If no specific request, use Kokkos default exec space
#ifdef HOMMEXX_CUDA_SPACE
using ExecSpace = Kokkos::Cuda;
// CUDA Can't have less than 32 threads per warp or less than 1 warp per block
static constexpr const int Default_Threads_Per_Team = 2;
static constexpr const int Default_Vectors_Per_Thread = 16;
#elif defined(HOMMEXX_OPENMP_SPACE)
using ExecSpace = Kokkos::OpenMP;
static constexpr const int Default_Threads_Per_Team = 1;
static constexpr const int Default_Vectors_Per_Thread = 1;
#elif defined(HOMMEXX_THREADS_SPACE)
using ExecSpace = Kokkos::Threads;
static constexpr const int Default_Threads_Per_Team = 1;
static constexpr const int Default_Vectors_Per_Thread = 1;
#elif defined(HOMMEXX_SERIAL_SPACE)
using ExecSpace = Kokkos::Serial;
static constexpr const int Default_Threads_Per_Team = 1;
static constexpr const int Default_Vectors_Per_Thread = 1;
#elif defined(HOMMEXX_DEFAULT_SPACE)
using ExecSpace = Kokkos::DefaultExecutionSpace::execution_space;
static constexpr const int Default_Threads_Per_Team = 1;
static constexpr const int Default_Vectors_Per_Thread = 1;
#else
#error "No valid execution space choice"
#endif

// The memory spaces
using ExecMemSpace    = ExecSpace::memory_space;
using ScratchMemSpace = ExecSpace::scratch_memory_space;
using HostMemSpace    = Kokkos::HostSpace;

// A team member type
using TeamMember = Kokkos::TeamPolicy<ExecSpace>::member_type;

// Short name for views with layout right
template <typename DataType, typename... Types>
using ViewType = Kokkos::View<DataType,Kokkos::LayoutRight,Types...>;

// Further specializations for execution space and managed/unmanaged memory
template<typename DataType>
using ExecViewManaged = ViewType<DataType,ExecMemSpace,Kokkos::MemoryManaged>;
template<typename DataType>
using ExecViewUnmanaged = ViewType<DataType,ExecMemSpace,Kokkos::MemoryUnmanaged>;

// Further specializations for host space.
template<typename DataType>
using HostViewManaged = ViewType<DataType,HostMemSpace,Kokkos::MemoryManaged>;
template<typename DataType>
using HostViewUnmanaged = ViewType<DataType,HostMemSpace,Kokkos::MemoryUnmanaged>;

// The scratch view type: always unmanaged, and always with c pointers
template<typename DataType>
using ScratchView = ViewType<DataType,ScratchMemSpace,Kokkos::MemoryUnmanaged>;

// To view the fully expanded name of a complicated template type T,
// just try to access some non-existent field of MyDebug<T>. E.g.:
// MyDebug<T>::type i;
template<typename T>
struct MyDebug {};

} // Homme

#endif // HOMMEXX_TYPES_HPP
