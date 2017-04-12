#ifndef HOMMEXX_TYPES_HPP
#define HOMMEXX_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <Hommexx_config.h>

namespace Homme {

// Usual typedef for real scalar type
using Real   = double;
using RCPtr  = Real* const;
using CRCPtr = const Real* const;

// Selecting the execution space. If no specific request, use Kokkos default exec space
#ifdef HOMMEXX_CUDA_SPACE
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
#endif

// The memory spaces
using ExecMemSpace    = ExecSpace::memory_space;
using ScratchMemSpace = ExecSpace::scratch_memory_space;
using HostMemSpace    = Kokkos::HostSpace;

// Short name for views with layout right
template <typename DataType, typename... Types>
using ViewType = Kokkos::View<DataType,Kokkos::LayoutRight,Types...>;

// Further specializations for execution space and managed/unmanaged memory
template<typename DataType>
using ExecViewManaged = ViewType<DataType,ExecMemSpace,Kokkos::MemoryManaged>;
template<typename DataType>
using ExecViewUnmanaged = ViewType<DataType,ExecMemSpace,Kokkos::MemoryUnmanaged>;
template<typename DataType>
using ExecViewF90 = Kokkos::View<DataType,Kokkos::LayoutLeft,ExecMemSpace,Kokkos::MemoryManaged>;

// Further specializations for host space.
// Note: views from f90 pointers are LayoutLeft and always unmanaged
template<typename DataType>
using HostViewManaged = ViewType<DataType,HostMemSpace,Kokkos::MemoryManaged>;
template<typename DataType>
using HostViewUnmanaged = ViewType<DataType,HostMemSpace,Kokkos::MemoryUnmanaged>;
template<typename DataType>
using HostViewF90 = Kokkos::View<DataType,Kokkos::LayoutLeft,HostMemSpace,Kokkos::MemoryUnmanaged>;

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
