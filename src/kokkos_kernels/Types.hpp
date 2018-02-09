#include <Kokkos_Core.hpp>
#include <Hommexx_config.h>
#include <dimensions.hpp>
#include <kinds.hpp>

#ifndef _SHALLOW_WATER_TYPES_HPP_
#define _SHALLOW_WATER_TYPES_HPP_

namespace Homme {

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
#warning "No valid execution space choice"
#endif

// Given the execution space selected above, get the memory spaces
using ExecMemSpace    = ExecSpace::memory_space;
using ScratchMemSpace = ExecSpace::scratch_memory_space;
using HostMemSpace    = Kokkos::HostSpace;

// Typedef the memory management options
using MemoryManaged   = Kokkos::MemoryManaged;
using MemoryUnmanaged = Kokkos::MemoryUnmanaged;

using Team_State = Kokkos::TeamPolicy<>::member_type;

// HommeView is a templated alias to Kokkos::View with
// LayoutLeft as Default
template <typename DataType, typename MemorySpace,
          typename MemoryManagement,
          typename LayoutType = Kokkos::LayoutLeft>
using HommeView =
    Kokkos::View<DataType, LayoutType, MemorySpace,
                 MemoryManagement>;

// Aliases for 1D-6D Homme views, with LayoutLeft
template <typename MemorySpace, typename MemoryManagement>
using HommeView1D =
    HommeView<real *, MemorySpace, MemoryManagement>;

template <typename MemorySpace, typename MemoryManagement>
using HommeView2D =
    HommeView<real **, MemorySpace, MemoryManagement>;

template <typename MemorySpace, typename MemoryManagement>
using HommeView3D =
    HommeView<real ***, MemorySpace, MemoryManagement>;

template <typename MemorySpace, typename MemoryManagement>
using HommeView4D =
    HommeView<real ****, MemorySpace, MemoryManagement>;

template <typename MemorySpace, typename MemoryManagement>
using HommeView5D =
    HommeView<real *****, MemorySpace, MemoryManagement>;

template <typename MemorySpace, typename MemoryManagement>
using HommeView6D =
    HommeView<real ******, MemorySpace, MemoryManagement>;

// Aliases for 1D-6D Homme views with host space memory,
// with LayoutLeft
template <typename MemoryManagement>
using HommeHostView1D =
    HommeView1D<HostMemSpace, MemoryManagement>;

template <typename MemoryManagement>
using HommeHostView2D =
    HommeView2D<HostMemSpace, MemoryManagement>;

template <typename MemoryManagement>
using HommeHostView3D =
    HommeView3D<HostMemSpace, MemoryManagement>;

template <typename MemoryManagement>
using HommeHostView4D =
    HommeView4D<HostMemSpace, MemoryManagement>;

template <typename MemoryManagement>
using HommeHostView5D =
    HommeView5D<HostMemSpace, MemoryManagement>;

template <typename MemoryManagement>
using HommeHostView6D =
    HommeView6D<HostMemSpace, MemoryManagement>;

// Aliases for 1D-6D Homme views on execuction space memory,
// with LayoutLeft and memory always managed
// Note: if for some reason you need memory unmanaged, you
// can
//       still use HommeViewXD<ExecSpace, MemoryUnmanaged>
using HommeExecView1D =
    HommeView1D<ExecSpace, MemoryManaged>;
using HommeExecView2D =
    HommeView2D<ExecSpace, MemoryManaged>;
using HommeExecView3D =
    HommeView3D<ExecSpace, MemoryManaged>;
using HommeExecView4D =
    HommeView4D<ExecSpace, MemoryManaged>;
using HommeExecView5D =
    HommeView5D<ExecSpace, MemoryManaged>;
using HommeExecView6D =
    HommeView6D<ExecSpace, MemoryManaged>;

// Aliases for 1D-6D Homme views on scratch memory,
// with LayoutLeft and memory always unmanaged
// Note: if for some reason you need memory managed, you can
//       still use HommeViewXD<ScratchMemSpace, MemoryManaged>
using HommeScratchView1D =
    HommeView1D<ScratchMemSpace, MemoryUnmanaged>;
using HommeScratchView2D =
    HommeView2D<ScratchMemSpace, MemoryUnmanaged>;
using HommeScratchView3D =
    HommeView3D<ScratchMemSpace, MemoryUnmanaged>;
using HommeScratchView4D =
    HommeView4D<ScratchMemSpace, MemoryUnmanaged>;
using HommeScratchView5D =
    HommeView5D<ScratchMemSpace, MemoryUnmanaged>;
using HommeScratchView6D =
    HommeView6D<ScratchMemSpace, MemoryUnmanaged>;

}  // namespace Homme

#endif //_SHALLOW_WATER_TYPES_HPP_
