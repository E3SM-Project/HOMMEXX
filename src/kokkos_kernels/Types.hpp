#include <Kokkos_Core.hpp>
#include <dimensions.hpp>
#include <kinds.hpp>

namespace Homme {

// Typedef the memory space options
using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
using HostSpace    = Kokkos::HostSpace;
using ExecSpace    = Kokkos::DefaultExecutionSpace::memory_space;

// Typedef the memory management options
using MemoryManaged   = Kokkos::MemoryManaged;
using MemoryUnmanaged = Kokkos::MemoryUnmanaged;

using Team_State = Kokkos::TeamPolicy<>::member_type;

// HommeView is a templated alias to Kokkos::View with LayoutLeft as Default
template<typename DataType, typename MemorySpace, typename MemoryManagement, typename LayoutType = Kokkos::LayoutLeft>
using HommeView = Kokkos::View<DataType, LayoutType, MemorySpace, MemoryManagement>;

// Aliases for 1D-6D Homme views, with LayoutLeft
template<typename MemorySpace, typename MemoryManagement>
using HommeView1D =  HommeView<real *,      MemorySpace, MemoryManagement>;

template<typename MemorySpace, typename MemoryManagement>
using HommeView2D =  HommeView<real **,     MemorySpace, MemoryManagement>;

template<typename MemorySpace, typename MemoryManagement>
using HommeView3D =  HommeView<real ***,    MemorySpace, MemoryManagement>;

template<typename MemorySpace, typename MemoryManagement>
using HommeView4D =  HommeView<real ****,   MemorySpace, MemoryManagement>;

template<typename MemorySpace, typename MemoryManagement>
using HommeView5D =  HommeView<real *****,  MemorySpace, MemoryManagement>;

template<typename MemorySpace, typename MemoryManagement>
using HommeView6D =  HommeView<real ******, MemorySpace, MemoryManagement>;

// Aliases for 1D-6D Homme views with host space memory, with LayoutLeft
template<typename MemoryManagement>
using HommeHostView1D =  HommeView1D<HostSpace, MemoryManagement>;

template<typename MemoryManagement>
using HommeHostView2D =  HommeView2D<HostSpace, MemoryManagement>;

template<typename MemoryManagement>
using HommeHostView3D =  HommeView3D<HostSpace, MemoryManagement>;

template<typename MemoryManagement>
using HommeHostView4D =  HommeView4D<HostSpace, MemoryManagement>;

template<typename MemoryManagement>
using HommeHostView5D =  HommeView5D<HostSpace, MemoryManagement>;

template<typename MemoryManagement>
using HommeHostView6D =  HommeView6D<HostSpace, MemoryManagement>;

// Aliases for 1D-6D Homme views on execuction space memory,
// with LayoutLeft and memory always managed
// Note: if for some reason you need memory unmanaged, you can
//       still use HommeViewXD<ExecSpace, MemoryUnmanaged>
using HommeExecView1D =  HommeView1D<ExecSpace, MemoryManaged>;
using HommeExecView2D =  HommeView2D<ExecSpace, MemoryManaged>;
using HommeExecView3D =  HommeView3D<ExecSpace, MemoryManaged>;
using HommeExecView4D =  HommeView4D<ExecSpace, MemoryManaged>;
using HommeExecView5D =  HommeView5D<ExecSpace, MemoryManaged>;
using HommeExecView6D =  HommeView6D<ExecSpace, MemoryManaged>;

// Aliases for 1D-6D Homme views on scratch memory,
// with LayoutLeft and memory always unmanaged
// Note: if for some reason you need memory managed, you can
//       still use HommeViewXD<ScratchSpace, MemoryManaged>
using HommeScratchView1D =  HommeView1D<ScratchSpace, MemoryUnmanaged>;
using HommeScratchView2D =  HommeView2D<ScratchSpace, MemoryUnmanaged>;
using HommeScratchView3D =  HommeView3D<ScratchSpace, MemoryUnmanaged>;
using HommeScratchView4D =  HommeView4D<ScratchSpace, MemoryUnmanaged>;
using HommeScratchView5D =  HommeView5D<ScratchSpace, MemoryUnmanaged>;
using HommeScratchView6D =  HommeView6D<ScratchSpace, MemoryUnmanaged>;

}  // namespace Homme
