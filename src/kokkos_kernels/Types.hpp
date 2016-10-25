
#include <Kokkos_Core.hpp>

#include <kinds.hpp>

namespace Homme {

  //using TeamMember  = Kokkos::TeamPolicy< Kokkos::IndexType<int> >::member_type;
  //using SharedSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;

using Dinv   = Kokkos::View<real*****,  Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;
  using P      = Kokkos::View<real*****,  Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;
using PS     = Kokkos::View<real***,    Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;
using Deriv  = Kokkos::View<real***,    Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;
using GradPS = Kokkos::View<real****,   Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;
using V      = Kokkos::View<real******, Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;
using Det    = Kokkos::View<real***,    Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;



} // namespace Homme
