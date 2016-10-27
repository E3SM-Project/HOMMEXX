
#include <Kokkos_Core.hpp>

#include <kinds.hpp>

namespace Homme {

using D = Kokkos::View<real*****,  Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;
using P = Kokkos::View<real*****,  Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;
using V = Kokkos::View<real******, Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;

} // namespace Homme
