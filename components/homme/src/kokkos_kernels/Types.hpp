
#include <Kokkos_Core.hpp>

#include <kinds.hpp>

namespace Homme {

using Alpha = Kokkos::View<real *, Kokkos::LayoutLeft,
			   Kokkos::MemoryUnmanaged>;
using D = Kokkos::View<real *****, Kokkos::LayoutLeft,
                       Kokkos::MemoryUnmanaged>;
using P = Kokkos::View<real *****, Kokkos::LayoutLeft,
                       Kokkos::MemoryUnmanaged>;
using V = Kokkos::View<real ******, Kokkos::LayoutLeft,
                       Kokkos::MemoryUnmanaged>;

using SphereMP = Kokkos::View<real ***, Kokkos::LayoutLeft,
                              Kokkos::MemoryUnmanaged>;
using PTens = Kokkos::View<real ****, Kokkos::LayoutLeft,
                           Kokkos::MemoryUnmanaged>;
using VTens = Kokkos::View<real *****, Kokkos::LayoutLeft,
                           Kokkos::MemoryUnmanaged>;

using D_noie = Kokkos::View<real ****, Kokkos::LayoutLeft,
                       Kokkos::MemoryUnmanaged>;
using P_noie = Kokkos::View<real ****, Kokkos::LayoutLeft,
                       Kokkos::MemoryUnmanaged>;
using V_noie = Kokkos::View<real *****, Kokkos::LayoutLeft,
                       Kokkos::MemoryUnmanaged>;
using SphereMP_noie = Kokkos::View<real **, Kokkos::LayoutLeft,
				   Kokkos::MemoryUnmanaged>;
using PTens_noie = Kokkos::View<real ***, Kokkos::LayoutLeft,
                           Kokkos::MemoryUnmanaged>;
using VTens_noie = Kokkos::View<real ****, Kokkos::LayoutLeft,
                           Kokkos::MemoryUnmanaged>;

}  // namespace Homme
