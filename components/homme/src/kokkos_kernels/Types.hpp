
#include <Kokkos_Core.hpp>

#include <kinds.hpp>

namespace Homme {

template<typename DataType, typename MemoryManagement>
using HommeView = Kokkos::View<DataType,Kokkos::LayoutLeft,MemoryManagement>;

template<typename MemoryManagement>
using HommeView1D =  HommeView<real *, MemoryManagement>;

template<typename MemoryManagement>
using HommeView2D =  HommeView<real **, MemoryManagement>;

template<typename MemoryManagement>
using HommeView3D =  HommeView<real ***, MemoryManagement>;

template<typename MemoryManagement>
using HommeView4D =  HommeView<real ****, MemoryManagement>;

template<typename MemoryManagement>
using HommeView5D =  HommeView<real *****, MemoryManagement>;

template<typename MemoryManagement>
using HommeView6D =  HommeView<real ******, MemoryManagement>;

typedef Kokkos::MemoryManaged     KMM;
typedef Kokkos::MemoryUnmanaged   KMU;

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
