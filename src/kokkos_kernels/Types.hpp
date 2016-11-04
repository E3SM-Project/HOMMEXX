
#include <Kokkos_Core.hpp>

#include <kinds.hpp>

namespace Homme {

template <typename T>
using HommeView = Kokkos::View<T, Kokkos::LayoutLeft,
                               Kokkos::MemoryUnmanaged>;

using Alpha = HommeView<real *>;
using D = HommeView<real *****>;
using P = HommeView<real *****>;
using PS = HommeView<real ***>;
using V = HommeView<real ******>;
using SphereMP = HommeView<real ***>;
using FCor = HommeView<real ***>;
using PTens = HommeView<real ****>;
using VTens = HommeView<real *****>;

enum Spherical_Polar_e { Radius, Lat, Lon };

struct Spherical_Polar {
  real radius;
  real lon;
  real lat;
};

using SphereP = HommeView<Kokkos::Array<real, 3> >;

using ULatLong = Kokkos::View<real ***, Kokkos::LayoutLeft>;
using Energy = Kokkos::View<real **, Kokkos::LayoutLeft>;
using PV = Kokkos::View<real **, Kokkos::LayoutLeft>;

struct derivative_t {
  real Dvv[np][np];
  real Dvv_diag[np][np]
  real Dvv_twt[np][np]
  real Mvv_twt[np][np]
  real Mfvm[np][nc+1)
  real Cfvm[np][nc]
  real legdg[np][np]
};

}  // namespace Homme
