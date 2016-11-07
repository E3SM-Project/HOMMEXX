
#include <Kokkos_Core.hpp>

#include <dimensions.hpp>
#include <kinds.hpp>

namespace Homme {

using Layout = Kokkos::LayoutLeft;

template <typename T>
using HommeView =
    Kokkos::View<T, Layout, Kokkos::MemoryUnmanaged>;

using Alpha = HommeView<real *>;
using D = HommeView<real *****>;
using P = HommeView<real *****>;
using PS = HommeView<real ***>;
using V = HommeView<real ******>;
using SphereMP = HommeView<real ***>;
using FCor = HommeView<real ***>;
using PTens = HommeView<real ****>;
using VTens = HommeView<real *****>;
using MetDet = HommeView<real ***>;

enum Spherical_Polar_e { Radius, Lat, Lon };

struct Spherical_Polar {
  real radius;
  real lon;
  real lat;
};

using SphereP = HommeView<Kokkos::Array<real, 3> >;

template <typename T>
using HommeLocal = Kokkos::View<T, Layout>;

// Scalar fields are scalar values over the np x np
// quadrature points
using ScalarField = HommeLocal<real **>;

// Vector fields are vector values over the np x np
// quadrature points
using VectorField = HommeLocal<real ***>;

struct derivative {
  real Dvv[np][np];
  real Dvv_diag[np][np];
  real Dvv_twt[np][np];
  real Mvv_twt[np][np];
  real Mfvm[np][nc + 1];
  real Cfvm[np][nc];
  real legdg[np][np];
};

}  // namespace Homme
