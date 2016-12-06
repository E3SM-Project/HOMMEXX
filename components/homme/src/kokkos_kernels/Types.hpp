
#include <Kokkos_Core.hpp>

#include <dimensions.hpp>
#include <kinds.hpp>

namespace Homme {

using Layout = Kokkos::LayoutLeft;

template <typename T>
using Homme_View = Kokkos::View<T, Layout>;

template <typename T>
using Homme_View_Host =
    Kokkos::View<T, Layout, Kokkos::HostSpace>;

using Alpha = Homme_View<real *>;
using Alpha_Host = Homme_View_Host<real *>;

using D = Homme_View<real *****>;
using D_Host = Homme_View_Host<real *****>;

using Dvv = Homme_View<real **>;
using Dvv_Host = Homme_View_Host<real **>;

using P = Homme_View<real *****>;
using P_Host = Homme_View_Host<real *****>;

using PS = Homme_View<real ***>;
using PS_Host = Homme_View_Host<real ***>;

using V = Homme_View<real ******>;
using V_Host = Homme_View_Host<real ******>;

using Sphere_MP = Homme_View<real ***>;
using Sphere_MP_Host = Homme_View_Host<real ***>;
using FCor = Homme_View<real ***>;
using FCor_Host = Homme_View_Host<real ***>;

using PTens = Homme_View<real ****>;
using PTens_Host = Homme_View_Host<real ****>;

using VTens = Homme_View<real *****>;
using VTens_Host = Homme_View_Host<real *****>;

using MetDet = Homme_View<real ***>;
using MetDet_Host = Homme_View_Host<real ***>;

enum Spherical_Polar_e { Radius, Lat, Lon };

struct Spherical_Polar {
  real radius;
  real lon;
  real lat;
};

using SphereP = Homme_View<Kokkos::Array<real, 3> >;

template <typename T>
using Homme_Local = Kokkos::View<T, Layout>;

// Scalar fields are scalar values over the np x np
// quadrature points
using Scalar_Field = Homme_Local<real **>;

// Vector fields are vector values over the np x np
// quadrature points
using Vector_Field = Homme_Local<real ***>;

}  // namespace Homme
