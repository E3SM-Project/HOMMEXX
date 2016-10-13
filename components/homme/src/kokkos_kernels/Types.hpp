#include <Kokkos_Core.hpp>


namespace Homme {

using Scalar = double;

// TODO set at compile time
constexpr int NP = 4;
constexpr int NLEV = 100;
constexpr int TIME_LEVELS = 2;

// TODO set to correct value
constexpr double RREARTH = 1.0;

using TeamMember  = Kokkos::TeamPolicy< Kokkos::IndexType<int> >::member_type;
using SharedSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;

#if 0
// TODO use when Extents becomes available in Kokkos
using Dinv   = Kokkos::View<Scalar, Kokkos::Extents<NP, NP, 2, 2, 0>, Kokkos::LayoutLeft>;
using P      = Kokkos::View<Scalar, Kokkos::Extents<NP, NP, NLEV, TIME_LEVELS, 0>, Kokkos::LayoutLeft>;
using PS     = Kokkos::View<Scalar, Kokkos::Extents<NP, NP, 0>, Kokkos::LayoutLeft>;
using Deriv  = Kokkos::View<Scalar, Kokkos::Extents<NP, NP, 0>, Kokkos::LayoutLeft>;
using GradPS = Kokkos::View<Scalar, Kokkos::Extents<NP, NP, 2, 0>, Kokkos::LayoutLeft>;
using V      = Kokkos::View<Scalar, Kokkos::Extents<NP, NP, 2, NLEV, TIME_LEVELS, 0>, Kokkos::LayoutLeft>;
using Det    = Kokkos::View<Scalar, Kokkos::Extents<NP, NP, 0>, Kokkos::LayoutLeft>;
#else
using Dinv   = Kokkos::View<Scalar*****,  Kokkos::LayoutLeft>;
using P      = Kokkos::View<Scalar*****,  Kokkos::LayoutLeft>;
using PS     = Kokkos::View<Scalar***,    Kokkos::LayoutLeft>;
using Deriv  = Kokkos::View<Scalar***,    Kokkos::LayoutLeft>;
using GradPS = Kokkos::View<Scalar****,   Kokkos::LayoutLeft>;
using V      = Kokkos::View<Scalar******, Kokkos::LayoutLeft>;
using Det    = Kokkos::View<Scalar***,    Kokkos::LayoutLeft>;
#endif



} // namespace Homme
