#include "Derivative.hpp"

#include "FortranArrayUtils.hpp"
#include "PhysicalConstants.hpp"

namespace Homme
{

Derivative::Derivative ()
 : m_dvv_exec           ("dvv")
{
  // Nothing to be done here
}

void Derivative::init (CF90Ptr& dvv_ptr)
{
  HostViewManaged<Real[NP][NP]> dvv_host ("dvv");
  flip_f90_array_2d_12<NP,NP> (dvv_ptr, dvv_host);
  Kokkos::deep_copy (m_dvv_exec, dvv_host);
}

void Derivative::random_init(std::mt19937_64 &engine) {
  std::uniform_real_distribution<Real> random_dist(16.0, 8192.0);
  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_host = Kokkos::create_mirror_view(m_dvv_exec);
  for(int igp = 0; igp < NP; ++igp) {
    for(int jgp = 0; jgp < NP; ++jgp) {
      dvv_host(igp, jgp) = random_dist(engine);
    }
  }
  Kokkos::deep_copy(m_dvv_exec, dvv_host);
}

void Derivative::dvv(Real *dvv_ptr) {
  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_cxx = Kokkos::create_mirror_view(m_dvv_exec),
    dvv_wrapper(dvv_ptr);
  Kokkos::deep_copy(dvv_cxx, m_dvv_exec);
  for(int igp = 0; igp < NP; ++igp) {
    for(int jgp = 0; jgp < NP; ++jgp) {
      dvv_wrapper(jgp, igp) = dvv_cxx(igp, jgp);
    }
  }
}

Derivative& get_derivative ()
{
  static Derivative deriv;

  return deriv;
}

} // namespace Homme
