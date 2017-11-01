#include "Derivative.hpp"

namespace Homme {

Derivative::Derivative()
    : m_dvv_exec("dvv")
{
  // Nothing to be done here
}

void Derivative::init(CF90Ptr &dvv_ptr) {
  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_host =
      Kokkos::create_mirror_view(m_dvv_exec);

  int k_dvv = 0;
  for (int igp = 0; igp < NP; ++igp) {
    for (int jgp = 0; jgp < NP; ++jgp, ++k_dvv) {
      dvv_host(igp, jgp) = dvv_ptr[k_dvv];
    }
  }

  Kokkos::deep_copy(m_dvv_exec, dvv_host);
}

void Derivative::random_init(std::mt19937_64 &engine) {
  std::uniform_real_distribution<Real> random_dist(16.0, 8192.0);
  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_host =
      Kokkos::create_mirror_view(m_dvv_exec);
  for (int igp = 0; igp < NP; ++igp) {
    for (int jgp = 0; jgp < NP; ++jgp) {
      dvv_host(igp, jgp) = random_dist(engine);
    }
  }
  Kokkos::deep_copy(m_dvv_exec, dvv_host);
}

void Derivative::dvv(Real *dvv_ptr) {
  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_f90(dvv_ptr);
  Kokkos::deep_copy(dvv_f90, m_dvv_exec);
}

} // namespace Homme
