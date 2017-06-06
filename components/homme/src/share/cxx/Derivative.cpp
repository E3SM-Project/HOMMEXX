#include "Derivative.hpp"

#include "FortranArrayUtils.hpp"
#include "BasicKernels.hpp"
#include "PhysicalConstants.hpp"

#include <random>

namespace Homme
{

Derivative::Derivative ()
 : m_dvv_exec           ("dvv")
 , m_integ_mat_exec     ("integration matrix")
 , m_bd_interp_mat_exec ("boundary interpolation matrix")
{
  // Nothing to be done here
}

void Derivative::init (CF90Ptr& dvv_ptr)
{
  HostViewManaged<Real[NP][NP]> dvv_host ("dvv");
  flip_f90_array_2d_12<NP,NP> (dvv_ptr, dvv_host);
  Kokkos::deep_copy (m_dvv_exec, dvv_host);
}

void Derivative::init (CF90Ptr& dvv_ptr, CF90Ptr& integration_mat_ptr, CF90Ptr& bd_interpolation_mat_ptr)
{
  HostViewManaged<Real[NP][NP]>     dvv_host ("dvv");
  HostViewManaged<Real[NC][NP]>     integ_mat_host ("dvv");
  HostViewManaged<Real[2][NC][NP]>  bd_interp_mat_host ("dvv");

  flip_f90_array_2d_12<NP,NP> (dvv_ptr, dvv_host);
  flip_f90_array_2d_12<NC,NP> (integration_mat_ptr, integ_mat_host);
  flip_f90_array_3d_213<NC,2,NP> (bd_interpolation_mat_ptr, bd_interp_mat_host);

  Kokkos::deep_copy (m_dvv_exec,           dvv_host);
  Kokkos::deep_copy (m_integ_mat_exec,     integ_mat_host);
  Kokkos::deep_copy (m_bd_interp_mat_exec, bd_interp_mat_host);
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

  ExecViewManaged<Real[NC][NP]>::HostMirror integ_mat_host = Kokkos::create_mirror_view(m_integ_mat_exec);
  ExecViewManaged<Real[2][NC][NP]>::HostMirror bd_interp_mat_host = Kokkos::create_mirror_view(m_bd_interp_mat_exec);
  for(int igp = 0; igp < NC; ++igp) {
    for(int jgp = 0; jgp < NP; ++jgp) {
      integ_mat_host(igp, jgp) = random_dist(engine);
      for(int dim = 0; dim < 2; ++dim) {
        bd_interp_mat_host(dim, igp, jgp) = random_dist(engine);
      }
    }
  }
  Kokkos::deep_copy(m_integ_mat_exec, integ_mat_host);
  Kokkos::deep_copy(m_bd_interp_mat_exec, bd_interp_mat_host);
}

void Derivative::dvv(Real *dvv_ptr) {
  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_cxx(dvv_ptr);
  Kokkos::deep_copy(dvv_cxx, m_dvv_exec);
}

void subcell_div_fluxes (const Kokkos::TeamPolicy<ExecSpace>::member_type& team_member,
                         const ExecViewUnmanaged<const Real[2][NP][NP]>    u,
                         const ExecViewUnmanaged<const Real[NP][NP]>       metdet,
                         ExecViewUnmanaged<Real[4][NC][NC]>                flux)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  const Derivative& deriv = get_derivative();

  // Helpers
  ExecViewUnmanaged<const Real[NC][NP]>    integ_mat     = deriv.get_integration_matrix();
  ExecViewUnmanaged<const Real[2][NC][NP]> bd_interp_mat = deriv.get_bd_interpolation_matrix();
  ExecViewUnmanaged<const Real[NC][NP]> bd_interp_mat_0 = subview (bd_interp_mat, 0, ALL(), ALL());
  ExecViewUnmanaged<const Real[NC][NP]> bd_interp_mat_1 = subview (bd_interp_mat, 1, ALL(), ALL());

  // Temporaries
  ExecViewManaged<Real[2][NP][NP]> v  ("v");
  ExecViewManaged<Real[NC][NP]>    tb ("tb");
  ExecViewManaged<Real[NC][NP]>    lr ("lr");
  ExecViewUnmanaged<Real[NC][NC]>  flux_b = subview (flux, 0, ALL(), ALL());
  ExecViewUnmanaged<Real[NC][NC]>  flux_r = subview (flux, 1, ALL(), ALL());
  ExecViewUnmanaged<Real[NC][NC]>  flux_t = subview (flux, 2, ALL(), ALL());
  ExecViewUnmanaged<Real[NC][NC]>  flux_l = subview (flux, 3, ALL(), ALL());

  Kokkos::parallel_for(
    Kokkos::ThreadVectorRange (team_member, NP * NP),
    KOKKOS_LAMBDA (const int idx)
    {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      v (0,igp,jgp) = u(0,igp,jgp) * metdet(igp,jgp);
      v (1,igp,jgp) = u(1,igp,jgp) * metdet(igp,jgp);
    }
  );

  ExecViewUnmanaged<const Real[NP][NP]> v2 = subview(v,1,ALL(),ALL());
  ExecViewUnmanaged<const Real[NP][NP]> v1 = subview(v,0,ALL(),ALL());

  // Top/bottom
  matrix_matrix<NC,NP,NP,NP> (team_member, integ_mat, v2, tb);
  matrix_matrix<NC,NP,NP,NP,false,true> (team_member, tb, bd_interp_mat_0, flux_b);
  matrix_matrix<NC,NP,NP,NP,false,true> (team_member, tb, bd_interp_mat_1, flux_t);

  // Left/right
  matrix_matrix<NC,NP,NP,NP,false,true> (team_member, v1, integ_mat, lr);
  matrix_matrix<NC,NP,NP,NP> (team_member, bd_interp_mat_0, lr, flux_l);
  matrix_matrix<NC,NP,NP,NP> (team_member, bd_interp_mat_1, lr, flux_r);

  Kokkos::parallel_for(
    Kokkos::ThreadVectorRange (team_member, NP * NP),
    KOKKOS_LAMBDA (const int idx)
    {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      flux_b (igp,jgp) *= -PhysicalConstants::rrearth;
      flux_r (igp,jgp) *=  PhysicalConstants::rrearth;
      flux_t (igp,jgp) *=  PhysicalConstants::rrearth;
      flux_l (igp,jgp) *= -PhysicalConstants::rrearth;
    }
  );
}

Derivative& get_derivative ()
{
  static Derivative deriv;

  return deriv;
}

} // namespace Homme
