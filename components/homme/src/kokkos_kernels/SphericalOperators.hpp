#ifndef HOMMEXX_SPHERICAL_OPERATORS_HPP
#define HOMMEXX_SPHERICAL_OPERATORS_HPP

#include <Derivative.hpp>
#include <ViewsPool.hpp>
#include <PhysicalConstants.hpp>
#include <ControlParameters.hpp>

#include <kinds.hpp>
#include <Kokkos_Core.hpp>
#include <Types.hpp>

#include <fortran_binding.hpp>
#include <SphericalOperators.hpp>

#include <iomanip>

namespace Homme
{

extern "C"
{

void gradient_sphere_c (const int& nets, const int& nete, const int& nelems,
                        real* const& scalar_field_ptr, real*& grad_field_ptr);

void divergence_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                             real* const& vector_field_ptr, real*& weak_div_field_ptr);

void laplace_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                          const int& variable_viscosity,
                          real* const& scalar_field_ptr, real*& weak_lapl_field_ptr);

void vorticity_sphere_c (const int& nets, const int& nete, const int& nelems,
                         real* const& vector_field_ptr, real*& vorticity_field_ptr);

void divergence_sphere_c (const int& nets, const int& nete, const int& nelems,
                          real* const& vector_field_ptr, real*& div_field_ptr);

void gradient_sphere_wk_testcov_c (const int& nets, const int& nete, const int& nelems,
                                   real* const& scalar_field_ptr, real*& grad_field);

void curl_sphere_wk_testcov_c (const int& nets, const int& nete, const int& nelems,
                               real* const& scalar_field_ptr, real*& curl_field);

void vlaplace_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                           const int& variable_viscosity, const real* nu_ratio,
                           real* const& vector_field_ptr, real*& lapl_weak_ptr);

void vlaplace_sphere_wk_cartesian_c (const int& nets, const int& nete, const int& nelems,
                                     const int& variable_viscosity,
                                     real* const& vector_field_ptr, real*& lapl_weak_ptr);

void vlaplace_sphere_wk_contra_c (const int& nets, const int& nete, const int& nelems,
                                  const int& variable_viscosity, const real* nu_ratio,
                                  real* const& vector_field_ptr, real*& lapl_weak_ptr);

} // extern "C"

template<typename M1, typename M2>
void gradient_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                             HommeView4D<M1> scalar_field, HommeView5D<M2> grad_field);

template<typename M1, typename M2>
void divergence_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                                  HommeView5D<M1> vector_field, HommeView4D<M2> weak_div_field);

template<typename M1, typename M2>
void laplace_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                               const int& variable_viscosity,
                               HommeView4D<M1> scalar_field, HommeView4D<M2> weak_lapl_field);

template<typename M1, typename M2>
void vorticity_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                              HommeView5D<M1> vector_field, HommeView4D<M2> vorticity_field);

template<typename M1, typename M2>
void divergence_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                               HommeView5D<M1> vector_field, HommeView4D<M2> divergence_field);

template<typename M1, typename M2>
void gradient_sphere_wk_testcov_kokkos (const int& nets, const int& nete, const int& nelems,
                                        HommeView4D<M1> scalar_field, HommeView5D<M2> grad_field);

template<typename M1, typename M2>
void curl_sphere_wk_testcov_kokkos (const int& nets, const int& nete, const int& nelems,
                                    HommeView4D<M1> scalar_field, HommeView5D<M2> curl_field);

template<typename M1, typename M2>
void vlaplace_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                                const int& variable_viscosity, const real* nu_ratio,
                                HommeView5D<M1> vector_field, HommeView5D<M2> lapl_weak);

template<typename M1, typename M2>
void vlaplace_sphere_wk_cartesian_kokkos (const int& nets, const int& nete, const int& nelems,
                                          const int& variable_viscosity,
                                          HommeView5D<M1> vector_field, HommeView5D<M2> lapl_weak);

template<typename M1, typename M2>
void vlaplace_sphere_wk_contra_kokkos (const int& nets, const int& nete, const int& nelems,
                                       const int& variable_viscosity, const real* nu_ratio,
                                       HommeView5D<M1> vector_field, HommeView5D<M2> lapl_weak);

// ======================= IMPLEMENTATION ======================= //

template<typename M1, typename M2>
void gradient_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                             HommeView4D<M1> scalar_field, HommeView5D<M2> grad_field)
{
  HommeView5D<KMU> Dinv  = get_views_pool_c()->get_Dinv();

  const real rrearth = get_physical_constants_c()->rrearth;
/*
  const int size = (nete-nets)*nlev;

  Kokkos::parallel_for(
    Kokkos::RangePolicy(0,size),
    KOKKOS_LAMBDA (int i)
    {
      const int ilevel = i / nlev;
      const int ielem  = i % nlev;
*/
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();
      //const int ilevel = team_member.team_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          real ds_dx, ds_dy;
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              // Computing local gradient
              ds_dx = ds_dy = 0.;
              for (int kgp=0; kgp<np; ++kgp)
              {
                ds_dx += get_derivative_c()->Dvv[igp][kgp]*scalar_field(kgp, jgp, ilevel, ielem);
                ds_dy += get_derivative_c()->Dvv[jgp][kgp]*scalar_field(igp, kgp, ilevel, ielem);
              }
              ds_dx *= rrearth;
              ds_dy *= rrearth;

              // Convert covarient to latlon
              grad_field(igp, jgp, 0, ilevel, ielem) = ( Dinv(igp, jgp, 0, 0, ielem)*ds_dx + Dinv(igp, jgp, 1, 0, ielem)*ds_dy );
              grad_field(igp, jgp, 1, ilevel, ielem) = ( Dinv(igp, jgp, 0, 1, ielem)*ds_dx + Dinv(igp, jgp, 1, 1, ielem)*ds_dy );
            }
          }
        }
      );
    }
  );
}

template<typename M1, typename M2>
void divergence_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                                  HommeView5D<M1> vector_field, HommeView4D<M2> weak_div_field)
{
  HommeView5D<KMU> Dinv     = get_views_pool_c()->get_Dinv();
  HommeView3D<KMU> spheremp = get_views_pool_c()->get_spheremp();

  const real rrearth = get_physical_constants_c()->rrearth;
/*
  const int size = (nete-nets)*nlev;

  Kokkos::paralell_for(
    Kokkos::RangePolocy(0,size),
    KOKKOS_LAMBDA(int i)
    {
      const int ilevel = i / nlev;
      const int ielem  = i % nlev;
*/
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();
      //const int ilevel = team_member.team_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          real vtemp[np][np][2];
          // Transform from latlon to contravarient
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              vtemp[igp][jgp][0] = Dinv(igp,jgp,0,0,ielem)*vector_field(igp, jgp, 0, ilevel, ielem) + Dinv(igp,jgp,0,1,ielem)*vector_field(igp, jgp, 1, ilevel, ielem);
              vtemp[igp][jgp][1] = Dinv(igp,jgp,1,0,ielem)*vector_field(igp, jgp, 0, ilevel, ielem) + Dinv(igp,jgp,1,1,ielem)*vector_field(igp, jgp, 1, ilevel, ielem);
            }
          }
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              weak_div_field(igp,jgp,ilevel,ielem) = 0.;
              for (int kgp=0; kgp<np; ++kgp)
              {
                weak_div_field(igp,jgp,ilevel,ielem) -= rrearth * (spheremp(kgp, jgp, ielem)*vtemp[kgp][jgp][0]*get_derivative_c()->Dvv[kgp][igp] +
                                                                   spheremp(igp, kgp, ielem)*vtemp[igp][kgp][1]*get_derivative_c()->Dvv[kgp][jgp]);
              }
            }
          }
        }
      );
    }
  );
}

template<typename M1, typename M2>
void laplace_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                               const int& variable_viscosity,
                               HommeView4D<M1> scalar_field, HommeView4D<M2> weak_lapl_field)
{
  HommeView5D<KMU> Dinv           = get_views_pool_c()->get_Dinv();
  HommeView3D<KMU> spheremp       = get_views_pool_c()->get_spheremp();
  HommeView3D<KMU> hyperviscosity = get_views_pool_c()->get_hypervisc();

  HommeView5D<KMM> grad_field ("field_gradient", np, np, 2, nlev, nelems);
  // Compute gradient
  gradient_sphere_kokkos (nets, nete, nelems, scalar_field, grad_field);

  if (variable_viscosity!=0)
  {
    if (get_control_parameters_c()->hypervisc_power!=0)
    {
/*
      const int size = (nete-nets)*nlev;
      Kokkos::parallel_for(
        Kokkos::RangePolicy(0,size),
        KOKKOS_LAMBDA (int i)
        {
          const int ilevel = i / nlev;
          const int ielem  = i % nlev;
*/
      Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nete-nets,nlev),
        KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
        {
          const int ielem  = nets + team_member.league_rank();

          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member,nlev),
            KOKKOS_LAMBDA (const unsigned int ilevel)
            {
              for (int igp=0; igp<np; ++igp)
              {
                for (int jgp=0; jgp<np; ++jgp)
                {
                  grad_field(igp, jgp, 0, ilevel, ielem) *= hyperviscosity(igp, jgp, ielem);
                  grad_field(igp, jgp, 1, ilevel, ielem) *= hyperviscosity(igp, jgp, ielem);
                }
              }
            }
          );
        }
      );
    }
    else if (get_control_parameters_c()->hypervisc_scaling)
    {
      HommeView5D<KMU> tensorVisc = get_views_pool_c()->get_tensor_visc();
      HommeView5D<KMM> tmp ("tmp", np, np, 2, nlev, nelems);

      // Compute D*grad(u), with D tensor viscosity
/*
      const int size = (nete-nets)*nlev;
      Kokkos::parallel_for(
        Kokkos::RangePolicy(0,size),
        KOKKOS_LAMBDA (int i)
        {
          const int ilevel = i / nlev;
          const int ielem  = i % nlev;
*/
      Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nete-nets,nlev),
        KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
        {
          const int ielem  = nets + team_member.league_rank();

          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member,nlev),
            KOKKOS_LAMBDA (const unsigned int ilevel)
            {
              for (int igp=0; igp<np; ++igp)
              {
                for (int jgp=0; jgp<np; ++jgp)
                {
                  tmp(igp,jgp,0,ilevel,ielem) = tensorVisc(igp,jgp,0,0,ielem)*grad_field(igp,jgp,0,ilevel,ielem)
                                              + tensorVisc(igp,jgp,0,1,ielem)*grad_field(igp,jgp,1,ilevel,ielem);
                  tmp(igp,jgp,1,ilevel,ielem) = tensorVisc(igp,jgp,1,0,ielem)*grad_field(igp,jgp,0,ilevel,ielem)
                                              + tensorVisc(igp,jgp,1,1,ielem)*grad_field(igp,jgp,1,ilevel,ielem);
                }
              }
            }
          );
        }
      );
      grad_field = tmp;
    }
  }

  divergence_sphere_wk_kokkos (nets, nete, nelems, grad_field, weak_lapl_field);
}

template<typename M1, typename M2>
void vorticity_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                              HommeView5D<M1> vector_field, HommeView4D<M2> vorticity_field)
{
  HommeView5D<KMU> D          = get_views_pool_c()->get_D();
  HommeView3D<KMU> rmetdet    = get_views_pool_c()->get_rmetdet();
  HommeView5D<KMM> vector_cov ("vector_cov", np, np, 2, nlev, nelems);

  // Convert to covariant form
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              vector_cov(igp,jgp,0,ilevel,ielem) = D(igp,jgp,0,0,ielem) * vector_field(igp,jgp,0,ilevel,ielem)
                                                 + D(igp,jgp,1,0,ielem) * vector_field(igp,jgp,1,ilevel,ielem);
              vector_cov(igp,jgp,1,ilevel,ielem) = D(igp,jgp,0,1,ielem) * vector_field(igp,jgp,0,ilevel,ielem)
                                                 + D(igp,jgp,1,1,ielem) * vector_field(igp,jgp,1,ilevel,ielem);
            }
          }
        }
      );
    }
  );

  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          real du_dy, dv_dx;
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              du_dy = dv_dx = 0.;
              for (int kgp=0; kgp<np; ++kgp)
              {
                du_dy += get_derivative_c()->Dvv[jgp][kgp] * vector_cov(igp, kgp, 0, ilevel, ielem);
                dv_dx += get_derivative_c()->Dvv[igp][kgp] * vector_cov(kgp, jgp, 1, ilevel, ielem);
              }

              vorticity_field (igp, jgp, ilevel, ielem) = ( dv_dx-du_dy ) * rmetdet(igp,jgp,ielem)*get_physical_constants_c()->rrearth;
            }
          }
        }
      );
    }
  );
}

template<typename M1, typename M2>
void divergence_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                               HommeView5D<M1> vector_field, HommeView4D<M2> divergence_field)
{
  HommeView5D<KMM> vector_contra_g ("vector_contra_g", np, np, 2, nlev, nelems);
  HommeView5D<KMU> Dinv            = get_views_pool_c()->get_Dinv();
  HommeView3D<KMU> metdet          = get_views_pool_c()->get_metdet();
  HommeView3D<KMU> rmetdet         = get_views_pool_c()->get_rmetdet();

  // Convert to contravariant and multiply by g
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets, nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              vector_contra_g(igp, jgp, 0, ilevel, ielem) = metdet(igp, jgp, ielem) * ( Dinv(igp,jgp,0,0,ielem)*vector_field(igp,jgp,0,ilevel,ielem)
                                                                                       +Dinv(igp,jgp,0,1,ielem)*vector_field(igp,jgp,1,ilevel,ielem) );
              vector_contra_g(igp, jgp, 1, ilevel, ielem) = metdet(igp, jgp, ielem) * ( Dinv(igp,jgp,1,0,ielem)*vector_field(igp,jgp,0,ilevel,ielem)
                                                                                       +Dinv(igp,jgp,1,1,ielem)*vector_field(igp,jgp,1,ilevel,ielem) );
            }
          }
        }
      );
    }
  );

  // Compute divergence
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets, nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          real du_dx, dv_dy;
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              du_dx = dv_dy = 0.;
              for (int kgp=0; kgp<np; ++kgp)
              {
                du_dx += get_derivative_c()->Dvv[igp][kgp] * vector_contra_g(kgp, jgp, 0, ilevel, ielem);
                dv_dy += get_derivative_c()->Dvv[jgp][kgp] * vector_contra_g(igp, kgp, 1, ilevel, ielem);
              }

              divergence_field (igp, jgp, ilevel, ielem) = (du_dx+dv_dy) * rmetdet(igp,jgp,ielem) * get_physical_constants_c()->rrearth;
            }
          }
        }
      );
    }
  );
}

template<typename M1, typename M2>
void gradient_sphere_wk_testcov_kokkos (const int& nets, const int& nete, const int& nelems,
                                        HommeView4D<M1> scalar_field, HommeView5D<M2> grad_field)
{
  HommeView5D<KMM> grad_contra ("grad_contra", np, np, 2, nlev, nelems);
  HommeView5D<KMU> metinv      = get_views_pool_c()->get_metinv();
  HommeView3D<KMU> metdet      = get_views_pool_c()->get_metdet();
  HommeView3D<KMU> mp          = get_views_pool_c()->get_mp();
  HommeView5D<KMU> D           = get_views_pool_c()->get_D();

  const real rrearth = get_physical_constants_c()->rrearth;

  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              grad_contra (igp,jgp,0,ilevel,ielem) = grad_contra (igp,jgp,1,ilevel,ielem) = 0;
              for (int kgp=0; kgp<np; ++kgp)
              {
                grad_contra (igp, jgp, 0, ilevel, ielem) -= rrearth * ( mp(kgp,jgp,ielem)*metinv(igp,jgp,0,0,ielem)*metdet(igp,jgp,ielem)*scalar_field(kgp,jgp,ilevel,ielem)*get_derivative_c()->Dvv[kgp][igp]
                                                                      + mp(igp,kgp,ielem)*metinv(igp,jgp,1,0,ielem)*metdet(igp,jgp,ielem)*scalar_field(igp,kgp,ilevel,ielem)*get_derivative_c()->Dvv[kgp][jgp]);
                grad_contra (igp, jgp, 1, ilevel, ielem) -= rrearth * ( mp(kgp,jgp,ielem)*metinv(igp,jgp,0,1,ielem)*metdet(igp,jgp,ielem)*scalar_field(kgp,jgp,ilevel,ielem)*get_derivative_c()->Dvv[kgp][igp]
                                                                      + mp(igp,kgp,ielem)*metinv(igp,jgp,1,1,ielem)*metdet(igp,jgp,ielem)*scalar_field(igp,kgp,ilevel,ielem)*get_derivative_c()->Dvv[kgp][jgp]);
              }
            }
          }
        }
      );
    }
  );

  // Convert contra->latlon
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              grad_field (igp, jgp, 0, ilevel, ielem) =  D(igp, jgp, 0, 0, ielem) * grad_contra(igp, jgp, 0, ilevel, ielem)
                                                        +D(igp, jgp, 0, 1, ielem) * grad_contra(igp, jgp, 1, ilevel, ielem);
              grad_field (igp, jgp, 1, ilevel, ielem) =  D(igp, jgp, 1, 0, ielem) * grad_contra(igp, jgp, 0, ilevel, ielem)
                                                        +D(igp, jgp, 1, 1, ielem) * grad_contra(igp, jgp, 1, ilevel, ielem);
            }
          }
        }
      );
    }
  );
}

template<typename M1, typename M2>
void curl_sphere_wk_testcov_kokkos (const int& nets, const int& nete, const int& nelems,
                                    HommeView4D<M1> scalar_field, HommeView5D<M2> curl_field)
{
  HommeView3D<KMU> mp = get_views_pool_c()->get_mp();
  HommeView5D<KMU> D  = get_views_pool_c()->get_D();
  HommeView5D<KMM> grad_contra ("grad_contra", np, np, 2, nlev, nelems);

  const real rrearth = get_physical_constants_c()->rrearth;
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              grad_contra (igp,jgp,0,ilevel,ielem) = grad_contra (igp,jgp,1,ilevel,ielem) = 0;
              for (int kgp=0; kgp<np; ++kgp)
              {
                grad_contra (igp, jgp, 0, ilevel, ielem) -= mp(igp,kgp,ielem)*scalar_field(igp,kgp,ilevel,ielem)*get_derivative_c()->Dvv[kgp][jgp]*rrearth;
                grad_contra (igp, jgp, 1, ilevel, ielem) += mp(kgp,jgp,ielem)*scalar_field(kgp,jgp,ilevel,ielem)*get_derivative_c()->Dvv[kgp][igp]*rrearth;
              }
            }
          }
        }
      );
    }
  );

  // Convert contra->latlon
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              curl_field(igp,jgp,0,ilevel,ielem) = D(igp,jgp,0,0,ielem)*grad_contra(igp,jgp,0,ilevel,ielem) + D(igp,jgp,0,1,ielem)*grad_contra(igp,jgp,1,ilevel,ielem);
              curl_field(igp,jgp,1,ilevel,ielem) = D(igp,jgp,1,0,ielem)*grad_contra(igp,jgp,0,ilevel,ielem) + D(igp,jgp,1,1,ielem)*grad_contra(igp,jgp,1,ilevel,ielem);
            }
          }
        }
      );
    }
  );
}

template<typename M1, typename M2>
void vlaplace_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                                const int& variable_viscosity, const real* nu_ratio,
                                HommeView5D<M1> vector_field, HommeView5D<M2> lapl_weak)
{
  if (get_control_parameters_c()->hypervisc_scaling && variable_viscosity)
  {
    if (nu_ratio!=0 && *nu_ratio!=1)
    {
      std::cerr << "ERROR: tensorHV can not be used when nu_div!=nu.\n";
      std::abort();
    }
    vlaplace_sphere_wk_cartesian_kokkos (nets,nete,nelems,variable_viscosity,vector_field,lapl_weak);
  }
  else
  {
    vlaplace_sphere_wk_contra_kokkos (nets,nete,nelems,variable_viscosity,nu_ratio,vector_field,lapl_weak);
  }
}

template<typename M1, typename M2>
void vlaplace_sphere_wk_cartesian_kokkos (const int& nets, const int& nete, const int& nelems,
                                          const int& variable_viscosity,
                                          HommeView5D<M1> vector_field, HommeView5D<M2> lapl_weak)
{
  HommeView4D<KMM> vector_cart[3];
  HommeView4D<KMM> lapl_cart[3];
  HommeView5D<KMU> vec_sphere2cart = get_views_pool_c()->get_vec_sphere2cart();
  HommeView3D<KMU> spheremp        = get_views_pool_c()->get_spheremp();

  for (int i=0; i<3; ++i)
  {
    vector_cart[i] = HommeView4D<KMM>("vec_cart_" +std::to_string(i), np, np, nlev, nelems);
    lapl_cart[i]   = HommeView4D<KMM>("lapl_cart_"+std::to_string(i), np, np, nlev, nelems);
  }

  // Transformation latlon -> cartesian
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              for (int icomp=0; icomp<3; ++icomp)
              {
                vector_cart[icomp](igp,jgp,ilevel,ielem) = vec_sphere2cart(igp,jgp,icomp,0,ielem)*vector_field(igp,jgp,0,ilevel,ielem)
                                                         + vec_sphere2cart(igp,jgp,icomp,1,ielem)*vector_field(igp,jgp,1,ilevel,ielem);
              }
            }
          }
        }
      );
    }
  );

  // Compute laplacian on cartesian components
  for (int icomp=0; icomp<3; ++icomp)
  {
    laplace_sphere_wk_kokkos (nets, nete, nelems, variable_viscosity, vector_cart[icomp], lapl_cart[icomp]);
  }

  // Transform back to latlon
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              lapl_weak(igp,jgp,0,ilevel,ielem) = vec_sphere2cart(igp,jgp,0,0,ielem) * lapl_cart[0](igp,jgp,ilevel,ielem)
                                                + vec_sphere2cart(igp,jgp,1,0,ielem) * lapl_cart[1](igp,jgp,ilevel,ielem)
                                                + vec_sphere2cart(igp,jgp,2,0,ielem) * lapl_cart[2](igp,jgp,ilevel,ielem);
              lapl_weak(igp,jgp,1,ilevel,ielem) = vec_sphere2cart(igp,jgp,0,1,ielem) * lapl_cart[0](igp,jgp,ilevel,ielem)
                                                + vec_sphere2cart(igp,jgp,1,1,ielem) * lapl_cart[1](igp,jgp,ilevel,ielem)
                                                + vec_sphere2cart(igp,jgp,2,1,ielem) * lapl_cart[2](igp,jgp,ilevel,ielem);
            }
          }
        }
      );
    }
  );

  // Add in correction so we don't damp rigid rotation
  double rrearth2 = std::pow(get_physical_constants_c()->rrearth,2);
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              lapl_weak(igp,jgp,0,ilevel,ielem) += 2*spheremp(igp,jgp,ielem)*vector_field(igp,jgp,0,ilevel,ielem) * rrearth2;
              lapl_weak(igp,jgp,1,ilevel,ielem) += 2*spheremp(igp,jgp,ielem)*vector_field(igp,jgp,1,ilevel,ielem) * rrearth2;
            }
          }
        }
      );
    }
  );
}

template<typename M1, typename M2>
void vlaplace_sphere_wk_contra_kokkos (const int& nets, const int& nete, const int& nelems,
                                       const int& variable_viscosity, const real* nu_ratio,
                                       HommeView5D<M1> vector_field, HommeView5D<M2> lapl_weak)
{
  HommeView4D<KMM> div  ("divergence", np, np, nlev, nelems);
  HommeView4D<KMM> vort ("vorticity",  np, np, nlev, nelems);

  divergence_sphere_kokkos (nets, nete, nelems, vector_field, div);
  vorticity_sphere_kokkos  (nets, nete, nelems, vector_field, vort);

  if (variable_viscosity!=0 && get_control_parameters_c()->hypervisc_power!=0)
  {
    HommeView3D<KMU> hyperviscosity = get_views_pool_c()->get_hypervisc();

    Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(nete-nets,nlev),
      KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
      {
        const int ielem  = nets + team_member.league_rank();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member,nlev),
          KOKKOS_LAMBDA (const unsigned int ilevel)
          {
            for (int igp=0; igp<np; ++igp)
            {
              for (int jgp=0; jgp<np; ++jgp)
              {
                div  (igp, jgp, ilevel, ielem) *= hyperviscosity(igp, jgp, ielem);
                vort (igp, jgp, ilevel, ielem) *= hyperviscosity(igp, jgp, ielem);
              }
            }
          }
        );
      }
    );
  }

  if (nu_ratio!=0)
  {
    real nu_ratio_val = *nu_ratio;

    Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(nete-nets,nlev),
      KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
      {
        const int ielem  = nets + team_member.league_rank();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member,nlev),
          KOKKOS_LAMBDA (const unsigned int ilevel)
          {
            for (int igp=0; igp<np; ++igp)
            {
              for (int jgp=0; jgp<np; ++jgp)
              {
                div(igp, jgp, ilevel, ielem)  *= nu_ratio_val;
              }
            }
          }
        );
      }
    );
  }

  // Compute grad-div and curl-vort
  HommeView5D<KMM> grad_div  ("grad_div",  np, np, 2, nlev, nelems);
  HommeView5D<KMM> curl_vort ("curl_vort", np, np, 2, nlev, nelems);

  gradient_sphere_wk_testcov_kokkos (nets, nete, nelems, div, grad_div);
  curl_sphere_wk_testcov_kokkos     (nets, nete, nelems, vort, curl_vort);

  // Add grad_div and curl_vort, and adding correction so we don't damp rigid rotation
  double rrearth2 = std::pow(get_physical_constants_c()->rrearth,2);
  auto spheremp = get_views_pool_c()->get_spheremp();

  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(nete-nets,nlev),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ielem  = nets + team_member.league_rank();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,nlev),
        KOKKOS_LAMBDA (const unsigned int ilevel)
        {
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              lapl_weak(igp,jgp,0,ilevel,ielem) = grad_div(igp, jgp, 0, ilevel, ielem) - curl_vort (igp, jgp, 0, ilevel, ielem)
                                                + 2*spheremp(igp,jgp,ielem)*vector_field(igp,jgp,0,ilevel,ielem) * rrearth2;
              lapl_weak(igp,jgp,1,ilevel,ielem) = grad_div(igp, jgp, 1, ilevel, ielem) - curl_vort (igp, jgp, 1, ilevel, ielem)
                                                + 2*spheremp(igp,jgp,ielem)*vector_field(igp,jgp,1,ilevel,ielem) * rrearth2;
            }
          }
        }
      );
    }
  );
}

} // Namespace Homme

#endif // HOMMEXX_SPHERICAL_OPERATORS_HPP
