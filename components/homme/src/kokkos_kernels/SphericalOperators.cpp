#include <Derivative.hpp>
#include <PointersPool.hpp>
#include <PhysicalConstants.hpp>
#include <ControlParameters.hpp>

#include <kinds.hpp>
#include <Kokkos_Core.hpp>

#include <fortran_binding.hpp>
#include <SphericalOperators.hpp>

#include <iomanip>
namespace Homme
{

extern "C"
{

// ======================= DIFFERENTIAL OPERATIONS ON SPHERE ===================================== //

void gradient_sphere_c (const int& nets, const int& nete, const int& nelems,
                        real* const& scalar_field_ptr, real*& grad_field_ptr)
{
  // Note: for each dimension, pass 'dim_length, dim_stride'
  Kokkos::LayoutStride layout_s(nelems,nlev*np*np,
                                nlev,np*np,
                                np,np,
                                np,1);
  Kokkos::LayoutStride layout_g(nelems, nlev*2*np*np,
                                nlev,   2*np*np,
                                2,      np*np,
                                np,     np,
                                np,     1);

  Kokkos::View<real*[nlev][np][np],    Kokkos::LayoutStride> scalar_field (scalar_field_ptr,layout_s);
  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> grad_field   (grad_field_ptr,layout_g);

  gradient_sphere_kokkos (nets, nete, nelems, scalar_field, grad_field);
}

void gradient_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                             Kokkos::View<real*[nlev][np][np],    Kokkos::LayoutStride> scalar_field,
                             Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> grad_field)
{
  Kokkos::View<real*[2][2][np][np]> Dinv (get_pointers_pool_c()->Dinv,nelems);

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
                ds_dx += get_derivative_c()->Dvv[igp][kgp]*scalar_field(ielem, ilevel, kgp, jgp);
                ds_dy += get_derivative_c()->Dvv[jgp][kgp]*scalar_field(ielem, ilevel, igp, kgp);
              }
              ds_dx *= rrearth;
              ds_dy *= rrearth;

              // Convert covarient to latlon
              grad_field(ielem, ilevel, 0, igp, jgp) = ( Dinv(ielem,0,0,igp,jgp)*ds_dx + Dinv(ielem,1,0,igp,jgp)*ds_dy );
              grad_field(ielem, ilevel, 1, igp, jgp) = ( Dinv(ielem,0,1,igp,jgp)*ds_dx + Dinv(ielem,1,1,igp,jgp)*ds_dy );
            }
          }
        }
      );
    }
  );
}

void divergence_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                             real* const& vector_field_ptr, real*& weak_div_field_ptr)
{
  // Note: for each dimension, pass 'dim_length, dim_stride'
  Kokkos::LayoutStride layout_d(nelems,nlev*np*np,
                                nlev,np*np,
                                np,np,
                                np,1);
  Kokkos::LayoutStride layout_v(nelems, nlev*2*np*np,
                                nlev,   2*np*np,
                                2,      np*np,
                                np,     np,
                                np,     1);
  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field   (vector_field_ptr,   layout_v);
  Kokkos::View<real*[nlev][np][np],    Kokkos::LayoutStride> weak_div_field (weak_div_field_ptr, layout_d);

  divergence_sphere_wk_kokkos (nets, nete, nelems, vector_field, weak_div_field);
}

void divergence_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                                  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride>  vector_field,
                                  Kokkos::View<real*[nlev][np][np],    Kokkos::LayoutStride>  weak_div_field)
{
  Kokkos::View<real*[2][2][np][np]> Dinv     (get_pointers_pool_c()->Dinv,     nelems);
  Kokkos::View<real*[np][np]>       spheremp (get_pointers_pool_c()->spheremp, nelems);

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
              vtemp[igp][jgp][0] = Dinv(ielem,0,0,igp,jgp)*vector_field(ielem, ilevel, 0, igp, jgp) + Dinv(ielem,0,1,igp,jgp)*vector_field(ielem, ilevel, 1, igp, jgp);
              vtemp[igp][jgp][1] = Dinv(ielem,1,0,igp,jgp)*vector_field(ielem, ilevel, 0, igp, jgp) + Dinv(ielem,1,1,igp,jgp)*vector_field(ielem, ilevel, 1, igp, jgp);
            }
          }
          for (int igp=0; igp<np; ++igp)
          {
            for (int jgp=0; jgp<np; ++jgp)
            {
              weak_div_field(ielem, ilevel, igp, jgp) = 0.;
              for (int kgp=0; kgp<np; ++kgp)
              {
                weak_div_field(ielem, ilevel, igp, jgp) -= rrearth * (spheremp(ielem, kgp, jgp)*vtemp[kgp][jgp][0]*get_derivative_c()->Dvv[kgp][igp] +
                                                                      spheremp(ielem, igp, kgp)*vtemp[igp][kgp][1]*get_derivative_c()->Dvv[kgp][jgp]);
              }
            }
          }
        }
      );
    }
  );
}

void laplace_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                          const int& variable_viscosity,
                          real* const& scalar_field_ptr, real*& weak_lapl_field_ptr)
{
  // Note: for each dimension, pass 'dim_length, dim_stride'
  Kokkos::LayoutStride layout(nelems,nlev*np*np,
                              nlev,np*np,
                              np,np,
                              np,1);

  Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride>  scalar_field (scalar_field_ptr, layout);
  Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride>  weak_lapl_field (weak_lapl_field_ptr, layout);

  laplace_sphere_wk_kokkos (nets, nete, nelems, variable_viscosity, scalar_field, weak_lapl_field);
}

void laplace_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                               const int& variable_viscosity,
                               Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride>  scalar_field,
                               Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride>  weak_lapl_field)
{
  Kokkos::View<real*[2][2][np][np]> Dinv           (get_pointers_pool_c()->Dinv,      nelems);
  Kokkos::View<real*[np][np]>       spheremp       (get_pointers_pool_c()->spheremp,  nelems);
  Kokkos::View<real*[np][np]>       hyperviscosity (get_pointers_pool_c()->hypervisc, nelems);

  // Note: for each dimension, pass 'dim_length, dim_stride'
  Kokkos::LayoutStride layout(nelems,nlev*2*np*np,
                              nlev,2*np*np,
                              2,np*np,
                              np,np,
                              np,1);
  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> grad_field ("field_gradient", layout);

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
                  grad_field(ielem, ilevel, 0, igp, jgp) *= hyperviscosity(ielem, igp, jgp);
                  grad_field(ielem, ilevel, 1, igp, jgp) *= hyperviscosity(ielem, igp, jgp);
                }
              }
            }
          );
        }
      );
    }
    else if (get_control_parameters_c()->hypervisc_scaling)
    {
      Kokkos::View<real*[2][2][np][np]>    tensorVisc (get_pointers_pool_c()->tensor_visc, nelems);
      Kokkos::View<real*[nlev][2][np][np]> tmp        ("tmp",                              nelems);

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
                  tmp(ielem,ilevel,0,igp,jgp) = tensorVisc(ielem,0,0,igp,jgp)*grad_field(ielem,ilevel,0,igp,jgp)
                                              + tensorVisc(ielem,0,1,igp,jgp)*grad_field(ielem,ilevel,1,igp,jgp);
                  tmp(ielem,ilevel,1,igp,jgp) = tensorVisc(ielem,1,0,igp,jgp)*grad_field(ielem,ilevel,0,igp,jgp)
                                              + tensorVisc(ielem,1,1,igp,jgp)*grad_field(ielem,ilevel,1,igp,jgp);
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

void vorticity_sphere_c (const int& nets, const int& nete, const int& nelems,
                         real* const& vector_field_ptr, real*& vorticity_field_ptr)
{
  // Note: for each dimension, pass 'dim_length, dim_stride'
  Kokkos::LayoutStride layout_w(nelems,nlev*np*np,
                                nlev,np*np,
                                np,np,
                                np,1);
  Kokkos::LayoutStride layout_v(nelems, nlev*2*np*np,
                                nlev,   2*np*np,
                                2,      np*np,
                                np,     np,
                                np,     1);
  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field    (vector_field_ptr,    layout_w);
  Kokkos::View<real*[nlev][np][np],    Kokkos::LayoutStride> vorticity_field (vorticity_field_ptr, layout_v);

  vorticity_sphere_kokkos (nets, nete, nelems, vector_field, vorticity_field);
}

void vorticity_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                              Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field,
                              Kokkos::View<real*[nlev][np][np],    Kokkos::LayoutStride> vorticity_field)
{
  Kokkos::View<real*[2][2][np][np]>    D          (get_pointers_pool_c()->D,       nelems);
  Kokkos::View<real*[np][np]>          rmetdet    (get_pointers_pool_c()->rmetdet, nelems);
  Kokkos::View<real*[nlev][2][np][np]> vector_cov ("vector_cov",                   nelems);

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
              vector_cov(ielem,ilevel,0,igp,jgp) = D(ielem,0,0,igp,jgp) * vector_field(ielem,ilevel,0,igp,jgp)
                                                 + D(ielem,1,0,igp,jgp) * vector_field(ielem,ilevel,1,igp,jgp);
              vector_cov(ielem,ilevel,1,igp,jgp) = D(ielem,0,1,igp,jgp) * vector_field(ielem,ilevel,0,igp,jgp)
                                                 + D(ielem,1,1,igp,jgp) * vector_field(ielem,ilevel,1,igp,jgp);
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
                du_dy += get_derivative_c()->Dvv[jgp][kgp] * vector_cov(ielem, ilevel, 0, igp, kgp);
                dv_dx += get_derivative_c()->Dvv[igp][kgp] * vector_cov(ielem, ilevel, 1, kgp, jgp);
              }

              vorticity_field (ielem, ilevel, igp, jgp) = ( dv_dx-du_dy ) * rmetdet(ielem,igp,jgp)*get_physical_constants_c()->rrearth;
            }
          }
        }
      );
    }
  );
}

void divergence_sphere_c (const int& nets, const int& nete, const int& nelems,
                          real* const& vector_field_ptr, real*& div_field_ptr)
{
  // Note: for each dimension, pass 'dim_length, dim_stride'
  Kokkos::LayoutStride layout_d(nelems,nlev*np*np,
                                nlev,np*np,
                                np,np,
                                np,1);
  Kokkos::LayoutStride layout_v(nelems, nlev*2*np*np,
                                nlev,   2*np*np,
                                2,      np*np,
                                np,     np,
                                np,     1);
  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field     (vector_field_ptr, layout_v);
  Kokkos::View<real*[nlev][np][np],    Kokkos::LayoutStride> divergence_field (div_field_ptr,    layout_d);

  divergence_sphere_kokkos (nets, nete, nelems, vector_field, divergence_field);
}

void divergence_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                               Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field,
                               Kokkos::View<real*[nlev][np][np],    Kokkos::LayoutStride> divergence_field)
{
  Kokkos::View<real*[nlev][2][np][np]> vector_contra_g ("vector_contra_g",              nelems);
  Kokkos::View<real*[2][2][np][np]>    Dinv            (get_pointers_pool_c()->Dinv,    nelems);
  Kokkos::View<real*[np][np]>          metdet          (get_pointers_pool_c()->metdet,  nelems);
  Kokkos::View<real*[np][np]>          rmetdet         (get_pointers_pool_c()->rmetdet, nelems);

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
              vector_contra_g(ielem, ilevel, 0, igp, jgp) = metdet(ielem, igp, jgp) * ( Dinv(ielem,0,0,igp,jgp)*vector_field(ielem,ilevel,0,igp,jgp)
                                                                                       +Dinv(ielem,0,1,igp,jgp)*vector_field(ielem,ilevel,1,igp,jgp) );
              vector_contra_g(ielem, ilevel, 1, igp, jgp) = metdet(ielem, igp, jgp) * ( Dinv(ielem,1,0,igp,jgp)*vector_field(ielem,ilevel,0,igp,jgp)
                                                                                       +Dinv(ielem,1,1,igp,jgp)*vector_field(ielem,ilevel,1,igp,jgp) );
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
                du_dx += get_derivative_c()->Dvv[igp][kgp] * vector_contra_g(ielem, ilevel, 0, kgp, jgp);
                dv_dy += get_derivative_c()->Dvv[jgp][kgp] * vector_contra_g(ielem, ilevel, 1, igp, kgp);
              }

              divergence_field (ielem, ilevel, igp, jgp) = (du_dx+dv_dy) * rmetdet(ielem, igp, jgp) * get_physical_constants_c()->rrearth;
            }
          }
        }
      );
    }
  );
}

void gradient_sphere_wk_testcov_c (const int& nets, const int& nete, const int& nelems,
                                   real* const& scalar_field_ptr, real*& grad_field_ptr)
{
  Kokkos::View<real*[nlev][np][np]>    scalar_field (scalar_field_ptr, nelems);
  Kokkos::View<real*[nlev][2][np][np]> grad_field   (grad_field_ptr,   nelems);

  gradient_sphere_wk_testcov_kokkos (nets, nete, nelems, scalar_field, grad_field);
}

void gradient_sphere_wk_testcov_kokkos (const int& nets, const int& nete, const int& nelems,
                                        Kokkos::View<real*[nlev][np][np]>    scalar_field,
                                        Kokkos::View<real*[nlev][2][np][np]> grad_field)
{
  Kokkos::View<real*[nlev][2][np][np]> grad_contra ("grad_contra",                 nelems);
  Kokkos::View<real*[2][2][np][np]>    metinv      (get_pointers_pool_c()->metinv, nelems);
  Kokkos::View<real*[np][np]>          metdet      (get_pointers_pool_c()->metdet, nelems);
  Kokkos::View<real*[np][np]>          mp          (get_pointers_pool_c()->mp,     nelems);
  Kokkos::View<real*[2][2][np][np]>    D           (get_pointers_pool_c()->D,      nelems);

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
              grad_contra (ielem, ilevel, 0, igp, jgp) = grad_contra (ielem, ilevel, 1, igp, jgp) = 0;
              for (int kgp=0; kgp<np; ++kgp)
              {
                grad_contra (ielem, ilevel, 0, igp, jgp) -= rrearth * ( mp(ielem,kgp,jgp)*metinv(ielem,0,0,igp,jgp)*metdet(ielem,igp,jgp)*scalar_field(ielem,ilevel,kgp,jgp)*get_derivative_c()->Dvv[kgp][igp]
                                                                      + mp(ielem,igp,kgp)*metinv(ielem,1,0,igp,jgp)*metdet(ielem,igp,jgp)*scalar_field(ielem,ilevel,igp,kgp)*get_derivative_c()->Dvv[kgp][jgp]);
                grad_contra (ielem, ilevel, 1, igp, jgp) -= rrearth * ( mp(ielem,kgp,jgp)*metinv(ielem,0,1,igp,jgp)*metdet(ielem,igp,jgp)*scalar_field(ielem,ilevel,kgp,jgp)*get_derivative_c()->Dvv[kgp][igp]
                                                                      + mp(ielem,igp,kgp)*metinv(ielem,1,1,igp,jgp)*metdet(ielem,igp,jgp)*scalar_field(ielem,ilevel,igp,kgp)*get_derivative_c()->Dvv[kgp][jgp]);
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
              grad_field (ielem, ilevel, 0, igp, jgp) =  D(ielem, 0, 0, igp, jgp) * grad_contra(ielem, ilevel, 0, igp, jgp)
                                                        +D(ielem, 0, 1, igp, jgp) * grad_contra(ielem, ilevel, 1, igp, jgp);
              grad_field (ielem, ilevel, 1, igp, jgp) =  D(ielem, 1, 0, igp, jgp) * grad_contra(ielem, ilevel, 0, igp, jgp)
                                                        +D(ielem, 1, 1, igp, jgp) * grad_contra(ielem, ilevel, 1, igp, jgp);
            }
          }
        }
      );
    }
  );
}

void curl_sphere_wk_testcov_c (const int& nets, const int& nete, const int& nelems,
                               real* const& scalar_field_ptr, real*& curl_field_ptr)
{
  Kokkos::View<real*[nlev][np][np]>    scalar_field (scalar_field_ptr, nelems);
  Kokkos::View<real*[nlev][2][np][np]> curl_field   (curl_field_ptr,   nelems);

  curl_sphere_wk_testcov_kokkos (nets, nete, nelems, scalar_field, curl_field);
}

void curl_sphere_wk_testcov_kokkos (const int& nets, const int& nete, const int& nelems,
                                    Kokkos::View<real*[nlev][np][np]>    scalar_field,
                                    Kokkos::View<real*[nlev][2][np][np]> curl_field)
{
  Kokkos::View<real*[np][np]>          mp          (get_pointers_pool_c()->mp, nelems);
  Kokkos::View<real*[2][2][np][np]>    D           (get_pointers_pool_c()->D,  nelems);
  Kokkos::View<real*[nlev][2][np][np]> grad_contra ("grad_contra",             nelems);

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
              grad_contra (ielem, ilevel, 0, igp, jgp) = grad_contra (ielem, ilevel, 1, igp, jgp) = 0;
              for (int kgp=0; kgp<np; ++kgp)
              {
                grad_contra (ielem, ilevel, 0, igp, jgp) -= mp(ielem,igp,kgp)*scalar_field(ielem,ilevel,igp,kgp)*get_derivative_c()->Dvv[kgp][jgp]*rrearth;
                grad_contra (ielem, ilevel, 1, igp, jgp) += mp(ielem,kgp,jgp)*scalar_field(ielem,ilevel,kgp,jgp)*get_derivative_c()->Dvv[kgp][igp]*rrearth;
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
              curl_field(ielem,ilevel,0,igp,jgp) = D(ielem,0,0,igp,jgp)*grad_contra(ielem,ilevel,0,igp,jgp) + D(ielem,0,1,igp,jgp)*grad_contra(ielem,ilevel,1,igp,jgp);
              curl_field(ielem,ilevel,1,igp,jgp) = D(ielem,1,0,igp,jgp)*grad_contra(ielem,ilevel,0,igp,jgp) + D(ielem,1,1,igp,jgp)*grad_contra(ielem,ilevel,1,igp,jgp);
            }
          }
        }
      );
    }
  );
}

void vlaplace_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                           const int& variable_viscosity, const real* nu_ratio,
                           real* const& vector_field_ptr, real*& lapl_weak_ptr)
{
  // Note: for each dimension, pass 'dim_length, dim_stride'
  Kokkos::LayoutStride layout_v(nelems, nlev*2*np*np,
                                nlev,   2*np*np,
                                2,      np*np,
                                np,     np,
                                np,     1);
  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field  (vector_field_ptr, layout_v);
  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> lapl_weak     (lapl_weak_ptr,    layout_v);

  vlaplace_sphere_wk_kokkos (nets, nete, nelems, variable_viscosity, nu_ratio, vector_field, lapl_weak);
}

void vlaplace_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                                const int& variable_viscosity, const real* nu_ratio,
                                Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field,
                                Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> lapl_weak)
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

void vlaplace_sphere_wk_cartesian_c (const int& nets, const int& nete, const int& nelems,
                                     const int& variable_viscosity,
                                     real* const& vector_field_ptr, real*& lapl_weak_ptr)
{
  Kokkos::View<real*[nlev][2][np][np]> vector_field (vector_field_ptr, nelems);
  Kokkos::View<real*[nlev][2][np][np]> lapl_weak    (lapl_weak_ptr,    nelems);

  vlaplace_sphere_wk_cartesian_kokkos (nets, nete, nelems, variable_viscosity, vector_field, lapl_weak);
}

void vlaplace_sphere_wk_cartesian_kokkos (const int& nets, const int& nete, const int& nelems,
                                          const int& variable_viscosity,
                                          Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field,
                                          Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> lapl_weak)
{
  Kokkos::View<real*[nlev][3][np][np]> vector_cart      ("tmp",                                  nelems);
  Kokkos::View<real*[3][2][np][np]>    vec_sphere2cart  (get_pointers_pool_c()->vec_sphere2cart, nelems);
  Kokkos::View<real*[np][np]>          spheremp         (get_pointers_pool_c()->spheremp,        nelems);
  Kokkos::View<real*[nlev][3][np][np]> lapl_cart        ("tmp_lapl",                             nelems);

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
                vector_cart(ielem,ilevel,icomp,igp,jgp) = vec_sphere2cart(ielem,icomp,0,igp,jgp)*vector_field(ielem,ilevel,0,igp,jgp)
                                                        + vec_sphere2cart(ielem,icomp,1,igp,jgp)*vector_field(ielem,ilevel,1,igp,jgp);
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
    Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride> vector_cart_i = Kokkos::subview(vector_cart, Kokkos::ALL(), Kokkos::ALL(), icomp, Kokkos::ALL(), Kokkos::ALL());
    Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride> lapl_cart_i   = Kokkos::subview(lapl_cart,   Kokkos::ALL(), Kokkos::ALL(), icomp, Kokkos::ALL(), Kokkos::ALL());

    laplace_sphere_wk_kokkos (nets, nete, nelems, variable_viscosity, vector_cart_i, lapl_cart_i);
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
              lapl_weak(ielem,ilevel,0,igp,jgp) = vec_sphere2cart(ielem,0,0,igp,jgp) * lapl_cart(ielem,ilevel,0,igp,jgp)
                                                + vec_sphere2cart(ielem,1,0,igp,jgp) * lapl_cart(ielem,ilevel,1,igp,jgp)
                                                + vec_sphere2cart(ielem,2,0,igp,jgp) * lapl_cart(ielem,ilevel,2,igp,jgp);
              lapl_weak(ielem,ilevel,1,igp,jgp) = vec_sphere2cart(ielem,0,1,igp,jgp) * lapl_cart(ielem,ilevel,0,igp,jgp)
                                                + vec_sphere2cart(ielem,1,1,igp,jgp) * lapl_cart(ielem,ilevel,1,igp,jgp)
                                                + vec_sphere2cart(ielem,2,1,igp,jgp) * lapl_cart(ielem,ilevel,2,igp,jgp);
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
              lapl_weak(ielem,ilevel,0,igp,jgp) += 2*spheremp(ielem,igp,jgp)*vector_field(ielem,ilevel,0,igp,jgp) * rrearth2;
              lapl_weak(ielem,ilevel,1,igp,jgp) += 2*spheremp(ielem,igp,jgp)*vector_field(ielem,ilevel,1,igp,jgp) * rrearth2;
            }
          }
        }
      );
    }
  );
}

void vlaplace_sphere_wk_contra_c (const int& nets, const int& nete, const int& nelems,
                                  const int& variable_viscosity, const real* nu_ratio,
                                  real* const& vector_field_ptr, real*& lapl_weak_ptr)
{
  // Note: for each dimension, pass 'dim_length, dim_stride'
  Kokkos::LayoutStride layout_v(nelems, nlev*2*np*np,
                                nlev,   2*np*np,
                                2,      np*np,
                                np,     np,
                                np,     1);
  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field (vector_field_ptr, layout_v);
  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> lapl_weak    (vector_field_ptr, layout_v);

  vlaplace_sphere_wk_contra_kokkos (nets, nete, nelems, variable_viscosity, nu_ratio, vector_field, lapl_weak);
}

void vlaplace_sphere_wk_contra_kokkos (const int& nets, const int& nete, const int& nelems,
                                       const int& variable_viscosity, const real* nu_ratio,
                                       Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field,
                                       Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> lapl_weak)
{
  Kokkos::View<real*[nlev][np][np]> div  ("divergence", nelems);
  Kokkos::View<real*[nlev][np][np]> vort ("vorticity",  nelems);

  divergence_sphere_kokkos (nets, nete, nelems, vector_field, div);
  vorticity_sphere_kokkos  (nets, nete, nelems, vector_field, vort);

  if (variable_viscosity!=0 && get_control_parameters_c()->hypervisc_power!=0)
  {
    Kokkos::View<real*[np][np]>  hyperviscosity  (get_pointers_pool_c()->hypervisc, nelems);

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
                div(ielem, ilevel, igp, jgp)  *= hyperviscosity(ielem, igp, jgp);
                vort(ielem, ilevel, igp, jgp) *= hyperviscosity(ielem, igp, jgp);
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
                div(ielem, ilevel, igp, jgp)  *= nu_ratio_val;
              }
            }
          }
        );
      }
    );
  }

  // Compute grad-div and curl-vort
  Kokkos::View<real*[nlev][2][np][np]> grad_div  ("grad_div",  nelems);
  Kokkos::View<real*[nlev][2][np][np]> curl_vort ("curl_vort", nelems);

  gradient_sphere_wk_testcov_kokkos (nets, nete, nelems, div, grad_div);
  curl_sphere_wk_testcov_kokkos     (nets, nete, nelems, vort, curl_vort);

  // Add grad_div and curl_vort, and adding correction so we don't damp rigid rotation
  double rrearth2 = std::pow(get_physical_constants_c()->rrearth,2);
  Kokkos::View<real*[np][np]> spheremp (get_pointers_pool_c()->spheremp, nelems);
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
              lapl_weak(ielem,ilevel,0,igp,jgp) = grad_div(ielem, ilevel, 0, igp, jgp) - curl_vort (ielem, ilevel, 0, igp, jgp) + 2*spheremp(ielem,igp,jgp)*vector_field(ielem,ilevel,0,igp,jgp) * rrearth2;
              lapl_weak(ielem,ilevel,1,igp,jgp) = grad_div(ielem, ilevel, 1, igp, jgp) - curl_vort (ielem, ilevel, 1, igp, jgp) + 2*spheremp(ielem,igp,jgp)*vector_field(ielem,ilevel,1,igp,jgp) * rrearth2;
            }
          }
        }
      );
    }
  );
}

} // extern "C"

} // Namespace Homme
