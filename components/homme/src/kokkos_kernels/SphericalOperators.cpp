#include <kinds.hpp>
#include <Kokkos_Core.hpp>
#include <Types.hpp>

#include <fortran_binding.hpp>
#include <SphericalOperators.hpp>

namespace Homme
{

extern "C"
{

// ======================= DIFFERENTIAL OPERATIONS ON SPHERE ===================================== //

void gradient_sphere_c (const int& nets, const int& nete, const int& nelems,
                        real* const& scalar_field_ptr, real*& grad_field_ptr)
{
  HommeView4D<KMU> scalar_field (scalar_field_ptr, np, np, nlev, nelems);
  HommeView5D<KMU> grad_field   (grad_field_ptr,   np, np, 2, nlev, nelems);

  gradient_sphere_kokkos (nets, nete, nelems, scalar_field, grad_field);
}

void divergence_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                             real* const& vector_field_ptr, real*& weak_div_field_ptr)
{
  HommeView5D<KMU> vector_field   (vector_field_ptr,   np, np, 2, nlev, nelems);
  HommeView4D<KMU> weak_div_field (weak_div_field_ptr, np, np, nlev, nelems);

  divergence_sphere_wk_kokkos (nets, nete, nelems, vector_field, weak_div_field);
}

void laplace_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                          const int& variable_viscosity,
                          real* const& scalar_field_ptr, real*& weak_lapl_field_ptr)
{
  HommeView4D<KMU> scalar_field    (scalar_field_ptr,    np, np, nlev, nelems);
  HommeView4D<KMU> weak_lapl_field (weak_lapl_field_ptr, np, np, nlev, nelems);

  laplace_sphere_wk_kokkos (nets, nete, nelems, variable_viscosity, scalar_field, weak_lapl_field);
}

void vorticity_sphere_c (const int& nets, const int& nete, const int& nelems,
                         real* const& vector_field_ptr, real*& vorticity_field_ptr)
{
  HommeView5D<KMU> vector_field    (vector_field_ptr,    np, np, 2, nlev, nelems);
  HommeView4D<KMU> vorticity_field (vorticity_field_ptr, np, np, nlev, nelems);

  vorticity_sphere_kokkos (nets, nete, nelems, vector_field, vorticity_field);
}

void divergence_sphere_c (const int& nets, const int& nete, const int& nelems,
                          real* const& vector_field_ptr, real*& div_field_ptr)
{
  HommeView5D<KMU> vector_field     (vector_field_ptr, np, np, 2, nlev, nelems);
  HommeView4D<KMU> divergence_field (div_field_ptr,    np, np, nlev, nelems);

  divergence_sphere_kokkos (nets, nete, nelems, vector_field, divergence_field);
}

void gradient_sphere_wk_testcov_c (const int& nets, const int& nete, const int& nelems,
                                   real* const& scalar_field_ptr, real*& grad_field_ptr)
{
  HommeView4D<KMU> scalar_field (scalar_field_ptr, np, np, nlev, nelems);
  HommeView5D<KMU> grad_field   (grad_field_ptr,   np, np, 2, nlev, nelems);

  gradient_sphere_wk_testcov_kokkos (nets, nete, nelems, scalar_field, grad_field);
}

void curl_sphere_wk_testcov_c (const int& nets, const int& nete, const int& nelems,
                               real* const& scalar_field_ptr, real*& curl_field_ptr)
{
  HommeView4D<KMU> scalar_field (scalar_field_ptr, np, np, nlev, nelems);
  HommeView5D<KMU> curl_field   (curl_field_ptr,   np, np, 2, nlev, nelems);

  curl_sphere_wk_testcov_kokkos (nets, nete, nelems, scalar_field, curl_field);
}

void vlaplace_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                           const int& variable_viscosity, const real* nu_ratio,
                           real* const& vector_field_ptr, real*& lapl_weak_ptr)
{
  HommeView5D<KMU> vector_field  (vector_field_ptr, np, np, 2, nlev, nelems);
  HommeView5D<KMU> lapl_weak     (lapl_weak_ptr,    np, np, 2, nlev, nelems);

  vlaplace_sphere_wk_kokkos (nets, nete, nelems, variable_viscosity, nu_ratio, vector_field, lapl_weak);
}

void vlaplace_sphere_wk_cartesian_c (const int& nets, const int& nete, const int& nelems,
                                     const int& variable_viscosity,
                                     real* const& vector_field_ptr, real*& lapl_weak_ptr)
{
  HommeView5D<KMU> vector_field (vector_field_ptr, np, np, 2, nlev, nelems);
  HommeView5D<KMU> lapl_weak    (lapl_weak_ptr,    np, np, 2, nlev, nelems);

  vlaplace_sphere_wk_cartesian_kokkos (nets, nete, nelems, variable_viscosity, vector_field, lapl_weak);
}

void vlaplace_sphere_wk_contra_c (const int& nets, const int& nete, const int& nelems,
                                  const int& variable_viscosity, const real* nu_ratio,
                                  real* const& vector_field_ptr, real*& lapl_weak_ptr)
{
  HommeView5D<KMU> vector_field (vector_field_ptr, np, np, 2, nlev, nelems);
  HommeView5D<KMU> lapl_weak    (vector_field_ptr, np, np, 2, nlev, nelems);

  vlaplace_sphere_wk_contra_kokkos (nets, nete, nelems, variable_viscosity, nu_ratio, vector_field, lapl_weak);
}

} // extern "C"

} // Namespace Homme
