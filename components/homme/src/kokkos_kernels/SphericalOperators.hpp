#ifndef HOMMEXX_SPHERICAL_OPERATORS_HPP
#define HOMMEXX_SPHERICAL_OPERATORS_HPP

namespace Homme
{

extern "C"
{

void gradient_sphere_c (const int& nets, const int& nete, const int& nelems,
                        real* const& scalar_field_ptr, real*& grad_field_ptr);
void gradient_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                             Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride> scalar_field,
                             Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> grad_field);

void divergence_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                             real* const& vector_field_ptr, real* weak_div_field_ptr);

void divergence_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                                  Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride>  vector_field,
                                  Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride>     weak_div_field);

void laplace_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                          const int& variable_viscosity,
                          real* const& scalar_field_ptr, real* weak_lapl_field_ptr);

void laplace_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                               const int& variable_viscosity,
                               Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride>  scalar_field,
                               Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride>  weak_lapl_field);

void vorticity_sphere_c (const int& nets, const int& nete, const int& nelems,
                         real* const& vector_field_ptr, real* vorticity_field_ptr);

void vorticity_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                              Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field,
                              Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride>    vorticity_field);

void divergence_sphere_c (const int& nets, const int& nete, const int& nelems,
                          real* const& vector_field_ptr, real*& div_field_ptr);

void divergence_sphere_kokkos (const int& nets, const int& nete, const int& nelems,
                               Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field,
                               Kokkos::View<real*[nlev][np][np], Kokkos::LayoutStride>    divergence_field);

void gradient_sphere_wk_testcov_c (const int& nets, const int& nete, const int& nelems,
                                   real* const& scalar_field_ptr, real*& grad_field);

void gradient_sphere_wk_testcov_kokkos (const int& nets, const int& nete, const int& nelems,
                                        Kokkos::View<real*[nlev][np][np]> scalar_field,
                                        Kokkos::View<real*[nlev][2][np][np]> grad_field);

void curl_sphere_wk_testcov_c (const int& nets, const int& nete, const int& nelems,
                               real* const& scalar_field_ptr, real*& curl_field);

void curl_sphere_wk_testcov_kokkos (const int& nets, const int& nete, const int& nelems,
                                    Kokkos::View<real*[nlev][np][np]>    scalar_field,
                                    Kokkos::View<real*[nlev][2][np][np]> curl_field);

void vlaplace_sphere_wk_c (const int& nets, const int& nete, const int& nelems,
                           const int& variable_viscosity, const real* nu_ratio,
                           real* const& vector_field_ptr, real*& lapl_weak_ptr);

void vlaplace_sphere_wk_kokkos (const int& nets, const int& nete, const int& nelems,
                                const int& variable_viscosity, const real* nu_ratio,
                                Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field,
                                Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> lapl_weak);

void vlaplace_sphere_wk_cartesian_c (const int& nets, const int& nete, const int& nelems,
                                     const int& variable_viscosity,
                                     real* const& vector_field_ptr, real*& lapl_weak_ptr);

void vlaplace_sphere_wk_cartesian_kokkos (const int& nets, const int& nete, const int& nelems,
                                          const int& variable_viscosity,
                                          Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field,
                                          Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> lapl_weak);

void vlaplace_sphere_wk_contra_c (const int& nets, const int& nete, const int& nelems,
                                  const int& variable_viscosity, const real* nu_ratio,
                                  real* const& vector_field_ptr, real*& lapl_weak_ptr);

void vlaplace_sphere_wk_contra_kokkos (const int& nets, const int& nete, const int& nelems,
                                       const int& variable_viscosity, const real* nu_ratio,
                                       Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> vector_field,
                                       Kokkos::View<real*[nlev][2][np][np], Kokkos::LayoutStride> lapl_weak);

} // extern "C"

} // Namespace Homme

#endif // HOMMEXX_SPHERICAL_OPERATORS_HPP
