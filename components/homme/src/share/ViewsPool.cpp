#include <ViewsPool.hpp>

#include <dimensions.hpp>

namespace Homme
{

extern "C"
{

ViewsPool* get_views_pool_c()
{
  static ViewsPool pool;
  return &pool;
}

void init_views_pool_c (const int& num_elems,
                        real* const& elem_state_ps_ptr, real* const& elem_state_p_ptr,
                        real* const& elem_state_v_ptr,  real* const& metdet_ptr,
                        real* const& rmetdet_ptr,       real* const& metinv_ptr,
                        real* const& mp_ptr,            real* const& vec_sphere2cart_ptr,
                        real* const& spheremp_ptr,      real* const& rspheremp_ptr,
                        real* const& hypervisc_ptr,     real* const& tensor_visc_ptr,
                        real* const& D_ptr,             real* const& Dinv_ptr)
{
  get_views_pool_c()->elem_state_ps   = HommeView3D<KMU> (elem_state_ps_ptr,   np, np, num_elems);
  get_views_pool_c()->elem_state_p    = HommeView5D<KMU> (elem_state_p_ptr,    np, np, nlev, timelevels, num_elems);
  get_views_pool_c()->elem_state_v    = HommeView6D<KMU> (elem_state_v_ptr,    np, np, 2, nlev, timelevels, num_elems);
  get_views_pool_c()->metdet          = HommeView3D<KMU> (metdet_ptr,          np, np, num_elems);
  get_views_pool_c()->rmetdet         = HommeView3D<KMU> (rmetdet_ptr,         np, np, num_elems);
  get_views_pool_c()->metinv          = HommeView5D<KMU> (metinv_ptr,          np, np, 2, 2, num_elems);
  get_views_pool_c()->spheremp        = HommeView3D<KMU> (spheremp_ptr,        np, np, num_elems);
  get_views_pool_c()->rspheremp       = HommeView3D<KMU> (rspheremp_ptr,       np, np, num_elems);
  get_views_pool_c()->mp              = HommeView3D<KMU> (mp_ptr,              np, np, num_elems);
  get_views_pool_c()->vec_sphere2cart = HommeView5D<KMU> (vec_sphere2cart_ptr, np, np, 3, 2, num_elems);
  get_views_pool_c()->hypervisc       = HommeView3D<KMU> (hypervisc_ptr,       np, np, num_elems);
  get_views_pool_c()->tensor_visc     = HommeView5D<KMU> (tensor_visc_ptr,     np, np, 2, 2, num_elems);
  get_views_pool_c()->D               = HommeView5D<KMU> (D_ptr,               np, np, 2, 2, num_elems);
  get_views_pool_c()->Dinv            = HommeView5D<KMU> (Dinv_ptr,            np, np, 2, 2, num_elems);
}

} // extern "C"

} // Namespace Homme
