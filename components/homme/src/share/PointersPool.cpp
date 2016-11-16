#include <PointersPool.hpp>

namespace Homme
{

extern "C"
{

PointersPool* get_pointers_pool_c()
{
  static PointersPool p;
  return &p;
}

void init_pointers_pool_c (real* const& elem_state_ps_ptr, real* const& elem_state_p_ptr,
                           real* const& elem_state_v_ptr,  real* const& ptens_ptr,
                           real* const& vtens_ptr,         real* const& metdet_ptr,
                           real* const& rmetdet_ptr,       real* const& metinv_ptr,
                           real* const& mp_ptr,            real* const& vec_sphere2cart_ptr,
                           real* const& spheremp_ptr,      real* const& rspheremp_ptr,
                           real* const& hypervisc_ptr,     real* const& tensor_visc_ptr,
                           real* const& D_ptr,             real* const& Dinv_ptr)
{
  get_pointers_pool_c()->elem_state_ps   = elem_state_ps_ptr;
  get_pointers_pool_c()->elem_state_p    = elem_state_p_ptr;
  get_pointers_pool_c()->elem_state_v    = elem_state_v_ptr;
  get_pointers_pool_c()->ptens           = ptens_ptr;
  get_pointers_pool_c()->vtens           = vtens_ptr;
  get_pointers_pool_c()->metdet          = metdet_ptr;
  get_pointers_pool_c()->rmetdet         = rmetdet_ptr;
  get_pointers_pool_c()->metinv          = metinv_ptr;
  get_pointers_pool_c()->mp              = mp_ptr;
  get_pointers_pool_c()->vec_sphere2cart = vec_sphere2cart_ptr;
  get_pointers_pool_c()->spheremp        = spheremp_ptr;
  get_pointers_pool_c()->rspheremp       = rspheremp_ptr;
  get_pointers_pool_c()->hypervisc       = hypervisc_ptr;
  get_pointers_pool_c()->tensor_visc     = tensor_visc_ptr;
  get_pointers_pool_c()->D               = D_ptr;
  get_pointers_pool_c()->Dinv            = Dinv_ptr;
}

} // extern "C"

} // Namespace Homme
