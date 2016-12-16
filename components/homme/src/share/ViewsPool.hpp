#ifndef HOMMEXX_VIEWS_POOL_HPP
#define HOMMEXX_VIEWS_POOL_HPP

#include <kinds.hpp>
#include <Types.hpp>

namespace Homme
{

extern "C"
{

class ViewsPool
{
public:

  const HommeView3D<KMU> & get_elem_state_ps ()   const {return elem_state_ps;}
  const HommeView5D<KMU> & get_elem_state_p ()    const {return elem_state_p;}
  const HommeView6D<KMU> & get_elem_state_v ()    const {return elem_state_v;}
  const HommeView3D<KMU> & get_metdet ()          const {return metdet;}
  const HommeView3D<KMU> & get_rmetdet ()         const {return rmetdet;}
  const HommeView5D<KMU> & get_metinv ()          const {return metinv;}
  const HommeView3D<KMU> & get_mp ()              const {return mp;}
  const HommeView5D<KMU> & get_vec_sphere2cart () const {return vec_sphere2cart;}
  const HommeView3D<KMU> & get_spheremp ()        const {return spheremp;}
  const HommeView3D<KMU> & get_rspheremp ()       const {return rspheremp;}
  const HommeView3D<KMU> & get_hypervisc ()       const {return hypervisc;}
  const HommeView5D<KMU> & get_tensor_visc ()     const {return tensor_visc;}
  const HommeView5D<KMU> & get_D ()               const {return D;}
  const HommeView5D<KMU> & get_Dinv ()            const {return Dinv;}

  friend void init_views_pool_c (const int& num_elems,
                                 real* const& elem_state_ps_ptr, real* const& elem_state_p_ptr,
                                 real* const& elem_state_v_ptr,  real* const& metdet_ptr,
                                 real* const& rmetdet_ptr,       real* const& metinv_ptr,
                                 real* const& mp_ptr,            real* const& vec_sphere2cart_ptr,
                                 real* const& spheremp_ptr,      real* const& rspheremp_ptr,
                                 real* const& hypervisc_ptr,     real* const& tensor_visc_ptr,
                                 real* const& D_ptr,             real* const& Dinv_ptr);

  friend ViewsPool* get_views_pool_c ();
private:

  ViewsPool () = default;

  HommeView3D<KMU>   elem_state_ps;
  HommeView5D<KMU>   elem_state_p;
  HommeView6D<KMU>   elem_state_v;
  HommeView3D<KMU>   metdet;
  HommeView3D<KMU>   rmetdet;
  HommeView5D<KMU>   metinv;
  HommeView3D<KMU>   spheremp;
  HommeView3D<KMU>   rspheremp;
  HommeView3D<KMU>   mp;
  HommeView5D<KMU>   vec_sphere2cart;
  HommeView3D<KMU>   hypervisc;
  HommeView5D<KMU>   tensor_visc;
  HommeView5D<KMU>   D;
  HommeView5D<KMU>   Dinv;
};

ViewsPool* get_views_pool_c ();
void init_views_pool_c (const int& num_elems,
                        real* const& elem_state_ps_ptr, real* const& elem_state_p_ptr,
                        real* const& elem_state_v_ptr,  real* const& metdet_ptr,
                        real* const& rmetdet_ptr,       real* const& metinv_ptr,
                        real* const& mp_ptr,            real* const& vec_sphere2cart_ptr,
                        real* const& spheremp_ptr,      real* const& rspheremp_ptr,
                        real* const& hypervisc_ptr,     real* const& tensor_visc_ptr,
                        real* const& D_ptr,             real* const& Dinv_ptr);

} // extern "C"

} // Namespace Homme

#endif // HOMMEXX_VIEWS_POOL_HPP
