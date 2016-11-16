#ifndef HOMMEXX_POINTERS_POOL_HPP
#define HOMMEXX_POINTERS_POOL_HPP

#include <kinds.hpp>

namespace Homme
{

extern "C"
{

struct PointersPool
{
  real* elem_state_ps;
  real* elem_state_p;
  real* elem_state_v;
  real* ptens;
  real* vtens;
  real* metdet;
  real* rmetdet;
  real* metinv;
  real* mp;
  real* vec_sphere2cart;
  real* spheremp;
  real* rspheremp;
  real* hypervisc;
  real* tensor_visc;
  real* D;
  real* Dinv;
};

PointersPool* get_pointers_pool_c ();

} // extern "C"

} // Namespace Homme

#endif // HOMMEXX_POINTERS_POOL_HPP
