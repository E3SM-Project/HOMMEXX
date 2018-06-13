#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

module prim_advection_mod
#ifndef USE_KOKKOS_KERNELS
  use prim_advection_mod_base, only: vertical_remap_interface, Prim_Advec_Tracers_remap, Prim_Advec_Tracers_remap_rk2
#endif
  use prim_advection_mod_base, only: prim_advec_init1, prim_advec_init2, prim_advec_init_deriv, deriv
  implicit none
end module prim_advection_mod
