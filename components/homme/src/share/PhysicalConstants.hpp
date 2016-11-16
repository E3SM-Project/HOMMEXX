#ifndef HOMMEXX_PHYSICAL_CONSTANTS_HPP
#define HOMMEXX_PHYSICAL_CONSTANTS_HPP

#include <kinds.hpp>

namespace Homme
{

extern "C"
{

struct PhysicalConstants
{
  real rearth        ;
  real g             ;
  real omega         ;
  real Rgas          ;
  real Cp            ;
  real p0            ;
  real MWDAIR        ;
  real Rwater_vapor  ;
  real Cpwater_vapor ;
  real kappa         ;
  real Rd_on_Rv      ;
  real Cpd_on_Cpv    ;
  real rrearth       ;
  real Lc            ;
  real pi            ;
};

PhysicalConstants* get_physical_constants_c ();

} // extern "C"

} // Namespace Homme

#endif // HOMMEXX_PHYSICAL_CONSTANTS_HPP
