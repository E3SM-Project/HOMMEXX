#ifndef HOMMEXX_CONTROL_PARAMETERS_HPP
#define HOMMEXX_CONTROL_PARAMETERS_HPP

#include <kinds.hpp>

namespace Homme
{

extern "C"
{

struct ControlParameters
{
  bool hypervisc_scaling;
  real hypervisc_power;
};

ControlParameters* get_control_parameters_c ();

} // extern "C"

} // Namespace Homme

#endif // HOMMEXX_CONTROL_PARAMETERS_HPP
