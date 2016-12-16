#include <ControlParameters.hpp>

namespace Homme
{

extern "C"
{

ControlParameters* get_control_parameters_c ()
{
  static ControlParameters cp;
  return &cp;
}

void init_control_parameters_c (const int& hypervisc_scaling, const double& hypervisc_power)
{
  get_control_parameters_c()->hypervisc_scaling  = !(hypervisc_scaling==0);
  get_control_parameters_c()->hypervisc_power    = hypervisc_power;
}

} // extern "C"

} // Namespace Homme
