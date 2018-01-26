#ifndef HOMMEXX_SIMULATION_PARAMS_HPP
#define HOMMEXX_SIMULATION_PARAMS_HPP

namespace Homme
{

enum class MoistDry {
  Moist,
  Dry
};

/*
 * A struct to hold simulation parameters.
 *
 * This differs from the 'Control' structure, in that these parameter do not generally
 * need to be fwd-ed to the device, but rather they are used to do setup and to decide
 * which kernel have to be dispatched. Some *may* also be needed inside kernels, so they
 * will appear also inside Control.
 */
struct SimulationParams
{
  int       time_step_type;
  int       rsplit;
  int       qsplit;
  MoistDry  moisture;

  bool      prescribed_wind;

  double    time_step;

  int       state_frequency;
  bool      energy_fixer;
  bool      disable_diagnostics;

  double    nu_p;
  double    ur_weight;
};

} // namespace Homme

#endif // HOMMEXX_SIMULATION_PARAMS_HPP
