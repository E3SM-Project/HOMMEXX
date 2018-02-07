#ifndef HOMMEXX_SIMULATION_PARAMS_HPP
#define HOMMEXX_SIMULATION_PARAMS_HPP

#include "HommexxEnums.hpp"

namespace Homme
{

/*
 * A struct to hold simulation parameters.
 *
 * This differs from the 'Control' structure, in that these parameter do not generally
 * need to be fwd-ed to the device, but rather they are used to do setup and to decide
 * which kernel have to be dispatched. Some *may* also be needed inside kernels, so they
 * will appear also inside Control (such as rsplit and qsplit); you can think of this
 * struct as the 'parameters needed on host', and Control as the 'parameter needed
 * on device' (or inside kernels, in general).
 * TODO: comments for each option!!!!
 */
struct SimulationParams
{
  SimulationParams() : params_set(false) {}

  int       time_step_type; // TODO: convert to enum
  int       rsplit;
  int       qsplit;
  int       qsize;

  MoistDry  moisture;
  RemapAlg  remap_alg;
  TestCase  test_case;

  int       limiter_option; // TODO: convert to enum

  bool      prescribed_wind;

  double    time_step;

  int       state_frequency;
  bool      energy_fixer;
  bool      disable_diagnostics;
  bool      use_semi_lagrangian_transport;

  double    nu;
  double    nu_p;
  double    nu_q;
  double    nu_s;
  double    nu_top;
  double    nu_div;
  int       hypervis_order;
  int       hypervis_subcycle;
  double    hypervis_scaling;

  double    ur_weight;

  // Use this member to check whether the struct has been initialized
  bool      params_set;
};

} // namespace Homme

#endif // HOMMEXX_SIMULATION_PARAMS_HPP
