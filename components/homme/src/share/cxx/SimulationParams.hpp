#ifndef HOMMEXX_SIMULATION_PARAMS_HPP
#define HOMMEXX_SIMULATION_PARAMS_HPP

namespace Homme
{

enum class MoistDry {
  MOIST,
  DRY
};

enum class RemapAlg {
  PPM_FIXED,
  PPM_MIRRORED
};

enum class TestCase {
  ASP_BAROCLINIC,
  ASP_GRAVITY_WAVE,
  ASP_MOUNTAIN,
  ASP_ROSSBY,
  ASP_TRACER,
  BAROCLINIC,
  DCMIP2012_TEST1_1,
  DCMIP2012_TEST1_2,
  DCMIP2012_TEST1_3,
  DCMIP2012_TEST2_0,
  DCMIP2012_TEST2_1,
  DCMIP2012_TEST2_2,
  DCMIP2012_TEST3,
  HELD_SUAREZ0,
  JW_BAROCLINIC
};

/*
 * A struct to hold simulation parameters.
 *
 * This differs from the 'Control' structure, in that these parameter do not generally
 * need to be fwd-ed to the device, but rather they are used to do setup and to decide
 * which kernel have to be dispatched. Some *may* also be needed inside kernels, so they
 * will appear also inside Control (such as rsplit and qsplit), but in general, these
 * parameters are used only on host.
 * TODO: comments!!!!
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
  double    nu_s;
  double    nu_p;
  int       hypervis_order;
  int       hypervis_subcycle;

  double    ur_weight;

  // Use this member to check whether the struct has been initialized
  bool      params_set;
};

} // namespace Homme

#endif // HOMMEXX_SIMULATION_PARAMS_HPP
