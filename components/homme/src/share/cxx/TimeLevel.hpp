#ifndef HOMMEXX_TIME_LEVEL_HPP
#define HOMMEXX_TIME_LEVEL_HPP

#include "mpi/ErrorDefs.hpp"

namespace Homme
{

enum class UpdateType {
  LEAPFROG,
  FORWARD
};

struct TimeLevel
{
  // Dynamics relative time levels
  int nm1;
  int n0;
  int np1;

  // Absolute time level since simulation start
  int nstep;

  // Absolute time level of first complete leapfrog timestep
  int nstep0;

  // Tracers relative time levels
  int n0_qdp;
  int np1_qdp;

  // Time passed since the start of the simulation
  // TODO: I think this is used only with when CAM is defined
  double tevolve;

  void update_dynamics_levels (UpdateType type) {
    int tmp;
    switch(type) {
      case UpdateType::LEAPFROG:
        tmp = np1;
        np1 = nm1;
        nm1 = n0;
        n0  = nm1;
        break;
      case UpdateType::FORWARD:
        tmp = np1;
        np1 = n0;
        n0  = tmp;
        break;
      default:
        Errors::runtime_abort("Unknown time level update type",
                              Errors::unknown_option);
    }
    ++nstep;
  }

  void update_tracers_levels (const int qsplit) {
    int i_temp = nstep/qsplit;
    if (i_temp%2 == 0) {
      n0_qdp  = 0;
      np1_qdp = 1;
    } else {
      n0_qdp  = 1;
      np1_qdp = 0;
    }
  }
};

} // namespace Homme

#endif // HOMMEXX_TIME_LEVEL_HPP
