#ifndef HOMMEXX_CONTROL_HPP
#define HOMMEXX_CONTROL_HPP

#include "Types.hpp"

#include <cstdlib>

namespace Homme {

struct Control {

  Control ()
  {
    // We start by setting
    team_size         = 1;
    default_team_size = 1;

    const char* var;
    var = getenv("OMP_NUM_THREADS");
    if (var!=0)
    {
      // the team size canno exceed the value of OMP_NUM_THREADS, so se note it down
      default_team_size = std::atoi(var);
    }

    var = getenv("HOMMEXX_TEAM_SIZE");
    if (var!=0)
    {
      // The user requested a team size for homme. We accept it, provided that
      // it does not exceed the value of OMP_NUM_THREADS. If it does exceed that,
      // we simply set it to OMP_NUM_THREADS.
      default_team_size = std::min(std::atoi(var),default_team_size);
    }
  }

  // This constructor should only be used by the host
  void init (const int nets, const int nete, const int num_elems,
             const int nm1,  const int n0,   const int np1,
             const int qn0,  const Real dt2, const Real ps0,
             const bool compute_diagonstics, const Real eta_ave_w,
             CRCPtr hybrid_a_ptr);

  // This method sets team_size if it wasn't already set via environment variable in the constructor
  void set_team_size ();

  // The desired team size for kokkos team policy
  int default_team_size;

  // The desired team size for kokkos team policy
  int team_size;

  // Range of element indices to be handled by this thread is [nets,nete)
  int nets;
  int nete;

  // The number of elements on this rank
  int num_elems;

  // States time levels indices
  int n0;
  int nm1;
  int np1;

  // Tracers timelevel, inclusive range of 0-1
  int qn0;

  // Number of tracers (may be lower than QSIZE_D)
  int qsize;

  // Time step
  Real dt;

  // Weight for eta_dot_dpdn mean flux
  Real eta_ave_w;

  int compute_diagonstics;

  Real ps0;

  // For vertically lagrangian dynamics,
  // apply remap every rsplit tracer timesteps
  int rsplit;

  // hybrid coefficients
  ExecViewManaged<Real[NUM_LEV_P]> hybrid_am;
  ExecViewManaged<Real[NUM_LEV_P+1]> hybrid_ai;
  ExecViewManaged<Real[NUM_LEV_P]> hybrid_bm;
  ExecViewManaged<Real[NUM_LEV_P+1]> hybrid_bi;

};

} // Namespace Homme

#endif // HOMMEXX_CONTROL_HPP
