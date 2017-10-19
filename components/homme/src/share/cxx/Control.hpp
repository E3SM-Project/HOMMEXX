#ifndef HOMMEXX_CAAR_CONTROL_HPP
#define HOMMEXX_CAAR_CONTROL_HPP

#include "Types.hpp"

#include <cstdlib>

namespace Homme {

struct Control {

  // This constructor should only be used by the host
  void init (const int nets, const int nete, const int num_elems,
             const int nm1,  const int n0,   const int np1,
             const int qn0,  const Real dt2, const Real ps0,
             const bool compute_diagonstics, const Real eta_ave_w,
             CRCPtr hybrid_a_ptr);

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

  // hybryd a
  ExecViewManaged<Real[NUM_LEV_P]> hybrid_a;
};

Control& get_control ();

} // Namespace Homme

#endif // HOMMEXX_CAAR_CONTROL_HPP
