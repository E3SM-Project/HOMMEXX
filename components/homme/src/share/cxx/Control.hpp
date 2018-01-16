#ifndef HOMMEXX_CONTROL_HPP
#define HOMMEXX_CONTROL_HPP

#include "Types.hpp"

#include <cstdlib>

namespace Homme {

struct Control {

  Control () = default;

  // This constructor should only be used by the host
  void init (const int nets, const int nete, const int num_elems,
             const int qn0,  const Real ps0, 
             const int rsplit,
             CRCPtr hybrid_am_ptr,
             CRCPtr hybrid_ai_ptr,
             CRCPtr hybrid_bm_ptr,
             CRCPtr hybrid_bi_ptr);

  void set_rk_stage_data (const int nm1, const int n0,   const int np1,
                          const Real dt, const Real eta_ave_w,
                          const bool compute_diagonstics);
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
  // or time level for moist temp?
  int qn0;

  // Number of tracers (may be lower than QSIZE_D)
  int qsize;

  // Time step
  // OG for dynamics?
  Real dt;

  // Weight for eta_dot_dpdn mean flux
  Real eta_ave_w;

  int compute_diagonstics;

  Real ps0;

  // For vertically lagrangian dynamics,
  // apply remap every rsplit tracer timesteps
  int rsplit;

  // hybrid coefficients
  //ExecViewManaged<Real[NUM_PHYSICAL_LEV]> hybrid_am;
  ExecViewManaged<Real[NUM_PHYSICAL_LEV+1]> hybrid_ai;
  //ExecViewManaged<Real[NUM_PHYSICAL_LEV]> hybrid_bm;
  ExecViewManaged<Real[NUM_PHYSICAL_LEV+1]> hybrid_bi;

};

} // Namespace Homme

#endif // HOMMEXX_CONTROL_HPP
