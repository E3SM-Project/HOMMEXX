#ifndef HOMMEXX_CONTROL_HPP
#define HOMMEXX_CONTROL_HPP

#include "Types.hpp"

#include <cstdlib>

namespace Homme {

struct Control {
  struct DSSOption {
    enum Enum { eta = 1, omega, div_vdp_ave };
    static Enum from(int);
  };

  Control() = default;

  // This constructor should only be used by the host
  void init (const int nets, const int nete, const int num_elems,
             const int qn0,  const Real ps0, 
             const int rsplit,
             CRCPtr hybrid_am_ptr,
             CRCPtr hybrid_ai_ptr,
             CRCPtr hybrid_bm_ptr,
             CRCPtr hybrid_bi_ptr);

  void random_init(int num_elems, int seed);

  void set_rk_stage_data(const int nm1, const int n0, const int np1,
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
  int np1_qdp;

  // Tracers options;
  DSSOption::Enum DSSopt;
  Real nu_q;
  int rhs_viss, rhs_multiplier;
  int limiter_option; // we handle = 8

  // Hyperviscosity options
  Real hypervis_scaling;
  Real nu_top;
  Real nu_ratio;
  Real nu, nu_p, nu_s;

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
  //ExecViewManaged<Real[NUM_PHYSICAL_LEV]> hybrid_bm;

  Real hybrid_ai0;
  // hybrid ai
  ExecViewManaged<Real[NUM_INTERFACE_LEV]> hybrid_ai;
  // hybrid bi
  ExecViewManaged<Real[NUM_INTERFACE_LEV]> hybrid_bi;
  ExecViewManaged<Scalar[NUM_LEV]> dp0;
};

} // Namespace Homme

#endif // HOMMEXX_CONTROL_HPP
