#ifndef HOMMEXX_CONTROL_HPP
#define HOMMEXX_CONTROL_HPP

#include "Types.hpp"

namespace Homme {

struct Control {

  Control() = default;

  // This method should only be called from the host
  void init_hvcoord(const Real ps0_in,
                    CRCPtr hybrid_am_ptr,
                    CRCPtr hybrid_ai_ptr,
                    CRCPtr hybrid_bm_ptr,
                    CRCPtr hybrid_bi_ptr);

  // This constructor should only be used by the host
  void init (const int num_elems, const int n0_qdp,  const int rsplit);

  void random_init(int num_elems, int seed);

  void set_rk_stage_data(const int nm1, const int n0, const int np1,
                         const Real dt, const Real eta_ave_w,
                         const bool compute_diagonstics);

  // The number of elements on this rank
  int num_elems;

  // States time levels indices
  int n0;
  int nm1;
  int np1;

  // Tracers timelevel, inclusive range of 0-1
  // or time level for moist temp?
  int n0_qdp;
  int np1_qdp;

  // Tracers options;
  Real nu_q;
  Real rhs_viss, rhs_multiplier;
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
  ExecViewManaged<Scalar[NUM_LEV]> hybrid_ai_delta;
  // hybrid bi
  ExecViewManaged<Real[NUM_INTERFACE_LEV]> hybrid_bi;
  ExecViewManaged<Scalar[NUM_LEV]> hybrid_bi_delta;
  ExecViewManaged<Scalar[NUM_LEV]> dp0;
};

} // Namespace Homme

#endif // HOMMEXX_CONTROL_HPP
