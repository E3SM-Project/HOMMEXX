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
  void init(const int nets, const int nete, const int num_elems,
            const int qn0_in, const Real ps0_in, int rsplit_in,
            CRCPtr hybrid_a_ptr, CRCPtr hybrid_b_ptr);

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
  int qn0;
  int np1_qdp;

  // Tracers options;
  DSSOption::Enum DSSopt;
  Real nu_p, nu_q;
  int rhs_viss, rhs_multiplier;
  int limiter_option; // we handle = 8

  // Hyperviscosity options
  Real hypervis_scaling;
  Real nu;
  Real nu_s;

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

  // hybrid a
  ExecViewManaged<Real[NUM_INTERFACE_LEV]> hybrid_a;
  // hybrid b
  ExecViewManaged<Real[NUM_INTERFACE_LEV]> hybrid_b;
  ExecViewManaged<Scalar[NUM_LEV]> dp0;
};

} // Namespace Homme

#endif // HOMMEXX_CONTROL_HPP
