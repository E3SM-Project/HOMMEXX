#ifndef HOMMEXX_TRACERS_HPP
#define HOMMEXX_TRACERS_HPP

#include "Types.hpp"

namespace Homme {

struct Tracers {
  Tracers() = default;
  Tracers(const int num_elems, const int num_tracers);

  void random_init();

  void pull_qdp(CF90Ptr &state_qdp);
  void push_qdp(F90Ptr &state_qdp) const;

  KOKKOS_INLINE_FUNCTION
  int num_tracers() const {
    return nt;
  }

  ExecViewManaged<Scalar*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]> qdp;
  ExecViewManaged<Scalar*[QSIZE_D][NP][NP][NUM_LEV]> qtens_biharmonic; // Also doubles as just qtens.
  ExecViewManaged<Scalar*[QSIZE_D][2][NUM_LEV]> qlim;
  ExecViewManaged<Scalar*[QSIZE_D][NP][NP][NUM_LEV]> q;

private:
  int nt;
};

} // namespace Homme

#endif // HOMMEXX_TRACERS_HPP