#ifndef HOMMEXX_TRACERS_HPP
#define HOMMEXX_TRACERS_HPP

#include "Types.hpp"

namespace Homme {

class Tracers {
public:
  Tracers(const int num_elems, const int num_tracers);

  void pull_qdp(CF90Ptr &state_qdp);
  void push_qdp(F90Ptr &state_qdp) const;

  struct Tracer {
    static constexpr size_t num_scalars() {
      return ((2 + 2) * (NP * NP) + 2) * NUM_LEV;
    }
    ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> qtens;
    ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> vstar_qdp;
    ExecViewUnmanaged<Scalar[2][NUM_LEV]> qlim;
    ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> q;
  };

private:
  ExecViewManaged<Tracer **> tracers;
  ExecViewManaged<Scalar *> buf;
};

} // namespace Homme

#endif // HOMMEXX_TRACERS_HPP
