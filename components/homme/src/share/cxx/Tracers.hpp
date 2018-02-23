#ifndef HOMMEXX_TRACERS_HPP
#define HOMMEXX_TRACERS_HPP

#include "Types.hpp"

namespace Homme {

class Tracers {
public:
  Tracers() = default;
  Tracers(const int num_elems, const int num_tracers);

  void random_init();

  void pull_qdp(CF90Ptr &state_qdp);
  void push_qdp(F90Ptr &state_qdp) const;

  struct Tracer {
    static constexpr size_t num_scalars() {
      return ((1 + 2) * (NP * NP) + 2) * NUM_LEV;
    }
    ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> qtens;
    ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> vstar_qdp;
    ExecViewUnmanaged<Scalar[2][NUM_LEV]> qlim;
  };

  HostViewUnmanaged<const Tracer **> tracers() const {
    return m_h_tracers;
  }

  // This should only be called from a kernel
  KOKKOS_INLINE_FUNCTION
  const Tracer &tracer(const int ie, const int iq) const {
    return m_tracers(ie, iq);
  }

  KOKKOS_INLINE_FUNCTION
  int num_tracers() const {
    return m_tracers.extent_int(1);
  }

  ExecViewManaged<Scalar * [QSIZE_D][NP][NP][NUM_LEV]> m_q;
  ExecViewManaged<Scalar * [Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]> m_qdp;
  ExecViewManaged<Scalar * [QSIZE_D][NP][NP][NUM_LEV]> qtens_biharmonic;

private:
  ExecViewManaged<Tracer **> m_tracers;
  HostViewManaged<Tracer **> m_h_tracers;
  ExecViewManaged<Scalar *> buf;
};

} // namespace Homme

#endif // HOMMEXX_TRACERS_HPP
