
#include "Tracers.hpp"

namespace Homme {

Tracers::Tracers(const int num_elems, const int num_tracers)
    : tracers("tracers structures", num_elems, num_tracers),
      buf("scalar tracers buffer", tracers.size() * Tracer::num_scalars()) {
  Scalar *mem_start = &buf.implementation_map().reference(0);
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int tracer = 0; tracer < num_tracers; ++tracer) {
      tracers(ie, tracer).qtens =
          ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>(mem_start);
      mem_start += NP * NP * NUM_LEV;

      tracers(ie, tracer).vstar_qdp =
          ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]>(mem_start);
      mem_start += 2 * NP * NP * NUM_LEV;

      tracers(ie, tracer).qlim =
          ExecViewUnmanaged<Scalar[2][NUM_LEV]>(mem_start);
      mem_start += 2 * NUM_LEV;

      tracers(ie, tracer).q =
          ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>(mem_start);
      mem_start += NP * NP * NUM_LEV;
    }
  }
}

} // namespace Homme
