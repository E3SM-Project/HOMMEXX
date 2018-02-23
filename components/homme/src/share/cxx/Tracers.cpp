
#include <random>

#include "Tracers.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/TestUtils.hpp"

namespace Homme {

Tracers::Tracers(const int num_elems, const int num_tracers)
    : m_tracers("tracers structures", num_elems, num_tracers),
      m_h_tracers(Kokkos::create_mirror_view(m_tracers)),
      buf("scalar tracers buffer", m_tracers.size() * Tracer::num_scalars()) {
  Scalar *const mem_start = &buf.implementation_map().reference(0);
  size_t mem_pos = 0;
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int tracer = 0; tracer < num_tracers; ++tracer) {
      auto &t = m_h_tracers(ie, tracer);
      assert(mem_pos < buf.size());
      t.qtens = ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>(mem_start + mem_pos);
      mem_pos += t.qtens.size();

      assert(mem_pos < buf.size());
      t.vstar_qdp = ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]>(mem_start + mem_pos);
      mem_pos += t.vstar_qdp.size();

      assert(mem_pos < buf.size());
      t.qlim = ExecViewUnmanaged<Scalar[2][NUM_LEV]>(mem_start + mem_pos);
      mem_pos += t.qlim.size();

      assert(mem_pos < buf.size());
      t.q = ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>(mem_start + mem_pos);
      mem_pos += t.q.size();

      assert(mem_pos < buf.size());
      t.qtens_biharmonic = ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>(mem_start + mem_pos);
      mem_pos += t.qtens_biharmonic.size();

      assert(mem_pos <= buf.size());
    }
  }
  Kokkos::deep_copy(m_tracers, m_h_tracers);
}

void Tracers::random_init() {
  constexpr Real min_value = 0.015625;
  std::random_device rd;
  std::mt19937_64 engine(rd());
  std::uniform_real_distribution<Real> random_dist(min_value, 1.0 / min_value);

  genRandArray(buf, engine, random_dist);
}

void Tracers::pull_qdp(CF90Ptr &state_qdp) {
  HostViewUnmanaged<
      const Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>
  state_qdp_f90(state_qdp, m_tracers.extent_int(0));
  sync_to_device(state_qdp_f90, m_qdp);
}

void Tracers::push_qdp(F90Ptr &state_qdp) const {
  HostViewUnmanaged<
      Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>
  state_qdp_f90(state_qdp, m_tracers.extent_int(0));
  sync_to_host(m_qdp, state_qdp_f90);
}

} // namespace Homme
