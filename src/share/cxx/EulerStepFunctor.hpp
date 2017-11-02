#ifndef HOMMEXX_EULER_STEP_FUNCTOR_HPP
#define HOMMEXX_EULER_STEP_FUNCTOR_HPP

#include "Elements.hpp"
#include "Derivative.hpp"
#include "Control.hpp"
#include "SphereOperators.hpp"

namespace Homme
{

struct EulerStepFunctor
{
  const Control     m_data;
  const Elements    m_elements;
  const Derivative  m_deriv;

  EulerStepFunctor (const Control& data)
   : m_data    (data)
   , m_elements(Context::singleton().get_elements())
   , m_deriv   (Context::singleton().get_derivative())
  {
    // Nothing to be done here
  }

  KOKKOS_INLINE_FUNCTION
  static size_t shmem_size(int /*team_size*/) {
    // One scalar buffer and two vector buffers
    return 0;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TeamMember& team) const {
    KernelVariables kv(team);

    using Kokkos::ALL;
    const ExecViewUnmanaged<const Real[NP][NP]>
      metdet = Kokkos::subview(m_elements.m_metdet, kv.ie, ALL, ALL);

    const auto NP2 = NP * NP;
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(team, NP2*m_data.qsize),
      [&] (const int loop_idx) {
        const int iq  = loop_idx / NP2;
        const int igp = (loop_idx % NP2) / NP;
        const int jgp = (loop_idx % NP2) % NP;

        const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]>
          qdp   = Kokkos::subview(m_elements.m_qdp, kv.ie, m_data.qn0, iq, ALL, ALL, ALL);
        const ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>
          q_buf = Kokkos::subview(m_elements.buffers.qtens, kv.ie, iq, ALL, ALL, ALL);
        const ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]>
          v_buf = Kokkos::subview(m_elements.buffers.vstar_qdp, kv.ie, iq, ALL, ALL, ALL, ALL);

        for (int ilev = 0; ilev < NUM_LEV; ++ilev) {
          v_buf(0,igp,jgp,ilev) = (m_elements.buffers.vstar(kv.ie, 0, igp, jgp, ilev) *
                                   qdp(igp, jgp, ilev));
          v_buf(1,igp,jgp,ilev) = (m_elements.buffers.vstar(kv.ie, 1, igp, jgp, ilev) *
                                   qdp(igp, jgp, ilev));
          q_buf(igp,jgp,ilev) = qdp(igp,jgp,ilev);
        }
      }
    );
    kv.team_barrier();

    const auto dvv = m_deriv.get_dvv();
    const ExecViewUnmanaged<const Real[2][2][NP][NP]>
      dinv = Kokkos::subview(m_elements.m_dinv, kv.ie, ALL, ALL, ALL, ALL);
    for (int iq = 0; iq < m_data.qsize; ++iq)
      divergence_sphere_update(
        kv, -m_data.dt, 1.0, dinv, metdet, dvv,
        Kokkos::subview(m_elements.buffers.vstar_qdp, kv.ie, iq, ALL, ALL, ALL, ALL),
        m_elements.buffers.vdp,
        Kokkos::subview(m_elements.buffers.qtens, kv.ie, iq, ALL, ALL, ALL));
  }

};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
