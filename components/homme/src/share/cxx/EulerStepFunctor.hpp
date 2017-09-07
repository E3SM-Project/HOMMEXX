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
  const Elements      m_elements;
  const Derivative  m_deriv;

  EulerStepFunctor (const Control& data)
   : m_data    (data)
   , m_elements  (get_elements())
   , m_deriv   (get_derivative())
  {
    // Nothing to be done here
  }

  KOKKOS_INLINE_FUNCTION
  static size_t shmem_size(int /*team_size*/) {
    // One scalar buffer and two vector buffers
    return 0;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (TeamMember team) const
  {
    KernelVariables kv(team);

    ExecViewUnmanaged<const Real[2][2][NP][NP]> dinv   = Homme::subview(m_elements.m_dinv,kv.ie);
    ExecViewUnmanaged<const Real[NP][NP]>       metdet = Homme::subview(m_elements.m_metdet,kv.ie);

    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(team,NUM_LEV*m_data.qsize),
      [&] (const int lev_q)
      {
        const int iq   = lev_q / NUM_LEV;
        kv.ilev = lev_q % NUM_LEV;

        ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> qdp   = Homme::subview(m_elements.m_qdp,kv.ie,m_data.qn0,iq);
        ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>       q_buf = Homme::subview(m_elements.buffers.qtens,kv.ie,iq);
        ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]>    v_buf = Homme::subview(m_elements.buffers.vstar_qdp,kv.ie,iq);

        Kokkos::parallel_for (
          Kokkos::ThreadVectorRange (team, NP*NP),
          [&] (const int idx)
          {
            const int igp = idx / NP;
            const int jgp = idx % NP;

            v_buf(0,igp,jgp,kv.ilev) = m_elements.buffers.vstar(kv.ie,0,igp,jgp,kv.ilev) * qdp(igp,jgp,kv.ilev);
            v_buf(1,igp,jgp,kv.ilev) = m_elements.buffers.vstar(kv.ie,1,igp,jgp,kv.ilev) * qdp(igp,jgp,kv.ilev);
            q_buf(igp,jgp,kv.ilev) = qdp(igp,jgp,kv.ilev);
          }
        );

        divergence_sphere_update(kv, -m_data.dt, 1.0, dinv, metdet,
                                 m_deriv.get_dvv(), v_buf, q_buf);
      }
    );
  }

};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
