#ifndef HOMMEXX_EULER_STEP_FUNCTOR_HPP
#define HOMMEXX_EULER_STEP_FUNCTOR_HPP

#include "Region.hpp"
#include "Derivative.hpp"
#include "Control.hpp"
#include "SphereOperators.hpp"

namespace Homme
{

struct EulerStepFunctor
{
  const Control     m_data;
  const Region      m_region;
  const Derivative  m_deriv;

  static constexpr int IDX_USTAR = 0;
  static constexpr int IDX_VSTAR = 1;

  static constexpr int IDX_VBUFF = 0;

  static constexpr int IDX_TBUFF = 0;

  EulerStepFunctor (const Control& data)
   : m_data    (data)
   , m_region  (get_region())
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

    ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> vector_buf = ::Homme::subview(m_region.buffers.vectors,kv.ie,IDX_VBUFF);

    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> ustar = ::Homme::subview(m_region.buffers.scalars,kv.ie,IDX_USTAR);
    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> vstar = ::Homme::subview(m_region.buffers.scalars,kv.ie,IDX_VSTAR);

    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(team,NUM_LEV*m_data.qsize),
      [&] (const int lev_q)
      {
        const int iq   = lev_q / NUM_LEV;
        kv.ilev = lev_q % NUM_LEV;

        ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> qdp   = ::Homme::subview(m_region.m_qdp,kv.ie,m_data.qn0,iq);
        ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>       q_buf = ::Homme::subview(m_region.buffers.tracers,kv.ie,IDX_TBUFF,iq);

        Kokkos::parallel_for (
          Kokkos::ThreadVectorRange (team, NP*NP),
          [&] (const int idx)
          {
            const int igp = idx / NP;
            const int jgp = idx % NP;

            vector_buf(0,igp,jgp,kv.ilev) = ustar(igp,jgp,kv.ilev) * qdp(igp,jgp,kv.ilev);
            vector_buf(1,igp,jgp,kv.ilev) = vstar(igp,jgp,kv.ilev) * qdp(igp,jgp,kv.ilev);
            q_buf(igp,jgp,kv.ilev)        = qdp(igp,jgp,kv.ilev);
          }
        );

        divergence_sphere_update(kv, -m_data.dt, 1.0,
                                 m_region.m_dinv, m_region.m_metdet,
                                 m_deriv.get_dvv(),
                                 vector_buf, q_buf);
      }
    );
  }

};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
