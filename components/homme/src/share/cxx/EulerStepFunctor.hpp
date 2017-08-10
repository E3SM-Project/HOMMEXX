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

  static constexpr int IDX_QBUFF = 0;
  static constexpr int IDX_UBUFF = 1;
  static constexpr int IDX_VBUFF = 2;


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

    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> ustar = ::Homme::subview(m_region.buffers.scalars,kv.ie,IDX_USTAR);
    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> vstar = ::Homme::subview(m_region.buffers.scalars,kv.ie,IDX_VSTAR);

    ExecViewUnmanaged<const Real[2][2][NP][NP]> dinv   = ::Homme::subview(m_region.m_dinv,kv.ie);
    ExecViewUnmanaged<const Real[NP][NP]>       metdet = ::Homme::subview(m_region.m_metdet,kv.ie);

    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(team,NUM_LEV*m_data.qsize),
      [&] (const int lev_q)
      {
        const int iq   = lev_q / NUM_LEV;
        kv.ilev = lev_q % NUM_LEV;

        ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> qdp   = ::Homme::subview(m_region.m_qdp,kv.ie,m_data.qn0,iq);
        ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>       q_buf = ::Homme::subview(m_region.buffers.tracers,kv.ie,IDX_QBUFF,iq);
        ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>       u_buf = ::Homme::subview(m_region.buffers.tracers,kv.ie,IDX_UBUFF,iq);
        ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>       v_buf = ::Homme::subview(m_region.buffers.tracers,kv.ie,IDX_VBUFF,iq);


        Kokkos::parallel_for (
          Kokkos::ThreadVectorRange (team, NP*NP),
          [&] (const int idx)
          {
            const int igp = idx / NP;
            const int jgp = idx % NP;

            u_buf(igp,jgp,kv.ilev) = ustar(igp,jgp,kv.ilev) * qdp(igp,jgp,kv.ilev);
            v_buf(igp,jgp,kv.ilev) = vstar(igp,jgp,kv.ilev) * qdp(igp,jgp,kv.ilev);
            q_buf(igp,jgp,kv.ilev) = qdp(igp,jgp,kv.ilev);
          }
        );

        divergence_sphere_update(kv, -m_data.dt, 1.0, dinv, metdet,
                                 m_deriv.get_dvv(), u_buf, v_buf, q_buf);
      }
    );
  }

};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
