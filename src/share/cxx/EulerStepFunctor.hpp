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

  static constexpr int USTAR = 0;
  static constexpr int VSTAR = 1;

  struct KernelVariables
  {
    KernelVariables(const TeamMember &team_in)
        : team(team_in),
          scalar_buf(allocate_thread<Real, Real[NP][NP]>()),
          vector_buf_1(allocate_thread<Real, Real[2][NP][NP]>()),
          vector_buf_2(allocate_thread<Real, Real[2][NP][NP]>()),
          ie(team.league_rank()), ilev(-1), iq(-1)
    {
      // Nothing else to be done here
    }

    template <typename Primitive, typename Data>
    KOKKOS_INLINE_FUNCTION
    Primitive* allocate_thread() const {
      ScratchView<Data> view(team.thread_scratch(0));
      return view.data();
    }

    KOKKOS_INLINE_FUNCTION
    static size_t shmem_size(int team_size) {
      // One scalar buffer and two vector buffers
      size_t mem_size = (sizeof(Real[NP][NP]) + 2*sizeof(Real[2][NP][NP])) * team_size;
      return mem_size;
    }

    const TeamMember&                   team;
    ExecViewUnmanaged<Real[NP][NP]>     scalar_buf;
    ExecViewUnmanaged<Real[2][NP][NP]>  vector_buf_1;
    ExecViewUnmanaged<Real[2][NP][NP]>  vector_buf_2;
    int ie, ilev, iq;
  };

  EulerStepFunctor (const Control& data)
   : m_data    (data)
   , m_region  (get_region())
   , m_deriv   (get_derivative())
  {
    // Nothing to be done here
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (TeamMember team) const
  {
    KernelVariables kv(team);

    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team,NUM_LEV*m_data.qsize),
      [&] (const int lev_q)
      {
        kv.iq   = lev_q / NUM_LEV;
        kv.ilev = lev_q % NUM_LEV;

        Kokkos::parallel_for (
          Kokkos::ThreadVectorRange (kv.team, NP*NP),
          KOKKOS_LAMBDA (const int idx)
          {
            const int igp = idx / NP;
            const int jgp = idx % NP;

            kv.vector_buf_1(0,igp,jgp) = m_region.get_3d_buffer(kv.ie,USTAR,kv.ilev,igp,jgp) * m_region.QDP(kv.ie,m_data.qn0,kv.iq,kv.ilev,igp,jgp);
            kv.vector_buf_1(1,igp,jgp) = m_region.get_3d_buffer(kv.ie,VSTAR,kv.ilev,igp,jgp) * m_region.QDP(kv.ie,m_data.qn0,kv.iq,kv.ilev,igp,jgp);
          }
        );

        divergence_sphere(kv.team, kv.vector_buf_1,m_deriv.get_dvv(),
                          m_region.METDET(kv.ie),m_region.DINV(kv.ie),
                          kv.vector_buf_2, kv.scalar_buf);

        Kokkos::parallel_for (
          Kokkos::ThreadVectorRange (kv.team, NP*NP),
          KOKKOS_LAMBDA (const int idx)
          {
            const int igp = idx / NP;
            const int jgp = idx % NP;

            m_region.get_q_buffer(kv.ie,kv.iq,kv.ilev,igp,jgp) = m_region.QDP(kv.ie,m_data.qn0,kv.iq,kv.ilev,igp,jgp) - m_data.dt*kv.scalar_buf(igp,jgp);
          }
        );
      }
    );
  }

};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
