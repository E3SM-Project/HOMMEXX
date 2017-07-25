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

  EulerStepFunctor (const Control& data)
   : m_data    (data)
   , m_region  (get_region())
   , m_deriv   (get_derivative())
  {
    // Nothing to be done here
  }

  KOKKOS_INLINE_FUNCTION
  static size_t shmem_size(int team_size) {
    // One scalar buffer and two vector buffers
    size_t mem_size = (sizeof(Real[NP][NP]) + 2*sizeof(Real[2][NP][NP])) * team_size;
    return mem_size;
  }

  template <typename Primitive, typename Data>
  KOKKOS_INLINE_FUNCTION
  Primitive* allocate_thread(const TeamMember& team) const {
    ScratchView<Data> view(team.thread_scratch(0));
    return view.data();
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (TeamMember team) const
  {
    const int ie = team.league_rank();
    ExecViewUnmanaged<Real[NP][NP]>     scalar_buf   (allocate_thread<Real,Real[NP][NP]>(team));
    ExecViewUnmanaged<Real[2][NP][NP]>  vector_buf_1 (allocate_thread<Real,Real[2][NP][NP]>(team));
    ExecViewUnmanaged<Real[2][NP][NP]>  vector_buf_2 (allocate_thread<Real,Real[2][NP][NP]>(team));

    ExecViewUnmanaged<const Real[NP][NP]>       metdet_ie = m_region.METDET(ie);
    ExecViewUnmanaged<const Real[2][2][NP][NP]> dinv_ie   = m_region.DINV(ie);

    ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> ustar = m_region.get_3d_buffer(ie,USTAR);
    ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> vstar = m_region.get_3d_buffer(ie,VSTAR);
    ExecViewUnmanaged<const Real[QSIZE_D][NUM_LEV][NP][NP]> qdp   = m_region.QDP(ie,m_data.qn0);
    ExecViewUnmanaged<Real[QSIZE_D][NUM_LEV][NP][NP]>       q_buf = m_region.get_q_buffer(ie);

    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(team,NUM_LEV*m_data.qsize),
      [&] (const int lev_q)
      {
        const int iq   = lev_q / NUM_LEV;
        const int ilev = lev_q % NUM_LEV;

        Kokkos::parallel_for (
          Kokkos::ThreadVectorRange (team, NP*NP),
          [&] (const int idx)
          {
            const int igp = idx / NP;
            const int jgp = idx % NP;

            vector_buf_1(0,igp,jgp) = ustar(ilev,igp,jgp) * qdp(iq,ilev,igp,jgp);
            vector_buf_1(1,igp,jgp) = vstar(ilev,igp,jgp) * qdp(iq,ilev,igp,jgp);
          }
        );

        divergence_sphere(team,vector_buf_1,m_deriv.get_dvv(),
                          metdet_ie,dinv_ie,
                          vector_buf_2, scalar_buf);

        Kokkos::parallel_for (
          Kokkos::ThreadVectorRange (team, NP*NP),
          [&] (const int idx)
          {
            const int igp = idx / NP;
            const int jgp = idx % NP;

            q_buf(iq,ilev,igp,jgp) = qdp(iq,ilev,igp,jgp) - m_data.dt*scalar_buf(igp,jgp);
          }
        );
      }
    );
  }

};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
