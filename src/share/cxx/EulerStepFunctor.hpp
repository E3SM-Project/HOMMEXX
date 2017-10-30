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
  Control           m_data;
  const Elements    m_elements;
  const Derivative  m_deriv;

  struct TagVstar {};
  struct TagDivUpdate {};
  struct TagFused {};

  EulerStepFunctor (const Control& data)
   : m_data    (data)
   , m_elements(get_elements())
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
  void operator() (const TagVstar&, const TeamMember& team) const {
    KernelVariables kv(team, m_data.qsize);
    compute_vstar_qdp(kv);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagDivUpdate&, const TeamMember& team) const {
    KernelVariables kv(team, m_data.qsize);
    compute_qtens(kv);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagFused&, const TeamMember& team) const {
    start_timer("esf compute");
    KernelVariables kv(team, m_data.qsize);
    compute_vstar_qdp(kv);
    compute_qtens(kv);
    stop_timer("esf compute");
  }

  static void run() {
    // Get control structure
    Control& data = get_control();

    // Create the functor
    EulerStepFunctor func(data);

    profiling_resume();
#if 1
    Kokkos::parallel_for(get_policy<TagFused>(data), func);
#else
    Kokkos::parallel_for(get_policy<TagVstar>(data), func);
    Kokkos::fence();
    Kokkos::parallel_for(get_policy<TagDivUpdate>(data), func);
#endif

    // Finalize
    ExecSpace::fence();
    profiling_pause();
  }

private:

  template <typename Tag>
  static Kokkos::TeamPolicy<ExecSpace, Tag> get_policy(const Control& data) {
    const int vectors_per_thread =
      DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
    //todo Need to rework threading setup.
    const int threads_per_team   =
      std::is_same<ExecSpace,Hommexx_Cuda>::value ?
      DefaultThreadsDistribution<ExecSpace>::threads_per_team(1) :
      std::max(1, (DefaultThreadsDistribution<ExecSpace>::threads_per_team(data.num_elems) /
                   data.qsize));
    Kokkos::TeamPolicy<ExecSpace, Tag> policy(data.num_elems * data.qsize,
                                              threads_per_team,
                                              vectors_per_thread);
    policy.set_chunk_size(1);
    return policy;
  }

  KOKKOS_INLINE_FUNCTION
  void compute_vstar_qdp (const KernelVariables& kv) const {
    using Kokkos::ALL;
    const auto NP2 = NP * NP;
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP2),
      [&] (const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;

        const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]>
          qdp   = Kokkos::subview(m_elements.m_qdp, kv.ie, m_data.qn0, kv.iq, ALL, ALL, ALL);
        const ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>
          q_buf = Kokkos::subview(m_elements.buffers.qtens, kv.ie, kv.iq, ALL, ALL, ALL);
        const ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]>
          v_buf = Kokkos::subview(m_elements.buffers.vstar_qdp, kv.ie, kv.iq, ALL, ALL, ALL, ALL);

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV), [&] (const int& ilev) {
          v_buf(0,igp,jgp,ilev) = (m_elements.buffers.vstar(kv.ie, 0, igp, jgp, ilev) *
                                   qdp(igp, jgp, ilev));
          v_buf(1,igp,jgp,ilev) = (m_elements.buffers.vstar(kv.ie, 1, igp, jgp, ilev) *
                                   qdp(igp, jgp, ilev));
          q_buf(igp,jgp,ilev) = qdp(igp,jgp,ilev);
        });
      }
    );
  }

  KOKKOS_INLINE_FUNCTION
  void compute_qtens (const KernelVariables& kv) const {
    using Kokkos::ALL;
    const auto dvv = m_deriv.get_dvv();
    const ExecViewUnmanaged<const Real[NP][NP]>
      metdet = Kokkos::subview(m_elements.m_metdet, kv.ie, ALL, ALL);
    const ExecViewUnmanaged<const Real[2][2][NP][NP]>
      dinv = Kokkos::subview(m_elements.m_dinv, kv.ie, ALL, ALL, ALL, ALL);
    divergence_sphere_update(
      kv, -m_data.dt, 1.0, dinv, metdet, dvv,
      Kokkos::subview(m_elements.buffers.vstar_qdp, kv.ie, kv.iq, ALL, ALL, ALL, ALL),
      m_elements.buffers.vdp,
      Kokkos::subview(m_elements.buffers.qtens, kv.ie, kv.iq, ALL, ALL, ALL));
  }

};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
