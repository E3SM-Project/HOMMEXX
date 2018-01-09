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

  struct TagFused {};

  EulerStepFunctor (const Control& data)
   : m_data    (data)
   , m_elements(Context::singleton().get_elements())
   , m_deriv   (Context::singleton().get_derivative())
  {}

  KOKKOS_INLINE_FUNCTION
  static size_t shmem_size(int /*team_size*/) {
    return 0;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagFused&, const TeamMember& team) const {
    start_timer("esf compute");
    KernelVariables kv(team, m_data.qsize);
    compute_vstar_qdp(kv);
    compute_qtens(kv);
    if (m_data.rhs_viss != 0)
      add_hyperviscosity(kv);
    if (m_data.limiter_option == 8)
      limiter_optim_iter_full(kv);
    apply_mass_matrix(kv);
    stop_timer("esf compute");
  }

  static void run() {
    Control& data = Context::singleton().get_control();
    EulerStepFunctor func(data);

    profiling_resume();
    start_timer("esf run");
    Kokkos::parallel_for(get_policy<TagFused>(data), func);
    stop_timer("esf run");

    ExecSpace::fence();
    profiling_pause();
  }

private:

  template <typename Tag>
  static Kokkos::TeamPolicy<ExecSpace, Tag> get_policy(const Control& data) {
    return Homme::get_default_team_policy<ExecSpace, Tag>(data.num_elems * data.qsize);
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
          qdp   = Homme::subview(m_elements.m_qdp, kv.ie, m_data.qn0, kv.iq);
        const ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>
          q_buf = Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq);
        const ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]>
          v_buf = Homme::subview(m_elements.buffers.vstar_qdp, kv.ie, kv.iq);

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
      metdet = Homme::subview(m_elements.m_metdet, kv.ie);
    const ExecViewUnmanaged<const Real[2][2][NP][NP]>
      dinv = Homme::subview(m_elements.m_dinv, kv.ie);
    divergence_sphere_update(
      kv, -m_data.dt, 1.0, dinv, metdet, dvv,
      Homme::subview(m_elements.buffers.vstar_qdp, kv.ie, kv.iq),
      Homme::subview(m_elements.buffers.qwrk, kv.ie, kv.iq),
      Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq));
  }

  KOKKOS_INLINE_FUNCTION
  void add_hyperviscosity (const KernelVariables& kv) const {
    using Kokkos::ALL;
    auto qtens = Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq);
    auto qtens_biharmonic = Homme::subview(m_elements.buffers.qtens_biharmonic, kv.ie, kv.iq);
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP * NP),
      [&] (const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
          [&] (const int& ilev) {
            qtens(igp, jgp, ilev) += qtens_biharmonic(igp, jgp, ilev);
          });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void limiter_optim_iter_full (const KernelVariables& kv) const {
    
  }

  /*
    ! apply mass matrix, overwrite np1 with solution:
    ! dont do this earlier, since we allow np1_qdp == n0_qdp
    ! and we dont want to overwrite n0_qdp until we are done using it
  */
  KOKKOS_INLINE_FUNCTION
  void apply_mass_matrix (const KernelVariables& kv) const {
    using Kokkos::ALL;
    auto qdp = Homme::subview(m_elements.m_qdp, kv.ie, m_data.np1_qdp, kv.iq);
    auto qtens = Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq);
    auto spheremp = Homme::subview(m_elements.m_spheremp, kv.ie);
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP * NP),
      [&] (const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
          [&] (const int& ilev) {
            qdp(igp, jgp, ilev) *= spheremp(igp, jgp);
          });
      });
  }
};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
