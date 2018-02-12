#ifndef HOMMEXX_EULER_STEP_FUNCTOR_HPP
#define HOMMEXX_EULER_STEP_FUNCTOR_HPP

#include "Context.hpp"
#include "Elements.hpp"
#include "Derivative.hpp"
#include "HybridVCoord.hpp"
#include "HommexxEnums.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"
#include "utilities/SubviewUtils.hpp"
#include "utilities/VectorUtils.hpp"
#include "ErrorDefs.hpp"
#include "BoundaryExchange.hpp"
#include "profiling.hpp"

namespace Homme {

class EulerStepFunctor {

  struct EulerStepData {
    int         qsize;
    int         limiter_option;
    Real        rhs_viss;
    Real        rhs_multiplier;

    Real        nu_p;
    Real        nu_q;

    Real        dt;
    int         np1_qdp;
    int         n0_qdp;

    DSSOption   DSSopt;
  };

  EulerStepData       m_data;
  const Elements      m_elements;
  const Derivative    m_deriv;
  const HybridVCoord  m_hvcoord;

  bool                m_kernel_will_run_limiters;

  enum { m_mem_per_team = 2 * NP * NP * sizeof(Real) };

public:

  EulerStepFunctor (const SimulationParams& params)
   : m_elements(Context::singleton().get_elements())
   , m_deriv   (Context::singleton().get_derivative())
   , m_hvcoord (Context::singleton().get_hvcoord())
  {
    m_data.limiter_option = params.limiter_option;
    m_data.qsize = params.qsize;
    m_data.rhs_viss = 0.0;
    m_data.nu_p = params.nu_p;
    m_data.nu_q = params.nu_q;

    if (m_data.limiter_option == 4) {
      Errors::runtime_abort("Limiter option 4 hasn't been implemented!",
                            Errors::err_not_implemented);
    }
  }

  static size_t limiter_team_shmem_size (const int team_size) {
    return Memory<ExecSpace>::on_gpu ?
      (team_size * m_mem_per_team) :
      0;
  }

  size_t team_shmem_size (const int team_size) const {
    return m_kernel_will_run_limiters ? limiter_team_shmem_size(team_size) : 0;
  }

  struct BIHPre {};
  struct BIHPost {};

  /*
    ! get new min/max values, and also compute biharmonic mixing term

    ! two scalings depending on nu_p:
    ! nu_p=0:    qtens_biharmonic *= dp0                   (apply viscosity only to q)
    ! nu_p>0):   qtens_biharmonc *= elem()%psdiss_ave      (for consistency, if nu_p=nu_q)
   */
  void compute_biharmonic_pre() {
    profiling_resume();
    GPTLstart("esf-bih-pre run");

    assert(m_data.rhs_multiplier == 2.0);
    m_data.rhs_viss = 3.0;

    Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace, BIHPre>(
                             m_elements.num_elems() * m_data.qsize),
                         *this);

    ExecSpace::fence();
    GPTLstop("esf-bih-pre run");
    profiling_pause();
  }

  void compute_biharmonic_post() {
    profiling_resume();
    GPTLstart("esf-bih-post run");

    assert(m_data.rhs_multiplier == 2.0);

    Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace, BIHPost>(
                             m_elements.num_elems() * m_data.qsize),
                         *this);

    ExecSpace::fence();
    GPTLstop("esf-bih-post run");
    profiling_pause();
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const BIHPre&, const TeamMember& team) const {
    start_timer("esf-bih-pre compute");
    const int ie = team.league_rank() / m_data.qsize;
    const int iq = team.league_rank() % m_data.qsize;
    const auto& e = m_elements;
    const auto qtens_biharmonic = Homme::subview(e.buffers.qtens_biharmonic, ie, iq);
    if (m_data.nu_p > 0) {
      const auto dpdiss_ave = Homme::subview(e.m_derived_dpdiss_ave, ie);
      Kokkos::parallel_for (
        Kokkos::TeamThreadRange(team, NP*NP),
        [&] (const int loop_idx) {
          const int i = loop_idx / NP;
          const int j = loop_idx % NP;
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team, NUM_LEV),
            [&] (const int& k) {
              qtens_biharmonic(i,j,k) = qtens_biharmonic(i,j,k) * dpdiss_ave(i,j,k) / m_hvcoord.dp0(k);
            });
        });
      team.team_barrier();
    }
    laplace_simple(team, m_deriv.get_dvv(),
                   Homme::subview(e.m_dinv,ie),
                   Homme::subview(e.m_spheremp,ie),
                   Homme::subview(e.buffers.vstar_qdp, ie, iq),
                   qtens_biharmonic,
                   Homme::subview(e.buffers.qwrk, ie, iq),
                   qtens_biharmonic);
    stop_timer("esf-bih-pre compute");
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const BIHPost&, const TeamMember& team) const {
    start_timer("esf-bih-post compute");
    const int ie = team.league_rank() / m_data.qsize;
    const int iq = team.league_rank() % m_data.qsize;
    const auto& e = m_elements;
    const auto qtens_biharmonic = Homme::subview(e.buffers.qtens_biharmonic, ie, iq);
    team.team_barrier();
    laplace_simple(team, m_deriv.get_dvv(),
                   Homme::subview(e.m_dinv,ie),
                   Homme::subview(e.m_spheremp,ie),
                   Homme::subview(e.buffers.vstar_qdp, ie, iq),
                   qtens_biharmonic,
                   Homme::subview(e.buffers.qwrk, ie, iq),
                   qtens_biharmonic);
    // laplace_simple provides the barrier.
    {
      const auto spheremp = Homme::subview(e.m_spheremp, ie);
      Kokkos::parallel_for (
        Kokkos::TeamThreadRange(team, NP*NP),
        [&] (const int loop_idx) {
          const int i = loop_idx / NP;
          const int j = loop_idx % NP;
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team, NUM_LEV),
            [&] (const int& k) {
              qtens_biharmonic(i,j,k) = (-m_data.rhs_viss * m_data.dt * m_data.nu_q *
                                         m_hvcoord.dp0(k) * qtens_biharmonic(i,j,k) /
                                         spheremp(i,j));
            });
        });
    }
    stop_timer("esf-bih-post compute");
  }

  struct AALSetupPhase {};
  struct AALTracerPhase {};
  struct AALFusedPhases {};

  void advect_and_limit() {
    profiling_resume();
    GPTLstart("esf-aal-tot run");
    if (OnGpu<ExecSpace>::value) {
      GPTLstart("esf-aal-noq run");
      Kokkos::parallel_for(
          Homme::get_default_team_policy<ExecSpace, AALSetupPhase>(
              m_elements.num_elems()),
          *this);
      ExecSpace::fence();
      GPTLstop("esf-aal-noq run");
      GPTLstart("esf-aal-q run");
      m_kernel_will_run_limiters = true;
      Kokkos::parallel_for(
          Homme::get_default_team_policy<ExecSpace, AALTracerPhase>(
              m_elements.num_elems() * m_data.qsize),
          *this);
      ExecSpace::fence();
      m_kernel_will_run_limiters = false;
      GPTLstop("esf-aal-q run");
    } else {
      Kokkos::parallel_for(
          Homme::get_default_team_policy<ExecSpace, AALFusedPhases>(
              m_elements.num_elems()),
          *this);
    }
    ExecSpace::fence();
    GPTLstop("esf-aal-tot run");
    profiling_pause();
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const AALSetupPhase&, const TeamMember& team) const {
    start_timer("esf-aal-noq compute");
    KernelVariables kv(team);
    run_setup_phase(kv);
    stop_timer("esf-aal-noq compute");
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const AALTracerPhase&, const TeamMember& team) const {
    start_timer("esf-aal-q compute");
    KernelVariables kv(team, m_data.qsize);
    run_tracer_phase(kv);
    stop_timer("esf-aal-q compute");
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const AALFusedPhases&, const TeamMember& team) const {
    start_timer("esf-aal-fused compute");
    KernelVariables kv(team);
    run_setup_phase(kv);
    for (kv.iq = 0; kv.iq < m_data.qsize; ++kv.iq)
    {
      run_tracer_phase(kv);
    }
    stop_timer("esf-aal-fused compute");
  }

  struct PrecomputeDivDp {};

  void precompute_divdp() {
    profiling_resume();
    GPTLstart("esf-precompute_divdp run");

    Kokkos::parallel_for(
        Homme::get_default_team_policy<ExecSpace, PrecomputeDivDp>(
            m_elements.num_elems()),
        *this);

    ExecSpace::fence();
    GPTLstop("esf-precompute_divdp run");
    profiling_pause();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const PrecomputeDivDp &, const TeamMember &team) const {
    start_timer("esf-precompute_divdp compute");
    const int ie = team.league_rank();
    divergence_sphere(team, m_deriv.get_dvv(),
                      Homme::subview(m_elements.m_dinv,ie),
                      Homme::subview(m_elements.m_metdet,ie),
                      Homme::subview(m_elements.m_derived_vn0, ie),
                      Homme::subview(m_elements.buffers.div_buf,ie),
                      Homme::subview(m_elements.m_derived_divdp, ie));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV),
                           [&](const int ilev) {
        m_elements.m_derived_divdp_proj(ie, igp, jgp, ilev) =
            m_elements.m_derived_divdp(ie, igp, jgp, ilev);
      });
    });

    stop_timer("esf-precompute_divdp compute");
  }

  void qdp_time_avg (const int n0_qdp, const int np1_qdp) {
    const auto qdp    = m_elements.m_qdp;
    const int qsize   = m_data.qsize;

    const Real rkstage = 3.0;
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,m_elements.num_elems()*m_data.qsize*NP*NP*NUM_LEV),
                         KOKKOS_LAMBDA(const int idx) {
      const int ie   = (((idx / NUM_LEV) / NP) / NP) / qsize;
      const int iq   = (((idx / NUM_LEV) / NP) / NP) % qsize;
      const int igp  =  ((idx / NUM_LEV) / NP) % NP;
      const int jgp  =   (idx / NUM_LEV) % NP;
      const int ilev =    idx % NUM_LEV;

      qdp(ie,np1_qdp,iq,igp,jgp,ilev) =
            (qdp(ie,n0_qdp,iq,igp,jgp,ilev) +
             (rkstage-1)*qdp(ie,np1_qdp,iq,igp,jgp,ilev)) / rkstage;
    });
  }

  void compute_qmin_qmax() {
    // Temporaries, due to issues capturing *this on device
    const int qsize = m_data.qsize;
    const Real rhs_multiplier = m_data.rhs_multiplier;
    const int n0_qdp = m_data.n0_qdp;
    const Real dt = m_data.dt;
    const auto qdp = m_elements.m_qdp;
    const auto qtens_biharmonic= m_elements.buffers.qtens_biharmonic;
    const auto qlim = m_elements.buffers.qlim;
    const auto derived_dp = m_elements.m_derived_dp;
    const auto derived_divdp_proj= m_elements.m_derived_divdp_proj;
    Kokkos::RangePolicy<ExecSpace> policy1(0, m_elements.num_elems() * m_data.qsize *
                                                  NP * NP * NUM_LEV);
    Kokkos::parallel_for(policy1, KOKKOS_LAMBDA(const int &loop_idx) {
      const int ie = (((loop_idx / NUM_LEV) / NP) / NP) / qsize;
      const int q = (((loop_idx / NUM_LEV) / NP) / NP) % qsize;
      const int igp = ((loop_idx / NUM_LEV) / NP) % NP;
      const int jgp = (loop_idx / NUM_LEV) % NP;
      const int lev = loop_idx % NUM_LEV;

      Scalar dp = derived_dp(ie, igp, jgp, lev) -
                  rhs_multiplier * dt *
                      derived_divdp_proj(ie, igp, jgp, lev);

      Scalar tmp = qdp(ie, n0_qdp, q, igp, jgp, lev) / dp;
      qtens_biharmonic(ie, q, igp, jgp, lev) = tmp;
    });
    ExecSpace::fence();

    Kokkos::RangePolicy<ExecSpace> policy2(0, m_elements.num_elems() * m_data.qsize *
                                                  NUM_LEV);
    if (m_data.rhs_multiplier == 1.0) {
      Kokkos::parallel_for(policy2, KOKKOS_LAMBDA(const int &loop_idx) {
        const int ie = (loop_idx / NUM_LEV) / qsize;
        const int q = (loop_idx / NUM_LEV) % qsize;
        const int lev = loop_idx % NUM_LEV;
        Scalar min_biharmonic = qtens_biharmonic(ie, q, 0, 0, lev);
        Scalar max_biharmonic = min_biharmonic;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            min_biharmonic = min(min_biharmonic, qtens_biharmonic(ie, q, igp, jgp, lev));
            max_biharmonic = max(max_biharmonic, qtens_biharmonic(ie, q, igp, jgp, lev));
          }
        }
        qlim(ie, q, 0, lev) = min(qlim(ie, q, 0, lev), min_biharmonic);
        qlim(ie, q, 1, lev) = max(qlim(ie, q, 1, lev), max_biharmonic);
      });
    } else {
      Kokkos::parallel_for(policy2, KOKKOS_LAMBDA(const int &loop_idx) {
        const int ie = (loop_idx / NUM_LEV) / qsize;
        const int q = (loop_idx / NUM_LEV) % qsize;
        const int lev = loop_idx % NUM_LEV;
        Scalar min_biharmonic = qtens_biharmonic(ie, q, 0, 0, lev);
        Scalar max_biharmonic = min_biharmonic;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            min_biharmonic = min(min_biharmonic, qtens_biharmonic(ie, q, igp, jgp, lev));
            max_biharmonic = max(max_biharmonic, qtens_biharmonic(ie, q, igp, jgp, lev));
          }
        }
        qlim(ie, q, 0, lev) = min_biharmonic;
        qlim(ie, q, 1, lev) = max_biharmonic;
      });
    }
    ExecSpace::fence();
  }

  void neighbor_minmax_start() {
    BoundaryExchange &be =
        *Context::singleton().get_boundary_exchange("min max Euler");
    be.pack_and_send_min_max();
  }

  void neighbor_minmax_finish() {
    BoundaryExchange &be =
        *Context::singleton().get_boundary_exchange("min max Euler");
    be.recv_and_unpack_min_max();
  }

  void minmax_and_biharmonic() {
    const auto be = Context::singleton().get_boundary_exchange(
        "Euler step: min/max & qtens_biharmonic");
    if (!be->is_registration_completed()) {
      be->set_buffers_manager(
          Context::singleton().get_buffers_manager(MPI_EXCHANGE));
      be->set_num_fields(0, 0, m_data.qsize);
      be->register_field(m_elements.buffers.qtens_biharmonic, m_data.qsize, 0);
      be->registration_completed();
    }
    neighbor_minmax_start();
    compute_biharmonic_pre();
    be->exchange(m_elements.m_rspheremp);
    compute_biharmonic_post();
    neighbor_minmax_finish();
  }

  void neighbor_minmax() {
    BoundaryExchange &be =
        *Context::singleton().get_boundary_exchange("min max Euler");
    assert(be.is_registration_completed());
    be.exchange_min_max();
  }

  void exchange_qdp_dss_var () {
    // Note: we have three separate BE structures, all of which register qdp.
    // They differ only in the last field registered.
    // This allows us to have a SINGLE mpi call to exchange qsize+1 fields,
    // rather than one for qdp and one for the last DSS variable.

    std::stringstream ss;
    ss << "exchange qdp " << (m_data.DSSopt == DSSOption::ETA
                                  ? "eta"
                                  : m_data.DSSopt == DSSOption::OMEGA
                                        ? "omega"
                                        : "div_vdp_ave") << " " << m_data.np1_qdp;

    const std::shared_ptr<BoundaryExchange> be_qdp_dss_var =
        Context::singleton().get_boundary_exchange(ss.str());

    be_qdp_dss_var->exchange(m_elements.m_rspheremp);
  }

  void euler_step(const int np1_qdp, const int n0_qdp, const Real dt,
                  const Real rhs_multiplier, const DSSOption DSSopt) {

    m_data.n0_qdp         = n0_qdp;
    m_data.np1_qdp        = np1_qdp;
    m_data.dt             = dt;
    m_data.rhs_multiplier = rhs_multiplier;
    m_data.DSSopt         = DSSopt;

    if (m_data.limiter_option == 8) {
      // when running lim8, we also need to limit the biharmonic, so that term
      // needs to be included in each euler step.  three possible algorithms
      // here:
      // most expensive:
      //   compute biharmonic (which also computes qmin/qmax) during all 3
      //   stages be sure to set rhs_viss=1 cost:  3 biharmonic steps with 3 DSS

      // cheapest:
      //   compute biharmonic (which also computes qmin/qmax) only on first
      //   stage be sure to set rhs_viss=3 reuse qmin/qmax for all following
      //   stages (but update based on local qmin/qmax) cost:  1 biharmonic
      //   steps with 1 DSS main concern: viscosity

      // compromise:
      //   compute biharmonic (which also computes qmin/qmax) only on last stage
      //   be sure to set rhs_viss=3
      //   compute qmin/qmax directly on first stage
      //   reuse qmin/qmax for 2nd stage stage (but update based on local
      //   qmin/qmax) cost:  1 biharmonic steps, 2 DSS

      //  NOTE  when nu_p=0 (no dissipation applied in dynamics to dp equation),
      //        we should apply dissipation to Q (not Qdp) to preserve Q=1
      //        i.e.  laplace(Qdp) ~  dp0 laplace(Q)
      //        for nu_p=nu_q>0, we need to apply dissipation to Q *diffusion_dp

      // initialize dp, and compute Q from Qdp(and store Q in Qtens_biharmonic)
      GPTLstart("bihmix_qminmax");

      compute_qmin_qmax();

      if (m_data.rhs_multiplier == 0.0) {
        GPTLstart("eus_neighbor_minmax1");
        neighbor_minmax();
        GPTLstop("eus_neighbor_minmax1");
      } else if (m_data.rhs_multiplier == 2.0) {
        minmax_and_biharmonic();
      }
      GPTLstop("bihmix_qminmax");
    }
    GPTLstart("eus_2d_advec");
    GPTLstart("advance_qdp");

    advect_and_limit();

    GPTLstop("advance_qdp");

    exchange_qdp_dss_var();

    GPTLstop("eus_2d_advec");
  }

private:

  KOKKOS_INLINE_FUNCTION
  void run_setup_phase (const KernelVariables& kv) const {
    compute_2d_advection_step(kv);
  }

  KOKKOS_INLINE_FUNCTION
  void run_tracer_phase (const KernelVariables& kv) const {
    compute_vstar_qdp(kv);
    compute_qtens(kv);
    kv.team_barrier();
    if (m_data.rhs_viss != 0.0) {
      add_hyperviscosity(kv);
      kv.team_barrier();
    }
    if (m_data.limiter_option == 8) {
      limiter_optim_iter_full(kv);
      kv.team_barrier();
    }
    apply_spheremp(kv);
  }

  KOKKOS_INLINE_FUNCTION
  void compute_2d_advection_step (const KernelVariables& kv) const {
    const auto& c = m_data;
    const auto& e = m_elements;
    const bool lim8 = c.limiter_option == 8;
    const bool add_ps_diss = c.nu_p > 0 && c.rhs_viss != 0.0;
    const Real diss_fac = add_ps_diss ? -c.rhs_viss * c.dt * c.nu_q : 0;
    const auto& f_dss = (c.DSSopt == DSSOption::ETA ?
                         e.m_eta_dot_dpdn :
                         c.DSSopt == DSSOption::OMEGA ?
                         e.m_omega_p :
                         e.m_derived_divdp_proj);
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP*NP),
      [&] (const int loop_idx) {
        const int i = loop_idx / NP;
        const int j = loop_idx % NP;
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
          [&] (const int& k) {
            //! derived variable divdp_proj() (DSS'd version of divdp) will only
            //! be correct on 2nd and 3rd stage but that's ok because
            //! rhs_multiplier=0 on the first stage:
            const auto dp = e.m_derived_dp(kv.ie,i,j,k) -
              c.rhs_multiplier * c.dt * e.m_derived_divdp_proj(kv.ie,i,j,k);
            e.buffers.vstar(kv.ie,0,i,j,k) = e.m_derived_vn0(kv.ie,0,i,j,k) / dp;
            e.buffers.vstar(kv.ie,1,i,j,k) = e.m_derived_vn0(kv.ie,1,i,j,k) / dp;
            if (lim8) {
              //! Note that the term dpdissk is independent of Q
              //! UN-DSS'ed dp at timelevel n0+1:
              e.buffers.dpdissk(kv.ie,i,j,k) = dp - c.dt * e.m_derived_divdp(kv.ie,i,j,k);
              if (add_ps_diss) {
                //! add contribution from UN-DSS'ed PS dissipation
                //!          dpdiss(:,:) = ( hvcoord%hybi(k+1) - hvcoord%hybi(k) ) *
                //!          elem(ie)%derived%psdiss_biharmonic(:,:)
                e.buffers.dpdissk(kv.ie,i,j,k) += diss_fac *
                  e.m_derived_dpdiss_biharmonic(kv.ie,i,j,k) / e.m_spheremp(kv.ie,i,j);
              }
            }
            //! also DSS extra field
            //! note: eta_dot_dpdn is actually dimension nlev+1, but nlev+1 data is
            //! all zero so we only have to DSS 1:nlev
            f_dss(kv.ie,i,j,k) *= e.m_spheremp(kv.ie,i,j);
          });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_vstar_qdp (const KernelVariables& kv) const {
    const auto NP2 = NP * NP;
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP2),
      [&] (const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;

        const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]>
          qdp   = Homme::subview(m_elements.m_qdp, kv.ie, m_data.n0_qdp, kv.iq);
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
    const auto dvv = m_deriv.get_dvv();
    const ExecViewUnmanaged<const Real[NP][NP]>
      metdet = Homme::subview(m_elements.m_metdet, kv.ie);
    const ExecViewUnmanaged<const Real[2][2][NP][NP]>
      dinv = Homme::subview(m_elements.m_dinv, kv.ie);
    divergence_sphere_update(
      kv.team, -m_data.dt, 1.0, dvv, dinv, metdet,
      Homme::subview(m_elements.buffers.vstar_qdp, kv.ie, kv.iq),
      Homme::subview(m_elements.buffers.qwrk, kv.ie, kv.iq),
      Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq));
  }

  KOKKOS_INLINE_FUNCTION
  void add_hyperviscosity (const KernelVariables& kv) const {
    const auto qtens = Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq);
    const auto qtens_biharmonic = Homme::subview(m_elements.buffers.qtens_biharmonic, kv.ie, kv.iq);
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
    const auto sphweights = Homme::subview(m_elements.m_spheremp, kv.ie);
    const auto dpmass = Homme::subview(m_elements.buffers.dpdissk, kv.ie);
    const auto ptens = Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq);
    const auto qlim = Homme::subview(m_elements.buffers.qlim, kv.ie, kv.iq);

    limiter_optim_iter_full(kv.team, sphweights, dpmass, qlim, ptens);
  }

  //! apply mass matrix, overwrite np1 with solution:
  //! dont do this earlier, since we allow np1_qdp == n0_qdp
  //! and we dont want to overwrite n0_qdp until we are done using it
  KOKKOS_INLINE_FUNCTION
  void apply_spheremp (const KernelVariables& kv) const {
    const auto qdp = Homme::subview(m_elements.m_qdp, kv.ie, m_data.np1_qdp, kv.iq);
    const auto qtens = Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq);
    const auto spheremp = Homme::subview(m_elements.m_spheremp, kv.ie);
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP * NP),
      [&] (const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
          [&] (const int& ilev) {
            qdp(igp, jgp, ilev) = spheremp(igp, jgp) * qtens(igp, jgp, ilev);
          });
      });
  }

  // Do all the setup and teardown associated with a limiter. Call a limiter
  // functor to do the actual math given the problem data (mass, minp, maxp, c,
  // x), where the limiter possibly alters x to place it in the constraint set
  //    {x: (i) minp <= x_k <= maxp and (ii) c'x = mass }.
  template <typename Limit, typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl>
  KOKKOS_INLINE_FUNCTION static void
  with_limiter_shell (const TeamMember& team, const Limit& limit,
                      const ArrayGll& sphweights, const ArrayGllLvl& dpmass,
                      const Array2Lvl& qlim, const ArrayGllLvl& ptens) {
    const int NP2 = NP * NP;

    // Size doesn't matter; just need to get a pointer to the start of the
    // shared memory.
    Real* const team_data = Memory<ExecSpace>::get_shmem<Real>(team);

    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(team, NUM_PHYSICAL_LEV),
      [&] (const int ilev) {
        const int vpi = ilev / VECTOR_SIZE, vsi = ilev % VECTOR_SIZE;

        const auto tvr = Kokkos::ThreadVectorRange(team, NP2);
        using Kokkos::parallel_for;

        Real* const data = team_data ?
          team_data + 2 * NP2 * team.team_rank() :
          nullptr;
        Memory<ExecSpace>::AutoArray<Real, NP2> x(data), c(data + NP2);

        parallel_for(tvr, [&] (const int& k) {
            const int i = k / NP, j = k % NP;
            const auto& dpm = dpmass(i,j,vpi)[vsi];
            c[k] = sphweights(i,j)*dpm;
            x[k] = ptens(i,j,vpi)[vsi]/dpm;
          });

        Real sumc = 0;
        Homme::parallel_reduce(team, tvr, [&] (const int& k, Real& isumc) { isumc += c[k]; }, sumc);
        if (sumc <= 0) return; //! this should never happen, but if it does, dont limit
        Real mass = 0;
        Homme::parallel_reduce(team, tvr, [&] (const int& k, Real& imass) { imass += x[k]*c[k]; }, mass);

        Real minp = qlim(0,vpi)[vsi], maxp = qlim(1,vpi)[vsi];

        // This is a slightly different spot than where this comment came from,
        // but it's logically equivalent to do it here.
        //! IMPOSE ZERO THRESHOLD.  do this here so it can be turned off for
        //! testing
        if (minp < 0)
          minp = qlim(0,vpi)[vsi] = 0;

        //! relax constraints to ensure limiter has a solution:
        //! This is only needed if running with the SSP CFL>1 or
        //! due to roundoff errors
        // This is technically a write race condition, but the same value is
        // being written, so it doesn't matter.
        if (mass < minp*sumc)
          minp = qlim(0,vpi)[vsi] = mass/sumc;
        if (mass > maxp*sumc)
          maxp = qlim(1,vpi)[vsi] = mass/sumc;

        limit(team, mass, minp, maxp, x.data(), c.data());

        parallel_for(tvr, [&] (const int& k) {
            const int i = k / NP, j = k % NP;
            ptens(i,j,vpi)[vsi] = x[k]*dpmass(i,j,vpi)[vsi];
          });
      });
  }

public: // Expose for unit testing.

  // limiter_option = 8.
  template <typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl>
  KOKKOS_INLINE_FUNCTION static void
  limiter_optim_iter_full (const TeamMember& team,
                           const ArrayGll& sphweights, const ArrayGllLvl& dpmass,
                           const Array2Lvl& qlim, const ArrayGllLvl& ptens) {
    struct Limit {
      KOKKOS_INLINE_FUNCTION void
      operator() (const TeamMember& team, const Real& mass,
                  const Real& minp, const Real& maxp,
                  Real* KOKKOS_RESTRICT const x,
                  Real const* KOKKOS_RESTRICT const c) const {
        const int NP2 = NP * NP;
        const int maxiter = NP*NP - 1;
        const Real tol_limiter = 5e-14;

        const auto tvr = Kokkos::ThreadVectorRange(team, NP2);
        using Kokkos::parallel_for;

        for (int iter = 0; iter < maxiter; ++iter) {
          Real addmass = 0;
          Homme::parallel_reduce(team, tvr, [&] (const int& k, Real& iaddmass) {
              if (x[k] > maxp) {
                iaddmass += (x[k] - maxp)*c[k];
                x[k] = maxp;
              } else if (x[k] < minp) {
                iaddmass += (x[k] - minp)*c[k];
                x[k] = minp;
              }
            }, addmass);

          if (std::abs(addmass) <= tol_limiter*std::abs(mass))
            break;

          Real weightssum = 0;
          if (addmass > 0) {
            Homme::parallel_reduce(team, tvr, [&] (const int& k, Real& iweightssum) {
                if (x[k] < maxp)
                  iweightssum += c[k];
              }, weightssum);
            const auto adw = addmass/weightssum;
            parallel_for(tvr, [&] (const int& k) {
                if (x[k] < maxp)
                  x[k] += adw;
              });
          } else {
            Homme::parallel_reduce(team, tvr, [&] (const int& k, Real& iweightssum) {
                if (x[k] > minp)
                  iweightssum += c[k];
              }, weightssum);
            const auto adw = addmass/weightssum;
            parallel_for(tvr, [&] (const int& k) {
                if (x[k] > minp)
                  x[k] += adw;
              });
          }
        }
      }
    };

    with_limiter_shell(team, Limit(), sphweights, dpmass, qlim, ptens);
  }

  // This is limiter_option = 9 in ACME master. For now, just unit test it.
  template <typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl>
  KOKKOS_INLINE_FUNCTION static void
  limiter_clip_and_sum (const TeamMember& team,
                        const ArrayGll& sphweights, const ArrayGllLvl& dpmass,
                        const Array2Lvl& qlim, const ArrayGllLvl& ptens) {
    struct Limit {
      KOKKOS_INLINE_FUNCTION void
      operator() (const TeamMember& team, const Real& mass,
                  const Real& minp, const Real& maxp,
                  Real* KOKKOS_RESTRICT const x,
                  Real const* KOKKOS_RESTRICT const c) const {
        const int NP2 = NP * NP;

        const auto tvr = Kokkos::ThreadVectorRange(team, NP2);
        using Kokkos::parallel_for;

        // Clip.
        Real addmass = 0;
        Homme::parallel_reduce(team, tvr, [&] (const int& k, Real& iaddmass) {
            if (x[k] > maxp) {
              iaddmass += (x[k] - maxp)*c[k];
              x[k] = maxp;
            } else if (x[k] < minp) {
              iaddmass += (x[k] - minp)*c[k];
              x[k] = minp;
            }
          }, addmass);

        // No need for a tol: this isn't iterative. If it happens to be exactly
        // 0, then return early.
        if (addmass == 0) return;

        Real fac = 0;
        if (addmass > 0) {
          // Get sum of weights. Don't store them; we don't want another array.
          Homme::parallel_reduce(team, tvr, [&] (const int& k, Real& ifac) {
              ifac += c[k]*(maxp - x[k]);
            }, fac);
          if (fac > 0) {
            // Update.
            fac = addmass/fac;
            parallel_for(tvr, [&] (const int& k) { x[k] += fac*(maxp - x[k]); });
          }
        } else {
          Homme::parallel_reduce(team, tvr, [&] (const int& k, Real& ifac) {
              ifac += c[k]*(x[k] - minp);
            }, fac);
          if (fac > 0) {
            fac = addmass/fac;
            parallel_for(tvr, [&] (const int& k) { x[k] += fac*(x[k] - minp); });
          }
        }
      }
    };

    with_limiter_shell(team, Limit(), sphweights, dpmass, qlim, ptens);
  }
};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
