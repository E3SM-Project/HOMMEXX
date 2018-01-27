#include "CaarFunctor.hpp"
#include "Control.hpp"
#include "Context.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "mpi/ErrorDefs.hpp"
#include "mpi/BoundaryExchange.hpp"
#include "mpi/BuffersManager.hpp"

namespace Homme
{

void u3_5stage_timestep_c(const Real, const bool);
void caar_monolithic_c(Elements& elements, CaarFunctor& functor, BoundaryExchange& be,
                       Kokkos::TeamPolicy<ExecSpace,CaarFunctor::TagPreExchange>  policy_pre,
                       MDRangePolicy<ExecSpace,4> policy_post);
void prim_advance_exp_c(const Real dt, const bool compute_diagnostics);
void prim_advance_hypervis_dp_c (const int np1, const Real dt, const Real eta_ave_w);

// -------------- IMPLEMENTATIONS -------------- //

void prim_advance_exp_c(const Real dt, const bool compute_diagnostics)
{
  // Get control and simulation params
  Control&          data   = Context::singleton().get_control();
  SimulationParams& params = Context::singleton().get_simulation_params();
  assert(params.params_set);

  // Get the time level info
  TimeLevel& tl = Context::singleton().get_time_level();

  // Assume dry. If not dry, compute current qdp timelevel
  tl.n0_qdp = -1;
  if (params.moisture==MoistDry::MOIST) {
    tl.update_tracers_levels(params.qsplit);
  }

  // Establish time advance method and set eta_ave_w
  int method = params.time_step_type;
  data.eta_ave_w = 1.0/params.qsplit;
  if (params.time_step_type==0 && tl.nstep==0) {
    // 0: use leapfrog, but RK2 on first step
    method = 1;
  } else if (params.time_step_type==1) {
    // 1: use leapfrog, but RK2 on first qsplit stage
    method = 0;
    int qsplit_stage = tl.nstep % params.qsplit;  // get qsplit stage
    method = qsplit_stage==0 ? 1 : 0;             // use RK2 on first stage

    std::vector<double> ur_weights(params.qsplit,0.0);
    if (params.qsplit%2==0) {
      ur_weights[0] = 1.0/params.qsplit;
      for (int q=2; q<params.qsplit; q+=2) {
        ur_weights[q] = 2.0/params.qsplit;
      }
    } else {
      for (int q=1; q<params.qsplit; q+=2) {
        ur_weights[q] = 2.0/params.qsplit;
      }
    }
    data.eta_ave_w = ur_weights[qsplit_stage]; // RK2 + LF scheme has tricky weights
  }

  if (params.prescribed_wind) {
    Errors::runtime_abort("'prescribed wind' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
  }

  switch (method) {
    case 5:
      u3_5stage_timestep_c(dt, compute_diagnostics);
      break;

    default:
      Errors::runtime_abort("Time advance method not supported (maybe not yet ported to C++?).\n",
                            Errors::functionality_not_yet_implemented);
  }

#ifdef ENERGY_DIAGNOSTICS
  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagonstic' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // do ie = nets,nete
    //   elem(ie)%accum%DIFF(:,:,:,:)=elem(ie)%state%v(:,:,:,:,np1)
    //   elem(ie)%accum%DIFFT(:,:,:)=elem(ie)%state%T(:,:,:,np1)
    // enddo
  }
#endif

  if (params.time_step_type==0) {
    Errors::runtime_abort("'advance hypervis lf' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    // call advance_hypervis_lf(edge3p1,elem,hvcoord,hybrid,deriv,nm1,n0,np1,nets,nete,dt_vis)

  } else if (method<=10) {
    prim_advance_hypervis_dp_c(tl.np1,dt,data.eta_ave_w);
  }

#ifdef ENERGY_DIAGNOSTICS
  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagonstic' functionality not yet available in C++ build.\n",
                          Errors::functionality_not_yet_implemented);
    //       do ie = nets,nete
    // #if (defined COLUMN_OPENMP)
    // !$omp parallel do private(k)
    // #endif
    //         do k=1,nlev  !  Loop index added (AAM)
    //           elem(ie)%accum%DIFF(:,:,:,k)=( elem(ie)%state%v(:,:,:,k,np1) -&
    //                elem(ie)%accum%DIFF(:,:,:,k) ) / dt_vis
    //           elem(ie)%accum%DIFFT(:,:,k)=( elem(ie)%state%T(:,:,k,np1) -&
    //                elem(ie)%accum%DIFFT(:,:,k) ) / dt_vis
    //         enddo
    //       enddo
  }
#endif

  tl.tevolve += params.time_step;
}

void u3_5stage_timestep_c(const Real dt, const bool compute_diagonstics)
{
  // Get control and elements structures
  Control& data  = Context::singleton().get_control();
  Elements& elements = Context::singleton().get_elements();

  // Get the time level info
  TimeLevel& tl = Context::singleton().get_time_level();

  // Setup the policies
  auto policy_pre = Homme::get_default_team_policy<ExecSpace,CaarFunctor::TagPreExchange>(data.num_elems);
  MDRangePolicy<ExecSpace,4> policy_post({0,0,0,0},{data.num_elems,NP,NP,NUM_LEV}, {1,1,1,1});

  // Create the functor
  CaarFunctor functor(data, Context::singleton().get_elements(), Context::singleton().get_derivative());

  // Setup the boundary exchange
  std::shared_ptr<BoundaryExchange> be[NUM_TIME_LEVELS];
  for (int tl=0; tl<NUM_TIME_LEVELS; ++tl) {
    std::stringstream ss;
    ss << "caar tl " << tl;
    be[tl] = Context::singleton().get_boundary_exchange(ss.str());

    // If it was not yet created, create it and set it up
    if (!be[tl]->is_registration_completed()) {
      std::shared_ptr<BuffersManager> buffers_manager = Context::singleton().get_buffers_manager(MPI_EXCHANGE);
      be[tl]->set_buffers_manager(buffers_manager);

      // Set the views of this time level into this time level's boundary exchange
      be[tl]->set_num_fields(0,0,4);
      be[tl]->register_field(elements.m_u,1,tl);
      be[tl]->register_field(elements.m_v,1,tl);
      be[tl]->register_field(elements.m_t,1,tl);
      be[tl]->register_field(elements.m_dp3d,1,tl);
      be[tl]->registration_completed();
    }
  }

  // ===================== RK STAGES ===================== //

  const int nm1 = tl.nm1;
  const int n0  = tl.n0;
  const int np1 = tl.np1;

  // Stage 1: u1 = u0 + dt/5 RHS(u0),          t_rhs = t
  functor.set_rk_stage_data(n0,n0,nm1,dt/5.0,data.eta_ave_w/4.0,compute_diagonstics);
  caar_monolithic_c(elements,functor,*be[nm1],policy_pre,policy_post);

  // Stage 2: u2 = u0 + dt/5 RHS(u1),          t_rhs = t + dt/5
  functor.set_rk_stage_data(n0,nm1,np1,dt/5.0,0.0,false);
  caar_monolithic_c(elements,functor,*be[np1],policy_pre,policy_post);

  // Stage 3: u3 = u0 + dt/3 RHS(u2),          t_rhs = t + dt/5 + dt/5
  functor.set_rk_stage_data(n0,np1,np1,dt/3.0,0.0,false);
  caar_monolithic_c(elements,functor,*be[np1],policy_pre,policy_post);

  // Stage 4: u4 = u0 + 2dt/3 RHS(u3),         t_rhs = t + dt/5 + dt/5 + dt/3
  functor.set_rk_stage_data(n0,np1,np1,2.0*dt/3.0,0.0,false);
  caar_monolithic_c(elements,functor,*be[np1],policy_pre,policy_post);

  // Compute (5u1-u0)/4 and store it in timelevel nm1
  Kokkos::Experimental::md_parallel_for(
    policy_post,
    KOKKOS_LAMBDA(int ie, int igp, int jgp, int ilev) {
       elements.m_t(ie,nm1,igp,jgp,ilev) = (5.0*elements.m_t(ie,nm1,igp,jgp,ilev)-elements.m_t(ie,n0,igp,jgp,ilev))/4.0;
       elements.m_u(ie,nm1,igp,jgp,ilev) = (5.0*elements.m_u(ie,nm1,igp,jgp,ilev)-elements.m_u(ie,n0,igp,jgp,ilev))/4.0;
       elements.m_v(ie,nm1,igp,jgp,ilev) = (5.0*elements.m_v(ie,nm1,igp,jgp,ilev)-elements.m_v(ie,n0,igp,jgp,ilev))/4.0;
       elements.m_dp3d(ie,nm1,igp,jgp,ilev) = (5.0*elements.m_dp3d(ie,nm1,igp,jgp,ilev)-elements.m_dp3d(ie,n0,igp,jgp,ilev))/4.0;
  });
  ExecSpace::fence();

  // Stage 5: u5 = (5u1-u0)/4 + 3dt/4 RHS(u4), t_rhs = t + dt/5 + dt/5 + dt/3 + 2dt/3
  functor.set_rk_stage_data(nm1,np1,np1,3.0*dt/4.0,3.0*data.eta_ave_w/4.0,false);
  caar_monolithic_c(elements,functor,*be[np1],policy_pre,policy_post);
}

void caar_monolithic_c(Elements& elements, CaarFunctor& functor, BoundaryExchange& be,
                       Kokkos::TeamPolicy<ExecSpace,CaarFunctor::TagPreExchange>  policy_pre,
                       MDRangePolicy<ExecSpace,4> policy_post)
{
  // --- Pre boundary exchange
  profiling_resume();
  Kokkos::parallel_for("caar loop pre-boundary exchange", policy_pre, functor);
  ExecSpace::fence();
  profiling_pause();

  // Do the boundary exchange
  start_timer("caar_bexchV");
  be.exchange();

  // --- Post boundary echange
  profiling_resume();
  Kokkos::parallel_for("caar loop post-boundary exchange", policy_post, functor);
  ExecSpace::fence();
  profiling_pause();
  stop_timer("caar_bexchV");
}


} // namespace Homme
