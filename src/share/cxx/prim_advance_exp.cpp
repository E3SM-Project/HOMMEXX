#include "CaarFunctor.hpp"
#include "Control.hpp"
#include "Context.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "ErrorDefs.hpp"
#include "mpi/BoundaryExchange.hpp"
#include "mpi/BuffersManager.hpp"
#include "Utility.hpp"

namespace Homme
{

void u3_5stage_timestep(const int nm1, const int n0, const int np1,
                        const Real dt, const Real eta_ave_w, const bool compute_diagnostics);
void caar_monolithic(Elements& elements, CaarFunctor& functor, BoundaryExchange& be,
                       Kokkos::TeamPolicy<ExecSpace,CaarFunctor::TagPreExchange>&  policy_pre,
                       Kokkos::RangePolicy<ExecSpace,CaarFunctor::TagPostExchange>& policy_post);
void advance_hypervis_dp (const int np1, const Real dt, const Real eta_ave_w);

// -------------- IMPLEMENTATIONS -------------- //

void prim_advance_exp (const int nm1, const int n0, const int np1,
                       const Real dt, const bool compute_diagnostics)
{
  // Get control and simulation params
  Control&          data   = Context::singleton().get_control();
  SimulationParams& params = Context::singleton().get_simulation_params();

  // Note: In the following, all the checks are superfluous, since we already check that
  //       the options are supported when we init the simulation params. However, this way
  //       we remind ourselves that in these cases there is some missing code to convert from Fortran

  // Get time level info, and determine the tracers time level
  TimeLevel& tl = Context::singleton().get_time_level();
  data.n0_qdp= -1;
  if (params.moisture == MoistDry::MOIST) {
    tl.update_tracers_levels(params.qsplit);
    data.n0_qdp = tl.n0_qdp;
  }

  // Set eta_ave_w
  int method = params.time_step_type;
  Real eta_ave_w = 1.0/params.qsplit;

  if (params.time_step_type==0) {
    Errors::runtime_abort("[prim_advance_exp_iter",Errors::err_not_implemented);
  } else if (params.time_step_type==1) {
    Errors::runtime_abort("[prim_advance_exp_iter",Errors::err_not_implemented);
  }

#ifndef CAM
  // if "prescribed wind" set dynamics explicitly and skip time-integration
  if (params.prescribed_wind) {
    Errors::runtime_abort("'prescribed wind' functionality not yet available in C++ build.\n",
                           Errors::err_not_implemented);
  }
#endif

  // Perform time-advance
  switch (method) {
    case 5:
      // Perform RK stages
      u3_5stage_timestep(nm1, n0, np1, dt, eta_ave_w, compute_diagnostics);
      break;
    default:
      Errors::runtime_abort("[prim_advance_exp_iter",Errors::err_not_implemented);
  }

#ifdef ENERGY_DIAGNOSTICS
  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
  }
#endif

  if (params.time_step_type==0) {
    Errors::runtime_abort("'advance hypervis lf' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
    // call advance_hypervis_lf(edge3p1,elem,hvcoord,hybrid,deriv,nm1,n0,np1,nets,nete,dt_vis)

  } else if (params.time_step_type<=10) {
    GPTLstart("advance_hypervis_dp");
    advance_hypervis_dp(np1,dt,eta_ave_w);
    GPTLstop("advance_hypervis_dp");
  }

#ifdef ENERGY_DIAGNOSTICS
  if (compute_diagnostics) {
    Errors::runtime_abort("'compute diagnostic' functionality not yet available in C++ build.\n",
                          Errors::err_not_implemented);
  }
#endif
}

void u3_5stage_timestep(const int nm1, const int n0, const int np1,
                        const Real dt, const Real eta_ave_w, const bool compute_diagnostics)
{
  GPTLstart("U3-5stage_timestep");
  // Get control and elements structures
  Control& data  = Context::singleton().get_control();
  Elements& elements = Context::singleton().get_elements();

  // Setup the policies
  auto policy_pre = Homme::get_default_team_policy<ExecSpace,CaarFunctor::TagPreExchange>(data.num_elems);
  Kokkos::RangePolicy<ExecSpace,CaarFunctor::TagPostExchange> policy_post(0, data.num_elems*NP*NP*NUM_LEV);

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
      be[tl]->register_field(elements.m_v,tl,2,0);
      be[tl]->register_field(elements.m_t,1,tl);
      be[tl]->register_field(elements.m_dp3d,1,tl);
      be[tl]->registration_completed();
    }
  }

  // ===================== RK STAGES ===================== //

  // Stage 1: u1 = u0 + dt/5 RHS(u0),          t_rhs = t
  functor.set_rk_stage_data(n0,n0,nm1,dt/5.0,eta_ave_w/4.0,compute_diagnostics);
  caar_monolithic(elements,functor,*be[nm1],policy_pre,policy_post);

  // Stage 2: u2 = u0 + dt/5 RHS(u1),          t_rhs = t + dt/5
  functor.set_rk_stage_data(n0,nm1,np1,dt/5.0,0.0,false);
  caar_monolithic(elements,functor,*be[np1],policy_pre,policy_post);

  // Stage 3: u3 = u0 + dt/3 RHS(u2),          t_rhs = t + dt/5 + dt/5
  functor.set_rk_stage_data(n0,np1,np1,dt/3.0,0.0,false);
  caar_monolithic(elements,functor,*be[np1],policy_pre,policy_post);

  // Stage 4: u4 = u0 + 2dt/3 RHS(u3),         t_rhs = t + dt/5 + dt/5 + dt/3
  functor.set_rk_stage_data(n0,np1,np1,2.0*dt/3.0,0.0,false);
  caar_monolithic(elements,functor,*be[np1],policy_pre,policy_post);

  // Compute (5u1-u0)/4 and store it in timelevel nm1
  Kokkos::parallel_for(
    policy_post,
    KOKKOS_LAMBDA(const CaarFunctor::TagPostExchange&, const int it) {
       const int ie = it / (NP*NP*NUM_LEV);
       const int igp = (it / (NP*NUM_LEV)) % NP;
       const int jgp = (it / NUM_LEV) % NP;
       const int ilev = it % NUM_LEV;
       elements.m_t(ie,nm1,igp,jgp,ilev) = (5.0*elements.m_t(ie,nm1,igp,jgp,ilev)-elements.m_t(ie,n0,igp,jgp,ilev))/4.0;
       elements.m_v(ie,nm1,0,igp,jgp,ilev) = (5.0*elements.m_v(ie,nm1,0,igp,jgp,ilev)-elements.m_v(ie,n0,0,igp,jgp,ilev))/4.0;
       elements.m_v(ie,nm1,1,igp,jgp,ilev) = (5.0*elements.m_v(ie,nm1,1,igp,jgp,ilev)-elements.m_v(ie,n0,1,igp,jgp,ilev))/4.0;
       elements.m_dp3d(ie,nm1,igp,jgp,ilev) = (5.0*elements.m_dp3d(ie,nm1,igp,jgp,ilev)-elements.m_dp3d(ie,n0,igp,jgp,ilev))/4.0;
  });
  ExecSpace::fence();

  // Stage 5: u5 = (5u1-u0)/4 + 3dt/4 RHS(u4), t_rhs = t + dt/5 + dt/5 + dt/3 + 2dt/3
  functor.set_rk_stage_data(nm1,np1,np1,3.0*dt/4.0,3.0*eta_ave_w/4.0,false);
  caar_monolithic(elements,functor,*be[np1],policy_pre,policy_post);
  GPTLstop("U3-5stage_timestep");
}

void caar_monolithic(Elements& elements, CaarFunctor& functor, BoundaryExchange& be,
                     Kokkos::TeamPolicy<ExecSpace,CaarFunctor::TagPreExchange>&  policy_pre,
                     Kokkos::RangePolicy<ExecSpace,CaarFunctor::TagPostExchange>& policy_post)
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
