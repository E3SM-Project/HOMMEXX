#include "Derivative.hpp"
#include "Elements.hpp"
#include "Control.hpp"
#include "Context.hpp"

#include "CaarFunctor.hpp"
#include "EulerStepFunctor.hpp"
#include "RemapFunctor.hpp"
#include "BoundaryExchange.hpp"
#include "BuffersManager.hpp"

#include "Utility.hpp"

#include "profiling.hpp"

namespace Homme
{

extern "C"
{

void init_control_caar_c(const int &nets, const int &nete, const int &num_elems,
                         const int &qn0, const Real &ps0, const int &rsplit,
                         CRCPtr &hybrid_a_ptr, CRCPtr &hybrid_b_ptr) {
  Control &control = Context::singleton().get_control();
  control.init(nets, nete, num_elems, qn0, ps0, rsplit, hybrid_a_ptr,
               hybrid_b_ptr);
}

void init_control_euler_c (const int& nets, const int& nete, const int& DSSopt,
                           const int& rhs_multiplier, const int& qn0, const int& qsize, const Real& dt,
                           const int& np1_qdp, const double& nu_p, const double& nu_q, const int& rhs_viss,
                           const int& limiter_option)
{
  Control& control = Context::singleton().get_control ();

  control.DSSopt = Control::DSSOption::from(DSSopt);
  control.rhs_multiplier = rhs_multiplier;
  control.nu_p = nu_p;
  control.nu_q = nu_q;
  control.rhs_viss = rhs_viss;
  control.limiter_option = limiter_option;

  // Adjust indices
  control.nets  = nets-1;
  control.nete  = nete;  // F90 ranges are closed, c ranges are open on the right, so this can stay the same
  control.qn0   = qn0-1;

  control.qsize = qsize;
  control.dt    = dt;

  control.np1_qdp = np1_qdp-1;
}

void init_derivative_c (CF90Ptr& dvv)
{
  Derivative& deriv = Context::singleton().get_derivative ();
  deriv.init(dvv);
}

void init_elements_2d_c (const int& num_elems, CF90Ptr& D, CF90Ptr& Dinv, CF90Ptr& fcor,
                         CF90Ptr& spheremp, CF90Ptr& rspheremp, CF90Ptr& metdet, CF90Ptr& phis)
{
  Elements& r = Context::singleton().get_elements ();
  r.init (num_elems);
  r.init_2d(D,Dinv,fcor,spheremp,rspheremp,metdet,phis);
}

void caar_pull_data_c (CF90Ptr& elem_state_v_ptr, CF90Ptr& elem_state_t_ptr, CF90Ptr& elem_state_dp3d_ptr,
                       CF90Ptr& elem_derived_phi_ptr, CF90Ptr& elem_derived_pecnd_ptr,
                       CF90Ptr& elem_derived_omega_p_ptr, CF90Ptr& elem_derived_vn0_ptr,
                       CF90Ptr& elem_derived_eta_dot_dpdn_ptr, CF90Ptr& elem_state_Qdp_ptr)
{
  Elements& r = Context::singleton().get_elements();
  // Copy data from f90 pointers to cxx views
  r.pull_from_f90_pointers(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr,
                           elem_derived_phi_ptr,elem_derived_pecnd_ptr,
                           elem_derived_omega_p_ptr,elem_derived_vn0_ptr,
                           elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr);
}

void caar_push_results_c (F90Ptr& elem_state_v_ptr, F90Ptr& elem_state_t_ptr, F90Ptr& elem_state_dp3d_ptr,
                          F90Ptr& elem_derived_phi_ptr, F90Ptr& elem_derived_pecnd_ptr,
                          F90Ptr& elem_derived_omega_p_ptr, F90Ptr& elem_derived_vn0_ptr,
                          F90Ptr& elem_derived_eta_dot_dpdn_ptr, F90Ptr& elem_state_Qdp_ptr)
{
  Elements& r = Context::singleton().get_elements();
  r.push_to_f90_pointers(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr,
                         elem_derived_phi_ptr,elem_derived_pecnd_ptr,
                         elem_derived_omega_p_ptr,elem_derived_vn0_ptr,
                         elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr);
}

void euler_pull_data_c (CF90Ptr& elem_derived_eta_dot_dpdn_ptr, CF90Ptr& elem_derived_omega_p_ptr,
                        CF90Ptr& elem_derived_divdp_proj_ptr, CF90Ptr& elem_derived_vn0_ptr,
                        CF90Ptr& elem_derived_dp_ptr, CF90Ptr& elem_derived_divdp_ptr,
                        CF90Ptr& elem_derived_dpdiss_biharmonic_ptr, CF90Ptr& elem_state_Qdp_ptr,
                        CF90Ptr& Qtens_biharmonic_ptr, CF90Ptr& qmin_ptr,
                        CF90Ptr& qmax_ptr)
{
  Elements& r = Context::singleton().get_elements();
  const Control& data = Context::singleton().get_control();

  sync_to_device(HostViewUnmanaged<const Real*[NUM_INTERFACE_LEV][NP][NP]>(
                   elem_derived_eta_dot_dpdn_ptr, data.num_elems),
                 r.m_eta_dot_dpdn);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_omega_p_ptr, data.num_elems),
                 r.m_omega_p);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_divdp_proj_ptr, data.num_elems),
                 r.m_derived_divdp_proj);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][2][NP][NP]>(
                   elem_derived_vn0_ptr, data.num_elems),
                 r.m_derived_vn0);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_dp_ptr, data.num_elems),
                 r.m_derived_dp);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_divdp_ptr, data.num_elems),
                 r.m_derived_divdp);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_dpdiss_biharmonic_ptr, data.num_elems),
                 r.m_derived_dpdiss_biharmonic);

  r.pull_qdp(elem_state_Qdp_ptr);

  sync_to_device(HostViewUnmanaged<const Real**[NUM_PHYSICAL_LEV][NP][NP]>(
                   Qtens_biharmonic_ptr, data.num_elems, data.qsize, NUM_PHYSICAL_LEV, NP, NP),
                 r.buffers.qtens_biharmonic);
  sync_to_device(HostViewUnmanaged<const Real**[NUM_PHYSICAL_LEV]>(
                   qmin_ptr, data.num_elems, data.qsize, NUM_PHYSICAL_LEV),
                 HostViewUnmanaged<const Real**[NUM_PHYSICAL_LEV]>(
                   qmax_ptr, data.num_elems, data.qsize, NUM_PHYSICAL_LEV),
                 r.buffers.qlim);
}

void euler_push_results_c (F90Ptr& elem_derived_eta_dot_dpdn_ptr, F90Ptr& elem_derived_omega_p_ptr,
                           F90Ptr& elem_derived_divdp_proj_ptr, F90Ptr& elem_state_Qdp_ptr,
                           F90Ptr& qmin_ptr, F90Ptr& qmax_ptr)
{
  Elements& r = Context::singleton().get_elements();
  const Control& data = Context::singleton().get_control();
  r.push_qdp(elem_state_Qdp_ptr);
  sync_to_host(r.buffers.qlim,
               HostViewUnmanaged<Real**[NUM_PHYSICAL_LEV]>(
                 qmin_ptr, data.num_elems, data.qsize, NUM_PHYSICAL_LEV),
               HostViewUnmanaged<Real**[NUM_PHYSICAL_LEV]>(
                 qmax_ptr, data.num_elems, data.qsize, NUM_PHYSICAL_LEV));
  sync_to_host(r.m_eta_dot_dpdn,
               HostViewUnmanaged<Real*[NUM_INTERFACE_LEV][NP][NP]>(
                 elem_derived_eta_dot_dpdn_ptr, data.num_elems));
  sync_to_host(r.m_omega_p,
               HostViewUnmanaged<Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                 elem_derived_omega_p_ptr, data.num_elems));
  sync_to_host(r.m_derived_divdp_proj,
               HostViewUnmanaged<Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                 elem_derived_divdp_proj_ptr, data.num_elems));
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

void u3_5stage_timestep_c(const int& nm1, const int& n0, const int& np1,
                          const Real& dt, const Real& eta_ave_w,
                          const bool& compute_diagonstics)
{
  // Get control and elements structures
  Control& data  = Context::singleton().get_control();
  Elements& elements = Context::singleton().get_elements();

  // Setup the policies
  auto policy_pre = Homme::get_default_team_policy<ExecSpace,CaarFunctor::TagPreExchange>(data.num_elems);
  MDRangePolicy<ExecSpace,4> policy_post({0,0,0,0},{data.num_elems,NP,NP,NUM_LEV}, {1,1,1,1});

  // Create the functor
  CaarFunctor functor(data, Context::singleton().get_elements(), Context::singleton().get_derivative());

  // Setup the boundary exchange
  std::shared_ptr<BoundaryExchange> be[NUM_TIME_LEVELS];
  std::map<std::string,std::shared_ptr<BoundaryExchange>>& be_map = Context::singleton().get_boundary_exchanges();
  for (int tl=0; tl<NUM_TIME_LEVELS; ++tl) {
    std::stringstream ss;
    ss << "caar tl " << tl;
    be[tl] = be_map[ss.str()];

    // If it was not yet created, create it and set it up
    if (!be[tl]) {
      std::shared_ptr<Connectivity> connectivity = Context::singleton().get_connectivity();
      std::shared_ptr<BuffersManager> buffers_manager = Context::singleton().get_buffers_manager();
      if (!buffers_manager->is_connectivity_set()) {
        // TODO: should we do this inside the get_buffers_manager in Context?
        buffers_manager->set_connectivity(connectivity);
      }

      // Set the views of this time level into this time level's boundary exchange
      be[tl] = std::make_shared<BoundaryExchange>(connectivity,buffers_manager);

      // Setup the boundary exchange
      be[tl]->set_num_fields(0,4);
      be[tl]->register_field(elements.m_u,1,tl);
      be[tl]->register_field(elements.m_v,1,tl);
      be[tl]->register_field(elements.m_t,1,tl);
      be[tl]->register_field(elements.m_dp3d,1,tl);
      be[tl]->registration_completed();

      // Set this BE in the Context's map
      be_map[ss.str()] = be[tl];
    }
  }

  // ===================== RK STAGES ===================== //

  // Stage 1: u1 = u0 + dt/5 RHS(u0),          t_rhs = t
  functor.set_rk_stage_data(n0,n0,nm1,dt/5.0,eta_ave_w/4.0,compute_diagonstics);
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
  functor.set_rk_stage_data(nm1,np1,np1,3.0*dt/4.0,3.0*eta_ave_w/4.0,false);
  caar_monolithic_c(elements,functor,*be[np1],policy_pre,policy_post);
}

void advance_qdp_c()
{
  EulerStepFunctor::run();
}

} // extern "C"

template <typename RemapAlg, bool rsplit>
void vertical_remap(Control &sim_state, Real *fort_ps_v) {
  RemapFunctor<RemapAlg, rsplit> remap(sim_state,
                                       Context::singleton().get_elements());
  remap.run_remap();
  remap.update_fortran_ps_v(fort_ps_v);
}

extern "C" {

// fort_ps_v is of type Real [NUM_ELEMS][NUM_TIME_LEVELS][NP][NP]
void vertical_remap_c(const int &remap_alg, const int &np1, const int &np1_qdp,
                      const Real &dt, Real *&fort_ps_v) {
  Control &sim_state = Context::singleton().get_control();
  sim_state.np1 = np1;
  sim_state.qn0 = np1_qdp;
  sim_state.dt = dt;
  const auto rsplit = sim_state.rsplit;
  if (remap_alg == PpmFixed::fortran_remap_alg) {
    if (rsplit != 0) {
      vertical_remap<PpmVertRemap<PpmFixed>, true>(sim_state, fort_ps_v);
    } else {
      vertical_remap<PpmVertRemap<PpmFixed>, false>(sim_state, fort_ps_v);
    }
  } else if (remap_alg == PpmMirrored::fortran_remap_alg) {
    if (rsplit != 0) {
      vertical_remap<PpmVertRemap<PpmMirrored>, true>(sim_state, fort_ps_v);
    } else {
      vertical_remap<PpmVertRemap<PpmMirrored>, false>(sim_state, fort_ps_v);
    }
  } else {
    MPI_Abort(0, -1);
  }
}

} // extern "C"

} // namespace Homme
