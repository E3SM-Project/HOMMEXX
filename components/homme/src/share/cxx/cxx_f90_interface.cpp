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
                           const int& np1_qdp, const double& nu_p, const double& nu_q,
                           const int& limiter_option)
{
  Control& control = Context::singleton().get_control ();

  control.DSSopt = Control::DSSOption::from(DSSopt);
  control.rhs_multiplier = rhs_multiplier;
  control.nu_p = nu_p;
  control.nu_q = nu_q;
  control.limiter_option = limiter_option;

  // Adjust indices
  control.nets  = nets-1;
  control.nete  = nete;  // F90 ranges are closed, c ranges are open on the right, so this can stay the same
  control.qn0   = qn0-1;

  control.qsize = qsize;
  control.dt    = dt;

  control.np1_qdp = np1_qdp-1;
}

void init_euler_neighbor_minmax_c (const int& qsize)
{
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange("min max Euler");
  if (!be.is_registration_completed()) {
    Elements& elements = Context::singleton().get_elements();

    std::shared_ptr<Connectivity> connectivity = Context::singleton().get_connectivity();
    std::shared_ptr<BuffersManager> buffers_manager = Context::singleton().get_buffers_manager(MPI_EXCHANGE_MIN_MAX);

    be.set_connectivity(connectivity);
    be.set_buffers_manager(buffers_manager);
    be.set_num_fields(qsize,0,0);
    be.register_min_max_fields(elements.buffers.qlim,qsize,0);
    be.registration_completed();
  }
}

void euler_neighbor_minmax_c (const int& nets, const int& nete)
{
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange("min max Euler");
  be.exchange_min_max(nets-1, nete);
}

void euler_neighbor_minmax_start_c (const int& nets, const int& nete)
{
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange("min max Euler");
  be.pack_and_send_min_max(nets-1, nete);
}

void euler_neighbor_minmax_finish_c (const int& nets, const int& nete)
{
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange("min max Euler");
  be.recv_and_unpack_min_max(nets-1, nete);
}

void init_derivative_c (CF90Ptr& dvv)
{
  Derivative& deriv = Context::singleton().get_derivative ();
  deriv.init(dvv);
}

void init_elements_2d_c (const int& num_elems, CF90Ptr& D, CF90Ptr& Dinv, CF90Ptr& fcor,
                         CF90Ptr& mp, CF90Ptr& spheremp, CF90Ptr& rspheremp,
                         CF90Ptr& metdet, CF90Ptr& metinv, CF90Ptr& vec_sph2cart,
                         CF90Ptr& tensorVisc, CF90Ptr& phis)
{
  Elements& r = Context::singleton().get_elements ();
  r.init (num_elems);
  r.init_2d(D,Dinv,fcor,mp,spheremp,rspheremp,metdet,metinv,vec_sph2cart,tensorVisc,phis);
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

void euler_pull_qmin_qmax_c (F90Ptr& qmin_ptr, F90Ptr& qmax_ptr)
{
  Elements& r = Context::singleton().get_elements();
  const Control& data = Context::singleton().get_control();

  sync_to_device(HostViewUnmanaged<const Real**[NUM_PHYSICAL_LEV]>(
                   qmin_ptr, data.num_elems, data.qsize, NUM_PHYSICAL_LEV),
                 HostViewUnmanaged<const Real**[NUM_PHYSICAL_LEV]>(
                   qmax_ptr, data.num_elems, data.qsize, NUM_PHYSICAL_LEV),
                 r.buffers.qlim);
}

void euler_pull_data_c (CF90Ptr& elem_derived_eta_dot_dpdn_ptr, CF90Ptr& elem_derived_omega_p_ptr,
                        CF90Ptr& elem_derived_divdp_proj_ptr, CF90Ptr& elem_derived_vn0_ptr,
                        CF90Ptr& elem_derived_dp_ptr, CF90Ptr& elem_derived_divdp_ptr,
                        CF90Ptr& elem_derived_dpdiss_biharmonic_ptr, CF90Ptr& elem_state_Qdp_ptr,
                        CF90Ptr& Qtens_biharmonic_ptr)
{
  Elements& elements = Context::singleton().get_elements();
  const Control& data = Context::singleton().get_control();

  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_eta_dot_dpdn_ptr, data.num_elems),
                 elements.m_eta_dot_dpdn);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_omega_p_ptr, data.num_elems),
                 elements.m_omega_p);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_divdp_proj_ptr, data.num_elems),
                 elements.m_derived_divdp_proj);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][2][NP][NP]>(
                   elem_derived_vn0_ptr, data.num_elems),
                 elements.m_derived_vn0);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_dp_ptr, data.num_elems),
                 elements.m_derived_dp);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_divdp_ptr, data.num_elems),
                 elements.m_derived_divdp);
  sync_to_device(HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                   elem_derived_dpdiss_biharmonic_ptr, data.num_elems),
                 elements.m_derived_dpdiss_biharmonic);

  elements.pull_qdp(elem_state_Qdp_ptr);

  sync_to_device(HostViewUnmanaged<const Real**[NUM_PHYSICAL_LEV][NP][NP]>(
                   Qtens_biharmonic_ptr, data.num_elems, data.qsize, NUM_PHYSICAL_LEV, NP, NP),
                 elements.buffers.qtens_biharmonic);
}

void euler_push_results_c (F90Ptr& elem_derived_eta_dot_dpdn_ptr, F90Ptr& elem_derived_omega_p_ptr,
                           F90Ptr& elem_derived_divdp_proj_ptr, F90Ptr& elem_state_Qdp_ptr,
                           F90Ptr& qmin_ptr, F90Ptr& qmax_ptr)
{
  Elements& elements = Context::singleton().get_elements();
  const Control& data = Context::singleton().get_control();
  elements.push_qdp(elem_state_Qdp_ptr);
  sync_to_host(elements.buffers.qlim,
               HostViewUnmanaged<Real**[NUM_PHYSICAL_LEV]>(
                 qmin_ptr, data.num_elems, data.qsize, NUM_PHYSICAL_LEV),
               HostViewUnmanaged<Real**[NUM_PHYSICAL_LEV]>(
                 qmax_ptr, data.num_elems, data.qsize, NUM_PHYSICAL_LEV));
  sync_to_host(elements.m_eta_dot_dpdn,
               HostViewUnmanaged<Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                 elem_derived_eta_dot_dpdn_ptr, data.num_elems));
  sync_to_host(elements.m_omega_p,
               HostViewUnmanaged<Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                 elem_derived_omega_p_ptr, data.num_elems));
  sync_to_host(elements.m_derived_divdp_proj,
               HostViewUnmanaged<Real*[NUM_PHYSICAL_LEV][NP][NP]>(
                 elem_derived_divdp_proj_ptr, data.num_elems));
}

void caar_monolithic_c(Elements& elements, CaarFunctor& functor, BoundaryExchange& be,
                       Kokkos::TeamPolicy<ExecSpace,CaarFunctor::TagPreExchange>  policy_pre,
                       Kokkos::RangePolicy<ExecSpace,CaarFunctor::TagPostExchange> policy_post)
{
  // --- Pre boundary exchange
  GPTLstart("caar_monolithic_c-pre");
  Kokkos::parallel_for("caar loop pre-boundary exchange", policy_pre, functor);
  ExecSpace::fence();
  GPTLstop("caar_monolithic_c-pre");

  // Do the boundary exchange
  GPTLstart("caar_bexchV");
  be.exchange();

  // --- Post boundary echange
  Kokkos::parallel_for("caar loop post-boundary exchange", policy_post, functor);
  ExecSpace::fence();
  GPTLstop("caar_bexchV");
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
  caar_monolithic_c(elements,functor,*be[np1],policy_pre,policy_post);
}

void advance_qdp_c(const int& rhs_viss)
{
  Control& control = Context::singleton().get_control ();
  control.rhs_viss = rhs_viss;

  EulerStepFunctor::run();
}

void euler_exchange_qdp_dss_var_c ()
{
  Control data = Context::singleton().get_control();
  Elements& elements = Context::singleton().get_elements();

  // Note: we have three separate BE structures, all of which register qdp. They
  //       differ only in the last field registered. This allows us to have a SINGLE
  //       mpi call to exchange qsize+1 fields, rather than one for qdp and one for the
  //       last DSS variable.
  // TODO: move this setup in init_control_euler and move that function one stack frame up
  //       of euler_step in F90, making it set the common parameters to all euler_steps
  //       calls (nets, nete, dt, nu_p, nu_q)

  std::stringstream ss;
  ss << "exchange qdp "
     << (data.DSSopt == Control::DSSOption::eta ?
         "eta" :
         data.DSSopt == Control::DSSOption::omega ?
         "omega" :
         "div_vdp_ave")
     << " " << data.np1_qdp;

  const std::shared_ptr<BoundaryExchange> be_qdp_dss_var =
    Context::singleton().get_boundary_exchange(ss.str());

  const auto& dss_var = (data.DSSopt==Control::DSSOption::eta ? elements.m_eta_dot_dpdn :
                         (data.DSSopt==Control::DSSOption::omega ? elements.m_omega_p   :
                          elements.m_derived_divdp_proj));

  if (!be_qdp_dss_var->is_registration_completed()) {
    // If it is the first time we call this method, we need to set up the BE
    std::shared_ptr<BuffersManager> buffers_manager = Context::singleton().get_buffers_manager(MPI_EXCHANGE);
    be_qdp_dss_var->set_buffers_manager(buffers_manager);
    be_qdp_dss_var->set_num_fields(0,0,data.qsize+1);
    be_qdp_dss_var->register_field(elements.m_qdp,data.np1_qdp,data.qsize,0);
    be_qdp_dss_var->register_field(dss_var);
    be_qdp_dss_var->registration_completed();
  }

  be_qdp_dss_var->exchange();

  EulerStepFunctor::apply_rspheremp();
}

} // extern "C"

template <bool rsplit, template <int, typename...> class RemapAlg,
          typename... RemapOptions>
void vertical_remap(Control &sim_state, Real *fort_ps_v) {
  RemapFunctor<rsplit, RemapAlg, RemapOptions...> remap(
      sim_state, Context::singleton().get_elements());
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
      vertical_remap<true, PpmVertRemap, PpmFixed>(sim_state, fort_ps_v);
    } else {
      vertical_remap<false, PpmVertRemap, PpmFixed>(sim_state, fort_ps_v);
    }
  } else if (remap_alg == PpmMirrored::fortran_remap_alg) {
    if (rsplit != 0) {
      vertical_remap<true, PpmVertRemap, PpmMirrored>(sim_state, fort_ps_v);
    } else {
      vertical_remap<false, PpmVertRemap, PpmMirrored>(sim_state, fort_ps_v);
    }
  } else {
    MPI_Abort(0, -1);
  }
}

} // extern "C"

} // namespace Homme
