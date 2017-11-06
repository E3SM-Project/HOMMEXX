#include "Derivative.hpp"
#include "Elements.hpp"
#include "Control.hpp"
#include "Context.hpp"

#include "CaarFunctor.hpp"
#include "EulerStepFunctor.hpp"
#include "BoundaryExchange.hpp"

#include "Utility.hpp"

#include "profiling.hpp"

namespace Homme
{

extern "C"
{

void init_control_caar_c (const int& nets, const int& nete, const int& num_elems,
                          const int& qn0, const Real& ps0, const int& rsplit, CRCPtr& hybrid_a_ptr)
{
  Control& control = Context::singleton().get_control ();

  control.init(nets, nete, num_elems, qn0, ps0, hybrid_a_ptr);

  control.rsplit = rsplit;
}

void init_control_euler_c (const int& nets, const int& nete, const int& qn0, const int& qsize, const Real& dt)
{
  Control& control = Context::singleton().get_control ();

  // Adjust indices
  control.nets  = nets-1;
  control.nete  = nete;  // F90 ranges are closed, c ranges are open on the right, so this can stay the same
  control.qn0   = qn0-1;

  control.qsize = qsize;
  control.dt    = dt;

  control.set_team_size();
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

void euler_pull_data_c (CF90Ptr& elem_state_Qdp_ptr, CF90Ptr& vstar_ptr)
{
  Elements& r = Context::singleton().get_elements();
  const Control& data = Context::singleton().get_control();

  // Copy data from f90 pointers to cxx views
  r.pull_qdp(elem_state_Qdp_ptr);

  ExecViewUnmanaged<Scalar *[2][NP][NP][NUM_LEV]>             vstar_exec = r.buffers.vstar;
  ExecViewUnmanaged<Scalar *[2][NP][NP][NUM_LEV]>::HostMirror vstar_host = Kokkos::create_mirror_view(vstar_exec);

  int iter=0;
  for (int ie=0; ie<data.num_elems; ++ie) {
    for (int k=0; k<NUM_PHYSICAL_LEV; ++k) {
      int ilev = k / VECTOR_SIZE;
      int iv   = k % VECTOR_SIZE;
      for (int idim=0; idim<2; ++idim) {
        for (int i=0; i<NP; ++i) {
          for (int j=0; j<NP; ++j, ++iter) {
            vstar_host(ie,idim,i,j,ilev)[iv] = vstar_ptr[iter];
          }
        }
      }
    }
  }
  Kokkos::deep_copy(vstar_exec, vstar_host);
}

void euler_push_results_c (F90Ptr& qtens_ptr)
{
  const Elements& r = Context::singleton().get_elements();
  const Control& data = Context::singleton().get_control();

  ExecViewUnmanaged<Scalar *[QSIZE_D][NP][NP][NUM_LEV]>             qtens_exec = r.buffers.qtens;
  ExecViewUnmanaged<Scalar *[QSIZE_D][NP][NP][NUM_LEV]>::HostMirror qtens_host = Kokkos::create_mirror_view(qtens_exec);
  Kokkos::deep_copy(qtens_host, qtens_exec);

  int iter=0;
  for (int ie=0; ie<data.num_elems; ++ie) {
    for (int iq=0; iq<data.qsize; ++iq) {
      for (int k=0; k<NUM_PHYSICAL_LEV; ++k) {
        int ilev = k / VECTOR_SIZE;
        int iv   = k % VECTOR_SIZE;
        for (int i=0; i<NP; ++i) {
          for (int j=0; j<NP; ++j, ++iter) {
             qtens_ptr[iter] = qtens_host(ie,iq,i,j,ilev)[iv];
          }
        }
      }
    }
  }
}

void caar_monolithic_c(Elements& elements, CaarFunctor& functor, BoundaryExchange& be,
                       Kokkos::TeamPolicy<ExecSpace,CaarFunctor::TagPreExchange>  policy_pre,
                       Kokkos::TeamPolicy<ExecSpace,CaarFunctor::TagPostExchange> policy_post)
{
  // --- Pre boundary exchange
  profiling_resume();
  Kokkos::parallel_for("caar loop pre-boundary exchange", policy_pre, functor);
  ExecSpace::fence();
  profiling_pause();

  // Do the boundary exchange
  Kokkos::deep_copy(elements.h_u,elements.m_u);
  Kokkos::deep_copy(elements.h_v,elements.m_v);
  Kokkos::deep_copy(elements.h_t,elements.m_t);
  Kokkos::deep_copy(elements.h_dp3d,elements.m_dp3d);

  be.exchange();

  Kokkos::deep_copy(elements.m_u,elements.h_u);
  Kokkos::deep_copy(elements.m_v,elements.h_v);
  Kokkos::deep_copy(elements.m_t,elements.h_t);
  Kokkos::deep_copy(elements.m_dp3d,elements.h_dp3d);

  // --- Post boundary echange
  profiling_resume();
  Kokkos::parallel_for("caar loop post-boundary exchange", policy_post, functor);
  ExecSpace::fence();
  profiling_pause();
}

void u3_5stage_timestep_c(const int& nm1, const int& n0, const int& np1,
                          const Real& dt, const Real& eta_ave_w,
                          const bool& compute_diagonstics)
{
  // Get control and elements structures
  Control& data  = Context::singleton().get_control();
  Elements& elements = Context::singleton().get_elements();

  // Retrieve the team size
  const int vectors_per_thread = DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  const int threads_per_team   = data.team_size;

  // Create the functor
  CaarFunctor functor(data, Context::singleton().get_elements(), Context::singleton().get_derivative());

  // Setup the policies
  Kokkos::TeamPolicy<ExecSpace,CaarFunctor::TagPreExchange>  policy_pre  (data.num_elems, threads_per_team, vectors_per_thread);
  Kokkos::TeamPolicy<ExecSpace,CaarFunctor::TagPostExchange> policy_post (data.num_elems, threads_per_team, vectors_per_thread);
  policy_pre.set_chunk_size(1);
  policy_post.set_chunk_size(1);

  // Setup the boundary exchange
  BoundaryExchange* be[NUM_TIME_LEVELS];
  for (int tl=0; tl<NUM_TIME_LEVELS; ++tl) {
    std::stringstream ss;
    ss << "caar tl " << tl;
    be[tl] = &get_boundary_exchange(ss.str());

    // Set the views of this time level into this time level's boundary exchange
    if (!be[tl]->is_registration_completed())
    {
      be[tl]->set_num_fields(0,4);
      be[tl]->register_field(elements.h_u,1,tl);
      be[tl]->register_field(elements.h_v,1,tl);
      be[tl]->register_field(elements.h_t,1,tl);
      be[tl]->register_field(elements.h_dp3d,1,tl);

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
  Kokkos::Experimental::md_parallel_for(
    MDRangePolicy<ExecSpace,4>({0,0,0,0},{data.num_elems,NP,NP,NUM_LEV}, {1,1,1,1}),
    [&](int ie, int igp, int jgp, int ilev) {
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
  // Get control structure
  Control& data = Context::singleton().get_control();

  // Retrieve the team size
  const int vectors_per_thread = DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  const int threads_per_team   = data.team_size;

  // Setup the policy
  Kokkos::TeamPolicy<ExecSpace> policy(data.num_elems, threads_per_team, vectors_per_thread);
  policy.set_chunk_size(1);

  // Create the functor
  EulerStepFunctor func(data);

  profiling_resume();
  // Dispatch parallel for
  Kokkos::parallel_for(policy, func);

  // Finalize
  ExecSpace::fence();
  profiling_pause();
}

} // extern "C"

} // namespace Homme
