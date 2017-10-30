#include "Derivative.hpp"
#include "Elements.hpp"
#include "Control.hpp"

#include "CaarFunctor.hpp"
#include "EulerStepFunctor.hpp"

#include "profiling.hpp"

namespace Homme
{

extern "C"
{

void init_control_caar_c (const int& nets, const int& nete, const int& num_elems,
                          const int& nm1, const int& n0, const int& np1,
                          const int& qn0, const Real& dt2, const Real& ps0,
                          const bool& compute_diagonstics, const Real& eta_ave_w,
                          CRCPtr& hybrid_a_ptr)
{
  Control& control = get_control ();

  // Adjust indices
  int nets_c = nets-1;
  int nete_c = nete;  // F90 ranges are closed, c ranges are open on the right
  int n0_c = n0-1;
  int nm1_c = nm1-1;
  int qn0_c = qn0==-1 ? qn0 : qn0-1;  // the -1 index has a special meaning, and -2 is not even contemplated
  int np1_c = np1-1;
  control.init(nets_c, nete_c, num_elems, nm1_c, n0_c, np1_c, qn0_c, dt2, ps0, compute_diagonstics, eta_ave_w, hybrid_a_ptr);
}

void init_control_euler_c (const int& nets, const int& nete, const int& qn0, const int& qsize, const Real& dt)
{
  Control& control = get_control ();

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
  Derivative& deriv = get_derivative ();
  deriv.init(dvv);
}

void init_elements_2d_c (const int& num_elems, CF90Ptr& D, CF90Ptr& Dinv, CF90Ptr& fcor,
                       CF90Ptr& spheremp, CF90Ptr& metdet, CF90Ptr& phis)
{
  Elements& r = get_elements ();
  r.init (num_elems);
  r.init_2d(D,Dinv,fcor,spheremp,metdet,phis);
}

void caar_pull_data_c (CF90Ptr& elem_state_v_ptr, CF90Ptr& elem_state_t_ptr, CF90Ptr& elem_state_dp3d_ptr,
                       CF90Ptr& elem_derived_phi_ptr, CF90Ptr& elem_derived_pecnd_ptr,
                       CF90Ptr& elem_derived_omega_p_ptr, CF90Ptr& elem_derived_vn0_ptr,
                       CF90Ptr& elem_derived_eta_dot_dpdn_ptr, CF90Ptr& elem_state_Qdp_ptr)
{
  Elements& r = get_elements();
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
  Elements& r = get_elements();
  r.push_to_f90_pointers(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr,
                         elem_derived_phi_ptr,elem_derived_pecnd_ptr,
                         elem_derived_omega_p_ptr,elem_derived_vn0_ptr,
                         elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr);
}

void euler_pull_data_c (CF90Ptr& elem_state_Qdp_ptr, CF90Ptr& vstar_ptr)
{
  Elements& r = get_elements();
  const Control& data = get_control();

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
  const Elements& r = get_elements();
  const Control& data = get_control();

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

void caar_pre_exchange_monolithic_c()
{
  // Get control structure
  Control& data  = get_control();

  // Retrieve the team size
  const int vectors_per_thread = DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  const int threads_per_team   = data.team_size;

  // Setup the policy
  Kokkos::TeamPolicy<ExecSpace> policy(data.num_elems, threads_per_team, vectors_per_thread);
  policy.set_chunk_size(1);

  // Create the functor
  CaarFunctor func(data);

  profiling_resume();
  // Dispatch parallel for
  Kokkos::parallel_for("main caar loop", policy, func);

  // Finalize
  ExecSpace::fence();
  profiling_pause();
}

//todo move to EulerStepFunctor as a static function
void advance_qdp_c()
{
  EulerStepFunctor::run();
}

} // extern "C"

} // namespace Homme
