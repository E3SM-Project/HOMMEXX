#include "Derivative.hpp"
#include "Region.hpp"
#include "Control.hpp"

#include "CaarFunctor.hpp"
#include "EulerStepFunctor.hpp"

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

void init_control_euler_c (const int& nets, const int& nete, const int& qn0)
{
  Control& control = get_control ();

  // Adjust indices
  control.nets = nets-1;
  control.nete = nete;  // F90 ranges are closed, c ranges are open on the right, so this can stay the same
  control.qn0  = qn0-1;
}

void init_derivative_c (CF90Ptr& dvv, CF90Ptr& integration_matrix, CF90Ptr& boundary_interp_matrix)
{
  Derivative& deriv = get_derivative ();
  deriv.init(dvv,integration_matrix,boundary_interp_matrix);
}

void init_region_2d_c (const int& num_elems, CF90Ptr& D, CF90Ptr& Dinv, CF90Ptr& fcor,
                       CF90Ptr& spheremp, CF90Ptr& metdet, CF90Ptr& phis)
{
  Region& r = get_region ();
  r.init (num_elems);
  r.init_2d(D,Dinv,fcor,spheremp,metdet,phis);
}

void caar_copy_f90_data_to_region_c (CF90Ptr& elem_state_v_ptr, CF90Ptr& elem_state_t_ptr, CF90Ptr& elem_state_dp3d_ptr,
                                     CF90Ptr& elem_derived_phi_ptr, CF90Ptr& elem_derived_pecnd_ptr,
                                     CF90Ptr& elem_derived_omega_p_ptr, CF90Ptr& elem_derived_vn0_ptr,
                                     CF90Ptr& elem_derived_eta_dot_dpdn_ptr, CF90Ptr& elem_state_Qdp_ptr)
{
  Region& r = get_region();
  // Copy data from f90 pointers to cxx views
  r.pull_from_f90_pointers(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr,
                           elem_derived_phi_ptr,elem_derived_pecnd_ptr,
                           elem_derived_omega_p_ptr,elem_derived_vn0_ptr,
                           elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr);
}

void caar_copy_region_data_to_f90_c (F90Ptr& elem_state_v_ptr, F90Ptr& elem_state_t_ptr, F90Ptr& elem_state_dp3d_ptr,
                                     F90Ptr& elem_derived_phi_ptr, F90Ptr& elem_derived_pecnd_ptr,
                                     F90Ptr& elem_derived_omega_p_ptr, F90Ptr& elem_derived_vn0_ptr,
                                     F90Ptr& elem_derived_eta_dot_dpdn_ptr, F90Ptr& elem_state_Qdp_ptr)
{
  Region& r = get_region();
  r.push_to_f90_pointers(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr,
                         elem_derived_phi_ptr,elem_derived_pecnd_ptr,
                         elem_derived_omega_p_ptr,elem_derived_vn0_ptr,
                         elem_derived_eta_dot_dpdn_ptr, elem_state_Qdp_ptr);
}

void euler_copy_f90_data_to_region_c (CF90Ptr& ustar, CF90Ptr& vstar)
{
  Region& r = get_region();
  // Copy data from f90 pointers to cxx views
  r.pull_from_f90_buffers(ustar, vstar, nullptr, nullptr);
}

void euler_copy_region_data_to_f90_c (F90Ptr& ustar, F90Ptr& vstar)
{
  Region& r = get_region();
  r.push_to_f90_buffers(ustar, vstar, nullptr, nullptr);
}

void caar_pre_exchange_monolithic_c()
{
  // Get control structure
  Control& data  = get_control();

  // Retrieve the team size
  DefaultThreadsDistribution<ExecSpace>::init();
  const int vectors_per_thread = DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  const int threads_per_team   = DefaultThreadsDistribution<ExecSpace>::threads_per_team(data.nete-data.nets);

  // Setup the policy
  Kokkos::TeamPolicy<ExecSpace> policy(data.num_elems, threads_per_team, vectors_per_thread);
  policy.set_chunk_size(1);

  // Create the functor
  CaarFunctor func(data);

  // Dispatch parallel for
  Kokkos::parallel_for(policy, func);

  // Finalize
  ExecSpace::fence();
}

void advance_qdp_c()
{
  // Get control structure
  Control& data = get_control();

  // Create the functor
  EulerStepFunctor func(data);

  // Retrieve the team size
  DefaultThreadsDistribution<ExecSpace>::init();
  const int vectors_per_thread = DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  const int threads_per_team   = DefaultThreadsDistribution<ExecSpace>::threads_per_team(data.nete-data.nets);

  // Setup the policy
  Kokkos::TeamPolicy<ExecSpace> policy(data.num_elems, threads_per_team, vectors_per_thread);
  policy.set_chunk_size(1);

  // Dispatch parallel for
  Kokkos::parallel_for(policy, func);

  // Finalize
  ExecSpace::fence();
}

} // extern "C"

} // namespace Homme
