#include "Derivative.hpp"
#include "Elements.hpp"
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

  int threads_per_team = DefaultThreadsDistribution<ExecSpace>::threads_per_team(num_elems);
  const char* var;
  var = getenv("OMP_NUM_THREADS");
  if (var!=0)
  {
    // the team size cannot exceed the value of OMP_NUM_THREADS, so se note it down
    threads_per_team = std::atoi(var);
  }

  var = getenv("HOMMEXX_TEAM_SIZE");
  if (var!=0)
  {
    // The user requested a team size for homme. We accept it, provided that
    // it does not exceed the value of OMP_NUM_THREADS. If it does exceed that,
    // we simply set it to OMP_NUM_THREADS.
    threads_per_team = std::min(std::atoi(var),threads_per_team);
  }

  int vectors_per_thread = DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  int teams_per_league   = ExecSpace::thread_pool_size() / (threads_per_team*vectors_per_thread);

  // Print the kokkos threads distribution
  std::cout << "-- Kokkos threads distribution --\n"
            << "   teams per league: " << teams_per_league << "\n"
            << "   threads per team: " << threads_per_team << "\n"
            << "   vectors per thread: " << vectors_per_thread << "\n"
            << "---------------------------------\n";
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

  ExecViewUnmanaged<Scalar *[NUM_LEV][2][NP][NP]>             vstar_exec = r.buffers.vstar;
  ExecViewUnmanaged<Scalar *[NUM_LEV][2][NP][NP]>::HostMirror vstar_host = Kokkos::create_mirror_view(vstar_exec);

  int iter=0;
  for (int ie=0; ie<data.num_elems; ++ie)
  {
    for (int ilev=0; ilev<NUM_LEV; ++ilev) {
      for (int iv=0; iv<VECTOR_SIZE; ++iv) {
        for (int idim=0; idim<2; ++idim) {
          for (int i=0; i<NP; ++i) {
            for (int j=0; j<NP; ++j, ++iter) {
              vstar_host(ie,idim,ilev,i,j)[iv] = vstar_ptr[iter];
            }
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

  ExecViewUnmanaged<Scalar *[QSIZE_D][NUM_LEV][NP][NP]>             qtens_exec = r.buffers.qtens;
  ExecViewUnmanaged<Scalar *[QSIZE_D][NUM_LEV][NP][NP]>::HostMirror qtens_host = Kokkos::create_mirror_view(qtens_exec);
  Kokkos::deep_copy(qtens_host, qtens_exec);

  int iter=0;
  for (int ie=0; ie<data.num_elems; ++ie)
  {
    for (int iq=0; iq<data.qsize; ++iq) {
      for (int ilev=0; ilev<NUM_LEV; ++ilev) {
        for (int iv=0; iv<VECTOR_SIZE; ++iv) {
          for (int i=0; i<NP; ++i) {
            for (int j=0; j<NP; ++j, ++iter) {
               qtens_ptr[iter] = qtens_host(ie,iq,ilev,i,j)[iv];
            }
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

  // Dispatch parallel for
  Kokkos::parallel_for("main caar loop", policy, func);

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
  const int vectors_per_thread = DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  const int threads_per_team   = data.team_size;

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
