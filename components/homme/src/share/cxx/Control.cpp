#include "Control.hpp"

namespace Homme {

void Control::init(const int nets_in, const int nete_in,
                   const int num_elems_in, const int nm1_in,
                   const int n0_in, const int np1_in, const int qn0_in,
                   const Real dt_in, const Real ps0_in,
                   const bool compute_diagonstics_in,
                   const Real eta_ave_w_in,
                   const int rsplit_in, 
                   CRCPtr hybrid_am_ptr,
                   CRCPtr hybrid_ai_ptr,
                   CRCPtr hybrid_bm_ptr,
                   CRCPtr hybrid_bi_ptr) {
  nets = nets_in;
  nete = nete_in;
  num_elems = num_elems_in;
  n0 = n0_in;
  nm1 = nm1_in;
  np1 = np1_in;
  qn0 = qn0_in;
  dt  = dt_in;
  ps0 = ps0_in;
  compute_diagonstics = compute_diagonstics_in;
  eta_ave_w = eta_ave_w_in;
  rsplit = rsplit_in;
  hybrid_am = ExecViewManaged<Real[NUM_PHYSICAL_LEV]>(
      "Hybrid coordinates; coefficient A_midpoints");
  hybrid_ai = ExecViewManaged<Real[NUM_PHYSICAL_LEV+1]>(
      "Hybrid coordinates; coefficient A_interfaces");
  hybrid_bm = ExecViewManaged<Real[NUM_PHYSICAL_LEV]>(
      "Hybrid coordinates; coefficient B_midpoints");
  hybrid_bi = ExecViewManaged<Real[NUM_PHYSICAL_LEV+1]>(
      "Hybrid coordinates; coefficient B_interfaces");

  HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV]> host_hybrid_am(hybrid_am_ptr);
  HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV+1]> host_hybrid_ai(hybrid_ai_ptr);
  HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV]> host_hybrid_bm(hybrid_bm_ptr);
  HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV+1]> host_hybrid_bi(hybrid_bi_ptr);

//dest, source
  Kokkos::deep_copy(hybrid_am, host_hybrid_am);
  Kokkos::deep_copy(hybrid_ai, host_hybrid_ai);
  Kokkos::deep_copy(hybrid_bm, host_hybrid_bm);
  Kokkos::deep_copy(hybrid_bi, host_hybrid_bi);

/*
std::cout << "printing hybi! in CONTROL!!!!!!!!!!!!!! \n";
for(int ii = 0; ii < NUM_PHYSICAL_LEV+1; ++ii)
std::cout << "hybrid_bi " << ii << " " << hybrid_bi(ii) << "\n";
std::cout << "printing host hybi! in CONTROL!!!!!!!!!!!!!! \n";
for(int ii = 0; ii < NUM_PHYSICAL_LEV+1; ++ii)
std::cout << "HOST hybrid_bi " << ii << " " << host_hybrid_bi(ii) << "\n";
*/

  set_team_size();
}

void Control::set_team_size()
{
  // If the size requested at the beginning
  team_size = std::max(DefaultThreadsDistribution<ExecSpace>::threads_per_team(nete - nets), default_team_size);
}

} // namespace Homme
