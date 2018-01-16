#include "Control.hpp"

namespace Homme {

void Control::init(const int nets_in, const int nete_in, const int num_elems_in,
                   const int qn0_in, const Real ps0_in, 
                   const int rsplit_in,
                   CRCPtr hybrid_am_ptr,
                   CRCPtr hybrid_ai_ptr,
                   CRCPtr hybrid_bm_ptr,
                   CRCPtr hybrid_bi_ptr) {
  nets = nets_in;
  nete = nete_in;
  num_elems = num_elems_in;
  qn0 = qn0_in;
  ps0 = ps0_in;
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
}

void Control::set_rk_stage_data(const int nm1_in, const int n0_in, const int np1_in,
                                const Real dt_in, const Real eta_ave_w_in,
                                const bool compute_diagonstics_in)
{
  n0 = n0_in;
  nm1 = nm1_in;
  np1 = np1_in;

  dt = dt_in;
  eta_ave_w = eta_ave_w_in;
  compute_diagonstics = compute_diagonstics_in;
}

} // namespace Homme
