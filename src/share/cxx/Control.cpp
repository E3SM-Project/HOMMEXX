#include "Control.hpp"

namespace Homme {

void Control::init(const int nets_in, const int nete_in, const int num_elems_in,
                   const int qn0_in, const Real ps0_in, CRCPtr hybrid_a_ptr) {
  nets = nets_in;
  nete = nete_in;
  num_elems = num_elems_in;
  qn0 = qn0_in;
  ps0 = ps0_in;
  hybrid_a = ExecViewManaged<Real[NUM_LEV_P]>(
      "Hybrid coordinates; translates between pressure and velocity");

  HostViewUnmanaged<const Real[NUM_LEV_P]> host_hybrid_a(hybrid_a_ptr);
  Kokkos::deep_copy(hybrid_a, host_hybrid_a);

  set_team_size();
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

void Control::set_team_size()
{
  // If the size requested at the beginning
  team_size = std::max(DefaultThreadsDistribution<ExecSpace>::threads_per_team(nete - nets), default_team_size);
}

} // namespace Homme