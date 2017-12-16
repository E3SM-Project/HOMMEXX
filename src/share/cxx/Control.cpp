#include "Control.hpp"
#include "Utility.hpp"

#include <random>

namespace Homme {

void Control::init(const int nets_in, const int nete_in, const int num_elems_in,
                   const int nm1_in, const int n0_in, const int np1_in,
                   const int qn0_in, const Real dt_in, const Real ps0_in,
                   const bool compute_diagonstics_in, const Real eta_ave_w_in,
                   CRCPtr hybrid_a_ptr, CRCPtr hybrid_b_ptr) {
  nets = nets_in;
  nete = nete_in;
  num_elems = num_elems_in;
  n0 = n0_in;
  nm1 = nm1_in;
  np1 = np1_in;
  qn0 = qn0_in;
  dt = dt_in;
  ps0 = ps0_in;
  compute_diagonstics = compute_diagonstics_in;
  eta_ave_w = eta_ave_w_in;
  hybrid_a = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid a coordinates; translates between pressure and velocity");
  hybrid_b = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid b coordinates; translates between pressure and velocity");

  HostViewUnmanaged<const Real[NUM_LEV_P]> host_hybrid_a(hybrid_a_ptr);
  Kokkos::deep_copy(hybrid_a, host_hybrid_a);

  HostViewUnmanaged<const Real[NUM_LEV_P]> host_hybrid_b(hybrid_b_ptr);
  Kokkos::deep_copy(hybrid_b, host_hybrid_b);

  set_team_size();
}

void Control::set_team_size() {
  // If the size requested at the beginning
  team_size = std::max(
      DefaultThreadsDistribution<ExecSpace>::threads_per_team(nete - nets),
      default_team_size);
}

void Control::random_init(int num_elems_in, int seed) {
  const int min_value = std::numeric_limits<Real>::epsilon();
  const int max_value = 1.0 - min_value;
  hybrid_a = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid a coordinates; translates between pressure and velocity");
  hybrid_b = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid b coordinates; translates between pressure and velocity");
  num_elems = num_elems_in;
  struct check_coords {
    check_coords(bool reversed) : m_reversed(reversed) {}

    bool operator()(HostViewUnmanaged<Real[NUM_INTERFACE_LEV]> coords) const {
      // Enforce the boundaries
      coords(0) = 0.0;
      coords(1) = 1.0;
      // Put them in order
      std::sort(coords.data(), coords.data() + coords.size(), *this);
      // Make certain they're all distinct
      for (int i = 1; i < NUM_INTERFACE_LEV; ++i) {
        if ((m_reversed == false &&
             coords(i) <= coords(i - 1) *
                              (1.0 - std::numeric_limits<Real>::epsilon())) ||
            (m_reversed == true &&
             coords(i) >= coords(i - 1) *
                              (1.0 - std::numeric_limits<Real>::epsilon()))) {
          return false;
        }
      }
      return true;
    }

    bool operator()(Real lhs, Real rhs) const {
      if (m_reversed) {
        return lhs > rhs;
      } else {
        return lhs < rhs;
      }
    }

    const bool m_reversed;
  };
  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<Real> pdf(min_value, max_value);
  genRandArray(hybrid_a, engine, pdf, check_coords(false));
  genRandArray(hybrid_b, engine, pdf, check_coords(true));
}

} // namespace Homme
