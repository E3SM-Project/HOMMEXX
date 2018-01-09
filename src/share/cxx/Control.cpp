#include "Control.hpp"
#include "Utility.hpp"

#include <random>

namespace Homme {

void Control::init(const int nets_in, const int nete_in, const int num_elems_in,
                   const int qn0_in, const Real ps0_in, const int rsplit_in,
                   CRCPtr hybrid_a_ptr, CRCPtr hybrid_b_ptr) {
  nets = nets_in;
  nete = nete_in;
  num_elems = num_elems_in;
  qn0 = qn0_in;
  ps0 = ps0_in;
  rsplit = rsplit_in;
  hybrid_a = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid a coordinates; translates between pressure and velocity");
  hybrid_b = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid b coordinates; translates between pressure and velocity");

  assert(hybrid_a_ptr != nullptr);
  assert(hybrid_b_ptr != nullptr);

  HostViewUnmanaged<const Real[NUM_INTERFACE_LEV]> host_hybrid_a(hybrid_a_ptr);
  Kokkos::deep_copy(hybrid_a, host_hybrid_a);

  HostViewUnmanaged<const Real[NUM_INTERFACE_LEV]> host_hybrid_b(hybrid_b_ptr);
  Kokkos::deep_copy(hybrid_b, host_hybrid_b);
}

void Control::random_init(int num_elems_in, int seed) {
  const int min_value = std::numeric_limits<Real>::epsilon();
  const int max_value = 1.0 - min_value;
  hybrid_a = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid a coordinates; translates between pressure and velocity");
  hybrid_b = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid b coordinates; translates between pressure and velocity");
  num_elems = num_elems_in;

  std::mt19937_64 engine(seed);
  ps0 = 1.0;

  // hybrid_a can technically range from 0 to 1 like hybrid_b,
  // but doing so makes enforcing the monotonicity of p = a + b difficult
  // So only go to 0.25
  genRandArray(hybrid_a, engine, std::uniform_real_distribution<Real>(
                                     min_value, max_value / 4.0));

  // p = a + b must be monotonically increasing
  const auto check_coords = [=](
      HostViewUnmanaged<Real[NUM_INTERFACE_LEV]> coords) {
    // Enforce the boundaries
    coords(0) = 0.0;
    coords(1) = 1.0;
    // Put them in order
    std::sort(coords.data(), coords.data() + coords.size());
    Real p_prev = hybrid_a(0) + coords(0);
    // Make certain they're all distinct
    for (int i = 1; i < NUM_INTERFACE_LEV; ++i) {
      if (coords(i) <=
          coords(i - 1) * (1.0 + std::numeric_limits<Real>::epsilon())) {
        return false;
      }
      Real p_cur = coords(i) + hybrid_a(i);
      if (p_cur <= p_prev) {
        return false;
      }
      p_prev = p_cur;
    }
    return true;
  };
  genRandArray(hybrid_b, engine,
               std::uniform_real_distribution<Real>(min_value, max_value),
               check_coords);
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
