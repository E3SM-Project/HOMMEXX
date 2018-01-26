#include "Control.hpp"
#include "Utility.hpp"

#include <random>

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

  //hybrid_am = ExecViewManaged<Real[NUM_PHYSICAL_LEV]>(
  //    "Hybrid coordinates; coefficient A_midpoints");
  hybrid_ai = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid coordinates; coefficient A_interfaces");
  //hybrid_bm = ExecViewManaged<Real[NUM_PHYSICAL_LEV]>(
  //    "Hybrid coordinates; coefficient B_midpoints");

  //HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV]> host_hybrid_am(hybrid_am_ptr);
  HostViewUnmanaged<const Real[NUM_INTERFACE_LEV]> host_hybrid_ai(hybrid_ai_ptr);
  //HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV]> host_hybrid_bm(hybrid_bm_ptr);

  //dest, source
  //Kokkos::deep_copy(hybrid_am, host_hybrid_am);
  Kokkos::deep_copy(hybrid_ai, host_hybrid_ai);
  //Kokkos::deep_copy(hybrid_bm, host_hybrid_bm);

  {
    // Only hybrid_ai(0) is needed.
    hybrid_ai0 = hybrid_ai_ptr[0];
  }

//  if (rsplit == 0) {
    hybrid_bi = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid coordinates; coefficient B_interfaces");
    HostViewUnmanaged<const Real[NUM_INTERFACE_LEV]> host_hybrid_bi(hybrid_bi_ptr);
    Kokkos::deep_copy(hybrid_bi, host_hybrid_bi);
//  }

//  assert(hybrid_am_ptr != nullptr);
  assert(hybrid_ai_ptr != nullptr);
//  assert(hybrid_bm_ptr != nullptr);
  assert(hybrid_bi_ptr != nullptr);
}

void Control::random_init(int num_elems_in, int seed) {
  const int min_value = std::numeric_limits<Real>::epsilon();
  const int max_value = 1.0 - min_value;
  hybrid_ai = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid a_interface coefs");
  hybrid_bi = ExecViewManaged<Real[NUM_INTERFACE_LEV]>(
      "Hybrid b_interface coefs");
  num_elems = num_elems_in;

  std::mt19937_64 engine(seed);
  ps0 = 1.0;

  // hybrid_a can technically range from 0 to 1 like hybrid_b,
  // but doing so makes enforcing the monotonicity of p = a + b difficult
  // So only go to 0.25
  genRandArray(hybrid_ai, engine, std::uniform_real_distribution<Real>(
                                     min_value, max_value / 4.0));

  HostViewManaged<Real[NUM_INTERFACE_LEV]> host_hybrid_ai("Host hybrid ai coefs");
  Kokkos::deep_copy(host_hybrid_ai, hybrid_ai);

  // p = a + b must be monotonically increasing
  // OG: what is this for? does a test require it?
  // (not critisizm, but i don't understand)
  const auto check_coords = [=](
      HostViewUnmanaged<Real[NUM_INTERFACE_LEV]> coords) {
    // Enforce the boundaries
    coords(0) = 0.0;
    coords(1) = 1.0;
    // Put them in order
    std::sort(coords.data(), coords.data() + coords.size());
//    Real p_prev = host_hybrid_ai(0) + coords(0);
    Real p_prev = hybrid_ai0 + coords(0);
    // Make certain they're all distinct
    for (int i = 1; i < NUM_INTERFACE_LEV; ++i) {
      if (coords(i) <=
          coords(i - 1) * (1.0 + std::numeric_limits<Real>::epsilon())) {
        return false;
      }
      Real p_cur = coords(i) + host_hybrid_ai(i);
      if (p_cur <= p_prev) {
        return false;
      }
      p_prev = p_cur;
    }
    return true;
  };
  genRandArray(hybrid_bi, engine,
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

Control::DSSOption::Enum Control::DSSOption::from (int DSSopt) {
  return static_cast<Enum>(DSSopt);
}

} // namespace Homme
