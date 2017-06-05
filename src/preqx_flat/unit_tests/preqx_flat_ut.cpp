#include <catch/catch.hpp>

Real compare_answers(Real target, Real computed, Real relative_coeff = 1.0) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target - computed) / denom;
}

TEST_CASE("monolithic compute_and_apply_rhs", "compute_energy_grad") {
  constexpr const Real threshold = 1E-15;
  // compute_energy_grad requires U, V, PHI, PECND, DINV, P,
  constexpr const int num_elems = 10;
  constexpr const int nm1 = 0;
  constexpr const int n0 = 1;
  constexpr const int np1 = 2;
  constexpr const int qn0 = -1;
  constexpr const int ps0 = 1;
  constexpr const Real dt2 = 1.0;
  constexpr const Real eta_ave_w = 1.0;

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());
  using udi_type = std::uniform_int_distribution<int>;

  int nets = udi_type(0, num_elems - 1)(engine);
  int nete = udi_type(nets + 1, num_elems)(engine);

  Real hybrid_a[NUM_LEV_P] = { 3.14159265 };

  CaarControl control;
  control.init(nets, nete, num_elems, nm1, n0, np1, qn0, dt2, ps0, false,
               eta_ave_w, &hybrid_a[0]);
  CaarRegion &region = get_region();
  region.init(control.num_elems);
  region.init_2d(D_ptr, Dinv_ptr, fcor_ptr, spheremp_ptr, metdet_ptr, phis_ptr);
  region.pull_from_f90_ptrs(state_v_ptr, state_t_ptr, state_dp3d_ptr,
                            derived_phi_ptr, derived_pecnd_ptr,
                            derived_omega_p_ptr, derived_v_ptr,
                            derived_eta_dot_dpdn_ptr, state_Qdp_ptr);
  CaarFunctor functor(control);
  Kokkos::TeamPolicy<ExecSpace> policy;
  Kokkos::parallel_for(
}
