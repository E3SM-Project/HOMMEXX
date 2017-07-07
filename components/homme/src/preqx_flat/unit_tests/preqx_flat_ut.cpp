#include <catch/catch.hpp>

#include <limits>

#include <CaarControl.hpp>
#include <CaarFunctor.hpp>
#include <CaarRegion.hpp>
#include <Dimensions.hpp>
#include <KernelVariables.hpp>
#include <Types.hpp>

#include <assert.h>
#include <stdio.h>

using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {
void caar_compute_energy_grad_c_int(const Real *dvv, const Real *Dinv,
                                    const Real *const &pecnd,
                                    const Real *const &phi,
                                    const Real *const &velocity,
                                    Real (&vtemp)[2][NP][NP]);
}

Real compare_answers(Real target, Real computed, Real relative_coeff = 1.0) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target - computed) / denom;
}

/* compute_subfunctor_test
 *
 * Randomly initializes all of the input data
 * Calls the static method test_functor in TestFunctor_T,
 * passing in a correctly initialized KernelVariables object
 *
 * Ideally to use this structure you won't touch anything in it
 */
template <typename TestFunctor_T> class compute_subfunctor_test {
public:
  compute_subfunctor_test(int num_elems)
      : functor(), velocity("Velocity", num_elems),
        temperature("Temperature", num_elems), dp3d("DP3D", num_elems),
        phi("Phi", num_elems), pecnd("PE_CND", num_elems),
        omega_p("Omega_P", num_elems), derived_v("Derived V?", num_elems),
        eta_dpdn("Eta dot dp/deta", num_elems), qdp("QDP", num_elems),
        dinv("DInv", num_elems), dvv("dvv"), nets(1), nete(num_elems) {
    Real hybrid_a[NUM_LEV_P] = { 0 };
    functor.m_data.init(0, num_elems, num_elems, nm1, n0, np1, qn0, ps0, dt2,
                        false, eta_ave_w, hybrid_a);

    get_derivative().dvv(dvv.data());

    get_region().push_to_f90_pointers(velocity.data(), temperature.data(),
                                      dp3d.data(), phi.data(), pecnd.data(),
                                      omega_p.data(), derived_v.data(),
                                      eta_dpdn.data(), qdp.data());
    for (int ie = 0; ie < num_elems; ++ie) {
      get_region().dinv(Kokkos::subview(dinv, ie, Kokkos::ALL, Kokkos::ALL,
                                        Kokkos::ALL, Kokkos::ALL).data(),
                        ie);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember team) const {
    KernelVariables kv(team);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                         [&](const int &level) {
      kv.ilev = level;
      TestFunctor_T::test_functor(functor, kv);
    });
  }

  void run_functor() const {
    Kokkos::TeamPolicy<ExecSpace> policy(functor.m_data.num_elems, 16, 4);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
  }

  CaarFunctor functor;

  // Arrays used to pass data to and from Fortran
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_LEV][2][NP][NP]> velocity;
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_LEV][NP][NP]> temperature;
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_LEV][NP][NP]> dp3d;
  HostViewManaged<Real * [NUM_LEV][NP][NP]> phi;
  HostViewManaged<Real * [NUM_LEV][NP][NP]> pecnd;
  HostViewManaged<Real * [NUM_LEV][NP][NP]> omega_p;
  HostViewManaged<Real * [NUM_LEV][2][NP][NP]> derived_v;
  HostViewManaged<Real * [NUM_LEV_P][NP][NP]> eta_dpdn;
  HostViewManaged<Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_LEV][NP][NP]> qdp;
  HostViewManaged<Real * [2][2][NP][NP]> dinv;
  HostViewManaged<Real[NP][NP]> dvv;

  const int nets;
  const int nete;

  static constexpr const int nm1 = 0;
  static constexpr const int n0 = 1;
  static constexpr const int np1 = 2;
  static constexpr const int qn0 = -1;
  static constexpr const int ps0 = 1;
  static constexpr const Real dt2 = 1.0;
  static constexpr const Real eta_ave_w = 1.0;
};

TEST_CASE("monolithic compute_and_apply_rhs", "compute_energy_grad") {
  constexpr const Real rel_threshold = 1E-15;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are initialized in the
  // singleton
  CaarRegion &region = get_region();
  region.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  class compute_energy_grad_test {
  public:
    KOKKOS_INLINE_FUNCTION
    static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
      functor.compute_energy_grad(kv);
    }
  };
  compute_subfunctor_test<compute_energy_grad_test> test_functor(num_elems);

  test_functor.run_functor();

  HostViewManaged<Real * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> u_vel("U",
                                                                   num_elems),
      v_vel("V", num_elems);
  Kokkos::deep_copy(u_vel, get_region().m_u);
  Kokkos::deep_copy(v_vel, get_region().m_v);

  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_LEV; ++level) {
      Real vtemp[2][NP][NP];
      caar_compute_energy_grad_c_int(
          test_functor.dvv.data(),
          Kokkos::subview(test_functor.dinv, ie, Kokkos::ALL, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL).data(),
          Kokkos::subview(test_functor.pecnd, ie, level, Kokkos::ALL,
                          Kokkos::ALL).data(),
          Kokkos::subview(test_functor.phi, ie, level, Kokkos::ALL, Kokkos::ALL)
              .data(),
          Kokkos::subview(test_functor.velocity, ie, test_functor.n0, level,
                          Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(),
          vtemp);
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          REQUIRE(!std::isnan(vtemp[0][igp][jgp]));
          REQUIRE(!std::isnan(vtemp[1][igp][jgp]));
          REQUIRE(!std::isnan(u_vel(ie, test_functor.n0, igp, jgp, level)));
          REQUIRE(!std::isnan(v_vel(ie, test_functor.n0, igp, jgp, level)));
          printf("% .17e  % .17e     % .17e  % .17e\n", vtemp[0][igp][jgp],
                 u_vel(ie, test_functor.n0, igp, jgp, level), vtemp[1][igp][jgp],
                 v_vel(ie, test_functor.n0, igp, jgp, level));
          REQUIRE(std::numeric_limits<Real>::epsilon() >=
                  compare_answers(vtemp[0][igp][jgp],
                                  u_vel(ie, test_functor.n0, igp, jgp, level),
                                  4.0));
          REQUIRE(std::numeric_limits<Real>::epsilon() >=
                  compare_answers(vtemp[1][igp][jgp],
                                  v_vel(ie, test_functor.n0, igp, jgp, level),
                                  4.0));
        }
      }
    }
  }
}
