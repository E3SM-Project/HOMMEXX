#include <catch/catch.hpp>

#include <limits>

#include "Control.hpp"
#include "CaarFunctor.hpp"
#include "Elements.hpp"
#include "Dimensions.hpp"
#include "KernelVariables.hpp"
#include "Types.hpp"

#include "preqx_flat_ut_sphere_op_ml.cpp"
#include "preqx_flat_ut_sphere_op_sl.cpp"

#include "utils_flat_ut.hpp"

#include <assert.h>
#include <stdio.h>
#include <random>


using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {

void caar_compute_energy_grad_c_int(
    const Real *const &dvv, const Real *const &Dinv,
    const Real *const &pecnd, const Real *const &phi,
    const Real *const &velocity,
    Real *const &vtemp);  //(&vtemp)[2][NP][NP]);
}  // extern C

/* compute_subfunctor_test
 *
 * Randomly initializes all of the input data
 * Calls the static method test_functor in TestFunctor_T,
 * passing in a correctly initialized KernelVariables object
 *
 * Ideally to use this structure you won't touch anything in
 * it
 */
template <typename TestFunctor_T>
class compute_subfunctor_test {
 public:
  compute_subfunctor_test(int num_elems)
      : functor(),
        velocity("Velocity", num_elems),
        temperature("Temperature", num_elems),
        dp3d("DP3D", num_elems),
        phi("Phi", num_elems),
        pecnd("PE_CND", num_elems),
        omega_p("Omega_P", num_elems),
        derived_v("Derived V?", num_elems),
        eta_dpdn("Eta dot dp/deta", num_elems),
        qdp("QDP", num_elems),
        dinv("DInv", num_elems),
        dvv("dvv"),
        nets(1),
        nete(num_elems) {
    Real hybrid_a[NUM_LEV_P] = {0};
    functor.m_data.init(0, num_elems, num_elems, nm1, n0,
                        np1, qn0, ps0, dt2, false,
                        eta_ave_w, hybrid_a);

    get_derivative().dvv(dvv.data());

    get_elements().push_to_f90_pointers(
        velocity.data(), temperature.data(), dp3d.data(),
        phi.data(), pecnd.data(), omega_p.data(),
        derived_v.data(), eta_dpdn.data(), qdp.data());
    for(int ie = 0; ie < num_elems; ++ie) {
      get_elements().dinv(
          Kokkos::subview(dinv, ie, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL,
                          Kokkos::ALL)
              .data(),
          ie);
    }
  }

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size(const int team_size) const {
    return KernelVariables::shmem_size(team_size);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember team) const {
    KernelVariables kv(team);
    TestFunctor_T::test_functor(functor, kv);
  }

  void run_functor() const {
    Kokkos::TeamPolicy<ExecSpace> policy(
        functor.m_data.num_elems, 16, 4);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
  }

  CaarFunctor functor;

  // host
  // Arrays used to pass data to and from Fortran
  HostViewManaged<
      Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP]>
      velocity;
  HostViewManaged<
      Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>
      temperature;
  HostViewManaged<
      Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>
      dp3d;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> phi;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> pecnd;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]>
      omega_p;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]>
      derived_v;
  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]>
      eta_dpdn;
  HostViewManaged<Real *
                  [Q_NUM_TIME_LEVELS]
                      [QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>
      qdp;
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

class compute_energy_grad_test {
 public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor,
                           KernelVariables &kv) {
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          functor.compute_energy_grad(kv);
        });
  }
};

TEST_CASE("monolithic compute_and_apply_rhs",
          "compute_energy_grad") {
  printf(
      "Q_NUM_TIME_LEVELS: %d\n"
      "QSIZE_D: %d\n"
      "NUM_PHYSICAL_LEV: %d\n"
      "VECTOR_SIZE: %d\n"
      "LEVEL_PADDING: %d\n"
      "NUM_LEV: %d\n",
      Q_NUM_TIME_LEVELS, QSIZE_D, NUM_PHYSICAL_LEV,
      VECTOR_SIZE, LEVEL_PADDING, NUM_LEV);
  constexpr const Real rel_threshold = 1E-15;
  constexpr const int num_elems = 1;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are
  // initialized in the singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  compute_subfunctor_test<compute_energy_grad_test>
      test_functor(num_elems);
  test_functor.run_functor();
  HostViewManaged<Scalar * [2][NP][NP][NUM_LEV]>
      energy_grad("energy_grad", num_elems);
  Kokkos::deep_copy(energy_grad,
                    elements.buffers.energy_grad);

  for(int ie = 0; ie < num_elems; ++ie) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real vtemp[2][NP][NP];

        caar_compute_energy_grad_c_int(
            reinterpret_cast<Real *>(
                test_functor.dvv.data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.dinv, ie,
                                Kokkos::ALL, Kokkos::ALL,
                                Kokkos::ALL, Kokkos::ALL)
                    .data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.pecnd, ie,
                                level * VECTOR_SIZE + v,
                                Kokkos::ALL, Kokkos::ALL)
                    .data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.phi, ie,
                                level * VECTOR_SIZE + v,
                                Kokkos::ALL, Kokkos::ALL)
                    .data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.velocity, ie,
                                test_functor.n0,
                                level * VECTOR_SIZE + v,
                                Kokkos::ALL, Kokkos::ALL,
                                Kokkos::ALL)
                    .data()),
            &vtemp[0][0][0]);
        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            REQUIRE(!std::isnan(vtemp[0][igp][jgp]));
            REQUIRE(!std::isnan(vtemp[1][igp][jgp]));
            REQUIRE(!std::isnan(
                energy_grad(ie, 0, igp, jgp, level)[v]));
            REQUIRE(!std::isnan(
                energy_grad(ie, 1, igp, jgp, level)[v]));
            REQUIRE(
                std::numeric_limits<Real>::epsilon() >=
                compare_answers(
                    vtemp[0][igp][jgp],
                    energy_grad(ie, 0, igp, jgp, level)[v],
                    128.0));
            REQUIRE(
                std::numeric_limits<Real>::epsilon() >=
                compare_answers(
                    vtemp[1][igp][jgp],
                    energy_grad(ie, 1, igp, jgp, level)[v],
                    128.0));
          }
        }
      }
    }
  }
  std::cout << "CaarFunctor: compute_energy_grad() test "
               "finished.\n";
};  // end of TEST_CASE(...,"compute_energy_grad")



