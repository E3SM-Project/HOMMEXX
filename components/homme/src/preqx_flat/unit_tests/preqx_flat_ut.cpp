#include <catch/catch.hpp>

#include <limits>
#include <random>
#include <type_traits>

#undef NDEBUG

#include "Control.hpp"
#include "CaarFunctor.hpp"
#include "Elements.hpp"
#include "Dimensions.hpp"
#include "KernelVariables.hpp"
#include "Types.hpp"
#include "Utility.hpp"

#include <assert.h>
#include <stdio.h>

using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {
void caar_compute_energy_grad_c_int(const Real *dvv,
                                    const Real *Dinv,
                                    const Real *pecnd,
                                    const Real *phi,
                                    const Real *velocity,
                                    Real *vtemp);

void preq_omega_ps_c_int(Real *omega_p, const Real *velocity,
                         const Real *pressure, const Real *div_vdp,
                         const Real *dinv, const Real *dvv);

void preq_hydrostatic_c_int(Real *phi, const Real *phis,
                            const Real *virt_temperature, const Real *pressure,
                            const Real *delta_pressure);

void caar_compute_dp3d_np1_c_int(int np1, int nm1, const Real &dt,
                                 const Real *spheremp, const Real *divdp,
                                 const Real *eta_dot_dpdn, Real *dp3d);

void caar_compute_divdp_c_int(const Real eta_ave_w, const Real *velocity,
                              const Real *dp3d, const Real *dinv,
                              const Real *metdet, const Real *dvv,
                              Real *derived_vn0, Real *vdp, Real *divdp);

void caar_compute_pressure_c_int(const Real &hyai, const Real &ps0,
                                 const Real *dp, Real *pressure);

void caar_compute_temperature_no_tracers_c_int(const Real *temperature,
                                               Real *virt_temperature);

void caar_compute_temperature_tracers_c_int(const Real *qdp, const Real *dp,
                                            const Real *temperature,
                                            Real *virt_temperature);

void caar_compute_omega_p_c_int(const Real eta_ave_w,
                                const Real *omega_p_buffer, Real *omega_p);

void caar_compute_temperature_c_int(const Real dt, const Real * spheremp,
                                    const Real *dinv, const Real *dvv,
                                    const Real *velocity, const Real *t_virt,
                                    const Real * omega_p, const Real * t_vadv,
                                    const Real *t_previous,
                                    const Real *t_current,
                                    Real *t_future);

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
template <typename TestFunctor_T> class compute_subfunctor_test {
public:
  compute_subfunctor_test(Elements &elements)
      : functor(), velocity("Velocity", elements.num_elems()),
        temperature("Temperature", elements.num_elems()),
        dp3d("DP3D", elements.num_elems()), phi("Phi", elements.num_elems()),
        phis("Phis?", elements.num_elems()),
        pecnd("Potential Energy CND?", elements.num_elems()),
        omega_p("Omega_P", elements.num_elems()),
        derived_v("Derived V?", elements.num_elems()),
        eta_dpdn("Eta dot dp/deta", elements.num_elems()),
        qdp("QDP", elements.num_elems()), metdet("metdet", elements.num_elems()),
        dinv("DInv", elements.num_elems()),
        spheremp("SphereMP", elements.num_elems()), dvv("dvv"), nets(1),
        nete(elements.num_elems()) {
    Real hybrid_a[NUM_LEV_P] = { 0 };
    functor.m_data.init(0, elements.num_elems(), elements.num_elems(), nm1, n0, np1,
                        qn0, ps0, dt, false, eta_ave_w, hybrid_a);

    get_derivative().dvv(dvv.data());

    elements.push_to_f90_pointers(velocity.data(), temperature.data(),
                                dp3d.data(), phi.data(), pecnd.data(),
                                omega_p.data(), derived_v.data(),
                                eta_dpdn.data(), qdp.data());

    Kokkos::deep_copy(spheremp, elements.m_spheremp);
    Kokkos::deep_copy(metdet, elements.m_metdet);

    for (int ie = 0; ie < elements.num_elems(); ++ie) {
      elements.dinv(Kokkos::subview(dinv, ie, Kokkos::ALL, Kokkos::ALL,
                                  Kokkos::ALL, Kokkos::ALL).data(),
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
    // Retrieve the team size
    const int vectors_per_thread = ThreadsDistribution<ExecSpace>::vectors_per_thread();
    const int threads_per_team   = ThreadsDistribution<ExecSpace>::threads_per_team(functor.m_data.num_elems);

    Kokkos::TeamPolicy<ExecSpace> policy(functor.m_data.num_elems, threads_per_team, vectors_per_thread);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
  }

  CaarFunctor functor;

  // host
  // Arrays used to pass data to and from Fortran
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP]>
  velocity;
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>
  temperature;
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]> dp3d;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> phi;
  HostViewManaged<Real * [NP][NP]> phis;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> pecnd;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> omega_p;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> derived_v;
  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]> eta_dpdn;
  HostViewManaged<Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>
  qdp;
  HostViewManaged<Real * [NP][NP]> metdet;
  HostViewManaged<Real * [2][2][NP][NP]> dinv;
  HostViewManaged<Real * [NP][NP]> spheremp;
  HostViewManaged<Real[NP][NP]> dvv;

  const int nets;
  const int nete;

  static constexpr int nm1 = 0;
  static constexpr int nm1_f90 = nm1 + 1;
  static constexpr int n0 = 1;
  static constexpr int n0_f90 = n0 + 1;
  static constexpr int np1 = 2;
  static constexpr int np1_f90 = np1 + 1;
  static constexpr int qn0 = 0;
  static constexpr Real ps0 = 1.0;
  static constexpr Real dt = 1.0;
  static constexpr Real eta_ave_w = 1.0;
};

class compute_energy_grad_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                         [&](const int &level) {
      kv.ilev = level;
      functor.compute_energy_grad(kv);
    });
  }
};

TEST_CASE("compute_energy_grad", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 64.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are
  // initialized in the singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  // Generate data
  ExecViewManaged<Scalar * [NUM_LEV][2][NP][NP]>::HostMirror energy_grad_in;
  energy_grad_in = Kokkos::create_mirror(elements.buffers.energy_grad);
  // Note: create_mirror, NOT create_mirror_view, since we need to keep the input intact for F90 code
  genRandArray(energy_grad_in, engine, std::uniform_real_distribution<Real>(0, 100.0));
  Kokkos::deep_copy(elements.buffers.energy_grad, energy_grad_in);

  // Run C++ version (on all elements)
  compute_subfunctor_test<compute_energy_grad_test> test_functor(elements);
  test_functor.run_functor();

  ExecViewManaged<Scalar * [NUM_LEV][2][NP][NP]>::HostMirror energy_grad_out;
  energy_grad_out = Kokkos::create_mirror_view(elements.buffers.energy_grad);
  Kokkos::deep_copy(energy_grad_out, elements.buffers.energy_grad);

  HostViewManaged<Real[2][NP][NP]> vtemp("vtemp");
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_LEV; ++level) {
      for (int v = 0; v < VECTOR_SIZE; ++v) {
        for (int i = 0; i < NP; i++) {
          for (int j = 0; j < NP; j++) {
            vtemp(0, i, j) = energy_grad_in(ie, level, 0, i, j)[v];
            vtemp(1, i, j) = energy_grad_in(ie, level, 1, i, j)[v];
          }
        }

        // Run F90 version (on this element)
        caar_compute_energy_grad_c_int(
            test_functor.dvv.data(),
            Homme::subview(test_functor.dinv, ie).data(),
            Homme::subview(test_functor.pecnd, ie, level * VECTOR_SIZE + v).data(),
            Homme::subview(test_functor.phi, ie, level * VECTOR_SIZE + v).data(),
            Homme::subview(test_functor.velocity, ie, test_functor.n0, level * VECTOR_SIZE + v).data(),
            vtemp.data());

        // Compare answers
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            const Real correct[2] = { vtemp(0, igp, jgp), vtemp(1, igp, jgp) };
            const Real computed[2] = {
              energy_grad_out(ie, level, 0, igp, jgp)[v],
              energy_grad_out(ie, level, 1, igp, jgp)[v]
            };
            for (int dim = 0; dim < 2; ++dim) {
              Real rel_error = compare_answers(correct[dim], computed[dim]);
              REQUIRE(!std::isnan(correct[dim]));
              REQUIRE(!std::isnan(computed[dim]));

              REQUIRE(rel_threshold >= rel_error);
            }
          }
        }
      }
    }
  }
} // end of TEST_CASE(...,"compute_energy_grad")

class preq_omega_ps_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    if (kv.team.team_rank() == 0) {
      functor.preq_omega_ps(kv);
    }
  }
};

TEST_CASE("preq_omega_ps", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 256.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  // Generate data on f90 views
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> pressure("host pressure", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("div_vdp", num_elems);
  genRandArray(div_vdp, engine, std::uniform_real_distribution<Real>(0, 100.0));
  genRandArray(pressure, engine, std::uniform_real_distribution<Real>(0, 100.0));

  // Syncing to device
  sync_to_device(div_vdp, elements.buffers.div_vdp);
  sync_to_device(pressure, elements.buffers.pressure);

  // Run C++ version (on all elements)
  compute_subfunctor_test<preq_omega_ps_test> test_functor(elements);
  test_functor.run_functor();

  // Results of the computation
  ExecViewManaged<Scalar * [NUM_LEV][NP][NP]>::HostMirror omega_p;
  omega_p = Kokkos::create_mirror_view(elements.buffers.omega_p);
  Kokkos::deep_copy(omega_p, elements.buffers.omega_p);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> omega_p_f90("Fortran omega_p");
  for (int ie = 0; ie < num_elems; ++ie) {

    // Run F90 version (on this element)
    preq_omega_ps_c_int(
        omega_p_f90.data(),
        Homme::subview(pressure, ie).data(),
        Homme::subview(test_functor.velocity, ie, test_functor.n0).data(),
        Homme::subview(div_vdp, ie).data(),
        Homme::subview(test_functor.dinv, ie).data(),
        test_functor.dvv.data());

    // Compare answers
    for (int level=0; level<NUM_PHYSICAL_LEV; ++level) {
      int ilev = level / VECTOR_SIZE;
      int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          REQUIRE(!std::isnan(omega_p(ie, ilev, igp, jgp)[ivec]));
          REQUIRE(!std::isnan(omega_p_f90(level, igp, jgp)));
          Real rel_error = compare_answers(omega_p_f90(level, igp, jgp),
                                           omega_p(ie, ilev, igp, jgp)[ivec]);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}

class preq_hydrostatic_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    if (kv.team.team_rank() == 0) {
      functor.preq_hydrostatic(kv);
    }
  }
};

TEST_CASE("preq_hydrostatic", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 4.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<preq_hydrostatic_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);

  // Generate data
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> temperature_virt("host virtual temperature", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> pressure("host pressure", num_elems);
  genRandArray(temperature_virt, engine, std::uniform_real_distribution<Real>(0.0125, 1.0));
  genRandArray(pressure, engine, std::uniform_real_distribution<Real>(0.0125, 1.0));

  // Syncing to device
  sync_to_device(temperature_virt, elements.buffers.temperature_virt);
  sync_to_device(pressure, elements.buffers.pressure);

  // Run C++ version (on all elements)
  TestType test_functor(elements);
  test_functor.run_functor();
  sync_to_host(elements.m_phi, test_functor.phi);

  // Copy input data to F90
  Kokkos::deep_copy(test_functor.phis, elements.m_phis);
  sync_to_host(elements.m_dp3d, test_functor.dp3d);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> phi_f90("Fortran phi");
  for (int ie = 0; ie < num_elems; ++ie) {

    // Run F90 version (on this element)
    preq_hydrostatic_c_int(
        phi_f90.data(),
        Homme::subview(test_functor.phis, ie).data(),
        Homme::subview(temperature_virt, ie).data(),
        Homme::subview(pressure, ie).data(),
        Homme::subview(test_functor.dp3d, ie, test_functor.n0).data());

    // Compare answers
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          const Real correct = phi_f90(level, igp, jgp);
          REQUIRE(!std::isnan(correct));
          const Real computed = test_functor.phi(ie, level, igp, jgp);
          REQUIRE(!std::isnan(computed));
          const Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}

class dp3d_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                         [&](const int &ilev) {
      kv.ilev = ilev;
      functor.compute_dp3d_np1(kv);
    });
  }
};

TEST_CASE("dp3d", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 128.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  // Generate data
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("host div_vdp", num_elems);
  genRandArray(div_vdp, engine, std::uniform_real_distribution<Real>(0, 100.0));

  // Syncing to device
  sync_to_device(div_vdp, elements.buffers.div_vdp);

  // Copy the initial state to F90 before running any of the test
  compute_subfunctor_test<dp3d_test> test_functor(elements);
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]> dp3d_f90(
      "dp3d fortran", num_elems);
  Kokkos::deep_copy(dp3d_f90, test_functor.dp3d);

  // Run C++ version (on all elements) and fetch results
  test_functor.run_functor();
  sync_to_host(elements.m_dp3d, test_functor.dp3d);

  for (int ie = 0; ie < num_elems; ++ie) {
    // Run F90 version (on this element)
    caar_compute_dp3d_np1_c_int(
        test_functor.np1_f90, test_functor.nm1_f90,
        test_functor.functor.m_data.dt,
        Homme::subview(test_functor.spheremp, ie).data(),
        Homme::subview(div_vdp, ie).data(),
        Homme::subview(test_functor.eta_dpdn, ie).data(),
        Homme::subview(dp3d_f90, ie).data());

    // Compare answers
    for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          Real correct =
              dp3d_f90(ie, test_functor.functor.m_data.np1, k, igp, jgp);
          REQUIRE(!std::isnan(correct));
          Real computed = test_functor.dp3d(ie, test_functor.functor.m_data.np1,
                                            k, igp, jgp);
          REQUIRE(!std::isnan(computed));
          Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}

class vdp_vn0_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                         [&](const int idx) {
      kv.ilev = idx;
      functor.compute_div_vdp(kv);
    });
  }
};

TEST_CASE("vdp_vn0", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 512.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  // Copy the initial state to F90 before running any of the test
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> vn0_f90("vn0 f90 results", num_elems);
  sync_to_host(elements.m_derived_un0, elements.m_derived_vn0, vn0_f90);

  // Run C++ version (on all elements) and fetch results
  compute_subfunctor_test<vdp_vn0_test> test_functor(elements);
  test_functor.run_functor();
  sync_to_host(elements.m_derived_un0, elements.m_derived_vn0, test_functor.derived_v);
  HostViewManaged<Scalar * [NUM_LEV][2][NP][NP]> vdp("vdp results", num_elems);
  HostViewManaged<Scalar * [NUM_LEV][NP][NP]> div_vdp("div_vdp results", num_elems);
  Kokkos::deep_copy(vdp, elements.buffers.vdp);
  Kokkos::deep_copy(div_vdp, elements.buffers.div_vdp);

  HostViewManaged<Real[2][NP][NP]> vdp_f90("vdp f90 results");
  HostViewManaged<Real[NP][NP]> div_vdp_f90("div_vdp f90 results");
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level=0; level<NUM_PHYSICAL_LEV; ++level) {
      int ilev = level / VECTOR_SIZE;
      int ivec = level % VECTOR_SIZE;
      // Run F90 version (on this element)
      caar_compute_divdp_c_int(
          compute_subfunctor_test<vdp_vn0_test>::eta_ave_w,
          Homme::subview(test_functor.velocity, ie, test_functor.n0, level).data(),
          Homme::subview(test_functor.dp3d, ie, test_functor.n0, level).data(),
          Homme::subview(test_functor.dinv, ie).data(),
          Homme::subview(test_functor.metdet, ie).data(),
          test_functor.dvv.data(),
          Homme::subview(vn0_f90, ie, level).data(),
          vdp_f90.data(), div_vdp_f90.data());
      // Compare answers
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          for (int hgp = 0; hgp < 2; ++hgp) {
            {
              // Check vdp
              Real correct = vdp_f90(hgp, igp, jgp);
              REQUIRE(!std::isnan(correct));
              Real computed = vdp(ie, ilev, hgp, igp, jgp)[ivec];
              REQUIRE(!std::isnan(computed));
              if (correct != 0.0) {
                Real rel_error = compare_answers(correct, computed);
                REQUIRE(rel_threshold >= rel_error);
              }
            }
            {
              // Check derived_vn0
              Real correct = vn0_f90(ie, level, hgp, igp, jgp);
              REQUIRE(!std::isnan(correct));
              Real computed =
                  test_functor.derived_v(ie, level, hgp, igp, jgp);
              REQUIRE(!std::isnan(computed));
              if (correct != 0.0) {
                Real rel_error = compare_answers(correct, computed);
                REQUIRE(rel_threshold >= rel_error);
              }
            }
          }
          {
            // Check div_vdp
            Real correct = div_vdp_f90(igp, jgp);
            REQUIRE(!std::isnan(correct));
            Real computed = div_vdp(ie, ilev, igp, jgp)[ivec];
            REQUIRE(!std::isnan(computed));
            Real rel_error = compare_answers(correct, computed);
            REQUIRE(rel_threshold >= rel_error);
          }
        }
      }
    }
  }
}

class pressure_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    if (kv.team.team_rank() == 0) {
      functor.compute_pressure(kv);
    }
  }
};

TEST_CASE("pressure", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 1.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<pressure_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  TestType test_functor(elements);

  // Generate data
  ExecViewManaged<Real[NUM_LEV_P]>::HostMirror hybrid_a_mirror("hybrid_a_host");
  genRandArray(hybrid_a_mirror, engine, std::uniform_real_distribution<Real>(0.0125, 1.0));
  test_functor.functor.m_data.init(1, num_elems, num_elems, TestType::nm1,
                                   TestType::n0, TestType::np1, TestType::qn0,
                                   TestType::dt, TestType::ps0, false,
                                   TestType::eta_ave_w, hybrid_a_mirror.data());

  // Run C++ version (on all elements) and fetch results
  test_functor.run_functor();
  HostViewManaged<Scalar * [NUM_LEV][NP][NP]> pressure_cxx("pressure_cxx", num_elems);
  Kokkos::deep_copy(pressure_cxx, elements.buffers.pressure);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> pressure_f90("pressure_f90");

  // Copy the initial state to F90
  sync_to_host(elements.m_dp3d, test_functor.dp3d);

  for (int ie = 0; ie < num_elems; ++ie) {
    // Run F90 version (on this element)
    caar_compute_pressure_c_int(
        hybrid_a_mirror(0), test_functor.functor.m_data.ps0,
        Homme::subview(test_functor.dp3d, ie, test_functor.n0).data(),
        pressure_f90.data());
    // Compare answers
    for (int level=0; level<NUM_PHYSICAL_LEV; ++level) {
      int ilev = level / VECTOR_SIZE;
      int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          const Real correct = pressure_f90(level, igp, jgp);
          const Real computed = pressure_cxx(ie, ilev, igp, jgp)[ivec];
          REQUIRE(!std::isnan(correct));
          REQUIRE(!std::isnan(computed));
          const Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}

class temperature_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
      [&](const int &idx) {
        kv.ilev = idx;
        functor.compute_temperature_np1(kv);
    });
  }
};

TEST_CASE("temperature", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 2.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<temperature_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  // Generate data
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> temperature_virt("Virtual temperature test", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> omega_p("Omega P test", num_elems);
  genRandArray(temperature_virt, engine, std::uniform_real_distribution<Real>(0, 1.0));
  genRandArray(omega_p, engine, std::uniform_real_distribution<Real>(0, 1.0));

  // Syncing to device
  sync_to_device(temperature_virt, elements.buffers.temperature_virt);
  sync_to_device(omega_p, elements.buffers.omega_p);

  // Run C++ version (on all elements) and fetch results
  TestType test_functor(elements);
  test_functor.run_functor();
  sync_to_host(elements.m_t, test_functor.temperature);

  HostViewManaged<Real [NP][NP]> temperature_vadv("Temperature Vertical Advection");
  for(int i = 0; i < NP; ++i) {
    for(int j = 0; j < NP; ++j) {
      temperature_vadv(i, j) = 0.0;
    }
  }
  HostViewManaged<Real [NP][NP]> temperature_f90("Temperature f90");
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      // Run F90 version (on this element)
      caar_compute_temperature_c_int(
        test_functor.dt,
        Homme::subview(test_functor.spheremp, ie).data(),
        Homme::subview(test_functor.dinv, ie).data(),
        test_functor.dvv.data(),
        Homme::subview(test_functor.velocity, ie, test_functor.n0, level).data(),
        Homme::subview(temperature_virt, ie, level).data(),
        Homme::subview(omega_p, ie, level).data(),
        temperature_vadv.data(),
        Homme::subview(test_functor.temperature, ie, test_functor.nm1, level).data(),
        Homme::subview(test_functor.temperature, ie, test_functor.n0, level).data(),
        temperature_f90.data());

      // Compare answers
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          const Real correct = temperature_f90(igp, jgp);
          const Real computed = test_functor.temperature(ie, int(test_functor.np1), level, igp, jgp);
          REQUIRE(!std::isnan(correct));
          REQUIRE(!std::isnan(computed));
          const Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}

class virtual_temperature_no_tracers_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                         [&](const int ilev) {
      kv.ilev = ilev;
      functor.compute_temperature_no_tracers_helper(kv);
    });
  }
};

TEST_CASE("virtual temperature no tracers",
          "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 1.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<virtual_temperature_no_tracers_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);

  TestType test_functor(elements);
  // Copy the initial state to F90 before running any of the test
  sync_to_host(elements.m_t, test_functor.temperature);

  // Run C++ version (on all elements) and fetch results
  test_functor.run_functor();
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> temperature_virt_cxx("virtual temperature cxx", num_elems);
  sync_to_host(elements.buffers.temperature_virt, temperature_virt_cxx);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> temperature_virt_f90("virtual temperature f90");
  for (int ie = 0; ie < num_elems; ++ie) {
    // Run F90 version (on this element)
    caar_compute_temperature_no_tracers_c_int(
        Homme::subview(test_functor.temperature, ie, test_functor.n0).data(),
        temperature_virt_f90.data());

    // Compare answers
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          const Real correct = temperature_virt_f90(level, igp, jgp);
          REQUIRE(!std::isnan(correct));
          const Real computed = temperature_virt_cxx(ie, level, igp, jgp);
          REQUIRE(!std::isnan(computed));
          const Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}

class virtual_temperature_with_tracers_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                         [&](const int ilev) {
      kv.ilev = ilev;
      functor.compute_temperature_tracers_helper(kv);
    });
  }
};

TEST_CASE("virtual temperature with tracers",
          "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 4.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType =
      compute_subfunctor_test<virtual_temperature_with_tracers_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);

  TestType test_functor(elements);
  // Copy the initial state to F90 before running any of the test
  sync_to_host(elements.m_qdp, test_functor.qdp);
  sync_to_host(elements.m_dp3d, test_functor.dp3d);
  sync_to_host(elements.m_t, test_functor.temperature);

  // Run C++ version (on all elements) and fetch results
  test_functor.run_functor();
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> temperature_virt_cxx("virtual temperature cxx", num_elems);
  sync_to_host(elements.buffers.temperature_virt, temperature_virt_cxx);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> temperature_virt_f90(
      "virtual temperature f90");
  for (int ie = 0; ie < num_elems; ++ie) {
    // Run F90 version (on this element)
    caar_compute_temperature_tracers_c_int(
        Homme::subview(test_functor.qdp, ie, test_functor.qn0, 0).data(),
        Homme::subview(test_functor.dp3d, ie, test_functor.n0).data(),
        Homme::subview(test_functor.temperature, ie, test_functor.n0).data(),
        temperature_virt_f90.data());

    // Compare answers
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          const Real correct = temperature_virt_f90(level, igp, jgp);
          REQUIRE(!std::isnan(correct));
          const Real computed = temperature_virt_cxx(ie, level, igp, jgp);
          REQUIRE(!std::isnan(computed));
          const Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}

class omega_p_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                         [&](const int ilev) {
      kv.ilev = ilev;
      functor.compute_omega_p(kv);
    });
  }
};

TEST_CASE("omega_p", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 0.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<omega_p_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);

  // Generate data
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> source_omega_p( "source omega p", num_elems);
  genRandArray(source_omega_p, engine, std::uniform_real_distribution<Real>(0.0125, 1.0));

  // Syncing to device
  sync_to_device(source_omega_p, elements.buffers.omega_p);

  // Copy the initial state to F90 before running any of the test
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> omega_p_f90("omega p f90", num_elems);
  sync_to_host(elements.m_omega_p, omega_p_f90);

  // Run C++ version (on all elements) and fetch results
  TestType test_functor(elements);
  test_functor.run_functor();
  sync_to_host(elements.m_omega_p, test_functor.omega_p);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> temperature_virt_cxx("virtual temperature cxx", num_elems);
  sync_to_host(elements.buffers.temperature_virt, temperature_virt_cxx);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> temperature_virt_f90("virtual temperature f90");
  for (int ie = 0; ie < num_elems; ++ie) {
    // Run F90 version (on this element)
    caar_compute_omega_p_c_int(
      test_functor.eta_ave_w,
      Homme::subview(source_omega_p, ie).data(),
      Homme::subview(omega_p_f90, ie).data());
    // Compare answers
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          const Real correct = omega_p_f90(ie, level, igp, jgp);
          REQUIRE(!std::isnan(correct));
          const Real computed = test_functor.omega_p(ie, level, igp, jgp);
          REQUIRE(!std::isnan(computed));
          const Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}
