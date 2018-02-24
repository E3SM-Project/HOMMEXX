#include <catch/catch.hpp>

#include <limits>
#include <random>
#include <type_traits>

#undef NDEBUG

#include "CaarFunctorImpl.hpp"
#include "EulerStepFunctorImpl.hpp"
#include "Elements.hpp"
#include "Tracers.hpp"
#include "HybridVCoord.hpp"
#include "Dimensions.hpp"
#include "KernelVariables.hpp"
#include "Types.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/TestUtils.hpp"
#include "Context.hpp"

#include <assert.h>
#include <stdio.h>

using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {
void caar_compute_energy_grad_c_int(const Real *dvv,
                                    const Real *Dinv,
                                    const Real *phi,
                                    const Real *velocity,
                                    Real *tvirt,
                                    Real *press,
                                    Real *press_grad,
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

void caar_compute_eta_dot_dpdn_vertadv_euler_c_int(Real *eta_dot_dpdn,
                                                   Real *sdot_sum,
                                                   const Real *divdp,
                                                   const Real *hybi);

void preq_vertadv(const Real *temperature, const Real *velocity,
                  const Real *eta_dot_dpdn, const Real *recipr_density,
                  Real *t_vadv, Real *v_vadv);

void caar_adjust_eta_dot_dpdn_c_int(const Real eta_ave_w,
                                    Real *eta_dot_total, const Real *eta_dot);

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
  compute_subfunctor_test(Elements &elements, Tracers &tracers,
                          const int rsplit_in = 0)
      : functor(
            elements, tracers, Context::singleton().get_derivative(),
            Context::singleton().get_hvcoord(),
            SphereOperators(elements, Context::singleton().get_derivative()),
            rsplit_in),
        policy(functor.m_elements.num_elems(), 16, 4),
        velocity("Velocity", elements.num_elems()),
        temperature("Temperature", elements.num_elems()),
        dp3d("DP3D", elements.num_elems()),
        phi("Phi", elements.num_elems()),
        phis("Phi_surf", elements.num_elems()),
        omega_p("Omega_P", elements.num_elems()),
        derived_v("Derived V", elements.num_elems()),
        eta_dpdn("Eta dot dp/deta", elements.num_elems()),
        qdp("QDP", elements.num_elems()),
        metdet("metdet", elements.num_elems()),
        dinv("DInv", elements.num_elems()),
        spheremp("SphereMP", elements.num_elems()), dvv("dvv"),
        rsplit(rsplit_in)
        {

    functor.m_sphere_ops.allocate_buffers(policy);
    functor.set_n0_qdp(n0_qdp);
    functor.set_rk_stage_data(nm1, n0, np1, dt, eta_ave_w, false);

    Context::singleton().get_derivative().dvv(dvv.data());

    elements.push_to_f90_pointers(velocity.data(), temperature.data(),
                                dp3d.data(), phi.data(),
                                omega_p.data(), derived_v.data(),
                                  eta_dpdn.data());
    tracers.push_qdp(qdp.data());

    Kokkos::deep_copy(spheremp, elements.m_spheremp);
    Kokkos::deep_copy(metdet, elements.m_metdet);

    for (int ie = 0; ie < elements.num_elems(); ++ie) {
      elements.dinv(Homme::subview(dinv, ie).data(),
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
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
  }

  CaarFunctorImpl functor;
  Kokkos::TeamPolicy<ExecSpace> policy;

  // host
  // Arrays used to pass data to and from Fortran
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP]>
  velocity;
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>
  temperature;
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]> dp3d;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> phi;
  HostViewManaged<Real * [NP][NP]> phis;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> omega_p;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> derived_v;
  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]> eta_dpdn;
  HostViewManaged<Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>
  qdp;
  HostViewManaged<Real * [NP][NP]> metdet;
  HostViewManaged<Real * [2][2][NP][NP]> dinv;
  HostViewManaged<Real * [NP][NP]> spheremp;
  HostViewManaged<Real[NP][NP]> dvv;

  static constexpr int nm1 = 0;
  static constexpr int nm1_f90 = nm1 + 1;
  static constexpr int n0 = 1;
  static constexpr int n0_f90 = n0 + 1;
  static constexpr int np1 = 2;
  static constexpr int np1_f90 = np1 + 1;
  static constexpr int n0_qdp = 0;
  static constexpr Real ps0 = 1.0;
  static constexpr Real dt = 1.0;
  static constexpr Real eta_ave_w = 1.0;

private:
  int rsplit;
};

class compute_energy_grad_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.compute_energy_grad(kv);
  }
};

TEST_CASE("compute_energy_grad", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 64.0;
  constexpr const int num_elems = 2;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are
  // initialized in the singleton
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  Context::singleton().get_derivative().random_init();

  HostViewManaged<Scalar * [NP][NP][NUM_LEV]> temperature_virt_in("temperature_virt input", num_elems);
  HostViewManaged<Scalar * [NP][NP][NUM_LEV]> pressure_in("pressure input", num_elems);
  HostViewManaged<Scalar * [2][NP][NP][NUM_LEV]> pressure_grad_in("pressure_grad input", num_elems);

  genRandArray(temperature_virt_in, engine, std::uniform_real_distribution<Real>(0.1, 100.0));
  genRandArray(pressure_in,         engine, std::uniform_real_distribution<Real>(0.1, 100.0));
  genRandArray(pressure_grad_in,    engine, std::uniform_real_distribution<Real>(0.1, 100.0));

  Kokkos::deep_copy(elements.buffers.temperature_virt, temperature_virt_in);
  Kokkos::deep_copy(elements.buffers.pressure, pressure_in);
  Kokkos::deep_copy(elements.buffers.pressure_grad, pressure_grad_in);

  compute_subfunctor_test<compute_energy_grad_test> test_functor(elements, tracers);
  test_functor.run_functor();

  HostViewManaged<Scalar * [2][NP][NP][NUM_LEV]> energy_grad_out(
      "energy_grad output", num_elems);
  Kokkos::deep_copy(energy_grad_out, elements.buffers.energy_grad);

  HostViewManaged<Real[2][NP][NP]> vtemp("vtemp");
  HostViewManaged<Real[NP][NP]> tvirt("tvirt");
  HostViewManaged<Real[NP][NP]> press("press");
  HostViewManaged<Real[2][NP][NP]> press_grad("press_grad");
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_LEV; ++level) {
      for (int v = 0; v < VECTOR_SIZE; ++v) {
        for (int i = 0; i < NP; i++) {
          for (int j = 0; j < NP; j++) {
            press_grad(0, i, j) = pressure_grad_in(ie, 0, i, j, level)[v];
            press_grad(1, i, j) = pressure_grad_in(ie, 1, i, j, level)[v];
            press(i, j) = pressure_in(ie, i, j, level)[v];
            tvirt(i, j) = temperature_virt_in(ie, i, j, level)[v];
            vtemp(0, i, j) = 0;
            vtemp(1, i, j) = 0;
          }
        }
        caar_compute_energy_grad_c_int(
            test_functor.dvv.data(),
                Homme::subview(test_functor.dinv, ie).data(),
                Homme::subview(test_functor.phi, ie, level * VECTOR_SIZE + v).data(),
                Homme::subview(test_functor.velocity, ie, test_functor.n0,
                                level * VECTOR_SIZE + v).data(),
                tvirt.data(),
                press.data(),
                press_grad.data(),
                vtemp.data());
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            const Real correct[2] = { vtemp(0, igp, jgp), vtemp(1, igp, jgp) };
            const Real computed[2] = {
              energy_grad_out(ie, 0, igp, jgp, level)[v],
              energy_grad_out(ie, 1, igp, jgp, level)[v]
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
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.preq_omega_ps(kv);
  }
};

TEST_CASE("preq_omega_ps", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 256.0;
  constexpr const int num_elems = 2;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  Context::singleton().get_derivative().random_init();

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> pressure("host pressure",
                                                              num_elems);
  genRandArray(pressure, engine,
               std::uniform_real_distribution<Real>(0, 100.0));
  sync_to_device(pressure, elements.buffers.pressure);

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("host div_vdp",
                                                             num_elems);
  genRandArray(div_vdp, engine, std::uniform_real_distribution<Real>(0, 100.0));
  sync_to_device(div_vdp, elements.buffers.div_vdp);

  compute_subfunctor_test<preq_omega_ps_test> test_functor(elements, tracers);
  test_functor.run_functor();
  // Results of the computation
  HostViewManaged<Scalar * [NP][NP][NUM_LEV]> omega_p("omega_p", num_elems);
  Kokkos::deep_copy(omega_p, elements.buffers.omega_p);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> omega_p_f90(
      "Fortran omega_p");
  for (int ie = 0; ie < num_elems; ++ie) {
    preq_omega_ps_c_int(
        omega_p_f90.data(), Homme::subview(pressure, ie).data(),
        Homme::subview(test_functor.velocity, ie, test_functor.n0).data(),
        Homme::subview(div_vdp, ie)
            .data(),
        Homme::subview(test_functor.dinv, ie).data(),
        test_functor.dvv.data());
    for (int k = 0, vec_lev = 0; vec_lev < NUM_LEV; ++vec_lev) {
      // Note this MUST be this loop so that k is set properly
      for (int v = 0; v < VECTOR_SIZE; ++k, ++v) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            REQUIRE(!std::isnan(omega_p(ie, igp, jgp, vec_lev)[v]));
            REQUIRE(!std::isnan(omega_p_f90(k, igp, jgp)));
            Real rel_error = compare_answers(omega_p_f90(k, igp, jgp),
                                             omega_p(ie, igp, jgp, vec_lev)[v]);
            REQUIRE(rel_threshold >= rel_error);
          }
        }
      }
    }
  }
}

class preq_hydrostatic_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.preq_hydrostatic(kv);
  }
};

TEST_CASE("preq_hydrostatic", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 4.0;
  constexpr const int num_elems = 2;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<preq_hydrostatic_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> temperature_virt(
      "host virtual temperature", num_elems);
  genRandArray(temperature_virt, engine,
               std::uniform_real_distribution<Real>(0.0125, 1.0));
  sync_to_device(temperature_virt, elements.buffers.temperature_virt);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> pressure("host pressure",
                                                              num_elems);
  genRandArray(pressure, engine,
               std::uniform_real_distribution<Real>(0.0125, 1.0));
  sync_to_device(pressure, elements.buffers.pressure);

  TestType test_functor(elements, tracers);
  Kokkos::deep_copy(test_functor.phis, elements.m_phis);
  sync_to_host(elements.m_dp3d, test_functor.dp3d);
  test_functor.run_functor();
  sync_to_host(elements.m_phi, test_functor.phi);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> phi_f90("Fortran phi");
  for (int ie = 0; ie < num_elems; ++ie) {
    preq_hydrostatic_c_int(
        phi_f90.data(),
        Homme::subview(test_functor.phis, ie).data(),
        Homme::subview(temperature_virt, ie).data(),
        Homme::subview(pressure, ie)
            .data(),
        Homme::subview(test_functor.dp3d, ie, test_functor.n0).data());
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
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.compute_dp3d_np1(kv);
  }
};

TEST_CASE("dp3d", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 128.0;
  constexpr const int num_elems = 2;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  Context::singleton().get_derivative().random_init();

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("host div_vdp",
                                                             num_elems);
  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]> eta_dot("host div_dp", num_elems);

  genRandArray(div_vdp, engine, std::uniform_real_distribution<Real>(0, 100.0));
  genRandArray(eta_dot, engine, std::uniform_real_distribution<Real>(-10.0, 10.0));

  sync_to_device(div_vdp, elements.buffers.div_vdp);
  sync_to_device(eta_dot, elements.buffers.eta_dot_dpdn_buf);

  compute_subfunctor_test<dp3d_test> test_functor(elements, tracers);

  // To ensure the Fortran doesn't pass without doing anything,
  // copy the initial state before running any of the test
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]> dp3d_f90(
      "dp3d fortran", num_elems);
  Kokkos::deep_copy(dp3d_f90, test_functor.dp3d);

  test_functor.run_functor();
  sync_to_host(elements.m_dp3d, test_functor.dp3d);

  for (int ie = 0; ie < num_elems; ++ie) {
    caar_compute_dp3d_np1_c_int(
        test_functor.np1_f90, test_functor.nm1_f90,
        test_functor.functor.m_data.dt,
        Homme::subview(test_functor.spheremp, ie)
            .data(),
        Homme::subview(div_vdp, ie)
            .data(),
        Homme::subview(eta_dot, ie).data(),
        Homme::subview(dp3d_f90, ie).data());
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
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.compute_div_vdp(kv);
  }
};

TEST_CASE("vdp_vn0", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 512.0;
  constexpr const int num_elems = 2;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  Context::singleton().get_derivative().random_init();

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> vn0_f90(
      "vn0 f90 results", num_elems);
  sync_to_host(elements.m_derived_vn0, vn0_f90);

  compute_subfunctor_test<vdp_vn0_test> test_functor(elements, tracers);
  test_functor.run_functor();

  sync_to_host(elements.m_derived_vn0, test_functor.derived_v);
  HostViewManaged<Scalar * [2][NP][NP][NUM_LEV]> vdp("vdp results", num_elems);
  Kokkos::deep_copy(vdp, elements.buffers.vdp);
  HostViewManaged<Scalar * [NP][NP][NUM_LEV]> div_vdp("div_vdp results",
                                                      num_elems);
  Kokkos::deep_copy(div_vdp, elements.buffers.div_vdp);

  HostViewManaged<Real[2][NP][NP]> vdp_f90("vdp f90 results");
  HostViewManaged<Real[NP][NP]> div_vdp_f90("div_vdp f90 results");
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int vec_lev = 0, level = 0; vec_lev < NUM_LEV; ++vec_lev) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        caar_compute_divdp_c_int(
            compute_subfunctor_test<vdp_vn0_test>::eta_ave_w,
            Homme::subview(test_functor.velocity, ie, test_functor.n0, level).data(),
            Homme::subview(test_functor.dp3d, ie, test_functor.n0, level).data(),
            Homme::subview(test_functor.dinv, ie).data(),
            Homme::subview(test_functor.metdet, ie)
                .data(),
            test_functor.dvv.data(),
            Homme::subview(vn0_f90, ie, level).data(),
            vdp_f90.data(), div_vdp_f90.data());
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            for (int hgp = 0; hgp < 2; ++hgp) {
              {
                // Check vdp
                Real correct = vdp_f90(hgp, igp, jgp);
                REQUIRE(!std::isnan(correct));
                Real computed = vdp(ie, hgp, igp, jgp, vec_lev)[vector];
                REQUIRE(!std::isnan(computed));

//og why is this if? what is special about correct=0?
//i see this can backfire
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
              Real computed = div_vdp(ie, igp, jgp, vec_lev)[vector];
              REQUIRE(!std::isnan(computed));
              Real rel_error = compare_answers(correct, computed);
              REQUIRE(rel_threshold >= rel_error);
            }
          }
        }
      }
    }
  }
}

class pressure_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.compute_pressure(kv);
  }
};

TEST_CASE("pressure", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 1.0;
  constexpr const int num_elems = 2;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<pressure_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  Context::singleton().get_derivative().random_init();

  ExecViewManaged<Real[NUM_PHYSICAL_LEV]>::HostMirror hybrid_am_mirror("hybrid_am_host");
  ExecViewManaged<Real[NUM_INTERFACE_LEV]>::HostMirror hybrid_ai_mirror("hybrid_ai_host");
  ExecViewManaged<Real[NUM_PHYSICAL_LEV]>::HostMirror hybrid_bm_mirror("hybrid_bm_host");
  ExecViewManaged<Real[NUM_INTERFACE_LEV]>::HostMirror hybrid_bi_mirror("hybrid_bi_host");

//OG coefficients A and B increase is not taken into here...
//probably, does not matter
//this test takes in only ai, so others should be made quiet_nans?
  genRandArray(hybrid_am_mirror, engine,
               std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_ai_mirror, engine,
               std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_bm_mirror, engine,
               std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_bi_mirror, engine,
               std::uniform_real_distribution<Real>(0.0125, 10.0));

  // Setup hvc BEFORE creating the test_functor, since hvcoord is const in CaarFunctorImpl
  HybridVCoord& hvc = Context::singleton().get_hvcoord();
  hvc.init(TestType::ps0,
           hybrid_am_mirror.data(),
           hybrid_ai_mirror.data(),
           hybrid_bm_mirror.data(),
           hybrid_bi_mirror.data());

//OG does init use any of hybrid coefficients? do they need to be generated?
//init makes device copies
  TestType test_functor(elements, tracers);

  test_functor.functor.set_n0_qdp(TestType::n0_qdp);
  test_functor.functor.set_rk_stage_data(TestType::nm1, TestType::n0, TestType::np1,
                                   TestType::dt, TestType::eta_ave_w, false);

  test_functor.run_functor();

  HostViewManaged<Scalar * [NP][NP][NUM_LEV]> pressure_cxx("pressure_cxx",
                                                           num_elems);
  Kokkos::deep_copy(pressure_cxx, elements.buffers.pressure);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> pressure_f90("pressure_f90");

  sync_to_host(elements.m_dp3d, test_functor.dp3d);

  for (int ie = 0; ie < num_elems; ++ie) {
    caar_compute_pressure_c_int(
        hybrid_ai_mirror(0), test_functor.functor.m_hvcoord.ps0,
        Homme::subview(test_functor.dp3d, ie, test_functor.n0).data(),
        pressure_f90.data());
    for (int vec_lev = 0, level = 0; vec_lev < NUM_LEV; ++vec_lev) {
      for (int vector = 0; vector < VECTOR_SIZE && level < NUM_PHYSICAL_LEV;
           ++vector, ++level) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            const Real correct = pressure_f90(level, igp, jgp);
            const Real computed = pressure_cxx(ie, igp, jgp, vec_lev)[vector];
            REQUIRE(!std::isnan(correct));
            REQUIRE(!std::isnan(computed));
            const Real rel_error = compare_answers(correct, computed);
            REQUIRE(rel_threshold >= rel_error);
          }
        }
      }
    }
  }
}

class temperature_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.compute_temperature_np1(kv);
  }
};

TEST_CASE("temperature", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 2.0;
  constexpr const int num_elems = 2;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<temperature_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  Context::singleton().get_derivative().random_init();

  ExecViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]>::HostMirror
    temperature_virt("Virtual temperature test", num_elems);
  genRandArray(temperature_virt, engine, std::uniform_real_distribution<Real>(0, 1.0));
  sync_to_device(temperature_virt, elements.buffers.temperature_virt);

  ExecViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]>::HostMirror
    omega_p("Omega P test", num_elems);
  genRandArray(omega_p, engine, std::uniform_real_distribution<Real>(0, 1.0));
  sync_to_device(omega_p, elements.buffers.omega_p);

  ExecViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]>::HostMirror t_vadv_f90("t_vadv", num_elems);
  genRandArray(t_vadv_f90, engine, std::uniform_real_distribution<Real>(-100, 100));
  sync_to_device(t_vadv_f90, elements.buffers.t_vadv_buf);

  TestType test_functor(elements, tracers);
  test_functor.run_functor();

  sync_to_host(elements.m_t, test_functor.temperature);

  HostViewManaged<Real [NP][NP]> temperature_f90("Temperature f90");
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {

      caar_compute_temperature_c_int(test_functor.dt,
                                     Homme::subview(test_functor.spheremp, ie).data(),
                                     Homme::subview(test_functor.dinv, ie).data(),
                                     test_functor.dvv.data(),
                                     Homme::subview(test_functor.velocity, ie,
                                                     test_functor.n0, level).data(),
                                     Homme::subview(temperature_virt, ie, level).data(),
                                     Homme::subview(omega_p, ie, level).data(),
                                     Homme::subview(t_vadv_f90, ie, level).data(),
                                     Homme::subview(test_functor.temperature, ie,
                                                     test_functor.nm1, level).data(),
                                     Homme::subview(test_functor.temperature, ie,
                                                     test_functor.n0, level).data(),
                                     temperature_f90.data());

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
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.compute_temperature_no_tracers_helper(kv);
  }
};

TEST_CASE("virtual temperature no tracers",
          "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 1.0;
  constexpr const int num_elems = 2;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<virtual_temperature_no_tracers_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  TestType test_functor(elements, tracers);
  sync_to_host(elements.m_t, test_functor.temperature);
  test_functor.run_functor();

  ExecViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]>::HostMirror
  temperature_virt_cxx("virtual temperature cxx", num_elems);

  sync_to_host(elements.buffers.temperature_virt, temperature_virt_cxx);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> temperature_virt_f90(
      "virtual temperature f90");

  for (int ie = 0; ie < num_elems; ++ie) {
    caar_compute_temperature_no_tracers_c_int(
        Homme::subview(test_functor.temperature, ie, test_functor.n0).data(),
        temperature_virt_f90.data());
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
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.compute_temperature_tracers_helper(kv);
  }
};


TEST_CASE("moist_virtual_temperature",
          "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 4.0;
  constexpr const int num_elems = 2;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType =
      compute_subfunctor_test<virtual_temperature_with_tracers_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  TestType test_functor(elements, tracers);
  sync_to_host(tracers.qdp, test_functor.qdp);
  sync_to_host(elements.m_dp3d, test_functor.dp3d);
  sync_to_host(elements.m_t, test_functor.temperature);
  test_functor.run_functor();

  ExecViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]>::HostMirror
  temperature_virt_cxx("virtual temperature cxx", num_elems);

  sync_to_host(elements.buffers.temperature_virt, temperature_virt_cxx);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> temperature_virt_f90(
      "virtual temperature f90");

  for (int ie = 0; ie < num_elems; ++ie) {
    caar_compute_temperature_tracers_c_int(
        Homme::subview(test_functor.qdp, ie, test_functor.n0_qdp, 0).data(),
        Homme::subview(test_functor.dp3d, ie, test_functor.n0).data(),
        Homme::subview(test_functor.temperature, ie, test_functor.n0).data(),
        temperature_virt_f90.data());
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
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.compute_omega_p(kv);
  }
};

TEST_CASE("omega_p", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 0.0;
  constexpr const int num_elems = 2;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<omega_p_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

//which omega_p should survive? buffers or m_omega_p?
//sort it out
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> source_omega_p(
      "source omega p", num_elems);
  genRandArray(source_omega_p, engine,
               std::uniform_real_distribution<Real>(0.0125, 1.0));
  sync_to_device(source_omega_p, elements.buffers.omega_p);

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> omega_p_f90("omega p f90",
                                                                 num_elems);

  TestType test_functor(elements, tracers);
  sync_to_host(elements.m_omega_p, omega_p_f90);
  test_functor.run_functor();
  sync_to_host(elements.m_omega_p, test_functor.omega_p);

  ExecViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]>::HostMirror
  temperature_virt_cxx("virtual temperature cxx", num_elems);

  sync_to_host(elements.buffers.temperature_virt, temperature_virt_cxx);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> temperature_virt_f90(
      "virtual temperature f90");

  for (int ie = 0; ie < num_elems; ++ie) {
    caar_compute_omega_p_c_int(test_functor.eta_ave_w,
                               Homme::subview(source_omega_p, ie).data(),
                               Homme::subview(omega_p_f90, ie).data());
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


class accumulate_eta_dot_dpdn_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.accumulate_eta_dot_dpdn(kv);
  }
};

TEST_CASE("accumulate eta_dot_dpdn", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 0.0;
  constexpr const int num_elems = 2;
  std::random_device rd;
  rngAlg engine(rd());
  using TestType = compute_subfunctor_test<accumulate_eta_dot_dpdn_test>;

  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]> eta_dot("eta dot", num_elems);
  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]> eta_dot_total_f90("total eta dot", num_elems);
  genRandArray(eta_dot, engine, std::uniform_real_distribution<Real>(-10.0, 10.0));

//check rsplit in m_data init!!! set to zero?
//wahts going on with eta_ave_w? should be random
//zeored, we don't need them in this test
  ExecViewManaged<Real[NUM_PHYSICAL_LEV]>::HostMirror hybrid_am_mirror("hybrid_am_host");
  ExecViewManaged<Real[NUM_INTERFACE_LEV]>::HostMirror hybrid_ai_mirror("hybrid_ai_host");
  ExecViewManaged<Real[NUM_PHYSICAL_LEV]>::HostMirror hybrid_bm_mirror("hybrid_bm_host");
  ExecViewManaged<Real[NUM_INTERFACE_LEV]>::HostMirror hybrid_bi_mirror("hybrid_bi_host");

  // Setup hvc BEFORE creating the test_functor, since hvcoord is const in CaarFunctorImpl
  HybridVCoord& hvc = Context::singleton().get_hvcoord();
  hvc.init(TestType::ps0,
           hybrid_am_mirror.data(),
           hybrid_ai_mirror.data(),
           hybrid_bm_mirror.data(),
           hybrid_bi_mirror.data());

  constexpr int rsplit = 0;
  TestType test_functor(elements, tracers, rsplit);

  test_functor.functor.set_n0_qdp(TestType::n0_qdp);

  test_functor.functor.set_rk_stage_data(TestType::nm1, TestType::n0, TestType::np1,
                                                TestType::dt, TestType::eta_ave_w, false);

  sync_to_device(eta_dot, elements.buffers.eta_dot_dpdn_buf);
  sync_to_host_p2i(elements.m_eta_dot_dpdn, eta_dot_total_f90);
  //will run on device
  test_functor.run_functor();

  sync_to_host_p2i(elements.m_eta_dot_dpdn, test_functor.eta_dpdn);

  for (int ie = 0; ie < num_elems; ++ie) {
    caar_adjust_eta_dot_dpdn_c_int(test_functor.eta_ave_w,
                               Homme::subview(eta_dot_total_f90, ie).data(),
                               Homme::subview(eta_dot, ie).data());
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          //compare total eta
          const Real correct = eta_dot_total_f90(ie, level, igp, jgp);
          REQUIRE(!std::isnan(correct));
          const Real computed = test_functor.eta_dpdn(ie, level, igp, jgp);
          REQUIRE(!std::isnan(computed));
          const Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }//level loop
  }//ie loop
};//end of accumulate_eta_dot_dpdn test


class eta_dot_dpdn_vertadv_euler_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.compute_eta_dot_dpdn_vertadv_euler(kv);
  }
};

// computing eta_dot_dpdn: (eta_dot, divdp, sdot_sum) --> (eta_dot, sdot_sum)
// affects only buf value of eta, sdot
TEST_CASE("eta_dot_dpdn", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 0.0;
  constexpr const int num_elems = 2;
  std::random_device rd;
  rngAlg engine(rd());
  using TestType = compute_subfunctor_test<eta_dot_dpdn_vertadv_euler_test>;
  // This must be a reference to ensure the views are initialized in the
  // singleton
  // on host first
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("host div_dp", num_elems);
  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]> eta_dot("host div_dp", num_elems);
  HostViewManaged<Real * [NP][NP]> sdot_sum("host sdot_sum", num_elems);
  //random init host views
  genRandArray(div_vdp, engine, std::uniform_real_distribution<Real>(-10.0, 10.0));
  genRandArray(eta_dot, engine, std::uniform_real_distribution<Real>(-10.0, 10.0));
  genRandArray(sdot_sum, engine, std::uniform_real_distribution<Real>(-10.0, 10.0));
  sync_to_device(div_vdp, elements.buffers.div_vdp);
  sync_to_device(eta_dot, elements.buffers.eta_dot_dpdn_buf);
  sync_to_device(sdot_sum, elements.buffers.sdot_sum);
//only hybi is used, should the rest be quiet_nans? yes
  ExecViewManaged<Real[NUM_PHYSICAL_LEV]>::HostMirror hybrid_am_mirror("hybrid_am_host");
  ExecViewManaged<Real[NUM_INTERFACE_LEV]>::HostMirror hybrid_ai_mirror("hybrid_ai_host");
  ExecViewManaged<Real[NUM_PHYSICAL_LEV]>::HostMirror hybrid_bm_mirror("hybrid_bm_host");
  ExecViewManaged<Real[NUM_INTERFACE_LEV]>::HostMirror hybrid_bi_mirror("hybrid_bi_host");
  genRandArray(hybrid_am_mirror, engine, std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_ai_mirror, engine, std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_bm_mirror, engine, std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_bi_mirror, engine, std::uniform_real_distribution<Real>(0.0125, 10.0));

  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]> eta_dot_f90("eta_dot f90", num_elems);
  HostViewManaged<Real * [NP][NP]> sdot_sum_f90("tavd f90", num_elems);

  deep_copy(eta_dot_f90, eta_dot);
  deep_copy(sdot_sum_f90, sdot_sum);

  // Setup hvc BEFORE creating the test_functor, since hvcoord is const in CaarFunctorImpl
  HybridVCoord& hvc = Context::singleton().get_hvcoord();
  hvc.init(TestType::ps0,
           hybrid_am_mirror.data(),
           hybrid_ai_mirror.data(),
           hybrid_bm_mirror.data(),
           hybrid_bi_mirror.data());

  constexpr int rsplit = 0;
  TestType test_functor(elements, tracers, rsplit);

  test_functor.functor.set_n0_qdp(TestType::n0_qdp);

  test_functor.functor.set_rk_stage_data(TestType::nm1, TestType::n0, TestType::np1,
                                         TestType::dt, TestType::eta_ave_w, false);


  //will run on device
  test_functor.run_functor();

  sync_to_host(elements.buffers.eta_dot_dpdn_buf, eta_dot);
  sync_to_host(elements.buffers.sdot_sum, sdot_sum);

  for (int ie = 0; ie < num_elems; ++ie) {
    caar_compute_eta_dot_dpdn_vertadv_euler_c_int(
                               Homme::subview(eta_dot_f90, ie).data(),
                               Homme::subview(sdot_sum_f90, ie).data(),
                               Homme::subview(div_vdp, ie).data(),
                               hybrid_bi_mirror.data());

    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          //compare eta
          const Real correct = eta_dot_f90(ie, level, igp, jgp);
          REQUIRE(!std::isnan(correct));
          const Real computed = eta_dot(ie, level, igp, jgp);
          REQUIRE(!std::isnan(computed));
          const Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);

        }
      }
    }//level loop

    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        //compare sdot
        const Real correct = sdot_sum_f90(ie, igp, jgp);
        REQUIRE(!std::isnan(correct));
        const Real computed = sdot_sum(ie, igp, jgp);
        REQUIRE(!std::isnan(computed));
        const Real rel_error = compare_answers(correct, computed);
        REQUIRE(rel_threshold >= rel_error);
      }
    }
  }//ie loop
};//end of compute_eta_dot_dpdn_vertadv_euler test


class preq_vertadv_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctorImpl &functor, KernelVariables &kv) {
    functor.preq_vertadv(kv);
  }
};

//preq_vertadv: (T, eta_dot, v, 1/dp3d) --> t_vadv, v_vadv
//takes in only buf value of eta, m values T, v, dp3d
TEST_CASE("preq_vertadv", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 0.0;
  constexpr const int num_elems = 2;
  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<preq_vertadv_test>;
  // This must be a reference to ensure the views are initialized in the
  // singleton
  // on host first
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  Tracers &tracers = Context::singleton().get_tracers();
  tracers.random_init();

  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]> eta_dot("host t_vadv", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> t_vadv("host t_vadv", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> v_vadv("host v_vadv", num_elems);

//what is the good strategy for NAN values? tails of NUM_INTERFACE_LEV vars
//should be init-ed to nans. the rest can be random...
//how to implement it -- to have a method that will assign quiet nans to tail of eta_dot
//in elments?
  genRandArray(t_vadv, engine, std::uniform_real_distribution<Real>(-100, 100));
  genRandArray(v_vadv, engine, std::uniform_real_distribution<Real>(-100, 100));
  genRandArray(eta_dot, engine, std::uniform_real_distribution<Real>(-100, 100));

  sync_to_device(eta_dot, elements.buffers.eta_dot_dpdn_buf);
  sync_to_device(t_vadv, elements.buffers.t_vadv_buf);
  sync_to_device(v_vadv, elements.buffers.v_vadv_buf);

  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]> eta_dot_f90("eta_dot f90", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> t_vadv_f90("tavd f90", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> v_vadv_f90("vavd f90", num_elems);
  HostViewManaged<Real [NUM_PHYSICAL_LEV][NP][NP]> rdp_f90("rdp f90", num_elems);

  deep_copy(eta_dot_f90, eta_dot);
  deep_copy(t_vadv_f90, t_vadv);
  deep_copy(v_vadv_f90, v_vadv);

  TestType test_functor(elements, tracers);
  test_functor.run_functor();

  const int n0 = test_functor.n0;

  //now copy buffer vals back to test values
  sync_to_host(elements.buffers.eta_dot_dpdn_buf, eta_dot);
  sync_to_host(elements.buffers.t_vadv_buf, t_vadv);
  sync_to_host(elements.buffers.v_vadv_buf, v_vadv);

  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        for (int i = 0; i < NP; i++) {
          for (int j = 0; j < NP; j++) {
            rdp_f90(level, i, j) = 1/test_functor.dp3d(ie, n0, level, i, j);
          }
        }
    }//level loop
    preq_vertadv(
        Homme::subview(test_functor.temperature, ie, n0).data(),
        Homme::subview(test_functor.velocity, ie, n0).data(),
        Homme::subview(eta_dot_f90, ie).data(),
        rdp_f90.data(),
        Homme::subview(t_vadv_f90, ie).data(),
        Homme::subview(v_vadv_f90, ie).data()
    );//preq vertadv call

    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          //errors for t_vadv
          const Real correct = t_vadv_f90(ie, level, igp, jgp);
          REQUIRE(!std::isnan(correct));
          const Real computed = t_vadv(ie, level, igp, jgp);
          REQUIRE(!std::isnan(computed));
          const Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);
          //errors for v_vadv
          for (int dim = 0; dim < 2; dim ++){
            const Real correct = v_vadv_f90(ie, level, dim, igp, jgp);
            REQUIRE(!std::isnan(correct));
            const Real computed = v_vadv(ie, level, dim, igp, jgp);
            REQUIRE(!std::isnan(computed));
            const Real rel_error = compare_answers(correct, computed);
            REQUIRE(rel_threshold >= rel_error);
          }//end of dim loop
        }
      }
    }//level loop
  }//ie loop
}//end of test case preq_vertadv

struct LimiterTester {
  static constexpr Real eps { std::numeric_limits<Real>::epsilon() };

  using HostGll = HostViewManaged<Real[NP][NP]>;
  using HostGllLvl = HostViewManaged<Scalar[NP][NP][NUM_LEV]>;
  using Host2Lvl = HostViewManaged<Scalar[2][NUM_LEV]>;
  using HostPlvl = HostViewManaged<Real[NUM_PHYSICAL_LEV]>;

  HostGll sphweights;
  HostGllLvl dpmass, ptens;
  Host2Lvl qlim;

  using DevGll = ExecViewManaged<Real[NP][NP]>;
  using DevGllLvl = ExecViewManaged<Scalar[NP][NP][NUM_LEV]>;
  using Dev2Lvl = ExecViewManaged<Scalar[2][NUM_LEV]>;

  DevGll sphweights_d;
  DevGllLvl dpmass_d, ptens_d;
  Dev2Lvl qlim_d;

  // For correctness checking.
  HostGllLvl ptens_orig;
  HostPlvl Qmass;

  void init () {
    sphweights = Kokkos::create_mirror_view(sphweights_d);
    dpmass = Kokkos::create_mirror_view(dpmass_d);
    ptens = Kokkos::create_mirror_view(ptens_d);
    qlim = Kokkos::create_mirror_view(qlim_d);
  }

  LimiterTester ()
    : sphweights_d("sphweights"), dpmass_d("dpmass"),
      ptens_d("ptens"), qlim_d("qlim"),
      ptens_orig("ptens_orig"), Qmass("Qmass")
  { init(); }

  void deep_copy (const LimiterTester& lv)
  {
    Kokkos::deep_copy(sphweights, lv.sphweights);
    Kokkos::deep_copy(dpmass, lv.dpmass);
    Kokkos::deep_copy(ptens, lv.ptens);
    Kokkos::deep_copy(qlim, lv.qlim);
    Kokkos::deep_copy(Qmass, lv.Qmass);
    Kokkos::deep_copy(ptens_orig, lv.ptens_orig);
    Kokkos::deep_copy(sphweights_d, lv.sphweights_d);
    Kokkos::deep_copy(dpmass_d, lv.dpmass_d);
    Kokkos::deep_copy(ptens_d, lv.ptens_d);
    Kokkos::deep_copy(qlim_d, lv.qlim_d);
  }

  void todevice () {
    Kokkos::deep_copy(sphweights_d, sphweights);
    Kokkos::deep_copy(dpmass_d, dpmass);
    Kokkos::deep_copy(ptens_d, ptens);
    Kokkos::deep_copy(qlim_d, qlim);
  }

  void fromdevice () {
    Kokkos::deep_copy(sphweights, sphweights_d);
    Kokkos::deep_copy(dpmass, dpmass_d);
    Kokkos::deep_copy(ptens, ptens_d);
    Kokkos::deep_copy(qlim, qlim_d);
  }

  template <typename Array>
  static void urand (Array& a, Real lo, Real hi) {
    std::random_device rd;
    rngAlg engine(rd());
    genRandArray(a, engine, std::uniform_real_distribution<Real>(lo, hi));
  }

  void init_feasible () {
    urand(sphweights, 1.0/16, 2.0/16);
    urand(dpmass, 0.5, 1);
    urand(ptens, 0, 1);
    urand(qlim, 0, 1);

    for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
      const int vi = k / VECTOR_SIZE, si = k % VECTOR_SIZE;

      // Order q limits.
      auto q0 = qlim(0,vi)[si], q1 = qlim(1,vi)[si];
      if (q1 < q0) {
        std::swap(q0, q1);
        qlim(0,vi)[si] = q0;
        qlim(1,vi)[si] = q1;
      }

      // Turn ptens into density.
      for (int i = 0; i < NP; ++i)
        for (int j = 0; j < NP; ++j)
          ptens(i,j,vi)[si] *= dpmass(i,j,vi)[si];

      // Make problem feasible by adjusting cell mass.
      Real m = 0, lo = 0, hi = 0;
      for (int i = 0; i < NP; ++i)
        for (int j = 0; j < NP; ++j) {
          m += sphweights(i,j) * ptens(i,j,vi)[si];
          lo += sphweights(i,j) * qlim(0,vi)[si] * dpmass(i,j,vi)[si];
          hi += sphweights(i,j) * qlim(1,vi)[si] * dpmass(i,j,vi)[si];
        }
      const Real dm = m < lo ? lo - m : m > hi ? hi - m : 0;
      if (dm)
        for (int i = 0; i < NP; ++i)
          for (int j = 0; j < NP; ++j)
            ptens(i,j,vi)[si] += (1 + 1e2*eps)*dm/(NP*NP*sphweights(i,j));
      m = 0;
      for (int i = 0; i < NP; ++i)
        for (int j = 0; j < NP; ++j)
          m += sphweights(i,j) * ptens(i,j,vi)[si];
      REQUIRE(m >= (1 - 1e1*eps)*lo);
      REQUIRE(m <= (1 + 1e1*eps)*hi);

      // For checking, record original values and true mass.
      Kokkos::deep_copy(ptens_orig, ptens);
      Qmass(k) = m;
    }

    todevice();
  }

  size_t team_shmem_size (const int team_size) const {
    return Homme::EulerStepFunctorImpl::limiter_team_shmem_size(team_size);
  }

  struct Lim8 {};
  KOKKOS_INLINE_FUNCTION void operator() (const Lim8&, const Homme::TeamMember& team) const {
    Homme::EulerStepFunctorImpl
      ::limiter_optim_iter_full(team, sphweights_d, dpmass_d, qlim_d, ptens_d);
  }

  struct CAAS {};
  KOKKOS_INLINE_FUNCTION void operator() (const CAAS&, const Homme::TeamMember& team) const {
    Homme::EulerStepFunctorImpl
      ::limiter_clip_and_sum(team, sphweights_d, dpmass_d, qlim_d, ptens_d);
  }

  void check () {
    for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
      const int vi = k / VECTOR_SIZE, si = k % VECTOR_SIZE;
      Real m = 0;
      for (int i = 0; i < NP; ++i)
        for (int j = 0; j < NP; ++j) {
          // Check that the mixing ratio is limited.
          REQUIRE(ptens(i,j,vi)[si] >=
                  (1 - 1e1*eps)*qlim(0,vi)[si]*dpmass(i,j,vi)[si]);
          REQUIRE(ptens(i,j,vi)[si] <=
                  (1 + 1e1*eps)*qlim(1,vi)[si]*dpmass(i,j,vi)[si]);
          m += sphweights(i,j) * ptens(i,j,vi)[si];
        }
      // Check mass conservation.
      REQUIRE(std::abs(m - Qmass(k)) <= 1e2*eps*Qmass(k));
    }
  }
};

TEST_CASE("lim=8 math correctness", "limiter") {
  LimiterTester lv;
  lv.init_feasible();
  Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace, LimiterTester::Lim8>(1), lv);
  lv.fromdevice();
  lv.check();
}

TEST_CASE("CAAS math correctness", "limiter") {
  LimiterTester lv;
  lv.init_feasible();
  Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace, LimiterTester::CAAS>(1), lv);
  lv.fromdevice();
  lv.check();
}

// The best thing to do here would be to check the KKT conditions for the
// 1-norm-minimization problem. That's a fair bit of programming. Instead, check
// that two very different methods that ought to provide 1-norm-minimal
// corrections have corrections that have the same 1-norm.
TEST_CASE("1-norm minimal", "limiters") {
  LimiterTester lv;
  lv.init_feasible();
  LimiterTester lv_deepcopy;
  lv_deepcopy.deep_copy(lv);
  std::vector<Real> lim8norm1(NUM_PHYSICAL_LEV), caasnorm1(NUM_PHYSICAL_LEV);

  auto get_norm1 = [&] (const LimiterTester& lv, std::vector<Real>& n1s) {
    for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
      const int vi = k / VECTOR_SIZE, si = k % VECTOR_SIZE;
      Real n1 = 0;
      for (int i = 0; i < NP; ++i)
        for (int j = 0; j < NP; ++j)
          n1 += std::abs(lv.ptens(i,j,vi)[si] - lv.ptens_orig(i,j,vi)[si])*lv.sphweights(i,j);
      n1s[k] = n1;
    }
  };

  Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace, LimiterTester::Lim8>(1), lv);
  lv.fromdevice();
  get_norm1(lv, lim8norm1);

  Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace, LimiterTester::CAAS>(1), lv_deepcopy);
  lv_deepcopy.fromdevice();
  get_norm1(lv_deepcopy, caasnorm1);

  for (int k = 0; k < NUM_PHYSICAL_LEV; ++k)
    REQUIRE(std::abs(lim8norm1[k] - caasnorm1[k]) <= 1e3*LimiterTester::eps*caasnorm1[k]);
}
