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
      : functor(elements, Context::singleton().get_derivative()),
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
        spheremp("SphereMP", elements.num_elems()), 
        dvv("dvv"), nets(1),
        nete(elements.num_elems()),
        rsplit(0) {

//make these random
    Real hybrid_am[NUM_LEV_P] = { 0 };
    Real hybrid_ai[NUM_LEV_P+1] = { 0 };
    Real hybrid_bm[NUM_LEV_P] = { 0 };
    Real hybrid_bi[NUM_LEV_P+1] = { 0 };

    functor.m_data.init(0, elements.num_elems(), elements.num_elems(), nm1, n0, np1,
                        qn0, ps0, dt, false, eta_ave_w, 0, //0 for rsplit for now 
                        hybrid_am, hybrid_ai, hybrid_bm, hybrid_bi);

//is this one random?
    Context::singleton().get_derivative().dvv(dvv.data());

    elements.push_to_f90_pointers(velocity.data(), temperature.data(),
                                dp3d.data(), phi.data(), 
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
    Kokkos::TeamPolicy<ExecSpace> policy(functor.m_data.num_elems, 16, 4);
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

private:
  int rsplit;

public:
  int return_rsplit(){
    return rsplit;
  }

  void set_rsplit(int _rsplit){
    assert(_rsplit >= 0);
    rsplit = _rsplit;
  };
};

class compute_energy_grad_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.compute_energy_grad(kv);
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
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);
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

  compute_subfunctor_test<compute_energy_grad_test> test_functor(elements);
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
                Kokkos::subview(test_functor.dinv, ie, Kokkos::ALL, Kokkos::ALL,
                                Kokkos::ALL, Kokkos::ALL).data(),
                Kokkos::subview(test_functor.phi, ie, level * VECTOR_SIZE + v,
                                Kokkos::ALL, Kokkos::ALL).data(),
                Kokkos::subview(test_functor.velocity, ie, test_functor.n0,
                                level * VECTOR_SIZE + v, Kokkos::ALL,
                                Kokkos::ALL, Kokkos::ALL).data(),
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
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.preq_omega_ps(kv);
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
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);
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

  compute_subfunctor_test<preq_omega_ps_test> test_functor(elements);
  test_functor.run_functor();
  // Results of the computation
  HostViewManaged<Scalar * [NP][NP][NUM_LEV]> omega_p("omega_p", num_elems);
  Kokkos::deep_copy(omega_p, elements.buffers.omega_p);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> omega_p_f90(
      "Fortran omega_p");
  for (int ie = 0; ie < num_elems; ++ie) {
    preq_omega_ps_c_int(
        omega_p_f90.data(), Kokkos::subview(pressure, ie, Kokkos::ALL,
                                            Kokkos::ALL, Kokkos::ALL).data(),
        Kokkos::subview(test_functor.velocity, ie, test_functor.n0, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(),
        Kokkos::subview(div_vdp, ie, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL)
            .data(),
        Kokkos::subview(test_functor.dinv, ie, Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL).data(),
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
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.preq_hydrostatic(kv);
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
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

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

  TestType test_functor(elements);
  Kokkos::deep_copy(test_functor.phis, elements.m_phis);
  sync_to_host(elements.m_dp3d, test_functor.dp3d);
  test_functor.run_functor();
  sync_to_host(elements.m_phi, test_functor.phi);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> phi_f90("Fortran phi");
  for (int ie = 0; ie < num_elems; ++ie) {
    preq_hydrostatic_c_int(
        phi_f90.data(),
        Kokkos::subview(test_functor.phis, ie, Kokkos::ALL, Kokkos::ALL).data(),
        Kokkos::subview(temperature_virt, ie, Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL).data(),
        Kokkos::subview(pressure, ie, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL)
            .data(),
        Kokkos::subview(test_functor.dp3d, ie, test_functor.n0, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL).data());
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
    functor.compute_dp3d_np1(kv);
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
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);
  Context::singleton().get_derivative().random_init();

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("host div_vdp",
                                                             num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV+1][NP][NP]> eta_dot("host div_dp", num_elems);

  genRandArray(div_vdp, engine, std::uniform_real_distribution<Real>(0, 100.0));
  genRandArray(eta_dot, engine, std::uniform_real_distribution<Real>(-10.0, 10.0));

  sync_to_device(div_vdp, elements.buffers.div_vdp);
  sync_to_device(eta_dot, elements.buffers.eta_dot_dpdn_buf);

  compute_subfunctor_test<dp3d_test> test_functor(elements);

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
        Kokkos::subview(test_functor.spheremp, ie, Kokkos::ALL, Kokkos::ALL)
            .data(),
        Kokkos::subview(div_vdp, ie, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL)
            .data(),
        Kokkos::subview(eta_dot, ie, Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL).data(),
        Kokkos::subview(dp3d_f90, ie, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL).data());
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
    functor.compute_div_vdp(kv);
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
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);
  Context::singleton().get_derivative().random_init();

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> vn0_f90(
      "vn0 f90 results", num_elems);
  sync_to_host(elements.m_derived_un0, elements.m_derived_vn0, vn0_f90);

  compute_subfunctor_test<vdp_vn0_test> test_functor(elements);
  test_functor.run_functor();

  sync_to_host(elements.m_derived_un0, elements.m_derived_vn0,
               test_functor.derived_v);
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
            Kokkos::subview(test_functor.velocity, ie, test_functor.n0, level,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(),
            Kokkos::subview(test_functor.dp3d, ie, test_functor.n0, level,
                            Kokkos::ALL, Kokkos::ALL).data(),
            Kokkos::subview(test_functor.dinv, ie, Kokkos::ALL, Kokkos::ALL,
                            Kokkos::ALL, Kokkos::ALL).data(),
            Kokkos::subview(test_functor.metdet, ie, Kokkos::ALL, Kokkos::ALL)
                .data(),
            test_functor.dvv.data(),
            Kokkos::subview(vn0_f90, ie, level, Kokkos::ALL, Kokkos::ALL,
                            Kokkos::ALL).data(),
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
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.compute_pressure(kv);
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
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);
  Context::singleton().get_derivative().random_init();

  TestType test_functor(elements);

  ExecViewManaged<Real[NUM_LEV_P]>::HostMirror hybrid_am_mirror("hybrid_am_host");
  ExecViewManaged<Real[NUM_LEV_P+1]>::HostMirror hybrid_ai_mirror("hybrid_ai_host");
  ExecViewManaged<Real[NUM_LEV_P]>::HostMirror hybrid_bm_mirror("hybrid_bm_host");
  ExecViewManaged<Real[NUM_LEV_P+1]>::HostMirror hybrid_bi_mirror("hybrid_bi_host");

//OG coefficients A and B increase is not taken into here...
//probably, does not matter
//this test takes in only ai, so other should be made quiet_nans
  genRandArray(hybrid_am_mirror, engine,
               std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_ai_mirror, engine,
               std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_bm_mirror, engine,
               std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_bi_mirror, engine,
               std::uniform_real_distribution<Real>(0.0125, 10.0));

//OG does init use any of hybrid coefficients? do they need to be generated?
//init makes device copies
  test_functor.functor.m_data.init(1, num_elems, num_elems, TestType::nm1,
                                   TestType::n0, TestType::np1, TestType::qn0,
                                   TestType::dt, TestType::ps0, false,
                                   TestType::eta_ave_w, 
                                   0, //0 for rsplit
                                   hybrid_am_mirror.data(),
                                   hybrid_ai_mirror.data(),
                                   hybrid_bm_mirror.data(),
                                   hybrid_bi_mirror.data());

  test_functor.run_functor();

  HostViewManaged<Scalar * [NP][NP][NUM_LEV]> pressure_cxx("pressure_cxx",
                                                           num_elems);
  Kokkos::deep_copy(pressure_cxx, elements.buffers.pressure);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> pressure_f90("pressure_f90");

  sync_to_host(elements.m_dp3d, test_functor.dp3d);

  for (int ie = 0; ie < num_elems; ++ie) {
    caar_compute_pressure_c_int(
        hybrid_ai_mirror(0), test_functor.functor.m_data.ps0,
        Kokkos::subview(test_functor.dp3d, ie, test_functor.n0, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL).data(),
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
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.compute_temperature_np1(kv);
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
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);
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

  TestType test_functor(elements);
  test_functor.run_functor();

  sync_to_host(elements.m_t, test_functor.temperature);

//  HostViewManaged<Real [NP][NP]> temperature_vadv("Temperature Vertical Advection");
//  for(int i = 0; i < NP; ++i) {
//    for(int j = 0; j < NP; ++j) {
//      temperature_vadv(i, j) = 0.0;
//    }
//  }

  HostViewManaged<Real [NP][NP]> temperature_f90("Temperature f90");
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {

      caar_compute_temperature_c_int(test_functor.dt,
                                     Kokkos::subview(test_functor.spheremp, ie,
                                                     Kokkos::ALL,
                                                     Kokkos::ALL).data(),
                                     Kokkos::subview(test_functor.dinv, ie,
                                                     Kokkos::ALL, Kokkos::ALL,
                                                     Kokkos::ALL, Kokkos::ALL).data(),
                                     test_functor.dvv.data(),
                                     Kokkos::subview(test_functor.velocity, ie,
                                                     test_functor.n0, level,
                                                     Kokkos::ALL, Kokkos::ALL,
                                                     Kokkos::ALL).data(),
                                     Kokkos::subview(temperature_virt, ie, level,
                                                     Kokkos::ALL, Kokkos::ALL).data(),
                                     Kokkos::subview(omega_p, ie, level,
                                                     Kokkos::ALL, Kokkos::ALL).data(),
                                     Kokkos::subview(t_vadv_f90, ie, level, 
                                                     Kokkos::ALL, Kokkos::ALL).data(),
                                     Kokkos::subview(test_functor.temperature, ie,
                                                     test_functor.nm1, level, Kokkos::ALL,
                                                     Kokkos::ALL).data(),
                                     Kokkos::subview(test_functor.temperature, ie,
                                                     test_functor.n0, level, Kokkos::ALL,
                                                     Kokkos::ALL).data(),
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
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.compute_temperature_no_tracers_helper(kv);
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
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  TestType test_functor(elements);
  sync_to_host(elements.m_t, test_functor.temperature);
  test_functor.run_functor();

  ExecViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]>::HostMirror
  temperature_virt_cxx("virtual temperature cxx", num_elems);

  sync_to_host(elements.buffers.temperature_virt, temperature_virt_cxx);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> temperature_virt_f90(
      "virtual temperature f90");

  for (int ie = 0; ie < num_elems; ++ie) {
    caar_compute_temperature_no_tracers_c_int(
        Kokkos::subview(test_functor.temperature, ie, test_functor.n0,
                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(),
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
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.compute_temperature_tracers_helper(kv);
  }
};

//is it meant moist?
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
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  TestType test_functor(elements);
  sync_to_host(elements.m_qdp, test_functor.qdp);
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
        Kokkos::subview(test_functor.qdp, ie, test_functor.qn0, 0, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL).data(),
        Kokkos::subview(test_functor.dp3d, ie, test_functor.n0, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL).data(),
        Kokkos::subview(test_functor.temperature, ie, test_functor.n0,
                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(),
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
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.compute_omega_p(kv);
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
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

//which omega_p should survive? buffers or m_omega_p?
//sort it out
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> source_omega_p(
      "source omega p", num_elems);
  genRandArray(source_omega_p, engine,
               std::uniform_real_distribution<Real>(0.0125, 1.0));
  sync_to_device(source_omega_p, elements.buffers.omega_p);

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> omega_p_f90("omega p f90",
                                                                 num_elems);

  TestType test_functor(elements);
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
                               Kokkos::subview(source_omega_p, ie, Kokkos::ALL,
                                               Kokkos::ALL, Kokkos::ALL).data(),
                               Kokkos::subview(omega_p_f90, ie, Kokkos::ALL,
                                               Kokkos::ALL,
                                               Kokkos::ALL).data());
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

class eta_dot_dpdn_vertadv_euler_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.compute_eta_dot_dpdn_vertadv_euler(kv);
  }
};


// computing eta_dot_dpdn: (eta_dot, divdp, sdot_sum) --> (eta_dot, sdot_sum)
// affects only buf value of eta, sdot
TEST_CASE("eta_dot_dpdn", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 0.0;
  constexpr const int num_elems = 10;
  std::random_device rd;
  rngAlg engine(rd());
  using TestType = compute_subfunctor_test<eta_dot_dpdn_vertadv_euler_test>;
  // This must be a reference to ensure the views are initialized in the
  // singleton
  // on host first
  Elements &elements = Context::singleton().get_elements();
  //element fields (except buffers) are randomly init-ed
  // will copy to device and randotm on device
  elements.random_init(num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("host div_dp", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV+1][NP][NP]> eta_dot("host div_dp", num_elems);
  HostViewManaged<Real * [NP][NP]> sdot_sum("host sdot_sum", num_elems);
  //random init host views
  genRandArray(div_vdp, engine, std::uniform_real_distribution<Real>(-10.0, 10.0));
  genRandArray(eta_dot, engine, std::uniform_real_distribution<Real>(-10.0, 10.0));
  genRandArray(sdot_sum, engine, std::uniform_real_distribution<Real>(-10.0, 10.0));
  //push host views to device
  sync_to_device(div_vdp, elements.buffers.div_vdp);
  sync_to_device(eta_dot, elements.buffers.eta_dot_dpdn_buf);
//source, dest
  sync_to_device(sdot_sum, elements.buffers.sdot_sum);
  //define host view for F input/output
//  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> divdp_f90("divdp f90", num_elems);
//  HostViewManaged<Real * [NP][NP]> sdot_sum_f90("sdot_sum f90", num_elems);
  //copy random host C views to F views
  //??? transposed or not?
//  deep_copy(divdp_f90, div_vdp);
//  deep_copy(sdot_sum_f90, sdot_sum);
//  why not copy from elements?
//only hybi is used, should the rest be quiet_nans? yes
  ExecViewManaged<Real[NUM_LEV_P]>::HostMirror hybrid_am_mirror("hybrid_am_host");
  ExecViewManaged<Real[NUM_LEV_P+1]>::HostMirror hybrid_ai_mirror("hybrid_ai_host");
  ExecViewManaged<Real[NUM_LEV_P]>::HostMirror hybrid_bm_mirror("hybrid_bm_host");
  ExecViewManaged<Real[NUM_LEV_P+1]>::HostMirror hybrid_bi_mirror("hybrid_bi_host");
  genRandArray(hybrid_am_mirror, engine, std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_ai_mirror, engine, std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_bm_mirror, engine, std::uniform_real_distribution<Real>(0.0125, 10.0));
  genRandArray(hybrid_bi_mirror, engine, std::uniform_real_distribution<Real>(0.0125, 10.0));


//we need these copies because vals test_functor will be modified
//now set up F output and make sure it has the same vals for _vadv before we run
////test_functor.
  HostViewManaged<Real * [NUM_PHYSICAL_LEV+1][NP][NP]> eta_dot_f90("eta_dot f90", num_elems);
  HostViewManaged<Real * [NP][NP]> sdot_sum_f90("tavd f90", num_elems);

//to from
  deep_copy(eta_dot_f90, eta_dot);
  deep_copy(sdot_sum_f90, sdot_sum);

//  const int rsplit = 0;
  TestType test_functor(elements);
  const int rsplit = 0;
  test_functor.set_rsplit(rsplit);

  test_functor.functor.m_data.init(1, num_elems, num_elems, TestType::nm1,
       TestType::n0, TestType::np1, TestType::qn0, TestType::dt, TestType::ps0, false,
       TestType::eta_ave_w, test_functor.return_rsplit(), 
       hybrid_am_mirror.data(), hybrid_ai_mirror.data(),
       hybrid_bm_mirror.data(), hybrid_bi_mirror.data());

// m value
//  HostViewManaged<Real * [NUM_PHYSICAL_LEV+1][NP][NP]> eta_dot_dpdn_f90("etadotdpdn f90", num_elems);
//

//we will always set etadot=0 before calling vertadv, but this makes it a better test
// m value
//  deep_copy(eta_dot_dpdn_f90, test_functor.eta_dpdn);
//


  //RUN subfunctor, why does it run on device?
  //will run on device
  test_functor.run_functor();
  //get C output from device to host to some var to store result
  //why does it need copying? is there any other mechanism to copy elements views 
  //than having external host views?

//in elements, m_eta_dot_dpdn should have a 'tail' of quiet nans,
//not sure where to init them.

//UNCOMMENT this later
  //source dest , copy C output to test_functor
  //

//m values only
//  sync_to_host(elements.m_eta_dot_dpdn, test_functor.eta_dpdn);
//
//
//

//save these in host values for comparison
  sync_to_host(elements.buffers.eta_dot_dpdn_buf, eta_dot);
  sync_to_host(elements.buffers.sdot_sum, sdot_sum);
  //if one of results is buffer variable, it needs another copy of host view,
  //see virt_temp for an example


//FORGOT TO CHECK SDOT!
  for (int ie = 0; ie < num_elems; ++ie) {
// input is eta, sdot, divdp, hybi
    caar_compute_eta_dot_dpdn_vertadv_euler_c_int(
                               Kokkos::subview(eta_dot_f90, ie, Kokkos::ALL,
                                               Kokkos::ALL, Kokkos::ALL).data(),
                               Kokkos::subview(sdot_sum_f90, ie, Kokkos::ALL,
                                               Kokkos::ALL).data(),
                               Kokkos::subview(div_vdp, ie, Kokkos::ALL,
                                               Kokkos::ALL, Kokkos::ALL).data(),
                               hybrid_bi_mirror.data());


//should be checking sdot_sum too!!!!!
    for (int level = 0; level < NUM_PHYSICAL_LEV+1; ++level) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          //compare eta
          const Real correct = eta_dot_f90(ie, level, igp, jgp);
          REQUIRE(!std::isnan(correct));
          const Real computed = eta_dot(ie, level, igp, jgp);
          REQUIRE(!std::isnan(computed));
//if( igp == 0 && jgp == 0){
//std::cout <<"ie="<< ie << " level="<< level <<" igp="<< igp << " jgp=" << jgp << "\n";
//std::cout << " F = " << correct << " C = " << computed << "\n";
//}
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
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.preq_vertadv(kv);
  }
};




//DONE
//og: it would be good if each test that uses buffers init-ed vars for buffers 
//the same way, but i don't thnk it is the case. or maybe it is?
//all emelents m_vars have copies in the test_functor, to move it to host?
//some buffer vars for C and F tests are inited as Real*, some are inited as Scalar*
//in the test body (even those with midlevels values only). why?
//
//
//preq_vertadv: (T, eta_dot, v, 1/dp3d) --> t_vadv, v_vadv
//takes in only buf value of eta, m values T, v, dp3d
TEST_CASE("preq_vertadv", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 0.0;
  constexpr const int num_elems = 10;
  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<preq_vertadv_test>;
  // This must be a reference to ensure the views are initialized in the
  // singleton
  // on host first
  Elements &elements = Context::singleton().get_elements();
  elements.random_init(num_elems);

  //preq_vertadv depends on eta_dot_dpdn, dp3d, T, v (all in elements, randomized by random_init),
  //modifies v_vadv, t_vadv (those are in buffers). 
  //We will assign nans to the buffer values:
  HostViewManaged<Real * [NUM_PHYSICAL_LEV+1][NP][NP]> eta_dot("host t_vadv", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> t_vadv("host t_vadv", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> v_vadv("host v_vadv", num_elems);

//we need quiet nans, this is just to debug the test itself
//  genRandArray(t_vadv, engine, std::uniform_real_distribution<Real>(-100, 100));
//  genRandArray(v_vadv, engine, std::uniform_real_distribution<Real>(-100, 100));
  genRandArray(eta_dot, engine, std::uniform_real_distribution<Real>(-100, 100));

  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      for (int i = 0; i < NP; i++) {
        for (int j = 0; j < NP; j++) {
          t_vadv(ie,level,i,j) = std::numeric_limits<Real>::quiet_NaN();
          v_vadv(ie,level,0,i,j) = std::numeric_limits<Real>::quiet_NaN();
          v_vadv(ie,level,1,i,j) = std::numeric_limits<Real>::quiet_NaN();
        }
      }
    } //level loop
  }//ie loop, end of assigning of quiet nans

//do we have this function for nlev+1 levels?
  sync_to_device(eta_dot, elements.buffers.eta_dot_dpdn_buf);
  sync_to_device(t_vadv, elements.buffers.t_vadv_buf);
  sync_to_device(v_vadv, elements.buffers.v_vadv_buf);

//now set up F output and make sure it has the same vals for _vadv before we run
//test_functor.
  HostViewManaged<Real * [NUM_PHYSICAL_LEV+1][NP][NP]> eta_dot_f90("eta_dot f90", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> t_vadv_f90("tavd f90", num_elems);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> v_vadv_f90("vavd f90", num_elems);
  HostViewManaged<Real [NUM_PHYSICAL_LEV][NP][NP]> rdp_f90("rdp f90", num_elems);

//to from
  deep_copy(eta_dot_f90, eta_dot);
  deep_copy(t_vadv_f90, t_vadv);
  deep_copy(v_vadv_f90, v_vadv);

  TestType test_functor(elements);
  //this test does not need m_data init?
  //this test will change t_vadv_buf, v_vadv_buf, eta_dot_dpdn_buf
  test_functor.run_functor();

  //now copy buffer vals back to test values
  sync_to_host(elements.buffers.eta_dot_dpdn_buf, eta_dot);
  sync_to_host(elements.buffers.t_vadv_buf, t_vadv);
  sync_to_host(elements.buffers.v_vadv_buf, v_vadv);

//std::cout << "in C NUM_PHYS_LEV="<< NUM_PHYSICAL_LEV << "\n";
//std::cout << "NUM_LEV, VECTOR_SIZE="<<NUM_LEV << ", " << VECTOR_SIZE << "\n";

  for (int ie = 0; ie < num_elems; ++ie) {
//         call preq_vertadv(elem(ie)%state%T(:,:,:,n0),elem(ie)%state%v(:,:,:,:,n0), &
//                       eta_dot_dpdn,rdp,T_vadv,v_vadv)
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        for (int i = 0; i < NP; i++) {
          for (int j = 0; j < NP; j++) {
            rdp_f90(level, i, j) = 1/test_functor.dp3d(ie, test_functor.n0, level, i, j);
          }
        }
    }//level loop
    preq_vertadv(
        Kokkos::subview(test_functor.temperature, ie, test_functor.n0, 
                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(),
        Kokkos::subview(test_functor.velocity, ie, test_functor.n0, 
                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(),
//        Kokkos::subview(test_functor.eta_dpdn, ie, 
//                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(), //if test takes m_value
        Kokkos::subview(eta_dot_f90, ie, 
                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(),
        rdp_f90.data(),
        Kokkos::subview(t_vadv_f90, ie, 
                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(),
        Kokkos::subview(v_vadv_f90, ie, 
                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data()
    );//preq vertadv call

    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          //errors for t_vadv
          const Real correct = t_vadv_f90(ie, level, igp, jgp);
          REQUIRE(!std::isnan(correct));
          const Real computed = t_vadv(ie, level, igp, jgp);

//if( igp == 0 && jgp == 0){
//std::cout <<"ie="<< ie << " level="<< level <<" igp="<< igp << " jgp=" << jgp << "\n";
//std::cout << " F = " << correct << " C = " << computed << "\n";
//}

          REQUIRE(!std::isnan(computed));
          const Real rel_error = compare_answers(correct, computed);
          REQUIRE(rel_threshold >= rel_error);


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

