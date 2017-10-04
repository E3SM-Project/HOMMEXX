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

void preq_omega_ps_c_int(Real *omega_p,
                         const Real *velocity,
                         const Real *pressure,
                         const Real *div_vdp,
                         const Real *dinv,
                         const Real *dvv);

void caar_compute_dp3d_np1_c_int(const int np1, const int nm1,
                                 const Real &dt2,
                                 const Real *spheremp,
                                 const Real *divdp,
                                 const Real *eta_dot_dpdn,
                                 Real *dp3d);
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
  compute_subfunctor_test(Elements &elements)
      : functor(),
        velocity("Velocity", elements.num_elems()),
        temperature("Temperature", elements.num_elems()),
        dp3d("DP3D", elements.num_elems()),
        phi("Phi", elements.num_elems()),
        pecnd("PE_CND", elements.num_elems()),
        omega_p("Omega_P", elements.num_elems()),
        derived_v("Derived V?", elements.num_elems()),
        eta_dpdn("Eta dot dp/deta", elements.num_elems()),
        qdp("QDP", elements.num_elems()),
        dinv("DInv", elements.num_elems()),
        spheremp("SphereMP", elements.num_elems()),
        dvv("dvv"),
        nets(1),
        nete(elements.num_elems()) {
    Real hybrid_a[NUM_LEV_P] = {0};
    functor.m_data.init(0, region.num_elems(), region.num_elems(),
                        nm1, n0, np1, qn0, ps0, dt2, false,
                        eta_ave_w, hybrid_a);

    get_derivative().dvv(dvv.data());

    elements.push_to_f90_pointers(
        velocity.data(), temperature.data(), dp3d.data(),
        phi.data(), pecnd.data(), omega_p.data(),
        derived_v.data(), eta_dpdn.data(), qdp.data());

    Kokkos::deep_copy(spheremp, region.m_spheremp);

    for(int ie = 0; ie < region.num_elems(); ++ie) {
      elements.dinv(
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
  static constexpr int qn0 = -1;
  static constexpr int ps0 = 1;
  static constexpr Real dt2 = 1.0;
  static constexpr Real eta_ave_w = 1.0;
};

void sync_to_host(ExecViewUnmanaged<Scalar *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]> source,
                  HostViewUnmanaged<Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]> dest) {
  ExecViewUnmanaged<Scalar *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror
    source_mirror(Kokkos::create_mirror_view(source));
  Kokkos::deep_copy(source_mirror, source);
  for(int ie = 0; ie < source.extent_int(0); ++ie) {
    for(int time = 0; time < NUM_TIME_LEVELS; ++time) {
      for(int vector_level = 0, level = 0; vector_level < NUM_LEV; ++vector_level) {
        for(int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
          for(int igp = 0; igp < NP; ++igp) {
            for(int jgp = 0; jgp < NP; ++jgp) {
              dest(ie, time, level, igp, jgp) =
                source_mirror(ie, time, igp, jgp, vector_level)[vector];
            }
          }
        }
      }
    }
  }
}

void sync_to_device(HostViewUnmanaged<Real *[NUM_PHYSICAL_LEV][NP][NP]> source,
                    ExecViewUnmanaged<Scalar *[NP][NP][NUM_LEV]> dest) {
  ExecViewUnmanaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror dest_mirror =
      Kokkos::create_mirror_view(dest);
  for(int ie = 0; ie < source.extent_int(0); ++ie) {
    for(int vector_level = 0, level = 0; vector_level < NUM_LEV; ++vector_level) {
      for(int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            dest_mirror(ie, igp, jgp, vector_level)[vector] =
              source(ie, level, igp, jgp);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}

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

TEST_CASE("compute_energy_grad", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 128.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are
  // initialized in the singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  compute_subfunctor_test<compute_energy_grad_test>
      test_functor(region);
  test_functor.run_functor();
  HostViewManaged<Scalar * [2][NP][NP][NUM_LEV]>
      energy_grad("energy_grad", num_elems);
  Kokkos::deep_copy(energy_grad,
                    elements.buffers.energy_grad);

  for(int ie = 0; ie < num_elems; ++ie) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real vtemp[2][NP][NP];
        for (int h = 0; h < 2; h++) {
          for (int i = 0; i < NP; i++) {
            for (int j = 0; j < NP; j++) {
              vtemp[h][i][j] = std::numeric_limits<Real>::quiet_NaN();
            }
          }
        }
        caar_compute_energy_grad_c_int(
            test_functor.dvv.data(),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.dinv, ie, Kokkos::ALL, Kokkos::ALL,
                                Kokkos::ALL, Kokkos::ALL).data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.pecnd, ie, level * VECTOR_SIZE + v,
                                Kokkos::ALL, Kokkos::ALL).data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.phi, ie, level * VECTOR_SIZE + v,
                                Kokkos::ALL, Kokkos::ALL).data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.velocity, ie, test_functor.n0,
                                level * VECTOR_SIZE + v, Kokkos::ALL,
                                Kokkos::ALL, Kokkos::ALL).data()),
            &vtemp[0][0][0]);
        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            REQUIRE(!std::isnan(vtemp[0][igp][jgp]));
            REQUIRE(!std::isnan(vtemp[1][igp][jgp]));
            REQUIRE(!std::isnan(energy_grad(ie, 0, igp, jgp, level)[v]));
            REQUIRE(!std::isnan(energy_grad(ie, 1, igp, jgp, level)[v]));
            Real rel_error = compare_answers(
                vtemp[0][igp][jgp], energy_grad(ie, 0, igp, jgp, level)[v]);
            REQUIRE(rel_threshold >= rel_error);

            rel_error = compare_answers(vtemp[1][igp][jgp],
                                        energy_grad(ie, 1, igp, jgp, level)[v]);
            REQUIRE(rel_threshold >= rel_error);
          }
        }
      }
    }
  }
}  // end of TEST_CASE(...,"compute_energy_grad")

class preq_omega_ps_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    functor.preq_omega_ps(kv);
  }
};

TEST_CASE("preq_omega_ps", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 128.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are initialized in the
  // singleton
  CaarRegion &region = get_region();
  region.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> pressure("host pressure", num_elems);
  genRandArray(reinterpret_cast<Real *>(pressure.data()),
               pressure.span(), engine,
               std::uniform_real_distribution<Real>(0, 100.0));
  sync_to_device(pressure, region.buffers.pressure);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("host div_vdp", num_elems);
  genRandArray(reinterpret_cast<Real *>(div_vdp.data()),
               div_vdp.span(), engine,
               std::uniform_real_distribution<Real>(0, 100.0));
  sync_to_device(div_vdp, region.buffers.div_vdp);

  compute_subfunctor_test<preq_omega_ps_test> test_functor(region);
  test_functor.run_functor();
  // Results of the computation
  HostViewManaged<Scalar * [NP][NP][NUM_LEV]> omega_p("omega_p", num_elems);
  Kokkos::deep_copy(omega_p, region.buffers.omega_p);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> omega_p_f90(
      "Fortran omega_p");
  for (int ie = 0; ie < num_elems; ++ie) {
    preq_omega_ps_c_int(omega_p_f90.data(),
                        Kokkos::subview(pressure, ie, Kokkos::ALL,
                                        Kokkos::ALL, Kokkos::ALL).data(),
                        Kokkos::subview(test_functor.velocity, ie, test_functor.n0,
                                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                        Kokkos::ALL).data(),
                        Kokkos::subview(div_vdp, ie, Kokkos::ALL,
                                        Kokkos::ALL, Kokkos::ALL).data(),
                        Kokkos::subview(test_functor.dinv, ie, Kokkos::ALL,
                                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL).data(),
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

class dp3d_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor, KernelVariables &kv) {
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
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
  CaarRegion &region = get_region();
  region.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("host div_vdp", num_elems);
  genRandArray(reinterpret_cast<Real *>(div_vdp.data()),
               div_vdp.span(), engine,
               std::uniform_real_distribution<Real>(0, 100.0));
  sync_to_device(div_vdp, region.buffers.div_vdp);

  compute_subfunctor_test<dp3d_test> test_functor(region);

  // To ensure the Fortran doesn't pass without doing anything,
  // copy the initial state before running any of the test
  HostViewManaged<Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]> dp3d_f90("dp3d fortran", num_elems);
  Kokkos::deep_copy(dp3d_f90, test_functor.dp3d);

  test_functor.run_functor();
  sync_to_host(region.m_dp3d, test_functor.dp3d);

  for(int ie = 0; ie < num_elems; ++ie) {
    caar_compute_dp3d_np1_c_int(test_functor.np1_f90,
                                test_functor.nm1_f90,
                                test_functor.functor.m_data.dt2,
                                Kokkos::subview(test_functor.spheremp, ie,
                                                Kokkos::ALL, Kokkos::ALL).data(),
                                Kokkos::subview(div_vdp, ie, Kokkos::ALL,
                                                Kokkos::ALL, Kokkos::ALL).data(),
                                Kokkos::subview(test_functor.eta_dpdn, ie,
                                                Kokkos::ALL, Kokkos::ALL,
                                                Kokkos::ALL).data(),
                                Kokkos::subview(dp3d_f90, ie, Kokkos::ALL,
                                                Kokkos::ALL, Kokkos::ALL,
                                                Kokkos::ALL).data());
    for(int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
      for(int igp = 0; igp < NP; ++igp) {
        for(int jgp = 0; jgp < NP; ++jgp) {
          REQUIRE(dp3d_f90(ie, test_functor.functor.m_data.nm1, k, igp, jgp) ==
                  test_functor.dp3d(ie, test_functor.functor.m_data.nm1, k, igp, jgp));
          Real correct = dp3d_f90(ie, test_functor.functor.m_data.np1, k, igp, jgp);
          REQUIRE(!std::isnan(correct));
          Real computed = test_functor.dp3d(ie, test_functor.functor.m_data.np1,
                                            k, igp, jgp);
          REQUIRE(!std::isnan(computed));
          Real rel_error = compare_answers(correct,
                                           computed);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}
