#include <catch/catch.hpp>

#include <limits>
#include <random>
#include <type_traits>

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

using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {
void caar_compute_energy_grad_c_int(const Real *const &dvv,
                                    const Real *const &Dinv,
                                    const Real *const &pecnd,
                                    const Real *const &phi,
                                    const Real *const &velocity,
                                    Real *const &vtemp); //(&vtemp)[2][NP][NP]);

void preq_omega_ps_c_int(Real *omega_p, const Real *velocity,
                         const Real *pressure, const Real *div_vdp,
                         const Real *dinv, const Real *dvv);

void preq_hydrostatic_c_int(Real *phi, const Real *phis,
                            const Real *virt_temperature, const Real *pressure,
                            const Real *delta_pressure);

void caar_compute_dp3d_np1_c_int(int np1, int nm1, const Real &dt2,
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

}  // extern C

Real compare_answers(Real target, Real computed, Real relative_coeff = 1.0) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }
  return std::fabs(target - computed) / denom;
} // end of definition of compare_answers()

void genRandArray(Real *arr, int arr_len, rngAlg &engine,
                  std::uniform_real_distribution<Real> pdf) {
  for (int i = 0; i < arr_len; ++i) {
    arr[i] = pdf(engine);
  }
} // end of definition of genRandArray()

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
                        qn0, ps0, dt2, false, eta_ave_w, hybrid_a);

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
  static constexpr Real dt2 = 1.0;
  static constexpr Real eta_ave_w = 1.0;
};

// Templates to verify at compile time that a view has the specified array type
template <typename ViewT, typename ArrayT> struct exec_view_mappable {
  using exec_view = ExecViewUnmanaged<ArrayT>;
  static constexpr bool value = Kokkos::Impl::ViewMapping<
      typename ViewT::traits, typename exec_view::traits, void>::is_assignable;
};

template <typename ViewT, typename ArrayT> struct host_view_mappable {
  using host_view = HostViewUnmanaged<ArrayT>;
  static constexpr bool value = Kokkos::Impl::ViewMapping<
      typename ViewT::traits, typename host_view::traits, void>::is_assignable;
};

// Kokkos views cannot be used to determine which overloaded function to call,
// so implement this check ourselves with enable_if.
// Despite the ugly templates, this provides much better error messages
// TODO: Move these to Utility.hpp
template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T, Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::
        value &&host_view_mappable<
            Dest_T, Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>::value,
    void>::type
sync_to_host(Source_T source, Dest_T dest) {
  ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror
  source_mirror(Kokkos::create_mirror_view(source));
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int time = 0; time < NUM_TIME_LEVELS; ++time) {
      for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
           ++vector_level) {
        for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              dest(ie, time, level, igp, jgp) =
                  source_mirror(ie, time, igp, jgp, vector_level)[vector];
            }
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T, Scalar * [NP][NP][NUM_LEV]>::value &&
        host_view_mappable<Dest_T, Real * [NUM_PHYSICAL_LEV][NP][NP]>::value,
    void>::type
sync_to_host(Source_T source, Dest_T dest) {
  ExecViewUnmanaged<Scalar * [NP][NP][NUM_LEV]>::HostMirror source_mirror(
      Kokkos::create_mirror_view(source));
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
         ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest(ie, level, igp, jgp) =
                source_mirror(ie, igp, jgp, vector_level)[vector];
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T,
                       Scalar * [Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>::
        value &&host_view_mappable<
            Dest_T, Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP]
                                              [NP]>::value,
    void>::type
sync_to_host(Source_T source, Dest_T dest) {
  typename Source_T::HostMirror source_mirror(
      Kokkos::create_mirror_view(source));
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int time = 0; time < Q_NUM_TIME_LEVELS; ++time) {
      for (int tracer = 0; tracer < QSIZE_D; ++tracer) {
        for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
             ++vector_level) {
          for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
            for (int igp = 0; igp < NP; ++igp) {
              for (int jgp = 0; jgp < NP; ++jgp) {
                dest(ie, time, tracer, level, igp, jgp) = source_mirror(
                    ie, time, tracer, igp, jgp, vector_level)[vector];
              }
            }
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T, Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::
        value &&host_view_mappable<
            Dest_T, Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>::value,
    void>::type
sync_to_host(Source_T source_1, Source_T source_2, Dest_T dest) {
  ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror
  source_1_mirror(Kokkos::create_mirror_view(source_1)),
      source_2_mirror(Kokkos::create_mirror_view(source_2));
  Kokkos::deep_copy(source_1_mirror, source_1);
  Kokkos::deep_copy(source_2_mirror, source_2);
  for (int ie = 0; ie < source_1.extent_int(0); ++ie) {
    for (int time = 0; time < NUM_TIME_LEVELS; ++time) {
      for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
           ++vector_level) {
        for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              dest(ie, time, level, 0, igp, jgp) =
                  source_1_mirror(ie, time, igp, jgp, vector_level)[vector];
              dest(ie, time, level, 1, igp, jgp) =
                  source_2_mirror(ie, time, igp, jgp, vector_level)[vector];
            }
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T, Scalar * [NP][NP][NUM_LEV]>::value &&
        host_view_mappable<Dest_T, Real * [NUM_PHYSICAL_LEV][2][NP][NP]>::value,
    void>::type
sync_to_host(Source_T source_1, Source_T source_2, Dest_T dest) {
  ExecViewUnmanaged<Scalar * [NP][NP][NUM_LEV]>::HostMirror source_1_mirror(
      Kokkos::create_mirror_view(source_1)),
      source_2_mirror(Kokkos::create_mirror_view(source_2));
  Kokkos::deep_copy(source_1_mirror, source_1);
  Kokkos::deep_copy(source_2_mirror, source_2);
  for (int ie = 0; ie < source_1.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
         ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest(ie, level, 0, igp, jgp) =
                source_1_mirror(ie, igp, jgp, vector_level)[vector];
            dest(ie, level, 1, igp, jgp) =
                source_2_mirror(ie, igp, jgp, vector_level)[vector];
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    host_view_mappable<Source_T, Real * [NUM_PHYSICAL_LEV][NP][NP]>::value &&
        exec_view_mappable<Dest_T, Scalar * [NP][NP][NUM_LEV]>::value,
    void>::type
sync_to_device(Source_T source, Dest_T dest) {
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
         ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
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
      std::numeric_limits<Real>::epsilon() * 32768.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are
  // initialized in the singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  compute_subfunctor_test<compute_energy_grad_test> test_functor(elements);
  test_functor.run_functor();
  HostViewManaged<Scalar * [2][NP][NP][NUM_LEV]> energy_grad("energy_grad",
                                                             num_elems);
  Kokkos::deep_copy(energy_grad, elements.buffers.energy_grad);

  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_LEV; ++level) {
      for (int v = 0; v < VECTOR_SIZE; ++v) {
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
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
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

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> pressure("host pressure",
                                                              num_elems);
  genRandArray(reinterpret_cast<Real *>(pressure.data()), pressure.span(),
               engine, std::uniform_real_distribution<Real>(0, 100.0));
  sync_to_device(pressure, elements.buffers.pressure);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("host div_vdp",
                                                             num_elems);
  genRandArray(reinterpret_cast<Real *>(div_vdp.data()), div_vdp.span(), engine,
               std::uniform_real_distribution<Real>(0, 100.0));
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
  CaarElements &elements = get_elements();
  elements.random_init(num_elems, engine);

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> temperature_virt(
      "host virtual temperature", num_elems);
  genRandArray(reinterpret_cast<Real *>(temperature_virt.data()),
               temperature_virt.span(), engine,
               std::uniform_real_distribution<Real>(0.0125, 1.0));
  sync_to_device(temperature_virt, elements.buffers.temperature_virt);
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> pressure("host pressure",
                                                              num_elems);
  genRandArray(reinterpret_cast<Real *>(pressure.data()), pressure.span(),
               engine, std::uniform_real_distribution<Real>(0.0125, 1.0));
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

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> div_vdp("host div_vdp",
                                                             num_elems);
  genRandArray(reinterpret_cast<Real *>(div_vdp.data()), div_vdp.span(), engine,
               std::uniform_real_distribution<Real>(0, 100.0));
  sync_to_device(div_vdp, elements.buffers.div_vdp);

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
        test_functor.functor.m_data.dt2,
        Kokkos::subview(test_functor.spheremp, ie, Kokkos::ALL, Kokkos::ALL)
            .data(),
        Kokkos::subview(div_vdp, ie, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL)
            .data(),
        Kokkos::subview(test_functor.eta_dpdn, ie, Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL).data(),
        Kokkos::subview(dp3d_f90, ie, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL).data());
    for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          REQUIRE(dp3d_f90(ie, test_functor.functor.m_data.nm1, k, igp, jgp) ==
                  test_functor.dp3d(ie, test_functor.functor.m_data.nm1, k, igp,
                                    jgp));
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

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> vn0_f90(
      "vn0 f90 results", num_elems);
  sync_to_host(elements.m_derived_un0, elements.m_derived_vn0, vn0_f90);

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

  ExecViewManaged<Real[NUM_LEV_P]>::HostMirror hybrid_a_mirror("hybrid_a_host");
  genRandArray(reinterpret_cast<Real *>(hybrid_a_mirror.data()),
               hybrid_a_mirror.span(), engine,
               std::uniform_real_distribution<Real>(0.0125, 1.0));
  test_functor.functor.m_data.init(1, num_elems, num_elems, TestType::nm1,
                                   TestType::n0, TestType::np1, TestType::qn0,
                                   TestType::dt2, TestType::ps0, false,
                                   TestType::eta_ave_w, hybrid_a_mirror.data());

  test_functor.run_functor();

  HostViewManaged<Scalar * [NP][NP][NUM_LEV]> pressure_cxx("pressure_cxx",
                                                           num_elems);
  Kokkos::deep_copy(pressure_cxx, elements.buffers.pressure);

  HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]> pressure_f90("pressure_f90");

  sync_to_host(elements.m_dp3d, test_functor.dp3d);

  for (int ie = 0; ie < num_elems; ++ie) {
    caar_compute_pressure_c_int(
        hybrid_a_mirror(0), test_functor.functor.m_data.ps0,
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

class temperature_no_tracers_test {
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

TEST_CASE("temperature no tracers", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 1.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<temperature_no_tracers_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);

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

class temperature_with_tracers_test {
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

TEST_CASE("temperature with tracers", "monolithic compute_and_apply_rhs") {
  constexpr const Real rel_threshold =
      std::numeric_limits<Real>::epsilon() * 4.0;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  using TestType = compute_subfunctor_test<temperature_with_tracers_test>;

  // This must be a reference to ensure the views are initialized in the
  // singleton
  Elements &elements = get_elements();
  elements.random_init(num_elems, engine);

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
  CaarRegion &region = get_region();
  region.random_init(num_elems, engine);

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> source_omega_p(
      "source omega p", num_elems);
  genRandArray(reinterpret_cast<Real *>(source_omega_p.data()),
               source_omega_p.span(), engine,
               std::uniform_real_distribution<Real>(0.0125, 1.0));
  sync_to_device(source_omega_p, region.buffers.omega_p);

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> omega_p_f90("omega p f90",
                                                                 num_elems);

  TestType test_functor(region);
  sync_to_host(region.m_omega_p, omega_p_f90);
  test_functor.run_functor();
  sync_to_host(region.m_omega_p, test_functor.omega_p);

  ExecViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]>::HostMirror
  temperature_virt_cxx("virtual temperature cxx", num_elems);

  sync_to_host(region.buffers.temperature_virt, temperature_virt_cxx);

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
