#include <catch/catch.hpp>

#include <random>
#include <iostream>

#include "Types.hpp"
#include "Derivative.hpp"
#include "SphereOperators.hpp"
#include "KernelVariables.hpp"
#include "Utility.hpp"

using namespace Homme;

extern "C" {

void init_deriv_f90(Real *dvv);

void gradient_sphere_c_callable(const Real *scalar, const Real *dvv,
                                const Real *DInv, Real *vector);

void divergence_sphere_c_callable(const Real *vector, const Real *dvv,
                                  const Real *metdet, const Real *DInv,
                                  Real *scalar);

void vorticity_sphere_c_callable(const Real *vector, const Real *dvv,
                                 const Real *metdet, const Real *D,
                                 Real *scalar);

} // extern "C"

// ====================== RANDOM INITIALIZATION ====================== //

TEST_CASE("dp3d_intervals", "Testing Elements::random_init") {
  constexpr int num_elems = 5;
  constexpr Real max_pressure = 32.0;
  constexpr Real rel_threshold = 128.0 * std::numeric_limits<Real>::epsilon();
  Elements elements;
  elements.random_init(num_elems, max_pressure);
  HostViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp3d("host dp3d",
                                                                    num_elems);
  Kokkos::deep_copy(dp3d, elements.m_dp3d);
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          HostViewUnmanaged<Scalar[NUM_LEV]> dp3d_col =
              Homme::subview(dp3d, ie, tl, igp, jgp);
          Real sum = 0.0;
          for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
            const int ilev = level / VECTOR_SIZE;
            const int vec_lev = level % VECTOR_SIZE;
            sum += dp3d_col(ilev)[vec_lev];
            REQUIRE(dp3d_col(ilev)[vec_lev] > 0.0);
          }
          Real rel_error = compare_answers(max_pressure, sum);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}

TEST_CASE("d_dinv_check", "Testing Elements::random_init") {
  constexpr int num_elems = 5;
  constexpr Real rel_threshold = 1e-10;
  Elements elements;
  elements.random_init(num_elems);
  HostViewManaged<Real * [2][2][NP][NP]> d("host d", num_elems);
  HostViewManaged<Real * [2][2][NP][NP]> dinv("host dinv", num_elems);
  Kokkos::deep_copy(d, elements.m_d);
  Kokkos::deep_copy(dinv, elements.m_dinv);
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        const Real det_1 = d(ie, 0, 0, igp, jgp) * d(ie, 1, 1, igp, jgp) -
                           d(ie, 0, 1, igp, jgp) * d(ie, 1, 0, igp, jgp);
        REQUIRE(det_1 > 0.0);
        const Real det_2 = dinv(ie, 0, 0, igp, jgp) * dinv(ie, 1, 1, igp, jgp) -
                           dinv(ie, 0, 1, igp, jgp) * dinv(ie, 1, 0, igp, jgp);
        REQUIRE(det_2 > 0.0);
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j) {
            Real pt_product = 0.0;
            for (int k = 0; k < 2; ++k) {
              pt_product += d(ie, i, k, igp, jgp) * dinv(ie, k, j, igp, jgp);
            }
            const Real expected = (i == j) ? 1.0 : 0.0;
            const Real rel_error = compare_answers(expected, pt_product);
            REQUIRE(rel_threshold >= rel_error);
          }
        }
      }
    }
  }
}

// ====================== SPHERE OPERATORS =========================== //

TEST_CASE("Multi_Level_Sphere_Operators",
          "Testing spherical differential operators") {
  // Short names
  using Kokkos::subview;
  using Kokkos::ALL;

  constexpr Real rel_threshold = std::numeric_limits<Real>::epsilon() * 1024.0;

  constexpr int nelems = 1;

  // Random numbers generators
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());
  std::uniform_real_distribution<Real> dreal(0.0125, 1);

  // Input host views
  HostViewManaged<Real * [2][2][NP][NP]> D_h("d_host", nelems);
  genRandArray(D_h, engine, dreal);
  ExecViewManaged<Real * [2][2][NP][NP]> D_exec("D_cxx_exec", nelems);
  Kokkos::deep_copy(D_exec, D_h);

  // Buffer View
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]> buffer("buffer_cxx", nelems);

  // Initialize derivative
  HostViewManaged<Real[NP][NP]> dvv_h("dvv_host");
  init_deriv_f90(dvv_h.data());
  ExecViewManaged<Real[NP][NP]> dvv_exec("dvv_exec");
  Kokkos::deep_copy(dvv_exec, dvv_h);

  // Execution policy
  auto policy = Homme::get_default_team_policy<ExecSpace>(nelems);

  SECTION("gradient sphere") {
    // Initialize input(s)
    HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> input_h("input host",
                                                               nelems);
    genRandArray(input_h, engine, dreal);
    ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> input_exec("input exec",
                                                           nelems);
    sync_to_device(input_h, input_exec);

    ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]> output_exec("output exec",
                                                               nelems);

    // Compute cxx
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamMember team_member) {
      KernelVariables kv(team_member);
      gradient_sphere(kv, D_exec, dvv_exec,
                      subview(input_exec, kv.ie, ALL, ALL, ALL), buffer,
                      subview(output_exec, kv.ie, ALL, ALL, ALL, ALL));
    });

    // Deep copy back to host
    HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> output_h(
        "output host", nelems);
    sync_to_host(output_exec, output_h);

    HostViewManaged<Real[2][NP][NP]> output_f90("output f90");
    for (int ie = 0; ie < nelems; ++ie) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        // Compute f90
        gradient_sphere_c_callable(subview(input_h, ie, level, ALL, ALL).data(),
                                   dvv_h.data(),
                                   subview(D_h, ie, ALL, ALL, ALL, ALL).data(),
                                   output_f90.data());

        // Check the answer
        for (int dim = 0; dim < 2; ++dim) {
          for (int j = 0; j < NP; ++j) {
            for (int i = 0; i < NP; ++i) {
              const Real correct = output_f90(dim, j, i);
              const Real computed = output_h(ie, level, dim, j, i);
              const Real rel_error = compare_answers(correct, computed);
              REQUIRE(rel_threshold >= rel_error);
            }
          }
        }
      }
    }
  }

  HostViewManaged<Real * [NP][NP]> metdet_h("metdet host", nelems);
  genRandArray(metdet_h, engine, dreal);
  ExecViewManaged<Real * [NP][NP]> metdet_exec("metdet exec", nelems);
  Kokkos::deep_copy(metdet_exec, metdet_h);

  SECTION("divergence sphere") {
    // Initialize input(s)
    HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> input_h("input host",
                                                                  nelems);
    genRandArray(input_h, engine, dreal);
    ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]> input_exec("input exec",
                                                              nelems);
    sync_to_device(input_h, input_exec);

    ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> output_exec("output exec",
                                                            nelems);

    // Compute cxx
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamMember team_member) {
      KernelVariables kv(team_member);
      divergence_sphere(kv, D_exec, metdet_exec,
                        dvv_exec,
                        subview(input_exec, kv.ie, ALL, ALL, ALL, ALL),
                        buffer, subview(output_exec, kv.ie, ALL, ALL, ALL));
    });

    // Deep copy back to host
    HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> output_h("output host",
                                                                nelems);
    sync_to_host(output_exec, output_h);

    HostViewManaged<Real[NP][NP]> output_f90("output f90");
    for (int ie = 0; ie < nelems; ++ie) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        // Compute f90
        divergence_sphere_c_callable(
            subview(input_h, ie, level, ALL, ALL, ALL).data(),
            dvv_h.data(),
            subview(metdet_h, ie, ALL, ALL).data(),
            subview(D_h, ie, ALL, ALL, ALL, ALL).data(), output_f90.data());

        // Check the answer
        for (int j = 0; j < NP; ++j) {
          for (int i = 0; i < NP; ++i) {
            const Real correct = output_f90(j, i);
            const Real computed = output_h(ie, level, j, i);
            const Real rel_error = compare_answers(correct, computed);
            REQUIRE(rel_threshold >= rel_error);
          }
        }
      }
    }
  }

  SECTION("vorticity sphere") {
    // Initialize input(s)
    HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> input_h("input host",
                                                                  nelems);
    genRandArray(input_h, engine, dreal);
    ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> input_1_exec("input 1 exec",
                                                             nelems);
    ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> input_2_exec("input 2 exec",
                                                             nelems);
    sync_to_device(input_h, input_1_exec, input_2_exec);

    ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> output_exec("output exec",
                                                            nelems);

    // Compute cxx
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamMember team_member) {
      KernelVariables kv(team_member);
      vorticity_sphere(kv, D_exec, metdet_exec,
                       dvv_exec,
                       subview(input_1_exec, kv.ie, ALL, ALL, ALL),
                       subview(input_2_exec, kv.ie, ALL, ALL, ALL), buffer,
                       subview(output_exec, kv.ie, ALL, ALL, ALL));
    });

    // Deep copy back to host
    HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> output_h("output host",
                                                                nelems);
    sync_to_host(output_exec, output_h);

    HostViewManaged<Real[NP][NP]> output_f90("output f90");
    for (int ie = 0; ie < nelems; ++ie) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        // Compute f90
        vorticity_sphere_c_callable(
            subview(input_h, ie, level, ALL, ALL, ALL).data(),
            dvv_h.data(),
            subview(metdet_h, ie, ALL, ALL).data(),
            subview(D_h, ie, ALL, ALL, ALL, ALL).data(), output_f90.data());

        // Check the answer
        for (int j = 0; j < NP; ++j) {
          for (int i = 0; i < NP; ++i) {
            const Real correct = output_f90(j, i);
            const Real computed = output_h(ie, level, j, i);
            const Real rel_error = compare_answers(correct, computed);
            REQUIRE(rel_threshold >= rel_error);
          }
        }
      }
    }
  }
}

TEST_CASE("Single_Level_Sphere_Operators",
          "Testing spherical differential operators") {
  // Short names
  using Kokkos::subview;
  using Kokkos::ALL;

  constexpr Real rel_threshold = std::numeric_limits<Real>::epsilon() * 32768.0;

  constexpr int nelems = 1;
  constexpr int num_rand_test = 10;

  // Fortran host views
  HostViewManaged<Real * [NP][NP]> scalar_h("scalar_host", nelems);
  HostViewManaged<Real * [2][NP][NP]> vector_h("vector_host", nelems);
  HostViewManaged<Real   [NP][NP]> dvv_h("dvv_host");
  HostViewManaged<Real * [2][2][NP][NP]> D_h("d_host", nelems);
  HostViewManaged<Real * [2][2][NP][NP]> DInv_h("dinv_host", nelems);
  HostViewManaged<Real * [NP][NP]> metdet_h("metdet_host", nelems);

  // Input exec views
  ExecViewManaged<Real * [NP][NP]> scalar_exec("scalar_cxx_exec", nelems);
  ExecViewManaged<Real * [2][NP][NP]> vector_exec("vector_cxx_exec", nelems);
  ExecViewManaged<Real * [2][2][NP][NP]> DInv_exec("DInv_cxx_exec", nelems);
  ExecViewManaged<Real * [2][2][NP][NP]> D_exec("D_cxx_exec", nelems);
  ExecViewManaged<Real * [NP][NP]> metdet_exec("metdet_cxx_exec", nelems);
  ExecViewManaged<Real * [2][NP][NP]> tmp_exec("tmp_cxx", nelems);
  ExecViewManaged<Real[2][NP][NP]> buffer("buffer_cxx");

  // Output exec views
  HostViewManaged<Real * [NP][NP]> scalar_output("scalar_exec_output", nelems);
  HostViewManaged<Real * [2][NP][NP]> vector_output("vector_exec_output",
                                                    nelems);

  // Random numbers generators
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());
  std::uniform_real_distribution<Real> dreal(0.0125, 1);

  // Initialize derivative
  init_deriv_f90(dvv_h.data());
  Derivative deriv;
  deriv.init(dvv_h.data());

  auto policy = Homme::get_default_team_policy<ExecSpace>(nelems);

  SECTION("gradient sphere single level") {
    for (int itest = 0; itest < num_rand_test; ++itest) {

      // Initialize input(s)
      genRandArray(scalar_h, engine, dreal);
      Kokkos::deep_copy(scalar_exec, scalar_h);

      genRandArray(DInv_h, engine, dreal);
      Kokkos::deep_copy(DInv_exec, DInv_h);

      // Compute cxx
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamMember team_member) {
        KernelVariables kv(team_member);

        gradient_sphere_sl(kv, DInv_exec, deriv.get_dvv(),
                           subview(scalar_exec, kv.ie, ALL, ALL), buffer,
                           subview(vector_exec, kv.ie, ALL, ALL, ALL));
      });

      // Deep copy back to host
      Kokkos::deep_copy(vector_output, vector_exec);

      for (int ie = 0; ie < nelems; ++ie) {
        // Compute f90
        gradient_sphere_c_callable(
            subview(scalar_h, ie, ALL, ALL).data(),
            dvv_h.data(),
            subview(DInv_h, ie, ALL, ALL, ALL, ALL).data(),
            subview(vector_h, ie, ALL, ALL, ALL).data());

        // Check the answer
        for (int dim = 0; dim < 2; ++dim) {
          for (int j = 0; j < NP; ++j) {
            for (int i = 0; i < NP; ++i) {
              const Real correct = vector_h(ie, dim, j, i);
              const Real computed = vector_output(ie, dim, j, i);
              const Real rel_error = compare_answers(correct, computed);
              REQUIRE(rel_threshold >= rel_error);
            }
          }
        }
      }
    }
  }

  SECTION("divergence sphere single level") {

    for (int itest = 0; itest < num_rand_test; ++itest) {
      // Initialize input(s)
      genRandArray(vector_h, engine, dreal);
      genRandArray(DInv_h, engine, dreal);
      genRandArray(metdet_h, engine, dreal);

      Kokkos::deep_copy(vector_exec, vector_h);
      Kokkos::deep_copy(DInv_exec, DInv_h);
      Kokkos::deep_copy(metdet_exec, metdet_h);

      // Compute cxx
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamMember team_member) {
        KernelVariables kv(team_member);

        ExecViewUnmanaged<Real[2][NP][NP]> vector_ie =
            subview(vector_exec, kv.ie, ALL, ALL, ALL);
        ExecViewUnmanaged<Real[NP][NP]> div_ie =
            subview(scalar_exec, kv.ie, ALL, ALL);

        divergence_sphere_sl(kv, DInv_exec, metdet_exec, deriv.get_dvv(),
                             vector_ie, buffer, div_ie);
      });

      // Deep copy back to host
      Kokkos::deep_copy(scalar_output, scalar_exec);

      for (int ie = 0; ie < nelems; ++ie) {
        // Compute f90
        divergence_sphere_c_callable(
            subview(vector_h, ie, ALL, ALL, ALL).data(),
            dvv_h.data(),
            subview(metdet_h, ie, ALL, ALL).data(),
            subview(DInv_h, ie, ALL, ALL, ALL, ALL).data(),
            subview(scalar_h, ie, ALL, ALL).data());
        // Check the answer
        for (int j = 0; j < NP; ++j) {
          for (int i = 0; i < NP; ++i) {
            const Real correct = scalar_h(ie, i, j);
            const Real computed = scalar_output(ie, i, j);
            const Real rel_error = compare_answers(correct, computed);
            REQUIRE(rel_threshold >= rel_error);
          }
        }
      }
    }
  }

  SECTION("vorticity sphere single level") {
    for (int itest = 0; itest < num_rand_test; ++itest) {
      // Initialize input(s)
      genRandArray(vector_h, engine, dreal);
      genRandArray(D_h, engine, dreal);
      genRandArray(metdet_h, engine, dreal);

      Kokkos::deep_copy(vector_exec, vector_h);
      Kokkos::deep_copy(D_exec, D_h);
      Kokkos::deep_copy(metdet_exec, metdet_h);

      // Compute cxx
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamMember team_member) {
        KernelVariables kv(team_member);

        ExecViewUnmanaged<Real[NP][NP]> vector_x_ie =
            subview(vector_exec, kv.ie, 0, ALL, ALL);
        ExecViewUnmanaged<Real[NP][NP]> vector_y_ie =
            subview(vector_exec, kv.ie, 1, ALL, ALL);
        ExecViewUnmanaged<Real[NP][NP]> vort_ie =
            subview(scalar_exec, kv.ie, ALL, ALL);

        vorticity_sphere_sl(kv, D_exec, metdet_exec, deriv.get_dvv(),
                            vector_x_ie, vector_y_ie, buffer, vort_ie);
      });

      // Deep copy back to host
      Kokkos::deep_copy(scalar_output, scalar_exec);

      for (int ie = 0; ie < nelems; ++ie) {
        // Compute f90
        vorticity_sphere_c_callable(subview(vector_h, ie, ALL, ALL, ALL).data(),
                                    dvv_h.data(),
                                    subview(metdet_h, ie, ALL, ALL).data(),
                                    subview(D_h, ie, ALL, ALL, ALL, ALL).data(),
                                    subview(scalar_h, ie, ALL, ALL).data());
        // Check the answer
        for (int i = 0; i < NP; ++i) {
          for (int j = 0; j < NP; ++j) {
            const Real correct = scalar_h(ie, i, j);
            const Real computed = scalar_output(ie, i, j);
            const Real rel_error =
                compare_answers(scalar_h(ie, i, j), scalar_output(ie, i, j));
            REQUIRE(rel_threshold >= rel_error);
          }
        }
      }
    }
  }
}

TEST_CASE("ExecSpaceDefs",
          "Test parallel machine parameterization.") {
  const int plev = 72;

  const auto test_basics = [=] (const ThreadPreferences& tp,
                                const std::pair<int, int>& tv) {
    REQUIRE(tv.first >= 1);
    REQUIRE(tv.second >= 1);
    if (tp.prefer_threads)
      REQUIRE((tv.first == tp.max_threads_usable || tv.second == 1));
    else
      REQUIRE((tv.first == 1 || tv.second > 1));
    if (tv.first * tv.second <= tp.max_threads_usable * tp.max_vectors_usable) {
      REQUIRE(tv.first <= tp.max_threads_usable);
      REQUIRE(tv.second <= tp.max_vectors_usable);
    }
  };

  SECTION("CPU/KNL") {
    for (int num_elem = 0; num_elem <= 3; ++num_elem) {
      for (int qsize = 1; qsize <= 30; ++qsize) {
        for (int pool = 1; pool <= 64; ++pool) {
          for (bool prefer_threads : {true, false}) {
            Homme::ThreadPreferences tp;
            tp.max_vectors_usable = plev;
            tp.prefer_threads = prefer_threads;
            const int npi = num_elem*qsize;
            const auto tv = Homme::Parallel::team_num_threads_vectors_from_pool(
              pool, npi, tp);
            // Correctness tests.
            test_basics(tp, tv);
            REQUIRE(tv.first * tv.second <= pool);
            // Tests for good behavior.
            if (npi >= pool)
              REQUIRE(tv.first * tv.second == 1);
          }
        }
      }
    }
    // Spot check some cases. Numbers are
    //     {#elem, qsize, pool, prefer_threads, #thread, #vector}.
    static const int cases[][6] = {{1, 1, 8, 1, 8, 1},
                                   {1, 30, 8, 1, 1, 1},
                                   {1, 1, 32, 1, 16, 2},
                                   {1, 1, 32, 0, 2, 16}};
    for (unsigned int i = 0; i < sizeof(cases)/sizeof(*cases); ++i) {
      const auto& c = cases[i];
      Homme::ThreadPreferences tp;
      tp.max_vectors_usable = plev/2;
      tp.prefer_threads = c[3];
      const auto tv = Homme::Parallel::team_num_threads_vectors_from_pool(
        c[2], c[0]*c[1], tp);
      REQUIRE(tv.first == c[4]);
      REQUIRE(tv.second == c[5]);
    }
  }

  SECTION("GPU") {
    static const int num_device_warps = 1792;
    static const int min_warps_per_team = 4, max_warps_per_team = 16;
    static const int num_threads_per_warp = 32;
    for (int num_elem = 0; num_elem <= 10000; num_elem += 10) {
      for (int qsize = 1; qsize <= 30; ++qsize) {
        for (bool prefer_threads : {true, false}) {
          Homme::ThreadPreferences tp;
          tp.prefer_threads = prefer_threads;
          tp.max_vectors_usable = plev;
          const int npi = num_elem*qsize;
          const auto tv = Homme::Parallel::team_num_threads_vectors_for_gpu(
            num_device_warps, num_threads_per_warp,
            min_warps_per_team, max_warps_per_team,
            npi, tp);
          // Correctness tests.
          test_basics(tp, tv);
          REQUIRE(Homme::Parallel::prevpow2(tv.second) == tv.second);
          REQUIRE(tv.first * tv.second >= num_threads_per_warp*min_warps_per_team);
          REQUIRE(tv.first * tv.second <= num_threads_per_warp*max_warps_per_team);
          // Tests for good behavior.
          REQUIRE(tv.first * tv.second >= min_warps_per_team * num_threads_per_warp);
          if (npi >= num_device_warps*num_threads_per_warp)
            REQUIRE(tv.first * tv.second == min_warps_per_team * num_threads_per_warp);
        }
      }
    }
    // Numbers are
    //     {#elem, qsize, prefer_threads, #thread, #vector}.
    static const int cases[][5] = {{1, 1, 1, 16, 32},
                                   {96, 30, 1, 16, 8},
                                   {96, 30, 0, 4, 32}};
    for (unsigned int i = 0; i < sizeof(cases)/sizeof(*cases); ++i) {
      const auto& c = cases[i];
      Homme::ThreadPreferences tp;
      tp.max_vectors_usable = plev/2;
      tp.prefer_threads = c[2];
      const auto tv = Homme::Parallel::team_num_threads_vectors_for_gpu(
        num_device_warps, num_threads_per_warp,
        min_warps_per_team, max_warps_per_team,
        c[0]*c[1], tp);
      REQUIRE(tv.first == c[3]);
      REQUIRE(tv.second == c[4]);
    }
  }
}
