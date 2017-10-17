#include <catch/catch.hpp>

#include <random>
#include <iostream>

#include "Types.hpp"
#include "FortranArrayUtils.hpp"
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

// ================================= TESTS ============================ //

TEST_CASE("flip_arrays", "flip arrays routines") {
  constexpr int N1 = 2;
  constexpr int N2 = 3;
  constexpr int N3 = 4;
  constexpr int N4 = 5;

  Real *A = new Real[N1 * N2];
  Real *B = new Real[N1 * N2 * N3];
  Real *C = new Real[N1 * N2 * N3 * N4];

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());
  std::uniform_real_distribution<Real> dreal(0, 1);

  constexpr int num_rand_test = 10;

  SECTION("f90->cxx") {
    // Same 'mathematical' shape. Simply change f90->cxx ordering

    genRandArray(A, N1 * N2, engine, dreal);
    genRandArray(B, N1 * N2 * N3, engine, dreal);

    HostViewManaged<Real[N1][N2]> A_cxx("A");
    HostViewManaged<Real[N1][N2][N3]> B_cxx("B");

    flip_f90_array_2d_12<N1, N2>(A, A_cxx);
    flip_f90_array_3d_123<N1, N2, N3>(B, B_cxx);

    int iter2d = 0;
    int iter3d = 0;
    for (int j = 0; j < N2; ++j) {
      for (int i = 0; i < N1; ++i, ++iter2d) {
        REQUIRE(compare_answers(A_cxx(i, j), A[iter2d]) == 0);
      }
    }

    for (int k = 0; k < N3; ++k) {
      for (int j = 0; j < N2; ++j) {
        for (int i = 0; i < N1; ++i, ++iter3d) {
          REQUIRE(compare_answers(B_cxx(i, j, k), B[iter3d]) == 0);
        }
      }
    }
  }

  SECTION("flip_f90_array_3d_213") {
    // Change f90->cxx odering and swap some dimensions
    genRandArray(B, N1 * N2 * N3, engine, dreal);
    HostViewManaged<Real[N2][N1][N3]> B_cxx("B");
    flip_f90_array_3d_213<N1, N2, N3>(B, B_cxx);

    int iter = 0;
    for (int k = 0; k < N3; ++k) {
      for (int j = 0; j < N2; ++j) {
        for (int i = 0; i < N1; ++i, ++iter) {
          REQUIRE(compare_answers(B_cxx(j, i, k), B[iter]) == 0);
        }
      }
    }
  }

  SECTION("flip_f90_array_3d_312") {
    // Change f90->cxx odering and swap some dimensions
    genRandArray(B, N1 * N2 * N3, engine, dreal);
    HostViewManaged<Real[N3][N1][N2]> B_cxx("B");
    flip_f90_array_3d_312<N1, N2, N3>(B, B_cxx);

    int iter = 0;
    for (int k = 0; k < N3; ++k) {
      for (int j = 0; j < N2; ++j) {
        for (int i = 0; i < N1; ++i, ++iter) {
          REQUIRE(compare_answers(B_cxx(k, i, j), B[iter]) == 0);
        }
      }
    }
  }

  SECTION("flip_f90_array_4d_3412") {
    // Change f90->cxx odering and swap some dimensions
    genRandArray(C, N1 * N2 * N3 * N4, engine, dreal);

    HostViewManaged<Real[N3][N4][N1][N2]> C_cxx("C");
    flip_f90_array_4d_3412<N1, N2, N3, N4>(C, C_cxx);

    int iter = 0;
    for (int l = 0; l < N4; ++l) {
      for (int k = 0; k < N3; ++k) {
        for (int j = 0; j < N2; ++j) {
          for (int i = 0; i < N1; ++i, ++iter) {
            REQUIRE(compare_answers(C_cxx(k, l, i, j), C[iter]) == 0);
          }
        }
      }
    }
  }

  // Cleanup
  delete[] A;
  delete[] B;
  delete[] C;
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
  ExecViewManaged<Scalar * [NUM_LEV][2][NP][NP]> buffer("buffer_cxx", nelems);

  // Initialize derivative
  HostViewManaged<Real[NP][NP]> dvv_h("dvv_host");
  init_deriv_f90(dvv_h.data());
  ExecViewManaged<Real[NP][NP]> dvv_exec("dvv_exec");
  Kokkos::deep_copy(dvv_exec, dvv_h);

  // Execution policy
  const int threads_per_team =
      DefaultThreadsDistribution<ExecSpace>::threads_per_team(nelems);
  const int vectors_per_thread =
      DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  Kokkos::TeamPolicy<ExecSpace> policy(nelems, threads_per_team,
                                       vectors_per_thread);

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
      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                           [&](const int level) {
        kv.ilev = level;
        gradient_sphere(kv, D_exec, dvv_exec,
                        subview(input_exec, kv.ie, ALL, ALL, ALL), buffer,
                        subview(output_exec, kv.ie, ALL, ALL, ALL, ALL));
      });
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
      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                           [&](const int level) {
        kv.ilev = level;
        divergence_sphere(kv, D_exec, metdet_exec,
                          dvv_exec,
                          subview(input_exec, kv.ie, ALL, ALL, ALL, ALL),
                          buffer, subview(output_exec, kv.ie, ALL, ALL, ALL));
      });
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
      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                           [&](const int level) {
        kv.ilev = level;
        vorticity_sphere(kv, D_exec, metdet_exec,
                         dvv_exec,
                         subview(input_1_exec, kv.ie, ALL, ALL, ALL),
                         subview(input_2_exec, kv.ie, ALL, ALL, ALL), buffer,
                         subview(output_exec, kv.ie, ALL, ALL, ALL));
      });
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

  // Execution policy
  const int vectors_per_thread =
      DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  const int threads_per_team = 1;
  Kokkos::TeamPolicy<ExecSpace> policy(nelems, threads_per_team,
                                       vectors_per_thread);
  policy.set_chunk_size(1);

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
