#include <catch/catch.hpp>

#include <random>
#include <iostream>

#include "Types.hpp"
#include "FortranArrayUtils.hpp"
#include "Derivative.hpp"
#include "SphereOperators.hpp"
#include "KernelVariables.hpp"

using namespace Homme;

extern "C" {

void init_deriv_f90(F90Ptr &dvv);
void gradient_sphere_f90(CF90Ptr &scalar_f90, CF90Ptr &DInv_f90,
                         F90Ptr &vector_f90, const int &nelems);
void divergence_sphere_f90(CF90Ptr &vector_f90, CF90Ptr &DInv_f90,
                           CF90Ptr &metdet_f90, F90Ptr &scalar_f90,
                           const int &nelems);
void vorticity_sphere_f90(CF90Ptr &vector_f90, CF90Ptr &D_f90,
                          CF90Ptr &metdet_f90, F90Ptr &scalar_f90,
                          const int &nelems);

} // extern "C"

template <typename rngAlg, typename PDF>
void genRandArray(Real *const x, int length, rngAlg &engine, PDF &pdf) {
  for (int i = 0; i < length; ++i) {
    x[i] = pdf(engine);
  }
}

template <typename ViewType, typename rngAlg, typename PDF>
void genRandArray(ViewType view, rngAlg &engine, PDF &pdf) {
  Real *data = view.data();
  for (size_t i = 0; i < view.size(); ++i) {
    data[i] = pdf(engine);
  }
}

Real compare_answers(Real target, Real computed, Real relative_coeff = 1.0) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target - computed) / denom;
}

// ================================= TESTS ============================ //

TEST_CASE("flip arrays", "flip arrays routines") {
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

TEST_CASE("SphereOperators", "Testing spherical differential operators") {
  // Short names
  using Kokkos::subview;
  using Kokkos::ALL;

  constexpr int nelems = 1;
  constexpr int num_rand_test = 10;

  // Fortran host views
  HostViewManaged<Real * [NP][NP]> scalar_h("scalar_host", nelems);
  HostViewManaged<Real * [2][NP][NP]> vector_h("vector_host", nelems);
  HostViewManaged<Real * [NP][NP]> dvv_h("dvv_host", nelems);
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
  // Output exec views
  HostViewManaged<Real * [NP][NP]> scalar_output("scalar_exec_output", nelems);
  HostViewManaged<Real * [2][NP][NP]> vector_output("vector_exec_output",
                                                    nelems);

  // Random numbers generators
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());
  std::uniform_real_distribution<Real> dreal(0, 1);

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

  SECTION("gradient sphere") {
    for (int itest = 0; itest < num_rand_test; ++itest) {

      // Initialize input(s)
      genRandArray(scalar_h, engine, dreal);
      genRandArray(DInv_h, engine, dreal);

      Kokkos::deep_copy(scalar_exec, scalar_h);
      Kokkos::deep_copy(DInv_exec, DInv_h);

      // Compute cxx
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamMember team_member) {
        KernelVariables kv(team_member);

        gradient_sphere_sl(kv, DInv_exec, deriv.get_dvv(),
                           Kokkos::subview(scalar_exec, kv.ie, ALL, ALL),
                           Kokkos::subview(vector_exec, kv.ie, ALL, ALL, ALL));
      });

      // Deep copy back to host
      Kokkos::deep_copy(vector_output, vector_exec);

      // Compute f90
      gradient_sphere_f90(scalar_h.data(), DInv_h.data(), vector_h.data(),
                          nelems);

      // Check the answer
      for (int ie = 0; ie < nelems; ++ie) {
        for (int dim = 0; dim < 2; ++dim) {
          for (int j = 0; j < NP; ++j) {
            for (int i = 0; i < NP; ++i) {
              REQUIRE(compare_answers(vector_h(ie, dim, j, i),
                                      vector_output(ie, dim, j, i)) == 0.0);
            }
          }
        }
      }
    }
  }

  SECTION("divergence sphere") {

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
                             vector_ie, div_ie);
      });

      // Deep copy back to host
      Kokkos::deep_copy(scalar_output, scalar_exec);

      // Compute f90
      divergence_sphere_f90(vector_h.data(), DInv_h.data(), metdet_h.data(),
                            scalar_h.data(), nelems);

      // Check the answer
      for (int ie = 0; ie < nelems; ++ie) {
        for (int j = 0; j < NP; ++j) {
          for (int i = 0; i < NP; ++i) {
            REQUIRE(compare_answers(scalar_h(ie, i, j),
                                    scalar_output(ie, i, j)) == 0.0);
          }
        }
      }
    }
  }

  SECTION("vorticity sphere") {
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
                            vector_x_ie, vector_y_ie, vort_ie);
      });

      // Compute f90
      vorticity_sphere_f90(vector_h.data(), D_h.data(), metdet_h.data(),
                           scalar_h.data(), nelems);

      // Deep copy back to host
      Kokkos::deep_copy(scalar_output, scalar_exec);

      // Check the answer
      for (int ie = 0; ie < nelems; ++ie) {
        for (int i = 0; i < NP; ++i) {
          for (int j = 0; j < NP; ++j) {
            REQUIRE(compare_answers(scalar_h(ie, i, j),
                                    scalar_output(ie, i, j)) == 0.0);
          }
        }
      }
    }
  }
}
