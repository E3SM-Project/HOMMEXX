#include <catch/catch.hpp>

#include <random>
#include <iostream>

#include "Types.hpp"
#include "FortranArrayUtils.hpp"
#include "Derivative.hpp"
#include "SphereOperators.hpp"

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

  constexpr int nelems = 10;
  constexpr int num_rand_test = 10;

  // Raw pointers
  Real *scalar_f90 = new Real[nelems * NP * NP];
  Real *vector_f90 = new Real[nelems * NP * NP * 2];
  Real *dvv_f90 = new Real[nelems * NP * NP];
  Real *D_f90 = new Real[nelems * NP * NP * 2 * 2];
  Real *DInv_f90 = new Real[nelems * NP * NP * 2 * 2];
  Real *metdet_f90 = new Real[nelems * NP * NP];

  // Host views
  HostViewManaged<Real * [NP][NP]> scalar_cxx("scalar_cxx", nelems);
  HostViewManaged<Real * [2][NP][NP]> vector_cxx("vector_cxx", nelems);
  HostViewManaged<Real * [2][2][NP][NP]> DInv_cxx("DInv_cxx", nelems);
  HostViewManaged<Real * [2][2][NP][NP]> D_cxx("D_cxx", nelems);
  HostViewManaged<Real * [NP][NP]> metdet_cxx("metdet_cxx", nelems);

  // Exec views
  ExecViewManaged<Real * [NP][NP]> scalar_cxx_exec("scalar_cxx_exec", nelems);
  ExecViewManaged<Real * [2][NP][NP]> vector_cxx_exec("vector_cxx_exec",
                                                      nelems);
  ExecViewManaged<Real * [2][2][NP][NP]> DInv_cxx_exec("DInv_cxx_exec", nelems);
  ExecViewManaged<Real * [2][2][NP][NP]> D_cxx_exec("D_cxx_exec", nelems);
  ExecViewManaged<Real * [NP][NP]> metdet_cxx_exec("metdet_cxx_exec", nelems);
  ExecViewManaged<Real * [2][NP][NP]> tmp_cxx("tmp_cxx", nelems);

  // Random numbers generators
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());
  std::uniform_real_distribution<Real> dreal(0, 1);

  // Initialize derivative
  init_deriv_f90(dvv_f90);
  Derivative &deriv = get_derivative();
  deriv.init(dvv_f90);

  // Execution policy
  DefaultThreadsDistribution<ExecSpace>::init();
  const int vectors_per_thread =
      DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  const int threads_per_team = 1;
  Kokkos::TeamPolicy<ExecSpace> policy(nelems, threads_per_team,
                                       vectors_per_thread);
  policy.set_chunk_size(1);

  CaarRegion region;
  SECTION("gradient sphere") {
    for (int itest = 0; itest < num_rand_test; ++itest) {

      // Initialize input(s)
      genRandArray(scalar_f90, nelems * NP * NP, engine, dreal);
      genRandArray(DInv_f90, nelems * NP * NP * 2 * 2, engine, dreal);

      // Flip inputs for cxx
      for (int ie = 0, it_s = 0, it_dinv = 0; ie < nelems; ++ie) {
        for (int jdim = 0; jdim < 2; ++jdim)
          for (int idim = 0; idim < 2; ++idim)
            for (int jp = 0; jp < NP; ++jp)
              for (int ip = 0; ip < NP; ++ip, ++it_dinv)
                DInv_cxx(ie, jdim, idim, jp, ip) = DInv_f90[it_dinv];
        for (int jp = 0; jp < NP; ++jp)
          for (int ip = 0; ip < NP; ++ip, ++it_s)
            scalar_cxx(ie, jp, ip) = scalar_f90[it_s];
      }
      // flip_f90_array_3d_312<NP,NP>       (scalar_f90, scalar_cxx);
      // flip_f90_array_5d_53412<NP,NP,2,2> (DInv_f90, DInv_cxx);

      Kokkos::deep_copy(scalar_cxx_exec, scalar_cxx);
      Kokkos::deep_copy(DInv_cxx_exec, DInv_cxx);

      // Compute cxx
      region.m_dinv = DInv_cxx_exec;
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamMember team_member) {
        const int ie = team_member.league_rank();

        ExecViewManaged<Real[NP][NP]> scalar_ie =
            subview(scalar_cxx_exec, ie, ALL, ALL);
        ExecViewManaged<Real[2][NP][NP]> grad_ie =
            subview(vector_cxx_exec, ie, ALL, ALL, ALL);

        gradient_sphere_sl(team_member, region, deriv.get_dvv(), scalar_ie,
                           grad_ie);
      });

      // Compute f90
      gradient_sphere_f90(scalar_f90, DInv_f90, vector_f90, nelems);

      // Deep copy back to host
      Kokkos::deep_copy(vector_cxx, vector_cxx_exec);

      // Check the answer
      int iter = 0;
      for (int ie = 0; ie < nelems; ++ie) {
        for (int dim = 0; dim < 2; ++dim) {
          for (int j = 0; j < NP; ++j) {
            for (int i = 0; i < NP; ++i, ++iter) {
              REQUIRE(compare_answers(vector_f90[iter],
                                      vector_cxx(ie, dim, j, i)) == 0.0);
            }
          }
        }
      }
    }
  }

  SECTION("divergence sphere") {

    DefaultThreadsDistribution<ExecSpace>::init();

    for (int itest = 0; itest < num_rand_test; ++itest) {
      // Initialize input(s)
      genRandArray(vector_f90, NP * NP * 2 * nelems, engine, dreal);
      genRandArray(DInv_f90, NP * NP * 2 * 2 * nelems, engine, dreal);
      genRandArray(metdet_f90, NP * NP * nelems, engine, dreal);

      // Flip inputs for cxx
      flip_f90_array_4d_4312<NP, NP, 2>(vector_f90, vector_cxx);
      flip_f90_array_5d_53412<NP, NP, 2, 2>(DInv_f90, DInv_cxx);
      flip_f90_array_3d_312<NP, NP>(metdet_f90, metdet_cxx);

      Kokkos::deep_copy(vector_cxx_exec, vector_cxx);
      Kokkos::deep_copy(DInv_cxx_exec, DInv_cxx);
      Kokkos::deep_copy(metdet_cxx_exec, metdet_cxx);

      // Compute cxx
      region.m_dinv = DInv_cxx_exec;
      region.m_metdet = metdet_cxx_exec;
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamMember team_member) {
        const int ie = team_member.league_rank();

        ExecViewUnmanaged<Real[2][NP][NP]> vector_ie =
            subview(vector_cxx_exec, ie, ALL, ALL, ALL);
        ExecViewUnmanaged<Real[NP][NP]> div_ie =
            subview(scalar_cxx_exec, ie, ALL, ALL);

        divergence_sphere_sl(team_member, region, deriv.get_dvv(), vector_ie,
                             div_ie);
      });

      // Compute f90
      divergence_sphere_f90(vector_f90, DInv_f90, metdet_f90, scalar_f90,
                            nelems);

      // Deep copy back to host
      Kokkos::deep_copy(scalar_cxx, scalar_cxx_exec);

      // Check the answer
      int iter = 0;
      for (int ie = 0; ie < nelems; ++ie) {
        for (int j = 0; j < NP; ++j) {
          for (int i = 0; i < NP; ++i, ++iter) {
            REQUIRE(compare_answers(scalar_f90[iter], scalar_cxx(ie, i, j)) ==
                    0.0);
          }
        }
      }
    }
  }

  SECTION("vorticity sphere") {
    for (int itest = 0; itest < num_rand_test; ++itest) {
      // Initialize input(s)
      genRandArray(vector_f90, nelems * NP * NP * 2, engine, dreal);
      genRandArray(D_f90, nelems * NP * NP * 2 * 2, engine, dreal);
      genRandArray(metdet_f90, nelems * NP * NP, engine, dreal);

      // Flip inputs for cxx
      flip_f90_array_4d_4312<NP, NP, 2>(vector_f90, vector_cxx);
      flip_f90_array_5d_53412<NP, NP, 2, 2>(D_f90, D_cxx);
      flip_f90_array_3d_312<NP, NP>(metdet_f90, metdet_cxx);

      Kokkos::deep_copy(vector_cxx_exec, vector_cxx);
      Kokkos::deep_copy(D_cxx_exec, D_cxx);
      Kokkos::deep_copy(metdet_cxx_exec, metdet_cxx);

      region.m_d = D_cxx_exec;
      region.m_metdet = metdet_cxx_exec;

      // Compute cxx
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamMember team_member) {
        const int ie = team_member.league_rank();

        ExecViewUnmanaged<Real[NP][NP]> vector_x_ie =
            subview(vector_cxx_exec, ie, 0, ALL, ALL);
        ExecViewUnmanaged<Real[NP][NP]> vector_y_ie =
            subview(vector_cxx_exec, ie, 1, ALL, ALL);
        ExecViewUnmanaged<Real[NP][NP]> vort_ie =
            subview(scalar_cxx_exec, ie, ALL, ALL);

        vorticity_sphere_sl(team_member, region, deriv.get_dvv(), vector_x_ie,
                            vector_y_ie, vort_ie);
      });

      // Compute f90
      vorticity_sphere_f90(vector_f90, D_f90, metdet_f90, scalar_f90, nelems);

      // Deep copy back to host
      Kokkos::deep_copy(scalar_cxx, scalar_cxx_exec);

      // Check the answer
      int iter = 0;
      for (int ie = 0; ie < nelems; ++ie) {
        for (int j = 0; j < NP; ++j) {
          for (int i = 0; i < NP; ++i, ++iter) {
            REQUIRE(compare_answers(scalar_f90[iter], scalar_cxx(ie, i, j)) ==
                    0.0);
          }
        }
      }
    }
  }

  // Cleanup
  delete[] scalar_f90;
  delete[] vector_f90;
  delete[] dvv_f90;
  delete[] DInv_f90;
  delete[] D_f90;
  delete[] metdet_f90;
}
