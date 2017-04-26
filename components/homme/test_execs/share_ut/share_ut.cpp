#include <catch/catch.hpp>

#include <random>
#include <iostream>

#include "Types.hpp"
#include "BasicKernels.hpp"
#include "FortranArrayUtils.hpp"
#include "Derivative.hpp"

using namespace Homme;

extern "C"
{

void matrix_matrix_f90 (CRCPtr& A_ptr, CRCPtr& B_ptr, RCPtr& C_ptr, const int& transpA, const int& transpB);
void subcell_div_fluxes_f90 (CRCPtr& u_ptr, CRCPtr& metdet_ptr, RCPtr& flux_ptr);
void init_deriv_arrays_f90 (RCPtr& dvv, RCPtr& integ_mat, RCPtr& bd_interp_mat);

} // extern "C"

template<bool transpA, bool transpB>
void matrix_matrix_c   (CRCPtr& A_ptr, CRCPtr& B_ptr, RCPtr& C_ptr)
{
  HostViewManaged<Real[NP][NP]> A_host ("A");
  HostViewManaged<Real[NP][NP]> B_host ("B");
  HostViewUnmanaged<Real[NP][NP]> C_host (C_ptr);

  HostViewManaged<Real[NP][NP]> A_exec ("A");
  HostViewManaged<Real[NP][NP]> B_exec ("B");
  HostViewManaged<Real[NP][NP]> C_exec ("C");

  flip_f90_array_2d_12<NP,NP> (A_ptr, A_host);
  flip_f90_array_2d_12<NP,NP> (B_ptr, B_host);

  Kokkos::deep_copy (A_exec, A_host);
  Kokkos::deep_copy (B_exec, B_host);

  Kokkos::parallel_for(
    Kokkos::TeamPolicy<ExecSpace>(1, ExecSpace::thread_pool_size()),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      Kokkos::single(
        Kokkos::PerTeam(team_member),
        KOKKOS_LAMBDA ()
        {
          matrix_matrix<NP,NP,NP,NP,transpA,transpB>(team_member,A_exec,B_exec,C_exec);
        }
      );
    }
  );

  Kokkos::deep_copy (C_host, C_exec);
}

void subcell_div_fluxes_c (CRCPtr& u_ptr, CRCPtr& metdet_ptr, RCPtr& flux_ptr)
{
  ExecViewManaged<Real[2][NP][NP]>   u_host      ("u");
  ExecViewManaged<Real[NP][NP]>      metdet_host ("metdet");
  ExecViewUnmanaged<Real[4][NC][NC]> flux_host   (flux_ptr);

  flip_f90_array_2d_12<NP,NP> (metdet_ptr, metdet_host);
  flip_f90_array_3d_312<NP,NP,2> (u_ptr, u_host);

  ExecViewManaged<Real[2][NP][NP]> u      ("u");
  ExecViewManaged<Real[NP][NP]>    metdet ("metdet");
  ExecViewManaged<Real[4][NC][NC]> flux   ("flux");

  Kokkos::deep_copy(u, u_host);
  Kokkos::deep_copy(metdet, metdet_host);

  Kokkos::parallel_for(
    Kokkos::TeamPolicy<ExecSpace>(1, ExecSpace::thread_pool_size()),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      Kokkos::single(
        Kokkos::PerTeam(team_member),
        KOKKOS_LAMBDA ()
        {
          subcell_div_fluxes (team_member, u, metdet, flux);
        }
      );
    }
  );

  Kokkos::deep_copy(flux_host, flux);
}

template <typename rngAlg, typename PDF>
void genRandArray(Real* const x, int length, rngAlg& engine, PDF& pdf)
{
  for(int i=0; i<length; ++i)
  {
    x[i] = pdf(engine);
  }
}

Real compare_answers (Real target, Real computed,
                      Real relative_coeff = 1.0)
{
  Real denom = 1.0;
  if (relative_coeff>0.0 && target!=0.0)
  {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target-computed) / denom;
}

// ================================= TESTS ============================ //

TEST_CASE ("flip arrays", "flip arrays routines")
{
  constexpr int N1 = 2;
  constexpr int N2 = 3;
  constexpr int N3 = 4;

  Real* A = new Real[N1*N2];
  Real* B = new Real[N1*N2*N3];

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());
  std::uniform_real_distribution<Real> dreal (0,1);

  constexpr int num_rand_test = 10;

  SECTION ("f90->cxx")
  {
    // Same 'mathematical' shape. Simply change f90->cxx ordering

    genRandArray (A, N1*N2, engine, dreal);
    genRandArray (B, N1*N2*N3, engine, dreal);

    HostViewManaged<Real[N1][N2]> A_cxx("A");
    HostViewManaged<Real[N1][N2][N3]> B_cxx("B");

    flip_f90_array_2d_12<N1,N2> (A, A_cxx);
    flip_f90_array_3d_123<N1,N2,N3> (B, B_cxx);

    for (int i=0; i<N1; ++i)
    {
      for (int j=0; j<N2; ++j)
      {
        REQUIRE (compare_answers(A_cxx(i,j),A[i+j*N1])==0);
        for (int k=0; k<N3; ++k)
        {
          REQUIRE (compare_answers(B_cxx(i,j,k),B[i+j*N1+k*N1*N2])==0);
        }
      }
    }
  }

  SECTION ("flip_f90_array_3d_213")
  {
    // Change f90->cxx odering and swap some dimensions
    genRandArray (B, N1*N2*N3, engine, dreal);

    HostViewManaged<Real[N2][N1][N3]> B_cxx("B");
    flip_f90_array_3d_213<N1,N2,N3> (B, B_cxx);

    for (int i=0; i<N1; ++i)
    {
      for (int j=0; j<N2; ++j)
      {
        for (int k=0; k<N3; ++k)
        {
          REQUIRE (compare_answers(B_cxx(j,i,k),B[i+j*N1+k*N1*N2])==0);
        }
      }
    }

    HostViewManaged<Real[N3][N1][N2]> C_cxx("C");

    flip_f90_array_3d_312<N1,N2,N3> (B, C_cxx);

    for (int i=0; i<N1; ++i)
    {
      for (int j=0; j<N2; ++j)
      {
        for (int k=0; k<N3; ++k)
        {
          REQUIRE (compare_answers(C_cxx(k,i,j),B[i+j*N1+k*N1*N2])==0);
        }
      }
    }
  }

  // Cleanup
  delete[] A;
  delete[] B;
}

TEST_CASE ("matrix_matrix", "matirx matrix multiplication")
{
  // We need two inputs, since C routines assume C ordering
  Real* A = new Real[NP*NP];
  Real* B = new Real[NP*NP];
  Real* C_f90 = new Real[NP*NP];
  Real* C_cxx = new Real[NP*NP];

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());
  std::uniform_real_distribution<Real> dreal (0,1);

  constexpr int num_rand_test = 10;

  SECTION ("A * B")
  {
    for (int itest=0; itest<num_rand_test; ++itest)
    {
      // Initialize input(s)
      genRandArray(A, NP*NP, engine, dreal);
      genRandArray(B, NP*NP, engine, dreal);

      // Compute
      matrix_matrix_f90 (A,B,C_f90,0,0);
      matrix_matrix_c<false,false> (A,B,C_cxx);

      // Check the answer
      for (int i=0; i<NP; ++i)
      {
        for (int j=0; j<NP; ++j)
        {
          REQUIRE(compare_answers(C_f90[i+j*NP],C_cxx[j+i*NP]) == 0.0);
        }
      }
    }
  }

  SECTION ("A * B'")
  {
    for (int itest=0; itest<num_rand_test; ++itest)
    {
      // Initialize input(s)
      genRandArray(A, NP*NP, engine, dreal);
      genRandArray(B, NP*NP, engine, dreal);

      // Compute
      matrix_matrix_f90 (A,B,C_f90,0,1);
      matrix_matrix_c<false,true> (A,B,C_cxx);

      // Check the answer
      for (int i=0; i<NP; ++i)
      {
        for (int j=0; j<NP; ++j)
        {
          REQUIRE(compare_answers(C_f90[i+j*NP],C_cxx[j+i*NP]) == 0.0);
        }
      }
    }
  }

  SECTION ("A' * B")
  {
    for (int itest=0; itest<num_rand_test; ++itest)
    {
      // Initialize input(s)
      genRandArray(A, NP*NP, engine, dreal);
      genRandArray(B, NP*NP, engine, dreal);

      // Compute
      matrix_matrix_f90 (A,B,C_f90,1,0);
      matrix_matrix_c<true,false> (A,B,C_cxx);

      // Check the answer
      for (int i=0; i<NP; ++i)
      {
        for (int j=0; j<NP; ++j)
        {
          REQUIRE(compare_answers(C_f90[i+j*NP],C_cxx[j+i*NP]) == 0.0);
        }
      }
    }
  }

  SECTION ("A' * B'")
  {
    for (int itest=0; itest<num_rand_test; ++itest)
    {
      // Initialize input(s)
      genRandArray(A, NP*NP, engine, dreal);
      genRandArray(B, NP*NP, engine, dreal);

      // Compute
      matrix_matrix_f90 (A,B,C_f90,1,1);
      matrix_matrix_c<true,true> (A,B,C_cxx);

      // Check the answer
      for (int i=0; i<NP; ++i)
      {
        for (int j=0; j<NP; ++j)
        {
          REQUIRE(compare_answers(C_f90[i+j*NP],C_cxx[j+i*NP]) == 0.0);
        }
      }
    }
  }

  // Cleanup
  delete[] A;
  delete[] B;
  delete[] C_f90;
  delete[] C_cxx;
}

TEST_CASE ("subcell_div_fluxes", "subcell_div_fluxes")
{
  Real* u      = new Real[2*NP*NP];
  Real* metdet = new Real[NP*NP];
  Real* flux_f90 = new Real[NC*NC*4];
  Real* flux_cxx = new Real[NC*NC*4];

  // Stuff needed by the Derivative
  Real* dvv = new Real[NP*NP];
  Real* integ_mat = new Real[NP*NC];
  Real* bd_interp_mat = new Real[NC*2*NP];

  init_deriv_arrays_f90 (dvv,integ_mat,bd_interp_mat);
  Derivative& deriv = get_derivative();
  deriv.init(dvv,integ_mat,bd_interp_mat);

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());
  std::uniform_real_distribution<Real> dreal (0,1);

  constexpr int num_rand_test = 10;

  // Note: pay attention to which indices run faster!
  //  array_f: (i,j,k) from [1,1,1] up to [nc,nc,4] => i is faster, then j, then k
  //  array_c: (k,i,j) from [0,0,0] up to (4,nc,nc) => j is faster, then i, then k
  SECTION ("subcell div fluxes")
  {
    for (int itest=0; itest<num_rand_test; ++itest)
    {
      // Initialize input(s)
      genRandArray(u, 2*NP*NP, engine, dreal);
      genRandArray(metdet, NP*NP, engine, dreal);

      // Compute
      subcell_div_fluxes_c(u,metdet,flux_cxx);
      subcell_div_fluxes_f90(u,metdet,flux_f90);

      // Check the answer
      for (int i=0; i<NC; ++i)
      {
        for (int j=0; j<NC; ++j)
        {
          for (int k=0; k<4; ++k)
          {
            REQUIRE(compare_answers(flux_f90[i+j*NC+k*NC*NC],flux_cxx[j+i*NC+k*NC*NC]) == 0.0);
          }
        }
      }
    }
  }
}
