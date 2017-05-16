#ifndef HOMMEXX_BASIC_KERNELS_HPP
#define HOMMEXX_BASIC_KERNELS_HPP

#include "Dimensions.hpp"
#include "Types.hpp"

namespace Homme
{

template<int ROWS_A, int COLS_A, int ROWS_B, int COLS_B, bool trA, bool trB>
struct MatrixMatrixDimensions
{
  static constexpr int ROWS_OUT = trA ? COLS_A : ROWS_A;
  static constexpr int COLS_OUT = trB ? ROWS_B : COLS_B;

  static constexpr int INNER_A = trA ? ROWS_A : COLS_A;
  static constexpr int INNER_B = trB ? COLS_B : ROWS_B;

  static_assert (INNER_A==INNER_B, "Error! Cannot multiply matrix of incompatible dimensions.\n");

  static constexpr int INNER_DIM = INNER_A;

  using type = ExecViewUnmanaged<Real[ROWS_OUT][COLS_OUT]>;
};

template<int ROWS_A, int COLS_A, int ROWS_B, int COLS_B, bool trA = false, bool trB = false>
KOKKOS_INLINE_FUNCTION
void matrix_matrix (const Kokkos::TeamPolicy<ExecSpace>::member_type& team_member,
                    const ExecViewUnmanaged<const Real[ROWS_A][COLS_A]> A,
                    const ExecViewUnmanaged<const Real[ROWS_B][COLS_B]> B,
                    typename MatrixMatrixDimensions<ROWS_A,COLS_A,ROWS_B,COLS_B,trA,trB>::type C)
{
  // Note: it is the user's responsibility to ensure that C is properly initialized
  //       to the correct value (usually, C(i,j)=0)

  constexpr int ROWS_OUT = MatrixMatrixDimensions<ROWS_A,COLS_A,ROWS_B,COLS_B,trA,trB>::ROWS_OUT;
  constexpr int COLS_OUT = MatrixMatrixDimensions<ROWS_A,COLS_A,ROWS_B,COLS_B,trA,trB>::COLS_OUT;

  constexpr int INNER_DIM = MatrixMatrixDimensions<ROWS_A,COLS_A,ROWS_B,COLS_B,trA,trB>::INNER_DIM;

  if (trA)
  {
    if (trB)
    {
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(team_member, ROWS_OUT * COLS_OUT),
        KOKKOS_LAMBDA (const int idx)
        {
          const int igp = idx / ROWS_OUT;
          const int jgp = idx % COLS_OUT;

          for (int kgp=0; kgp<INNER_DIM; ++kgp)
          {
            C(igp,jgp) += A(kgp,igp)*B(jgp,kgp);
          }
        }
      );
    }
    else
    {
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(team_member, ROWS_OUT * COLS_OUT),
        KOKKOS_LAMBDA (const int idx)
        {
          const int igp = idx / ROWS_OUT;
          const int jgp = idx % COLS_OUT;

          for (int kgp=0; kgp<INNER_DIM; ++kgp)
          {
            C(igp,jgp) += A(kgp,igp)*B(kgp,jgp);
          }
        }
      );
    }
  }
  else
  {
    if (trB)
    {
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(team_member, ROWS_OUT * COLS_OUT),
        KOKKOS_LAMBDA (const int idx)
        {
          const int igp = idx / ROWS_OUT;
          const int jgp = idx % COLS_OUT;

          for (int kgp=0; kgp<INNER_DIM; ++kgp)
          {
            C(igp,jgp) += A(igp,kgp)*B(jgp,kgp);
          }
        }
      );
    }
    else
    {
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(team_member, ROWS_OUT * COLS_OUT),
        KOKKOS_LAMBDA (const int idx)
        {
          const int igp = idx / ROWS_OUT;
          const int jgp = idx % COLS_OUT;

          for (int kgp=0; kgp<INNER_DIM; ++kgp)
          {
            C(igp,jgp) += A(igp,kgp)*B(kgp,jgp);
          }
        }
      );
    }

  }
}

} // namespace Homme

#endif // HOMMEXX_BASIC_KERNELS_HPP
