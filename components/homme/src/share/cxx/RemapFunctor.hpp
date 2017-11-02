#ifndef HOMMEXX_REMAP_FUNCTOR_HPP
#define HOMMEXX_REMAP_FUNCTOR_HPP

#include <type_traits>
#include <memory>

#include <Kokkos_Array.hpp>
#include <Kokkos_Pair.hpp>

#include "Types.hpp"

#include "profiling.hpp"

namespace Homme {

static constexpr size_t remap_dim = 3 + QSIZE_D;

struct Vert_Remap_Alg {};

// Piecewise Parabolic Method stencil
struct PPM_Vert_Remap : public Vert_Remap_Alg {
  using PPM_Grid_Indices = Kokkos::Array<Kokkos::pair<int, int>, 2>;
  using PPM_Indices = Kokkos::Array<Kokkos::pair<int, int>, 3>;

  KOKKOS_INLINE_FUNCTION
  static void compute_grids(
      KernelVariables &kv, const int &num_remap,
      const PPM_Grid_Indices &indices,
      const Kokkos::Array<ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]>,
                          remap_dim> &dx,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2][10]>,
                          remap_dim> &grids) {
    const int var_range_0 = indices[0].second - indices[0].first;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, num_remap * var_range_0),
        [&](const int &loop_idx) {
          const int var = loop_idx / var_range_0;
          const int j = (loop_idx % var_range_0) + indices[0].first;

          grids[var](j, 0) =
              dx[var](j + 1) / (dx[var](j) + dx[var](j + 1) + dx[var](j + 2));

          grids[var](j, 1) = (2.0 * dx[var](j) + dx[var](j + 1)) /
                             (dx[var](j + 1) + dx[var](j + 2));

          grids[var](j, 2) = (dx[var](j + 1) + 2.0 * dx[var](j + 2)) /
                             (dx[var](j) + dx[var](j + 1));
        });

    const int var_range_1 = indices[1].second - indices[1].first;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,
                                                 num_remap * var_range_1),
                         [&](const int &loop_idx) {
      const int var = loop_idx / var_range_1;
      const int j = (loop_idx % var_range_1) + indices[1].first;

      grids[var](j, 3) = dx[var](j + 1) / (dx[var](j + 1) + dx[var](j + 2));

      grids[var](j, 4) =
          1.0 / (dx[var](j) + dx[var](j + 1) + dx[var](j + 2) + dx[var](j + 3));

      grids[var](j, 5) = (2.0 * dx[var](j + 1) * dx[var](j + 2)) /
                         (dx[var](j + 1) + dx[var](j + 2));

      grids[var](j, 6) = (dx[var](j) + dx[var](j + 1)) /
                         (2.0 * dx[var](j + 1) + dx[var](j + 2));

      grids[var](j, 7) = (2.0 * dx[var](j + 1) * dx[var](j + 2)) /
                         (dx[var](j + 1) + dx[var](j + 2));

      grids[var](j, 8) = dx[var](j + 1) * (dx[var](j) + dx[var](j + 1)) /
                         (2.0 * dx[var](j + 1) + dx[var](j + 2));

      grids[var](j, 9) = dx[var](j + 2) * (dx[var](j + 2) + dx[var](j + 3)) /
                         (dx[var](j + 1) + 2.0 * dx[var](j + 2));
    });
  }

  KOKKOS_INLINE_FUNCTION
  static void compute_ppm(
      KernelVariables &kv, const int &num_remap, const PPM_Indices &indices,
      const Kokkos::Array<ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]>,
                          remap_dim> &cell_means,
      const Kokkos::Array<
          ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 2][10]>, remap_dim> &
          dx,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2]>,
                          remap_dim> &dma,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 1]>,
                          remap_dim> &ai,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV][3]>,
                          remap_dim> &parabola_coeffs) {
    const int var_range_0 = indices[0].second - indices[0].first;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,
                                                 num_remap * var_range_0),
                         [&](const int &loop_idx) {
      const int var = loop_idx / var_range_0;
      const int j = (loop_idx % var_range_0) + indices[0].first;
      if ((cell_means[var](j + 2) - cell_means[var](j + 1)) *
              (cell_means[var](j + 1) - cell_means[var](j)) >
          0.0) {
        Real da =
            dx[var](j, 0) *
            (dx[var](j, 1) * (cell_means[var](j + 2) - cell_means[var](j + 1)) +
             dx[var](j, 2) * (cell_means[var](j + 1) - cell_means[var](j)));
        dma[var](j) =
            min(min(fabs(da),
                    2.0 * fabs(cell_means[var](j + 1) - cell_means[var](j))),
                2.0 * fabs(cell_means[var](j + 2) - cell_means[var](j + 1))) *
            copysign(1.0, da);
      } else {
        dma[var](j) = 0.0;
      }
    });

    const int var_range_1 = indices[1].second - indices[1].first;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,
                                                 num_remap * var_range_1),
                         [&](const int &loop_idx) {
      const int var = loop_idx / var_range_1;
      const int j = (loop_idx % var_range_1) + indices[1].first;
      ai[var](j) =
          cell_means[var](j + 1) +
          dx[var](j, 3) * (cell_means[var](j + 2) - cell_means[var](j + 1)) +
          dx[var](j, 4) *
              (dx[var](j, 5) * (dx[var](j, 6) - dx[var](j, 7)) *
                   (cell_means[var](j + 2) - cell_means[var](j + 1)) -
               dx[var](j, 8) * dma[var](j + 1) + dx[var](j, 9) * dma[var](j));
    });

    const int var_range_2 = indices[2].second - indices[2].first;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,
                                                 num_remap * var_range_2),
                         [&](const int &loop_idx) {
      const int var = loop_idx / var_range_2;
      const int j = (loop_idx % var_range_2) + indices[1].first;

      Real al = ai[var](j - 1), ar = ai[var](j);
      if ((ar - cell_means[var](j + 1)) * (cell_means[var](j + 1) - al) <= 0.) {
        al = cell_means[var](j + 1);
        ar = cell_means[var](j + 1);
      }
      if ((ar - al) * (cell_means[var](j + 1) - (al + ar) / 2.0) >
          (ar - al) * (ar - al) / 6.0) {
        al = 3.0 * cell_means[var](j + 1) - 2.0 * ar;
      }
      if ((ar - al) * (cell_means[var](j + 1) - (al + ar) / 2.0) <
          -(ar - al) * (ar - al) / 6.0) {
        ar = 3.0 * cell_means[var](j + 1) - 2.0 * al;
      }

      // Computed these coefficients from the edge values and
      // cell mean in Maple. Assumes normalized coordinates:
      // xi=(x-x0)/dx
      parabola_coeffs[var](j - 1, 0) =
          1.5 * cell_means[var](j + 1) - (al + ar) / 4.0;
      parabola_coeffs[var](j - 1, 1) = ar - al;
      parabola_coeffs[var](j - 1, 2) =
          -6.0 * cell_means[var](j + 1) + 3.0 * (al + ar);
    });
  }
};

// Corresponds to remap alg = 1
struct PPM_Mirrored_Vert_Remap : public PPM_Vert_Remap {
  KOKKOS_INLINE_FUNCTION
  static void compute_grids(
      KernelVariables &kv, const int &num_remap,
      const Kokkos::Array<ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]>,
                          remap_dim> &dx,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2][10]>,
                          remap_dim> &grids) {
    constexpr PPM_Grid_Indices indices = {
      Kokkos::pair<int, int>{ 0, NUM_PHYSICAL_LEV },
      Kokkos::pair<int, int>{ 2, NUM_PHYSICAL_LEV + 1 }
    };
    PPM_Vert_Remap::compute_grids(kv, num_remap, indices, dx, grids);
  }

  KOKKOS_INLINE_FUNCTION
  static void compute_ppm(
      KernelVariables &kv, const int &num_remap,
      const Kokkos::Array<ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]>,
                          remap_dim> &cell_means,
      const Kokkos::Array<
          ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 2][10]>, remap_dim> &
          dx,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2]>,
                          remap_dim> &dma,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 1]>,
                          remap_dim> &ai,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV][3]>,
                          remap_dim> &parabola_coeffs) {
    constexpr PPM_Indices indices = {
      Kokkos::pair<int, int>{ 0, NUM_PHYSICAL_LEV + 2 },
      Kokkos::pair<int, int>{ 0, NUM_PHYSICAL_LEV + 1 },
      Kokkos::pair<int, int>{ 1, NUM_PHYSICAL_LEV + 1 }
    };
    PPM_Vert_Remap::compute_ppm(kv, num_remap, indices, cell_means, dx, dma, ai,
                                parabola_coeffs);
  }
};

// Corresponds to remap alg = 2
struct PPM_Unmirrored_Vert_Remap : public PPM_Vert_Remap {
  KOKKOS_INLINE_FUNCTION
  static void compute_grids(
      KernelVariables &kv, const int &num_remap,
      const Kokkos::Array<ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]>,
                          remap_dim> &dx,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2][10]>,
                          remap_dim> &grids) {
    constexpr PPM_Grid_Indices indices = {
      Kokkos::pair<int, int>{ 0, NUM_PHYSICAL_LEV },
      Kokkos::pair<int, int>{ 2, NUM_PHYSICAL_LEV + 1 }
    };
    PPM_Vert_Remap::compute_grids(kv, num_remap, indices, dx, grids);
  }

  KOKKOS_INLINE_FUNCTION
  static void compute_ppm(
      KernelVariables &kv, const int &num_remap,
      const Kokkos::Array<ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]>,
                          remap_dim> &cell_means,
      const Kokkos::Array<
          ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 2][10]>, remap_dim> &
          dx,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2]>,
                          remap_dim> &dma,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 1]>,
                          remap_dim> &ai,
      const Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV][3]>,
                          remap_dim> &parabola_coeffs) {
    constexpr PPM_Indices indices = {
      Kokkos::pair<int, int>{ 2, NUM_PHYSICAL_LEV },
      Kokkos::pair<int, int>{ 2, NUM_PHYSICAL_LEV - 1 },
      Kokkos::pair<int, int>{ 3, NUM_PHYSICAL_LEV - 1 }
    };
    PPM_Vert_Remap::compute_ppm(kv, num_remap, indices, cell_means, dx, dma, ai,
                                parabola_coeffs);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, num_remap),
                         [&](const int &var) {
      parabola_coeffs[var](0, 0) = cell_means[var](2);
      parabola_coeffs[var](1, 0) = cell_means[var](3);

      parabola_coeffs[var](NUM_PHYSICAL_LEV - 2, 0) =
          cell_means[var](NUM_PHYSICAL_LEV);
      parabola_coeffs[var](NUM_PHYSICAL_LEV - 1, 0) =
          cell_means[var](NUM_PHYSICAL_LEV + 1);

      parabola_coeffs[var](0, 1) = 0.0;
      parabola_coeffs[var](1, 1) = 0.0;
      parabola_coeffs[var](0, 2) = 0.0;
      parabola_coeffs[var](1, 2) = 0.0;

      parabola_coeffs[var](NUM_PHYSICAL_LEV - 2, 1) = 0.0;
      parabola_coeffs[var](NUM_PHYSICAL_LEV - 1, 1) = 0.0;
      parabola_coeffs[var](NUM_PHYSICAL_LEV - 2, 2) = 0.0;
      parabola_coeffs[var](NUM_PHYSICAL_LEV - 1, 2) = 0.0;
    });
  }
};

template <typename T> constexpr bool Is_Remap_Algorithm() {
  return std::is_base_of<Vert_Remap_Alg, T>::value;
};

template <typename remap_type> struct RemapFunctor {
  static_assert(Is_Remap_Algorithm<remap_type>,
                "RemapFunctor not given a remap algorithm to use");

  Control m_data;
  const Elements m_elements;

  static constexpr auto ALL = Kokkos::ALL;

  RemapFunctor(const Control &data, const Elements &elements)
      : m_data(data), m_elements(elements) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMember &team) {
    start_timer("Remap functor");
    KernelVariables kv(team);

    if (m_data.rsplit == 0) {
      // No remapping here, just dpstar check
    } else {
      // remap_Q_ppm t(np1) * dp_star
      // remap_Q_ppm v(np1) * dp_star
    }

    if (m_data.qsize > 0) {
      // remap_Q_ppm Qdp
    }

    stop_timer("Remap functor");
  }

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size(const int team_size) const {
    return KernelVariables::shmem_size(team_size);
  }
};
}

#endif // HOMMEXX_REMAP_FUNCTOR_HPP
