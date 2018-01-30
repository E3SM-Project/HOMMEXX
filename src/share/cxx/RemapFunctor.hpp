#ifndef HOMMEXX_REMAP_FUNCTOR_HPP
#define HOMMEXX_REMAP_FUNCTOR_HPP

#include <memory>
#include <type_traits>

#include <Kokkos_Array.hpp>

#include "mpi/ErrorDefs.hpp"

#include "Control.hpp"
#include "Elements.hpp"
#include "Types.hpp"

#include "profiling.hpp"

namespace Homme {

// All VertRemapAlg types must provide the following methods:
// compute_grids_phase, and compute_remap_phase
//
// compute_grids_phase is expected to have less parallelism available and to
// compute quantities which are independent of the tracers,
// based on the computed partitions
//
// compute_remap_phase remaps each of the tracers based on the quantities
// previously computed in compute_grids_phase.
// It is also expected to have a large amount of parallelism, specifically
// qsize * num_elems
struct VertRemapAlg {};

struct PpmBoundaryConditions {};

// Corresponds to remap alg = 1
struct PpmMirrored : public PpmBoundaryConditions {
  static constexpr int fortran_remap_alg = 1;

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> grid_indices_1() {
    return Loop_Range<int>(0, NUM_PHYSICAL_LEV + 2);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> grid_indices_2() {
    return Loop_Range<int>(0, NUM_PHYSICAL_LEV + 1);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> ppm_indices_1() {
    return Loop_Range<int>(0, NUM_PHYSICAL_LEV + 2);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> ppm_indices_2() {
    return Loop_Range<int>(0, NUM_PHYSICAL_LEV + 1);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> ppm_indices_3() {
    return Loop_Range<int>(1, NUM_PHYSICAL_LEV + 1);
  }

  KOKKOS_INLINE_FUNCTION
  static void apply_ppm_boundary(
      ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]> cell_means,
      ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV][3]> parabola_coeffs) {}

  static constexpr const char *name() { return "Mirrored PPM"; }
};

// Corresponds to remap alg = 2
struct PpmFixed : public PpmBoundaryConditions {
  static constexpr int fortran_remap_alg = 2;

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> grid_indices_1() {
    return Loop_Range<int>(2, NUM_PHYSICAL_LEV);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> grid_indices_2() {
    return Loop_Range<int>(2, NUM_PHYSICAL_LEV - 1);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> ppm_indices_1() {
    return Loop_Range<int>(2, NUM_PHYSICAL_LEV);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> ppm_indices_2() {
    return Loop_Range<int>(2, NUM_PHYSICAL_LEV - 1);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> ppm_indices_3() {
    return Loop_Range<int>(3, NUM_PHYSICAL_LEV - 1);
  }

  KOKKOS_INLINE_FUNCTION
  static void apply_ppm_boundary(
      ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]> cell_means,
      ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV][3]> parabola_coeffs) {
    parabola_coeffs(0, 0) = cell_means(2);
    parabola_coeffs(1, 0) = cell_means(3);

    parabola_coeffs(NUM_PHYSICAL_LEV - 2, 0) = cell_means(NUM_PHYSICAL_LEV);
    parabola_coeffs(NUM_PHYSICAL_LEV - 1, 0) = cell_means(NUM_PHYSICAL_LEV + 1);

    parabola_coeffs(0, 1) = 0.0;
    parabola_coeffs(1, 1) = 0.0;
    parabola_coeffs(0, 2) = 0.0;
    parabola_coeffs(1, 2) = 0.0;

    parabola_coeffs(NUM_PHYSICAL_LEV - 2, 1) = 0.0;
    parabola_coeffs(NUM_PHYSICAL_LEV - 1, 1) = 0.0;
    parabola_coeffs(NUM_PHYSICAL_LEV - 2, 2) = 0.0;
    parabola_coeffs(NUM_PHYSICAL_LEV - 1, 2) = 0.0;
  }

  static constexpr const char *name() { return "Fixed PPM"; }
};

// Piecewise Parabolic Method stencil
template <int _remap_dim, typename boundaries>
struct PpmVertRemap : public VertRemapAlg {
  static_assert(std::is_base_of<PpmBoundaryConditions, boundaries>::value,
                "PpmVertRemap requires a valid PPM "
                "boundary condition");
  static constexpr auto remap_dim = _remap_dim;
  const int gs = 2;

  explicit PpmVertRemap(const Control &data)
      : dpo(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>(
            "dpo", data.num_elems)),
        pio(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2]>(
            "pio", data.num_elems)),
        pin(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>(
            "pin", data.num_elems)),
        ppmdx(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2][10]>(
            "ppmdx", data.num_elems)),
        z2(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]>("z2",
                                                              data.num_elems)),
        kid(ExecViewManaged<int * [NP][NP][NUM_PHYSICAL_LEV]>("kid",
                                                              data.num_elems)) {
    for (int i = 0; i < remap_dim; ++i) {
      ao[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>(
          "a0", data.num_elems);
      mass_o[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>(
          "mass_o", data.num_elems);
      dma[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2]>(
          "dma", data.num_elems);
      ai[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>(
          "ai", data.num_elems);
      parabola_coeffs[i] =
          ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV][3]>(
              "Coefficients for the interpolating parabola", data.num_elems);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void compute_grids_phase(
      KernelVariables &kv,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness)
      const {
    start_timer("remap ppm grids phase");
    compute_partitions(kv, src_layer_thickness, tgt_layer_thickness);
    compute_integral_bounds(kv);
    stop_timer("remap ppm grids phase");
  }

  KOKKOS_INLINE_FUNCTION
  void compute_remap_phase(KernelVariables &kv, const int remap_idx,
                           ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> remap_var)
      const {
    start_timer("remap ppm Q phase");

    // From here, we loop over tracers for only those portions which depend on
    // tracer data, which includes PPM limiting and mass accumulation
    // More parallelism than we need here, maybe break it up?
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
                           [&](const int k) {
        const int ilevel = k / VECTOR_SIZE;
        const int ivector = k % VECTOR_SIZE;
        ao[remap_idx](kv.ie, igp, jgp, k + 2) =
            remap_var(igp, jgp, ilevel)[ivector] / dpo(kv.ie, igp, jgp, k + 2);
      });

      // Scan region
      Kokkos::single(Kokkos::PerThread(kv.team), [&]() {
        // Accumulate the old mass up to old grid cell interface locations to
        // simplify integration during remapping. Also, divide out the grid
        // spacing so we're working with actual tracer values and can conserve
        // mass.
        mass_o[remap_idx](kv.ie, igp, jgp, 0) = 0.0;
        for (int k = 0; k < NUM_PHYSICAL_LEV; k++) {
          const int ilevel = k / VECTOR_SIZE;
          const int ivector = k % VECTOR_SIZE;

          mass_o[remap_idx](kv.ie, igp, jgp, k + 1) =
              mass_o[remap_idx](kv.ie, igp, jgp, k) +
              remap_var(igp, jgp, ilevel)[ivector];
        } // end k loop
      });

      // Reflect the real values across the top and bottom boundaries into the
      // ghost cells
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, gs),
                           [&](const int &k_0) {
        ao[remap_idx](kv.ie, igp, jgp, 1 - k_0 - 1 + 1) =
            ao[remap_idx](kv.ie, igp, jgp, k_0 + 1 + 1);

        ao[remap_idx](kv.ie, igp, jgp, NUM_PHYSICAL_LEV + k_0 + 1 + 1) =
            ao[remap_idx](kv.ie, igp, jgp, NUM_PHYSICAL_LEV + 1 - k_0 - 1 + 1);
      }); // end ghost cell loop

      // Computes a monotonic and conservative PPM reconstruction
      compute_ppm(kv, Homme::subview(ao[remap_idx], kv.ie, igp, jgp),
                  Homme::subview(ppmdx, kv.ie, igp, jgp),
                  Homme::subview(dma[remap_idx], kv.ie, igp, jgp),
                  Homme::subview(ai[remap_idx], kv.ie, igp, jgp),
                  Homme::subview(parabola_coeffs[remap_idx], kv.ie, igp, jgp));
      compute_remap(kv, Homme::subview(kid, kv.ie, igp, jgp),
                    Homme::subview(z2, kv.ie, igp, jgp),
                    Homme::subview(parabola_coeffs[remap_idx], kv.ie, igp, jgp),
                    Homme::subview(mass_o[remap_idx], kv.ie, igp, jgp),
                    Homme::subview(dpo, kv.ie, igp, jgp),
                    Homme::subview(remap_var, igp, jgp));
    }); // End team thread range
    kv.team_barrier();
    stop_timer("remap ppm Q phase");
  }

  KOKKOS_INLINE_FUNCTION
  Real compute_mass(ExecViewUnmanaged<const Real[3]> parabola_coeffs,
                    const Real prev_mass, const Real prev_dp,
                    const Real x2) const {
    // This remapping assumes we're starting from the left interface of an
    // old grid cell
    // In fact, we're usually integrating very little or almost all of the
    // cell in question
    const Real x1 = -0.5;
    const Real integral = integrate_parabola(parabola_coeffs, x1, x2);
    const Real mass = prev_mass + integral * prev_dp;
    return mass;
  }

  KOKKOS_INLINE_FUNCTION
  void compute_remap(
      KernelVariables &kv, ExecViewUnmanaged<const int[NUM_PHYSICAL_LEV]> k_id,
      ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV]> integral_bounds,
      ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV][3]> parabola_coeffs,
      ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 1]> prev_mass,
      ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]> prev_dp,
      ExecViewUnmanaged<Scalar[NUM_LEV]> remap_var) const {
    // Compute tracer values on the new grid by integrating from the old cell
    // bottom to the new cell interface to form a new grid mass accumulation.
    // Taking the difference between accumulation at successive interfaces
    // gives the mass inside each cell. Since Qdp is supposed to hold the full
    // mass this needs no normalization.
    // This could be serialized on OpenMP to reduce the work by half,
    // but the parallel gain on CUDA is >> 2
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
                         [&](const int k) {
      // Using an immediately invoked function expression (IIFE, another
      // annoying C++ lingo acronym) lets us make mass_1 constant
      // This also provides better scoping for the variables inside it
      const Real mass_1 = [=]() {
        if (k > 0) {
          const Real x2_prev_lev = integral_bounds(k - 1);
          const int kk_prev_lev = k_id(k - 1) - 1;
          return compute_mass(Homme::subview(parabola_coeffs, kk_prev_lev),
                              prev_mass(kk_prev_lev), prev_dp(kk_prev_lev + 2),
                              x2_prev_lev);
        } else {
          return 0.0;
        }
      }();

      const Real x2_cur_lev = integral_bounds(k);

      const int kk_cur_lev = k_id(k) - 1;
      assert(kk_cur_lev >= 0);
      assert(kk_cur_lev < parabola_coeffs.extent_int(0));

      const Real mass_2 = compute_mass(
          Homme::subview(parabola_coeffs, kk_cur_lev), prev_mass(kk_cur_lev),
          prev_dp(kk_cur_lev + 2), x2_cur_lev);

      const int ilevel = k / VECTOR_SIZE;
      const int ivector = k % VECTOR_SIZE;
      remap_var(ilevel)[ivector] = mass_2 - mass_1;
    }); // k loop
  }

  KOKKOS_INLINE_FUNCTION
  void compute_grids(
      KernelVariables &kv,
      const ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]> dx,
      const ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2][10]> grids) const {
    {
      auto bounds = boundaries::grid_indices_1();
      Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, bounds.iterations()),
          [&](const int zoffset_j) {
            const int j = zoffset_j + *bounds.begin();
            grids(j, 0) = dx(j + 1) / (dx(j) + dx(j + 1) + dx(j + 2));

            grids(j, 1) = (2.0 * dx(j) + dx(j + 1)) / (dx(j + 1) + dx(j + 2));

            grids(j, 2) = (dx(j + 1) + 2.0 * dx(j + 2)) / (dx(j) + dx(j + 1));
          });
    }

    {
      auto bounds = boundaries::grid_indices_2();
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,
                                                     bounds.iterations()),
                           [&](const int zoffset_j) {
        const int j = zoffset_j + *bounds.begin();
        grids(j, 3) = dx(j + 1) / (dx(j + 1) + dx(j + 2));

        grids(j, 4) = 1.0 / (dx(j) + dx(j + 1) + dx(j + 2) + dx(j + 3));

        grids(j, 5) = (2.0 * dx(j + 1) * dx(j + 2)) / (dx(j + 1) + dx(j + 2));

        grids(j, 6) = (dx(j) + dx(j + 1)) / (2.0 * dx(j + 1) + dx(j + 2));

        grids(j, 7) = (dx(j + 3) + dx(j + 2)) / (2.0 * dx(j + 2) + dx(j + 1));

        grids(j, 8) =
            dx(j + 1) * (dx(j) + dx(j + 1)) / (2.0 * dx(j + 1) + dx(j + 2));

        grids(j, 9) =
            dx(j + 2) * (dx(j + 2) + dx(j + 3)) / (dx(j + 1) + 2.0 * dx(j + 2));
      });
    }
  }

  KOKKOS_INLINE_FUNCTION
  void compute_ppm(
      KernelVariables &kv,
      // input  views
      ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]> cell_means,
      ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 2][10]> dx,
      // buffer views
      ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2]> dma,
      ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 1]> ai,
      // result view
      ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV][3]> parabola_coeffs) const {
    {
      auto bounds = boundaries::ppm_indices_1();
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,
                                                     bounds.iterations()),
                           [&](const int zoffset_j) {
        const int j = zoffset_j + *bounds.begin();
        if ((cell_means(j + 2) - cell_means(j + 1)) *
                (cell_means(j + 1) - cell_means(j)) >
            0.0) {
          Real da =
              dx(j, 0) * (dx(j, 1) * (cell_means(j + 2) - cell_means(j + 1)) +
                          dx(j, 2) * (cell_means(j + 1) - cell_means(j)));

          dma(j) =
              min(fabs(da), 2.0 * fabs(cell_means(j + 1) - cell_means(j)),
                  2.0 * fabs(cell_means(j + 2) - cell_means(j + 1))) *
              copysign(1.0, da);
        } else {
          dma(j) = 0.0;
        }
                           });
    }
    {
      auto bounds = boundaries::ppm_indices_2();
      Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, bounds.iterations()),
          [&](const int zoffset_j) {
            const int j = zoffset_j + *bounds.begin();
            ai(j) = cell_means(j + 1) +
                    dx(j, 3) * (cell_means(j + 2) - cell_means(j + 1)) +
                    dx(j, 4) * (dx(j, 5) * (dx(j, 6) - dx(j, 7)) *
                                    (cell_means(j + 2) - cell_means(j + 1)) -
                                dx(j, 8) * dma(j + 1) + dx(j, 9) * dma(j));
          });
    }
    // TODO: Figure out and fix the issue which needs the Kokkos::single,
    // and parallelize over the bounds provided
    // This costs about 15-20% more on GPU than a fully parallel ppm remap
    Kokkos::single(Kokkos::PerThread(kv.team), [&]() {
      {
        auto bounds = boundaries::ppm_indices_3();
        // Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,
        //                                                bounds.iterations()),
        //                      [&](const int zoffset_j) {
        //   const int j = zoffset_j + *bounds.begin();
        for (auto j : bounds) {
          Real al = ai(j - 1);
          Real ar = ai(j);
          if ((ar - cell_means(j + 1)) * (cell_means(j + 1) - al) <= 0.) {
            al = cell_means(j + 1);
            ar = cell_means(j + 1);
          }
          if ((ar - al) * (cell_means(j + 1) - (al + ar) / 2.0) >
              (ar - al) * (ar - al) / 6.0) {
            al = 3.0 * cell_means(j + 1) - 2.0 * ar;
          }
          if ((ar - al) * (cell_means(j + 1) - (al + ar) / 2.0) <
              -(ar - al) * (ar - al) / 6.0) {
            ar = 3.0 * cell_means(j + 1) - 2.0 * al;
          }

          // Computed these coefficients from the edge values
          // and cell mean in Maple. Assumes normalized
          // coordinates: xi=(x-x0)/dx

          assert(parabola_coeffs.data() != nullptr);
          assert(j - 1 < parabola_coeffs.extent_int(0));
          assert(2 < parabola_coeffs.extent_int(1));

          parabola_coeffs(j - 1, 0) =
              1.5 * cell_means(j + 1) - (al + ar) / 4.0;
          parabola_coeffs(j - 1, 1) = ar - al;
          parabola_coeffs(j - 1, 2) =
              3.0 * (-2.0 * cell_means(j + 1) + (al + ar));
        }
      }
    });
    Kokkos::single(Kokkos::PerThread(kv.team), [&]() {
      boundaries::apply_ppm_boundary(cell_means, parabola_coeffs);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_partitions(
      KernelVariables &kv,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness)
      const {
    start_timer("remap compute_partitions");
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
                           [&](const int &k) {
        int ilevel = k / VECTOR_SIZE;
        int ivector = k % VECTOR_SIZE;
        dpo(kv.ie, igp, jgp, k + 2) =
            src_layer_thickness(igp, jgp, ilevel)[ivector];
      });
    });
    kv.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      Kokkos::single(Kokkos::PerThread(kv.team), [&]() {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        pin(kv.ie, igp, jgp, 0) = 0.0;
        pio(kv.ie, igp, jgp, 0) = 0.0;
        // scan region
        for (int k = 1; k <= NUM_PHYSICAL_LEV; k++) {
          const int layer_vlevel = (k - 1) / VECTOR_SIZE;
          const int layer_vector = (k - 1) % VECTOR_SIZE;
          pio(kv.ie, igp, jgp, k) =
              pio(kv.ie, igp, jgp, k - 1) +
              src_layer_thickness(igp, jgp, layer_vlevel)[layer_vector];
          pin(kv.ie, igp, jgp, k) =
              pin(kv.ie, igp, jgp, k - 1) +
              tgt_layer_thickness(igp, jgp, layer_vlevel)[layer_vector];
        } // k loop

        // This is here to allow an entire block of k
        // threads to run in the remapping phase. It makes
        // sure there's an old interface value below the
        // domain that is larger.
        assert(fabs(pio(kv.ie, igp, jgp, NUM_PHYSICAL_LEV) -
                    pin(kv.ie, igp, jgp, NUM_PHYSICAL_LEV)) < 1.0);
        pio(kv.ie, igp, jgp, NUM_PHYSICAL_LEV + 1) =
            pio(kv.ie, igp, jgp, NUM_PHYSICAL_LEV) + 1.0;

        // The total mass in a column does not change.
        // Therefore, the pressure of that mass cannot
        // either.
        pin(kv.ie, igp, jgp, NUM_PHYSICAL_LEV) =
            pio(kv.ie, igp, jgp, NUM_PHYSICAL_LEV);
      });
    });

    // Fill in the ghost regions with mirrored values.
    // if vert_remap_q_alg is defined, this is of no
    // consequence.
    // Note that the range of k makes this completely parallel,
    // without any data dependencies
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      const int _gs = gs;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, _gs),
                           [&](const int &k) {
        dpo(kv.ie, igp, jgp, 1 - k) = dpo(kv.ie, igp, jgp, k + 2);
        dpo(kv.ie, igp, jgp, NUM_PHYSICAL_LEV + k + 2) =
            dpo(kv.ie, igp, jgp, NUM_PHYSICAL_LEV + 1 - k);
      });
    });
    kv.team_barrier();

    stop_timer("remap compute_partitions");
  }

  KOKKOS_INLINE_FUNCTION
  void compute_integral_bounds(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
                           [&](const int k) {
        // Compute remapping intervals once for all
        // tracers. Find the old grid cell index in which
        // the k-th new cell interface resides. Then
        // integrate from the bottom of that old cell to
        // the new interface location. In practice, the
        // grid never deforms past one cell, so the search
        // can be simplified by this. Also, the interval
        // of integration is usually of magnitude close to
        // zero or close to dpo because of minimial
        // deformation. Numerous tests confirmed that the
        // bottom and top of the grids match to machine
        // precision, so set them equal to each other.
        int kk = k + 1;
        // This reduces the work required to find the index where this fails
        // at, and is typically less than NUM_PHYSICAL_LEV^2
        // Since the top bounds match anyway, the value of the coefficients
        // don't matter, so enforcing kk <= NUM_PHYSICAL_LEV doesn't affect
        // anything important
        //
        // Note that because we set
        // pio(:, :, :, NUM_PHYSICAL_LEV + 1) = pio(:, :, :, NUM_PHYSICAL_LEV) +
        // 1.0
        // and pin(:, :, :, NUM_PHYSICAL_LEV) = pio(:, :, :, NUM_PHYSICAL_LEV)
        // this loop ensures kk <= NUM_PHYSICAL_LEV + 2
        // Furthermore, since we set
        // pio(:, :, :, 0) = 0.0 and pin(:, :, :, 0) = 0.0
        // kk must be incremented at least once
        assert(pio(kv.ie, igp, jgp, NUM_PHYSICAL_LEV + 1) >
               pin(kv.ie, igp, jgp, k + 1));
        while (pio(kv.ie, igp, jgp, kk - 1) <= pin(kv.ie, igp, jgp, k + 1)) {
          kk++;
          assert(kk - 1 < pio.extent_int(3));
        }

        kk--;
        // This is to keep the indices in bounds.
        if (kk == NUM_PHYSICAL_LEV + 1) {
          kk = NUM_PHYSICAL_LEV;
        }
        // kk is now the cell index we're integrating over.

        // Save kk for reuse
        kid(kv.ie, igp, jgp, k) = kk;
        // PPM interpolants are normalized to an independent coordinate domain
        // [-0.5, 0.5].
        assert(kk - 1 >= 0);
        assert(kk < pio.extent_int(3));
        z2(kv.ie, igp, jgp, k) =
            (pin(kv.ie, igp, jgp, k + 1) -
             (pio(kv.ie, igp, jgp, kk - 1) + pio(kv.ie, igp, jgp, kk)) * 0.5) /
            dpo(kv.ie, igp, jgp, kk + 1);
      });

      ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 4]> point_dpo =
          Homme::subview(dpo, kv.ie, igp, jgp);
      ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2][10]> point_ppmdx =
          Homme::subview(ppmdx, kv.ie, igp, jgp);
      compute_grids(kv, point_dpo, point_ppmdx);
    });
  }

  KOKKOS_INLINE_FUNCTION Real
  integrate_parabola(ExecViewUnmanaged<const Real[3]> coeffs, Real x1,
                     Real x2) const {
    const Real a0 = coeffs(0);
    const Real a1 = coeffs(1);
    const Real a2 = coeffs(2);
    return (a0 * (x2 - x1) + a1 * (x2 * x2 - x1 * x1) / 2.0) +
           a2 * (x2 * x2 * x2 - x1 * x1 * x1) / 3.0;
  }

  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>,
                remap_dim> ao;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]> dpo;
  // pio corresponds to the points in each layer of the source layer thickness
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2]> pio;
  // pin corresponds to the points in each layer of the target layer thickness
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]> pin;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2][10]> ppmdx;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]> z2;
  ExecViewManaged<int * [NP][NP][NUM_PHYSICAL_LEV]> kid;

  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>,
                remap_dim> mass_o;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2]>,
                remap_dim> dma;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>,
                remap_dim> ai;

  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV][3]>,
                remap_dim> parabola_coeffs;
};

template <bool nonzero_rsplit> struct _RemapFunctorRSplit {
  static_assert(nonzero_rsplit == false, "The template specialization for "
                                         "_RemapFunctorRSplit seems to have "
                                         "been removed.");

  static constexpr int num_states_remap = 0;
  static constexpr int remap_dim = num_states_remap + QSIZE_D;

  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> m_src_layer_thickness;
  explicit _RemapFunctorRSplit(const int &num_elems)
      : m_src_layer_thickness("Source layer thickness", num_elems) {}

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> get_source_thickness(
      const int ie, const int np1,
      ExecViewUnmanaged<const Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp3d)
      const {
    return Homme::subview(m_src_layer_thickness, ie);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> compute_source_thickness(
      KernelVariables &kv, const int &np1, const Real &dt,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness,
      ExecViewUnmanaged<const Scalar * [NP][NP][NUM_LEV]> eta_dot_dpdn,
      ExecViewUnmanaged<const Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp3d)
      const {
    start_timer("remap compute_source_thickness");
    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness =
        get_source_thickness(kv.ie, np1, dp3d);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
                           [&](const int &level) {
        const int ilev = level / VECTOR_SIZE;
        const int vlev = level % VECTOR_SIZE;
        const int next_ilev = (level + 1) / VECTOR_SIZE;
        const int next_vlev = (level + 1) % VECTOR_SIZE;
        const auto eta_dot_dpdn_next = (level+1 < NUM_PHYSICAL_LEV ?
                                        eta_dot_dpdn(kv.ie, igp, jgp, next_ilev)[next_vlev] :
                                        0);
        const Real delta_dpdn =
            eta_dot_dpdn_next -
            eta_dot_dpdn(kv.ie, igp, jgp, ilev)[vlev];
        src_layer_thickness(igp, jgp, ilev)[vlev] =
            tgt_layer_thickness(igp, jgp, ilev)[vlev] + dt * delta_dpdn;
        if (kv.ie == 0 && igp == 0 && jgp == 0) {
          DEBUG_PRINT("src/tgt %d (%d %d): % .17e vs % .17e\n", level, ilev,
                      vlev, src_layer_thickness(igp, jgp, ilev)[vlev],
                      tgt_layer_thickness(igp, jgp, ilev)[vlev]);
        }
      });
    });
    kv.team_barrier();
    stop_timer("remap compute_source_thickness");
    return src_layer_thickness;
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>, num_states_remap>
  remap_states_array(const KernelVariables &kv, const Elements &elements,
                     int np1) const {
    // The states which need to be remapped
    return Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>,
                         num_states_remap>();
  }
};

template <> struct _RemapFunctorRSplit<true> {
  explicit _RemapFunctorRSplit(const int &num_elems) {}

  static constexpr int num_states_remap = 3;
  static constexpr int remap_dim = num_states_remap + QSIZE_D;

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> get_source_thickness(
      const int ie, const int np1,
      ExecViewUnmanaged<const Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp3d)
      const {
    return Homme::subview(dp3d, ie, np1);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> compute_source_thickness(
      KernelVariables &kv, const int np1, const Real dt,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness,
      ExecViewUnmanaged<const Scalar * [NP][NP][NUM_LEV]> eta_dot_dpdn,
      ExecViewUnmanaged<const Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp3d)
      const {
    return get_source_thickness(kv.ie, np1, dp3d);
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>, num_states_remap>
  remap_states_array(const KernelVariables &kv, const Elements &elements,
                     int np1) const {
    // The states which need to be remapped
    Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>, num_states_remap>
    state_remap{ { Homme::subview(elements.m_v, kv.ie, np1, 0),
                   Homme::subview(elements.m_v, kv.ie, np1, 1),
                   Homme::subview(elements.m_t, kv.ie, np1) } };
    return state_remap;
  }
};

template <bool nonzero_rsplit, template <int, typename...> class _RemapType,
          typename... RemapOptions>
struct RemapFunctor : public _RemapFunctorRSplit<nonzero_rsplit> {
  static constexpr int remap_dim =
      _RemapFunctorRSplit<nonzero_rsplit>::remap_dim;

  using RemapType = _RemapType<remap_dim, RemapOptions...>;

  static_assert(std::is_base_of<VertRemapAlg, RemapType>::value,
                "RemapFunctor not given a remap algorithm to use");

  Control m_data;
  const Elements m_elements;

  // Surface pressure
  ExecViewManaged<Real * [NP][NP]> m_ps_v;

  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> m_tgt_layer_thickness;

  ExecViewManaged<bool *> valid_layer_thickness;
  typename decltype(valid_layer_thickness)::HostMirror host_valid_input;

  RemapType m_remap;

  explicit RemapFunctor(const Control &data, const Elements &elements)
      : _RemapFunctorRSplit<nonzero_rsplit>(data.num_elems), m_data(data),
        m_elements(elements), m_ps_v("Surface pressure", data.num_elems),
        m_tgt_layer_thickness("Target Layer Thickness", data.num_elems),
        valid_layer_thickness(
            "Check for whether the surface thicknesses are positive",
            data.num_elems),
        host_valid_input(Kokkos::create_mirror_view(valid_layer_thickness)),
        m_remap(data) {}

  // fort_ps_v is of type Real [NUM_ELEMS][NUM_TIME_LEVELS][NP][NP]
  // This method only updates the np1 timelevel
  void update_fortran_ps_v(Real *fort_ps_v_ptr) {
    assert(fort_ps_v_ptr != nullptr);
    HostViewUnmanaged<Real * [NUM_TIME_LEVELS][NP][NP]> fort_ps_v(
        fort_ps_v_ptr, m_data.num_elems);
    for (int ie = 0; ie < m_data.num_elems; ++ie) {
      Kokkos::deep_copy(Homme::subview(fort_ps_v, ie, m_data.np1),
                        Homme::subview(m_ps_v, ie));
    }
  }

  void input_valid_assert() {
    Kokkos::deep_copy(host_valid_input, valid_layer_thickness);
    for (int ie = 0; ie < m_data.num_elems; ++ie) {
      if (host_valid_input(ie) == false) {
        Errors::runtime_abort("Negative layer thickness detected, aborting!",
                              Errors::err_negative_layer_thickness);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  int num_to_remap() const { return this->num_states_remap + m_data.qsize; }

  KOKKOS_INLINE_FUNCTION int
  build_remap_array(KernelVariables &kv,
                    Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>,
                                  remap_dim> &remap_vals) const {
    if (nonzero_rsplit == true) {
      const int num_states = build_remap_array_states(kv, remap_vals);
      const int num_tracers =
          build_remap_array_tracers(kv, num_states, remap_vals);
      return num_tracers + num_states;
    } else {
      return build_remap_array_tracers(kv, 0, remap_vals);
    }
  }

  KOKKOS_INLINE_FUNCTION int build_remap_array_states(
      KernelVariables &kv,
      Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>, remap_dim> &
          remap_vals) const {

    auto state_remap = this->remap_states_array(kv, m_elements, m_data.np1);

    // This must be done for every thread
    for (int state_idx = 0; state_idx < static_cast<int>(state_remap.size());
         ++state_idx) {
      remap_vals[state_idx] = state_remap[state_idx];
    }
    return state_remap.size();
  }

  KOKKOS_INLINE_FUNCTION int build_remap_array_tracers(
      KernelVariables &kv, const int prev_filled,
      Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>, remap_dim> &
          remap_vals) const {

    for (int q = 0; q < m_data.qsize; ++q) {
      remap_vals[prev_filled + q] =
          Homme::subview(m_elements.m_qdp, kv.ie, m_data.qn0, q);
    }
    return m_data.qsize;
  }

  struct ComputeThicknessTag {};
  struct ComputeGridsTag {};
  struct ComputeRemapTag {};
  // Computes the extrinsic values of the states in the initial map
  // i.e. velocity -> momentum
  struct ComputeExtrinsicsTag {};
  // Computes the intrinsic values of the states in the final map
  // i.e. momentum -> velocity
  struct ComputeIntrinsicsTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeThicknessTag, const TeamMember &team) const {
    KernelVariables kv(team);
    if (kv.ie == 0) {
      Kokkos::single(Kokkos::PerTeam(kv.team),
                     []() { DEBUG_PRINT("computing thickness\n"); });
    }
    compute_ps_v(kv, Homme::subview(m_elements.m_dp3d, kv.ie, m_data.np1),
                 Homme::subview(m_ps_v, kv.ie));

    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness =
        compute_target_thickness(kv);

    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness =
        this->compute_source_thickness(
            kv, m_data.np1, m_data.dt, tgt_layer_thickness,
            m_elements.m_eta_dot_dpdn, m_elements.m_dp3d);

    check_source_thickness(kv, src_layer_thickness);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeExtrinsicsTag, const TeamMember &team) const {
    KernelVariables kv(team);

    assert(this->num_states_remap > 0);
    const int den = (this->num_states_remap > 0) ? this->num_states_remap : 1;
    const int var = kv.ie % den;
    kv.ie /= den;
    assert(kv.ie < m_data.num_elems);

    auto state_remap = this->remap_states_array(kv, m_elements, m_data.np1);
    compute_extrinsic_state(
        kv, this->get_source_thickness(kv.ie, m_data.np1, m_elements.m_dp3d),
        state_remap[var]);
  }

  // This functor is the only one guaranteed to be run
  // (in rare cases, no tracers and rsplit==0),
  // so it needs to be separated from the others to reduce latency on the GPU
  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeGridsTag, const TeamMember &team) const {
    if (num_to_remap() > 0) {
      KernelVariables kv(team);
      if (kv.ie == 0) {
        Kokkos::single(Kokkos::PerTeam(kv.team),
                       []() { DEBUG_PRINT("computing grids\n"); });
      }
      m_remap.compute_grids_phase(
          kv, this->get_source_thickness(kv.ie, m_data.np1, m_elements.m_dp3d),
          Homme::subview(m_tgt_layer_thickness, kv.ie));
    }
  }

  // This asserts if num_to_remap() == 0
  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeRemapTag, const TeamMember &team) const {
    KernelVariables kv(team);
    if (kv.ie == 0) {
      Kokkos::single(Kokkos::PerTeam(kv.team),
                     []() { DEBUG_PRINT("computing remap<true>\n"); });
    }

    assert(num_to_remap() != 0);
    const int var = kv.ie % num_to_remap();
    const int den = (num_to_remap() > 0) ? num_to_remap() : 1;
    kv.ie /= num_to_remap();
    assert(kv.ie < m_data.num_elems);

    auto tgt_layer_thickness = Homme::subview(m_tgt_layer_thickness, kv.ie);
    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness =
        this->get_source_thickness(kv.ie, m_data.np1, m_elements.m_dp3d);

    Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>, remap_dim>
    remap_vals;

    DEBUG_EXPECT(build_remap_array(kv, remap_vals), num_to_remap());

    this->m_remap.compute_remap_phase(kv, var, remap_vals[var]);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeIntrinsicsTag, const TeamMember &team) const {
    KernelVariables kv(team);

    assert(this->num_states_remap != 0);
    const int den = (this->num_states_remap > 0) ? this->num_states_remap : 1;
    const int var = kv.ie % den;
    kv.ie /= den;
    assert(kv.ie < m_data.num_elems);

    auto state_remap = this->remap_states_array(kv, m_elements, m_data.np1);
    auto tgt_layer_thickness = Homme::subview(m_tgt_layer_thickness, kv.ie);
    compute_intrinsic_state(kv, tgt_layer_thickness, state_remap[var]);
  }

  void run_remap() {
    // This runs the remap algorithm after determining it needs to
    // It also verifies the state of the simulation is valid
    // If there's nothing to remap, it will only perform the verification
    run_functor<ComputeThicknessTag>("Remap Compute Grids Functor",
                                     this->m_data.num_elems);
    if (num_to_remap() > 0) {
      // We don't want the latency of launching an empty kernel
      if (nonzero_rsplit) {
        run_functor<ComputeExtrinsicsTag>("Remap Scale States Functor",
                                          m_data.num_elems *
                                              this->num_states_remap);
      }
      run_functor<ComputeGridsTag>("Remap Compute Grids Functor",
                                   this->m_data.num_elems);
      run_functor<ComputeRemapTag>("Remap Compute Remap Functor",
                                   this->m_data.num_elems * num_to_remap());
      if (nonzero_rsplit) {
        run_functor<ComputeIntrinsicsTag>("Remap Rescale States Functor",
                                          m_data.num_elems *
                                              this->num_states_remap);
      }
    }
    this->input_valid_assert();
  }

private:
  template <typename FunctorTag>
  void run_functor(const std::string functor_name, int num_exec) {
    Kokkos::TeamPolicy<ExecSpace, FunctorTag> policy =
        Homme::get_default_team_policy<ExecSpace, FunctorTag>(num_exec);
    // Timers don't work on CUDA, so place them here
    GPTLstart(functor_name.c_str());
    profiling_resume();
    Kokkos::parallel_for("vertical remap", policy, *this);
    ExecSpace::fence();
    profiling_pause();
    GPTLstop(functor_name.c_str());
  }

  KOKKOS_INLINE_FUNCTION
  void compute_ps_v(KernelVariables &kv,
                    ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> dp3d,
                    ExecViewUnmanaged<Real[NP][NP]> ps_v) const {
    start_timer("remap compute_ps_v");
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      // Using parallel_reduce to calculate this doesn't seem to work,
      // even when accumulating into a Scalar and computing the final sum in
      // serial, and is likely a Kokkos bug since ivdep shouldn't matter.
      Kokkos::single(Kokkos::PerThread(kv.team), [&]() {
        ps_v(igp, jgp) = 0.0;
        for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
          const int ilev = level / VECTOR_SIZE;
          const int vlev = level % VECTOR_SIZE;
          ps_v(igp, jgp) += dp3d(igp, jgp, ilev)[vlev];
        }
        ps_v(igp, jgp) += m_data.hybrid_ai0 * m_data.ps0;
      });
    });
    kv.team_barrier();
    stop_timer("remap compute_ps_v");
  }

  KOKKOS_INLINE_FUNCTION void compute_extrinsic_state(
      KernelVariables &kv,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness,
      ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> state_remap) const {
    start_timer("remap rescale_states");
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &ilev) {
        state_remap(igp, jgp, ilev) *= src_layer_thickness(igp, jgp, ilev);
      });
    });
    stop_timer("remap rescale_states");
  }

  KOKKOS_INLINE_FUNCTION
  void compute_intrinsic_state(
      KernelVariables &kv,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness,
      ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> state_remap) const {
    start_timer("remap rescale_states");
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &ilev) {
        state_remap(igp, jgp, ilev) /= tgt_layer_thickness(igp, jgp, ilev);
      });
    });
    stop_timer("remap rescale_states");
  }

  KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]>
  compute_target_thickness(KernelVariables &kv) const {
    start_timer("remap compute_target_thickness");
    auto tgt_layer_thickness = Homme::subview(m_tgt_layer_thickness, kv.ie);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
                           [&](const int &ilevel) {
        const int ilev = ilevel / VECTOR_SIZE;
        const int vec_lev = ilevel % VECTOR_SIZE;
        if (kv.ie == 0 && igp == 0 && jgp == 0) {
          DEBUG_PRINT(
              "%d (%d, %d) ps0: % .17e, ps_v: % .17e, hybrid ai: % .17e, "
              "hybrid bi: % .17e\n",
              ilevel, ilev, vec_lev, m_data.ps0, m_ps_v(kv.ie, igp, jgp),
              m_data.hybrid_ai(ilevel), m_data.hybrid_bi(ilevel));
        }
        tgt_layer_thickness(igp, jgp, ilev)[vec_lev] =
            (m_data.hybrid_ai(ilevel + 1) - m_data.hybrid_ai(ilevel)) *
                m_data.ps0 +
            (m_data.hybrid_bi(ilevel + 1) - m_data.hybrid_bi(ilevel)) *
                m_ps_v(kv.ie, igp, jgp);
      });
    });
    kv.team_barrier();
    stop_timer("remap compute_target_thickness");
    return tgt_layer_thickness;
  }

  KOKKOS_INLINE_FUNCTION void check_source_thickness(
      KernelVariables &kv,
      ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness)
      const {
    start_timer("remap check_source_thickness");
    // Kokkos parallel reduce doesn't support bool as a reduction type, so use
    // int instead
    // Reduce starts with false (0), making that the default state
    // If there is an error, this becomes true
    int invalid = false;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(kv.team, NP * NP * NUM_PHYSICAL_LEV),
        [&](const int &loop_idx, int &is_invalid) {
          const int igp = (loop_idx / NUM_PHYSICAL_LEV) / NP;
          const int jgp = (loop_idx / NUM_PHYSICAL_LEV) % NP;
          const int level = loop_idx % NUM_PHYSICAL_LEV;
          const int ilev = level / VECTOR_SIZE;
          const int vlev = level % VECTOR_SIZE;
          is_invalid |= (src_layer_thickness(igp, jgp, ilev)[vlev] < 0.0);
        },
        invalid);
    valid_layer_thickness(kv.ie) = !invalid;
    stop_timer("remap check_source_thickness");
  }
};

} // namespace Homme

#endif // HOMMEXX_REMAP_FUNCTOR_HPP
