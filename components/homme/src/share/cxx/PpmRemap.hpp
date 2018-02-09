#ifndef HOMMEXX_PPM_REMAP_HPP
#define HOMMEXX_PPM_REMAP_HPP

#include "ErrorDefs.hpp"

#include "RemapFunctor.hpp"

#include "Control.hpp"
#include "Elements.hpp"
#include "KernelVariables.hpp"
#include "Types.hpp"
#include "utilities/LoopsUtils.hpp"
#include "utilities/MathUtils.hpp"
#include "utilities/SubviewUtils.hpp"
#include "utilities/SyncUtils.hpp"

#include "profiling.hpp"

namespace Homme {
namespace Remap {
namespace Ppm {

static constexpr int AO_PHYSICAL_LEV = NUM_PHYSICAL_LEV + 4;
static constexpr int AO_LEV = AO_PHYSICAL_LEV / VECTOR_SIZE;

static constexpr int DPO_PHYSICAL_LEV = NUM_PHYSICAL_LEV + 4;
static constexpr int DPO_LEV = DPO_PHYSICAL_LEV / VECTOR_SIZE;

static constexpr int PIO_PHYSICAL_LEV = NUM_PHYSICAL_LEV + 2;
static constexpr int PIO_LEV = PIO_PHYSICAL_LEV / VECTOR_SIZE;

static constexpr int PIN_PHYSICAL_LEV = NUM_PHYSICAL_LEV + 1;
static constexpr int PIN_LEV = PIN_PHYSICAL_LEV / VECTOR_SIZE;

static constexpr int PPMDX_PHYSICAL_LEV = NUM_PHYSICAL_LEV + 2;
static constexpr int PPMDX_LEV = PPMDX_PHYSICAL_LEV / VECTOR_SIZE;

static constexpr int MASS_O_PHYSICAL_LEV = NUM_PHYSICAL_LEV + 1;
static constexpr int MASS_O_LEV = MASS_O_PHYSICAL_LEV / VECTOR_SIZE;

static constexpr int DMA_PHYSICAL_LEV = NUM_PHYSICAL_LEV + 2;
static constexpr int DMA_LEV = DMA_PHYSICAL_LEV / VECTOR_SIZE;

static constexpr int AI_PHYSICAL_LEV = NUM_PHYSICAL_LEV + 1;
static constexpr int AI_LEV = AI_PHYSICAL_LEV / VECTOR_SIZE;

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
      ExecViewUnmanaged<const Real[AO_PHYSICAL_LEV]> cell_means,
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
      ExecViewUnmanaged<const Real[AO_PHYSICAL_LEV]> cell_means,
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
      : dpo(ExecViewManaged<Real * [NP][NP][DPO_PHYSICAL_LEV]>("dpo",
                                                               data.num_elems)),
        pio(ExecViewManaged<Real * [NP][NP][PIO_PHYSICAL_LEV]>("pio",
                                                               data.num_elems)),
        pin(ExecViewManaged<Real * [NP][NP][PIN_PHYSICAL_LEV]>("pin",
                                                               data.num_elems)),
        ppmdx(ExecViewManaged<Real * [NP][NP][PPMDX_PHYSICAL_LEV][10]>(
            "ppmdx", data.num_elems)),
        z2(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]>("z2",
                                                              data.num_elems)),
        kid(ExecViewManaged<int * [NP][NP][NUM_PHYSICAL_LEV]>("kid",
                                                              data.num_elems)) {
    for (int i = 0; i < remap_dim; ++i) {
      ao[i] = ExecViewManaged<Real * [NP][NP][AO_PHYSICAL_LEV]>("a0",
                                                                data.num_elems);
      mass_o[i] = ExecViewManaged<Real * [NP][NP][MASS_O_PHYSICAL_LEV]>(
          "mass_o", data.num_elems);
      dma[i] = ExecViewManaged<Real * [NP][NP][DMA_PHYSICAL_LEV]>(
          "dma", data.num_elems);
      ai[i] = ExecViewManaged<Real * [NP][NP][AI_PHYSICAL_LEV]>("ai",
                                                                data.num_elems);
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
      ExecViewUnmanaged<const Real[MASS_O_PHYSICAL_LEV]> prev_mass,
      ExecViewUnmanaged<const Real[DPO_PHYSICAL_LEV]> prev_dp,
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
      const ExecViewUnmanaged<const Real[DPO_PHYSICAL_LEV]> dx,
      const ExecViewUnmanaged<Real[PPMDX_PHYSICAL_LEV][10]> grids) const {
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
  void compute_ppm(KernelVariables &kv,
                   // input  views
                   ExecViewUnmanaged<const Real[AO_PHYSICAL_LEV]> cell_means,
                   ExecViewUnmanaged<const Real[PPMDX_PHYSICAL_LEV][10]> dx,
                   // buffer views
                   ExecViewUnmanaged<Real[DMA_PHYSICAL_LEV]> dma,
                   ExecViewUnmanaged<Real[AI_PHYSICAL_LEV]> ai,
                   // result view
                   ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV][3]> parabola_coeffs)
      const {
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

          dma(j) = min(fabs(da), 2.0 * fabs(cell_means(j + 1) - cell_means(j)),
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

          parabola_coeffs(j - 1, 0) = 1.5 * cell_means(j + 1) - (al + ar) / 4.0;
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
        pio(kv.ie, igp, jgp, PIO_PHYSICAL_LEV - 1) =
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
        assert(pio(kv.ie, igp, jgp, PIO_PHYSICAL_LEV - 1) >
               pin(kv.ie, igp, jgp, k + 1));
        while (pio(kv.ie, igp, jgp, kk - 1) <= pin(kv.ie, igp, jgp, k + 1)) {
          kk++;
          assert(kk - 1 < pio.extent_int(3));
        }

        kk--;
        // This is to keep the indices in bounds.
        if (kk == PIN_PHYSICAL_LEV) {
          kk = PIN_PHYSICAL_LEV - 1;
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

      ExecViewUnmanaged<Real[DPO_PHYSICAL_LEV]> point_dpo =
          Homme::subview(dpo, kv.ie, igp, jgp);
      ExecViewUnmanaged<Real[PPMDX_PHYSICAL_LEV][10]> point_ppmdx =
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

  Kokkos::Array<ExecViewManaged<Real * [NP][NP][AO_PHYSICAL_LEV]>, remap_dim>
  ao;
  ExecViewManaged<Real * [NP][NP][DPO_PHYSICAL_LEV]> dpo;
  // pio corresponds to the points in each layer of the source layer thickness
  ExecViewManaged<Real * [NP][NP][PIO_PHYSICAL_LEV]> pio;
  // pin corresponds to the points in each layer of the target layer thickness
  ExecViewManaged<Real * [NP][NP][PIN_PHYSICAL_LEV]> pin;
  ExecViewManaged<Real * [NP][NP][PPMDX_PHYSICAL_LEV][10]> ppmdx;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]> z2;
  ExecViewManaged<int * [NP][NP][NUM_PHYSICAL_LEV]> kid;

  Kokkos::Array<ExecViewManaged<Real * [NP][NP][MASS_O_PHYSICAL_LEV]>,
                remap_dim> mass_o;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][DMA_PHYSICAL_LEV]>, remap_dim>
  dma;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][AI_PHYSICAL_LEV]>, remap_dim>
  ai;

  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV][3]>,
                remap_dim> parabola_coeffs;
};

} // namespace Ppm
} // namespace Remap
} // namespace Homme

#endif // HOMMEXX_PPM_REMAP_HPP
