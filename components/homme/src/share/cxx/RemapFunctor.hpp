#ifndef HOMMEXX_REMAP_FUNCTOR_HPP
#define HOMMEXX_REMAP_FUNCTOR_HPP

#include <memory>
#include <type_traits>

#include <Kokkos_Array.hpp>

#include "Control.hpp"
#include "Elements.hpp"
#include "Types.hpp"

#include "profiling.hpp"

namespace Homme {

struct Vert_Remap_Alg {};

struct PPM_Boundary_Conditions {};

// Corresponds to remap alg = 1
template <int _remap_dim> struct PPM_Mirrored : public PPM_Boundary_Conditions {
  static constexpr auto remap_dim = _remap_dim;
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

  static constexpr char *name() { return "Mirrored PPM"; }
};

// Corresponds to remap alg = 2
template <int _remap_dim> struct PPM_Fixed : public PPM_Boundary_Conditions {
  static constexpr auto remap_dim = _remap_dim;
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

  static constexpr char *name() { return "Fixed PPM"; }
};

// Piecewise Parabolic Method stencil
template <typename boundaries> struct PPM_Vert_Remap : public Vert_Remap_Alg {
  static_assert(std::is_base_of<PPM_Boundary_Conditions, boundaries>::value,
                "PPM_Vert_Remap requires a valid PPM "
                "boundary condition");
  static constexpr auto remap_dim = boundaries::remap_dim;

  PPM_Vert_Remap(const Control &data)
      : dpo(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>(
            "dpo", data.num_elems)),
        dpn(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>(
            "dpn", data.num_elems)),
        pio(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2]>(
            "pio", data.num_elems)),
        pin(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>(
            "pin", data.num_elems)),
        ppmdx(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2][10]>(
            "ppmdx", data.num_elems)),
        kid(ExecViewManaged<int * [NP][NP][NUM_PHYSICAL_LEV]>("kid",
                                                              data.num_elems)),
        z1(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]>("z1",
                                                              data.num_elems)),
        z2(ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]>("z2",
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
  void compute_grids(
      KernelVariables &kv,
      const ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]> dx,
      const ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2][10]> grids) const {
    for (int j : boundaries::grid_indices_1()) {
      grids(j, 0) = dx(j + 1) / (dx(j) + dx(j + 1) + dx(j + 2));

      grids(j, 1) = (2.0 * dx(j) + dx(j + 1)) / (dx(j + 1) + dx(j + 2));

      grids(j, 2) = (dx(j + 1) + 2.0 * dx(j + 2)) / (dx(j) + dx(j + 1));
    }

    for (int j : boundaries::grid_indices_2()) {
      grids(j, 3) = dx(j + 1) / (dx(j + 1) + dx(j + 2));

      grids(j, 4) = 1.0 / (dx(j) + dx(j + 1) + dx(j + 2) + dx(j + 3));

      grids(j, 5) = (2.0 * dx(j + 1) * dx(j + 2)) / (dx(j + 1) + dx(j + 2));

      grids(j, 6) = (dx(j) + dx(j + 1)) / (2.0 * dx(j + 1) + dx(j + 2));

      grids(j, 7) = (dx(j + 3) + dx(j + 2)) / (2.0 * dx(j + 2) + dx(j + 1));

      grids(j, 8) =
          dx(j + 1) * (dx(j) + dx(j + 1)) / (2.0 * dx(j + 1) + dx(j + 2));

      grids(j, 9) =
          dx(j + 2) * (dx(j + 2) + dx(j + 3)) / (dx(j + 1) + 2.0 * dx(j + 2));
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
    for (int j : boundaries::ppm_indices_1()) {
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
    }

    for (int j : boundaries::ppm_indices_2()) {
      ai(j) = cell_means(j + 1) +
              dx(j, 3) * (cell_means(j + 2) - cell_means(j + 1)) +
              dx(j, 4) * (dx(j, 5) * (dx(j, 6) - dx(j, 7)) *
                              (cell_means(j + 2) - cell_means(j + 1)) -
                          dx(j, 8) * dma(j + 1) + dx(j, 9) * dma(j));
    }

    for (int j : boundaries::ppm_indices_3()) {
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
      parabola_coeffs(j - 1, 0) = 1.5 * cell_means(j + 1) - (al + ar) / 4.0;
      parabola_coeffs(j - 1, 1) = ar - al;
      parabola_coeffs(j - 1, 2) = -6.0 * cell_means(j + 1) + 3.0 * (al + ar);
    }

    boundaries::apply_ppm_boundary(cell_means, parabola_coeffs);
  }

  KOKKOS_INLINE_FUNCTION
  void
  remap(KernelVariables &kv, const int num_remap,
        ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> src_layer_thickness,
        ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness,
        Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>, remap_dim>
            remap_vals) const {
    constexpr int gs = 2; // ghost cells
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_PHYSICAL_LEV * NP * NP),
        [&](const int &loop_idx) {
          const int k = (loop_idx / NP) / NP;
          const int igp = (loop_idx / NP) % NP;
          const int jgp = loop_idx % NP;
          int ilevel = k / VECTOR_SIZE;
          int ivector = k % VECTOR_SIZE;
          dpn(kv.ie, igp, jgp, k + 2) =
              tgt_layer_thickness(igp, jgp, ilevel)[ivector];
          dpo(kv.ie, igp, jgp, k + 2) =
              src_layer_thickness(igp, jgp, ilevel)[ivector];
        });
    kv.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      pin(kv.ie, igp, jgp, 0) = 0.0;
      pio(kv.ie, igp, jgp, 0) = 0.0;
      // scan region
      for (int k = 1; k <= NUM_PHYSICAL_LEV; k++) {
        pin(kv.ie, igp, jgp, k) =
            pin(kv.ie, igp, jgp, k - 1) + dpn(kv.ie, igp, jgp, k + 1);
        pio(kv.ie, igp, jgp, k) =
            pio(kv.ie, igp, jgp, k - 1) + dpo(kv.ie, igp, jgp, k + 1);
      } // k loop

      // This is here to allow an entire block of k
      // threads to run in the remapping phase. It makes
      // sure there's an old interface value below the
      // domain that is larger.
      pio(kv.ie, igp, jgp, NUM_PHYSICAL_LEV + 1) =
          pio(kv.ie, igp, jgp, NUM_PHYSICAL_LEV) + 1.0;

      // The total mass in a column does not change.
      // Therefore, the pressure of that mass cannot
      // either.
      pin(kv.ie, igp, jgp, NUM_PHYSICAL_LEV) =
          pio(kv.ie, igp, jgp, NUM_PHYSICAL_LEV);
    });
    kv.team_barrier();

    // TODO: Move this above the previous region and verify to reduce
    // team_barrier usage
    //
    // Fill in the ghost regions with mirrored values.
    // if vert_remap_q_alg is defined, this is of no
    // consequence.
    // Note that the range of k makes this completely parallel,
    // without any data dependencies
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, gs * NP * NP),
                         [&](const int &loop_idx) {
      const int k = (loop_idx / NP) / NP;
      const int igp = (loop_idx / NP) % NP;
      const int jgp = loop_idx % NP;
      dpo(kv.ie, igp, jgp, 1 - k) = dpo(kv.ie, igp, jgp, k + 2);
      dpo(kv.ie, igp, jgp, NUM_PHYSICAL_LEV + k + 2) =
          dpo(kv.ie, igp, jgp, NUM_PHYSICAL_LEV + 1 - k);
    });
    kv.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
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
      for (int k = 1; k <= NUM_PHYSICAL_LEV; k++) {
        int kk = k;
        // This reduces the work required to find the index where this fails at,
        // and is typically less than NUM_PHYSICAL_LEV^2
        while (pio(kv.ie, igp, jgp, kk - 1) <= pin(kv.ie, igp, jgp, k)) {
          kk++;
        }

        kk--;
        if (kk == NUM_PHYSICAL_LEV + 1) {
          kk = NUM_PHYSICAL_LEV;
        }
        kid(kv.ie, igp, jgp, k - 1) = kk;
        z1(kv.ie, igp, jgp, k - 1) = -0.5;
        z2(kv.ie, igp, jgp, k - 1) =
            (pin(kv.ie, igp, jgp, k) -
             (pio(kv.ie, igp, jgp, kk - 1) + pio(kv.ie, igp, jgp, kk)) * 0.5) /
            dpo(kv.ie, igp, jgp, kk + 1);
      } // k loop

      ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 4]> point_dpo =
          Homme::subview(dpo, kv.ie, igp, jgp);
      ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2][10]> point_ppmdx =
          Homme::subview(ppmdx, kv.ie, igp, jgp);

      compute_grids(kv, point_dpo, point_ppmdx);
    });
    kv.team_barrier();
    // Verified to here

    // More parallelism than we need here, maybe break it up into a
    // ThreadVectorRange as well?
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, num_remap * NP * NP),
                         [&](const int &loop_idx) {
      const int var = (loop_idx / NP) / NP;
      const int igp = (loop_idx / NP) % NP;
      const int jgp = loop_idx % NP;

      mass_o[var](kv.ie, igp, jgp, 0) = 0.0;
      for (int k = 0; k < NUM_PHYSICAL_LEV; k++) {
        const int ilevel = k / VECTOR_SIZE;
        const int ivector = k % VECTOR_SIZE;
        ao[var](kv.ie, igp, jgp, k + 2) =
            remap_vals[var](igp, jgp, ilevel)[ivector];
        mass_o[var](kv.ie, igp, jgp, k + 1) =
            mass_o[var](kv.ie, igp, jgp, k) + ao[var](kv.ie, igp, jgp, k + 2);
        ao[var](kv.ie, igp, jgp, k + 2) /= dpo(kv.ie, igp, jgp, k + 2);
      } // end k loop

      for (int k = 1; k <= gs; k++) {
        ao[var](kv.ie, igp, jgp, 1 - k + 1) = ao[var](kv.ie, igp, jgp, k + 1);
        ao[var](kv.ie, igp, jgp, NUM_PHYSICAL_LEV + k + 1) =
            ao[var](kv.ie, igp, jgp, NUM_PHYSICAL_LEV + 1 - k + 1);
      } // k loop

      compute_ppm(kv, Homme::subview(ao[var], kv.ie, igp, jgp),
                  Homme::subview(ppmdx, kv.ie, igp, jgp),
                  Homme::subview(dma[var], kv.ie, igp, jgp),
                  Homme::subview(ai[var], kv.ie, igp, jgp),
                  Homme::subview(parabola_coeffs[var], kv.ie, igp, jgp));

      if (kv.ie == 0 && var == 0 && igp == 0 && jgp == 0) {
        for(int k = 0; k < NUM_PHYSICAL_LEV
      }

      Real massn1 = 0.0;

      // Maybe just recompute massn1 and double our work
      // to get significantly more threads?
      for (int k = 0; k < NUM_PHYSICAL_LEV; k++) {
        const int ilevel = k / VECTOR_SIZE;
        const int ivector = k % VECTOR_SIZE;
        const int kk = kid(kv.ie, igp, jgp, k);
        const Real a0 = parabola_coeffs[var](kv.ie, igp, jgp, kk - 1, 0),
                   a1 = parabola_coeffs[var](kv.ie, igp, jgp, kk - 1, 1),
                   a2 = parabola_coeffs[var](kv.ie, igp, jgp, kk - 1, 2);
        const Real x1 = z1(kv.ie, igp, jgp, k);
        const Real x2 = z2(kv.ie, igp, jgp, k);
        // to make this bfb with F,  divide by 2
        // change F later
        const Real integrate_par = a0 * (x2 - x1) +
                                   a1 * (x2 * x2 - x1 * x1) / 2.0 +
                                   a2 * (x2 * x2 * x2 - x1 * x1 * x1) / 3.0;
        const Real massn2 = mass_o[var](kv.ie, igp, jgp, kk - 1) +
                            integrate_par * dpo(kv.ie, igp, jgp, kk + 1);
        remap_vals[var](igp, jgp, ilevel)[ivector] = massn2 - massn1;
        massn1 = massn2;
      } // k loop
    }); // End team thread range
  }

  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>,
                remap_dim> ao;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]> dpo;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]> dpn;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2]> pio;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]> pin;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2][10]> ppmdx;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>,
                remap_dim> mass_o;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]> z1;
  ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]> z2;
  ExecViewManaged<int * [NP][NP][NUM_PHYSICAL_LEV]> kid;

  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2]>,
                remap_dim> dma;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>,
                remap_dim> ai;

  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV][3]>,
                remap_dim> parabola_coeffs;
};

template <typename remap_type> struct Remap_Functor {
  static_assert(std::is_base_of<Vert_Remap_Alg, remap_type>::value,
                "RemapFunctor not given a remap algorithm to use");

  Control m_data;
  const Elements m_elements;

  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> tgt_layer_thickness;

  // Surface pressure
  ExecViewManaged<Real * [NP][NP]> ps_v;

  // ???
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> dp;

  remap_type remap;

  Remap_Functor(const Control &data, const Elements &elements)
      : m_data(data), m_elements(elements),
        tgt_layer_thickness("Target layer thickness", data.num_elems),
        ps_v("Surface pressure", data.num_elems), dp("dp", data.num_elems),
        remap(data) {}

  // Just implemented for CUDA until Kyungjoo's work is
  // finished
  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMember &team) const {
    start_timer("Remap functor");
    KernelVariables kv(team);

    int num_remap = 0;
    Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>,
                  remap_type::remap_dim> remap_vals;

    // The states which need to be remapped
    Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>, 3> state_remap{
      { Homme::subview(m_elements.m_u, kv.ie, m_data.np1),
        Homme::subview(m_elements.m_v, kv.ie, m_data.np1),
        Homme::subview(m_elements.m_t, kv.ie, m_data.np1) }
    };
    if (m_data.rsplit == 0) {
      // No remapping here, just dpstar check
    } else {
      // Compute ps_v separately, as it requires a parallel
      // reduce
      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                           [&](const int &loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;

        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
            [&](const int &ilevel, Real &accumulator) {
              const int ilev = ilevel / VECTOR_SIZE;
              const int vec_lev = ilevel % VECTOR_SIZE;
              accumulator +=
                  m_elements.m_dp3d(kv.ie, m_data.np1, igp, jgp, ilev)[vec_lev];
            },
            ps_v(kv.ie, igp, jgp));
        ps_v(kv.ie, igp, jgp) += m_data.hybrid_a(0) * m_data.ps0;
      });
      kv.team_barrier();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                           [&](const int &idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;

        // Until Kyungjoo's vectorization works on CUDA
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,
                                                       NUM_PHYSICAL_LEV),
                             [&](const int &ilevel) {
          const int ilev = ilevel / VECTOR_SIZE;
          const int vec_lev = ilevel % VECTOR_SIZE;
          dp(kv.ie, igp, jgp, ilev)[vec_lev] =
              (m_data.hybrid_a(ilevel + 1) - m_data.hybrid_a(ilevel)) *
                  m_data.ps0 +
              (m_data.hybrid_b(ilevel + 1) - m_data.hybrid_b(ilevel)) *
                  ps_v(kv.ie, igp, jgp, ilevel);

          tgt_layer_thickness(kv.ie, igp, jgp, ilev) =
              (m_data.hybrid_a(ilev + 1) - m_data.hybrid_a(ilev) * m_data.ps0) +
              (m_data.hybrid_b(ilev + 1) - m_data.hybrid_b(ilev)) *
                  ps_v(kv.ie, igp, jgp, ilev);
        });
      });

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(kv.team, state_remap.size()),
          [&](const int &var) { remap_vals[var] = state_remap[var]; });

      kv.team_barrier();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(
                               kv.team, state_remap.size() * NP * NP * NUM_LEV),
                           [&](const int &loop_idx) {
        const int var = ((loop_idx / NUM_LEV) / NP) / NP;
        const int igp = ((loop_idx / NUM_LEV) / NP) % NP;
        const int jgp = (loop_idx / NUM_LEV) % NP;
        const int ilev = loop_idx % NUM_LEV;
        remap_vals[var](igp, jgp, ilev) *=
            m_elements.m_dp3d(kv.ie, m_data.np1, igp, jgp, ilev);
      });
      kv.team_barrier();
    } // rsplit == 0

    if (m_data.qsize > 0) {
      // remap_Q_ppm Qdp
      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, m_data.qsize),
                           [&](const int &i) {
        // Need to verify this is the right tracer
        // timelevel
        remap_vals[num_remap + i] =
            Homme::subview(m_elements.m_qdp, kv.ie, m_data.qn0, i);
      });
      num_remap += m_data.qsize;
      kv.team_barrier();
    }

    if (num_remap > 0) {
      remap.remap(kv, num_remap,
                  Homme::subview(m_elements.m_dp3d, kv.ie, m_data.np1),
                  Homme::subview(tgt_layer_thickness, kv.ie), remap_vals);
      kv.team_barrier();
    }
    if (m_data.rsplit != 0) {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(kv.team, state_remap.size() * NP * NP),
          [&](const int &loop_idx) {
            const int var = (loop_idx / NP) / NP;
            const int igp = (loop_idx / NP) % NP;
            const int jgp = loop_idx % NP;
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                                 [&](const int &ilev) {
              state_remap[var](igp, jgp, ilev) /= dp(kv.ie, igp, jgp, ilev);
            });
          });
    }

    stop_timer("Remap functor");
  }

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size(const int team_size) const {
    return KernelVariables::shmem_size(team_size);
  }
};
} // namespace Homme

#endif // HOMMEXX_REMAP_FUNCTOR_HPP
