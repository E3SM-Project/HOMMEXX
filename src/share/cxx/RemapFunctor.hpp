#ifndef HOMMEXX_REMAP_FUNCTOR_HPP
#define HOMMEXX_REMAP_FUNCTOR_HPP

#include <type_traits>
#include <memory>

#include <Kokkos_Array.hpp>

#include "Types.hpp"

#include "profiling.hpp"

namespace Homme {

struct Vert_Remap_Alg {};

struct PPM_Boundary_Conditions {};

// Corresponds to remap alg = 1
template <int _remap_dim> struct PPM_Mirrored : public PPM_Boundary_Conditions {
  static constexpr auto remap_dim = _remap_dim;

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> grid_indices_1() {
    return Loop_Range<int>(0, NUM_PHYSICAL_LEV);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> grid_indices_2() {
    return Loop_Range<int>(2, NUM_PHYSICAL_LEV + 1);
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
      KernelVariables &kv,
      Kokkos::Array<ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV]>, remap_dim>
          cell_means, Kokkos::Array<ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV]>,
                                    remap_dim> &parabola_coeffs) {}
};

// Corresponds to remap alg = 2
template <int _remap_dim> struct PPM_Fixed : public PPM_Boundary_Conditions {
  static constexpr auto remap_dim = _remap_dim;

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> grid_indices_1() {
    return Loop_Range<int>(0, NUM_PHYSICAL_LEV);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Loop_Range<int> grid_indices_2() {
    return Loop_Range<int>(2, NUM_PHYSICAL_LEV + 1);
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
      ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV]> cell_means,
      ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV]> parabola_coeffs) {
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
};

// Piecewise Parabolic Method stencil
template <typename boundaries> struct PPM_Vert_Remap : public Vert_Remap_Alg {
  static_assert(std::is_base_of<PPM_Boundary_Conditions, boundaries>::value,
                "PPM_Vert_Remap requires a valid PPM boundary condition");
  static constexpr auto remap_dim = boundaries::remap_dim;

  PPM_Vert_Remap(const Control &data) {
    for (int i = 0; i < remap_dim; ++i) {
      ao[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>(
          "a0", data.num_elems);
      dpo[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>(
          "dpo", data.num_elems);
      dpn[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>(
          "dpn", data.num_elems);
      pio[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2]>(
          "pio", data.num_elems);
      pin[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>(
          "pin", data.num_elems);
      mass_o[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>(
          "mass_o", data.num_elems);
      z1[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]>(
          "z1", data.num_elems);
      z2[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]>(
          "z2", data.num_elems);
      ppmdx[i] = ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2][10]>(
          "ppmdx", data.num_elems);
      kid[i] = ExecViewManaged<int * [NP][NP][NUM_PHYSICAL_LEV]>(
          "kid", data.num_elems);
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
  static void
  compute_grids(KernelVariables &kv,
                const ExecViewUnmanaged<const Real[NUM_PHYSICAL_LEV + 4]> dx,
                const ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV + 2][10]> grids) {
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

      grids(j, 7) = (2.0 * dx(j + 1) * dx(j + 2)) / (dx(j + 1) + dx(j + 2));

      grids(j, 8) =
          dx(j + 1) * (dx(j) + dx(j + 1)) / (2.0 * dx(j + 1) + dx(j + 2));

      grids(j, 9) =
          dx(j + 2) * (dx(j + 2) + dx(j + 3)) / (dx(j + 1) + 2.0 * dx(j + 2));
    }
  }

  KOKKOS_INLINE_FUNCTION
  static void
  compute_ppm(KernelVariables &kv,
              ExecViewUnmanaged<const Real * [NUM_PHYSICAL_LEV + 4]> cell_means,
              ExecViewUnmanaged<const Real * [NUM_PHYSICAL_LEV + 2][10]> dx,
              ExecViewUnmanaged<Real * [NUM_PHYSICAL_LEV + 2]> dma,
              ExecViewUnmanaged<Real * [NUM_PHYSICAL_LEV + 1]> ai,
              ExecViewUnmanaged<Real * [NUM_PHYSICAL_LEV][3]> parabola_coeffs) {
    for (int j : boundaries::ppm_indices_1()) {
      if ((cell_means(kv.ie, j + 2) - cell_means(kv.ie, j + 1)) *
              (cell_means(kv.ie, j + 1) - cell_means(kv.ie, j)) >
          0.0) {
        Real da = dx(kv.ie, j, 0) *
                  (dx(kv.ie, j, 1) *
                       (cell_means(kv.ie, j + 2) - cell_means(kv.ie, j + 1)) +
                   dx(kv.ie, j, 2) *
                       (cell_means(kv.ie, j + 1) - cell_means(kv.ie, j)));
        dma(kv.ie, j) = min(min(fabs(da), 2.0 * fabs(cell_means(kv.ie, j + 1) -
                                                     cell_means(kv.ie, j))),
                            2.0 * fabs(cell_means(kv.ie, j + 2) -
                                       cell_means(kv.ie, j + 1))) *
                        copysign(1.0, da);
      } else {
        dma(kv.ie, j) = 0.0;
      }
    }

    for (int j : boundaries::ppm_indices_2()) {
      ai(kv.ie, j) =
          cell_means(kv.ie, j + 1) +
          dx(kv.ie, j, 3) *
              (cell_means(kv.ie, j + 2) - cell_means(kv.ie, j + 1)) +
          dx(kv.ie, j, 4) *
              (dx(kv.ie, j, 5) * (dx(kv.ie, j, 6) - dx(kv.ie, j, 7)) *
                   (cell_means(kv.ie, j + 2) - cell_means(kv.ie, j + 1)) -
               dx(kv.ie, j, 8) * dma(kv.ie, j + 1) +
               dx(kv.ie, j, 9) * dma(kv.ie, j));
    }

    for (int j : boundaries::ppm_indices_3()) {
      Real al = ai(kv.ie, j - 1);
      Real ar = ai(kv.ie, j);
      if ((ar - cell_means(kv.ie, j + 1)) * (cell_means(kv.ie, j + 1) - al) <=
          0.) {
        al = cell_means(kv.ie, j + 1);
        ar = cell_means(kv.ie, j + 1);
      }
      if ((ar - al) * (cell_means(kv.ie, j + 1) - (al + ar) / 2.0) >
          (ar - al) * (ar - al) / 6.0) {
        al = 3.0 * cell_means(kv.ie, j + 1) - 2.0 * ar;
      }
      if ((ar - al) * (cell_means(kv.ie, j + 1) - (al + ar) / 2.0) <
          -(ar - al) * (ar - al) / 6.0) {
        ar = 3.0 * cell_means(kv.ie, j + 1) - 2.0 * al;
      }

      // Computed these coefficients from the edge values and
      // cell mean in Maple. Assumes normalized coordinates:
      // xi=(x-x0)/dx
      parabola_coeffs(kv.ie, j - 1, 0) =
          1.5 * cell_means(kv.ie, j + 1) - (al + ar) / 4.0;
      parabola_coeffs(kv.ie, j - 1, 1) = ar - al;
      parabola_coeffs(kv.ie, j - 1, 2) =
          -6.0 * cell_means(kv.ie, j + 1) + 3.0 * (al + ar);
    }

    boundaries::apply_ppm_boundary(kv, cell_means, parabola_coeffs);
  }

  KOKKOS_INLINE_FUNCTION
  void remap(KernelVariables &kv, const int &num_remap,
             ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> src_layer_thickness,
             ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> tgt_layer_thickness,
             Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>,
                           remap_dim> remap_vals) {
    constexpr int gs = 2; // ?
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, num_remap * NP * NP),
                         [&](const int &loop_idx) {
      const int var = loop_idx / (NP * NP);
      const int igp = (loop_idx / NP) % NP;
      const int jgp = loop_idx % NP;
      pin[var](kv.ie, igp, jgp, 0) = 0.0;
      pio[var](kv.ie, igp, jgp, 0) = 0.0;
      for (int k = 1; k <= NUM_PHYSICAL_LEV; k++) {
        int ilevel = (k - 1) / VECTOR_SIZE;
        int ivector = (k - 1) / VECTOR_SIZE;
        dpn[var](kv.ie, igp, jgp, k + 1) =
            tgt_layer_thickness(ilevel, igp, jgp, ilevel)[ivector];
        dpo[var](kv.ie, igp, jgp, k + 1) =
            src_layer_thickness(ilevel, igp, jgp, ilevel)[ivector];
        pin[var](kv.ie, igp, jgp, k) =
            pin[var](kv.ie, igp, jgp, k - 1) + dpn[var](kv.ie, igp, jgp, k + 1);
        pio[var](kv.ie, igp, jgp, k) =
            pio[var](kv.ie, igp, jgp, k - 1) + dpo[var](kv.ie, igp, jgp, k + 1);
      } // k loop

      // This is here to allow an entire block of k threads to run in the
      // remapping phase.
      // It makes sure there's an old interface value below the domain that is
      // larger.
      pio[var](kv.ie, igp, jgp, NUM_PHYSICAL_LEV + 1) =
          pio[var](kv.ie, igp, jgp, NUM_PHYSICAL_LEV) + 1.0;

      // The total mass in a column does not change.
      // Therefore, the pressure of that mass cannot either.
      pin[var](kv.ie, igp, jgp, NUM_PHYSICAL_LEV) =
          pio[var](kv.ie, igp, jgp, NUM_PHYSICAL_LEV);

      // Fill in the ghost regions with mirrored values.
      // if vert_remap_q_alg is defined, this is of no consequence.
      for (int k = 1; k <= gs; k++) {
        dpo[var](kv.ie, igp, jgp, 1 - k + 1) = dpo[var](kv.ie, igp, jgp, k + 1);
        dpo[var](kv.ie, igp, jgp, NUM_PHYSICAL_LEV + k + 1) =
            dpo[var](kv.ie, igp, jgp, NUM_PHYSICAL_LEV + 1 - k + 1);
      } // end k loop

      // Compute remapping intervals once for all tracers. Find the old grid
      // cell index in which the
      // k-th new cell interface resides. Then integrate from the bottom of
      // that old cell to the new
      // interface location. In practice, the grid never deforms past one
      // cell, so the search can be
      // simplified by this. Also, the interval of integration is usually of
      // magnitude close to zero
      // or close to dpo because of minimial deformation.
      // Numerous tests confirmed that the bottom and top of the grids match
      // to machine precision, so
      // I set them equal to each other.
      for (int k = 1; k <= NUM_PHYSICAL_LEV; k++) {
        int kk = k;
        while (pio[var](kv.ie, igp, jgp, kk - 1) <=
               pin[var](kv.ie, igp, jgp, k)) {
          kk++;
        }

        kk--;
        if (kk == NUM_PHYSICAL_LEV + 1) {
          kk = NUM_PHYSICAL_LEV;
        }
        kid[var](kv.ie, igp, jgp, k - 1) = kk;
        z1[var](kv.ie, igp, jgp, k - 1) = -0.5;
        z2[var](kv.ie, igp, jgp, k - 1) =
            (pin[var](kv.ie, igp, jgp, k) - (pio[var](kv.ie, igp, jgp, kk - 1) +
                                             pio[var](kv.ie, igp, jgp, kk)) *
                                                0.5) /
            dpo[var](kv.ie, igp, jgp, kk + 1);
      } // k loop

      compute_grids(kv, Homme::subview(dpo[var], kv.ie, igp, jgp),
                    Homme::subview(ppmdx[var], kv.ie, igp, jgp));

      for (int q = 0; q < num_remap; q++) {
        mass_o[var](kv.ie, igp, jgp, 0) = 0.0;
        for (int k = 0; k < NUM_PHYSICAL_LEV; k++) {
          const int ilevel = k / VECTOR_SIZE;
          const int ivector = k % VECTOR_SIZE;
          ao[var](kv.ie, k + 2) =
              remap_vals[var](kv.ie, q, igp, jgp, ilevel)[ivector];
          mass_o[var](kv.ie, igp, jgp, k + 1) =
              mass_o[var](kv.ie, igp, jgp, k) + ao[var](kv.ie, k + 2);
          ao[var](kv.ie, igp, jgp, k + 2) /= dpo[var](kv.ie, igp, jgp, k + 2);
        } // end k loop

        for (int k = 1; k <= gs; k++) {
          ao[var](kv.ie, igp, jgp, 1 - k + 1) = ao[var](kv.ie, igp, jgp, k + 1);
          ao[var](kv.ie, igp, jgp, NUM_PHYSICAL_LEV + k + 1) =
              ao[var](kv.ie, igp, jgp, NUM_PHYSICAL_LEV + 1 - k + 1);
        } // k loop

        compute_ppm(kv, Homme::subview(ao[var], kv.ie, igp, jgp),
                    Homme::subview(ppmdx[var], igp, jgp),
                    Homme::subview(dma[var], igp, jgp),
                    Homme::subview(ai[var], igp, jgp),
                    Homme::subview(parabola_coeffs[var], igp, jgp));

        Real massn1 = 0.0;

        // Maybe just recompute massn1 and double our work to get significantly
        // more threads?
        for (int k = 0; k < NUM_PHYSICAL_LEV; k++) {
          const int kk = kid[var](kv.ie, k);
          const Real a0 = parabola_coeffs[var](kv.ie, igp, jgp, kk - 1, 0),
                     a1 = parabola_coeffs[var](kv.ie, igp, jgp, kk - 1, 1),
                     a2 = parabola_coeffs[var](kv.ie, igp, jgp, kk - 1, 2);
          const Real x1 = z1[var](kv.ie, igp, jgp, k);
          const Real x2 = z2[var](kv.ie, igp, jgp, k);
          // to make this bfb with F,  divide by 2
          // change F later
          const Real integrate_par = a0 * (x2 - x1) +
                                     a1 * (x2 * x2 - x1 * x1) / 2.0 +
                                     a2 * (x2 * x2 * x2 - x1 * x1 * x1) / 3.0;
          const Real massn2 = mass_o[var](kv.ie, igp, jgp, kk - 1) +
                              integrate_par * dpo[var](kv.ie, igp, jgp, kk + 1);
          remap_vals[var](kv.ie, q, igp, jgp, k) = massn2 - massn1;
          massn1 = massn2;
        } // k loop
      }   // end q loop
    });   // End team thread range
  }

private:
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>,
                remap_dim> ao;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>,
                remap_dim> dpo;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 4]>,
                remap_dim> dpn;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2]>,
                remap_dim> pio;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>,
                remap_dim> pin;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 1]>,
                remap_dim> mass_o;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]>, remap_dim>
  z1;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV]>, remap_dim>
  z2;
  Kokkos::Array<ExecViewManaged<Real * [NP][NP][NUM_PHYSICAL_LEV + 2][10]>,
                remap_dim> ppmdx;
  Kokkos::Array<ExecViewManaged<int * [NP][NP][NUM_PHYSICAL_LEV]>, remap_dim>
  kid;

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

  remap_type remap;

  Remap_Functor(const Control &data, const Elements &elements)
      : m_data(data), m_elements(elements), remap(data) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMember &team) {
    start_timer("Remap functor");
    KernelVariables kv(team);

    int num_remap = 0;
    Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>,
                  remap_type::remap_dim> remap_vals;
    if (m_data.rsplit == 0) {
      // No remapping here, just dpstar check
    } else {
      remap_vals[0] = Homme::subview(m_elements.m_u, kv.ie, m_data.np1);
      remap_vals[1] = Homme::subview(m_elements.m_v, kv.ie, m_data.np1);
      remap_vals[2] = Homme::subview(m_elements.m_t, kv.ie, m_data.np1);
      num_remap = 3;
    }

    if (m_data.qsize > 0) {
      // remap_Q_ppm Qdp
      for (int i = 0; i < m_data.qsize; i++, num_remap++) {
        // Need to verify this is the right tracer timelevel
        remap_vals[num_remap] =
            Homme::subview(m_elements.m_qdp, kv.ie, m_data.qn0, i);
      }
    }
    if (num_remap > 0) {
      remap_type::remap(kv, num_remap,
                        Homme::subview(m_elements.m_dp3d, kv.ie, m_data.np1),
                        remap_vals);
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
