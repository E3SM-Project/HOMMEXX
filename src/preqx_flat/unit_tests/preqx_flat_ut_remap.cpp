#include <catch/catch.hpp>

#include "dimensions_remap_tests.hpp"

#include "remap.hpp"
#include "utilities/SubviewUtils.hpp"
#include "utilities/TestUtils.hpp"

#include "KernelVariables.hpp"
#include "Types.hpp"
#include "PpmRemap.hpp"
#include "RemapFunctor.hpp"
#include "Types.hpp"

#include <algorithm>
#include <assert.h>
#include <limits>
#include <random>
#include <stdio.h>

using namespace Homme;
using namespace Remap;
using namespace Ppm;

using rngAlg = std::mt19937_64;

extern "C" {

// sort out const here
void compute_ppm_grids_c_callable(const Real *dx, Real *rslt, const int &alg);

void compute_ppm_c_callable(const Real *a, const Real *dx, Real *coefs,
                            const int &alg);

// F.o object files have only small letters in names
void remap_q_ppm_c_callable(Real *Qdp, const int &nx, const int &qsize,
                            const Real *dp1, const Real *dp2, const int &alg);

}; // extern C

/* This object is meant for testing different configurations of the PPM vertical
 * remap method.
 * boundary_cond needs to be one of the PPM boundary condition objects,
 * which provide the indexes to loop over
 */
template <typename boundary_cond, int _remap_dim> class ppm_remap_functor_test {
  static_assert(std::is_base_of<PpmBoundaryConditions, boundary_cond>::value,
                "PPM Remap test must have a valid boundary condition");

public:
  static constexpr int num_remap = _remap_dim;

  ppm_remap_functor_test(const int num_elems)
      : ne(num_elems), remap(num_elems),
        src_layer_thickness_kokkos("source layer thickness", num_elems),
        tgt_layer_thickness_kokkos("target layer thickness", num_elems) {
    for (int var = 0; var < num_remap; ++var) {
      remap_vals[var] =
          ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("remap var", num_elems);
    }
  }

  struct TagGridTest {};
  struct TagPPMTest {};
  struct TagRemapTest {};

  void test_grid() {
    std::random_device rd;
    rngAlg engine(rd());
    genRandArray(remap.dpo, engine,
                 std::uniform_real_distribution<Real>(0.125, 1000));
    Kokkos::parallel_for(
        Homme::get_default_team_policy<ExecSpace, TagGridTest>(ne), *this);
    ExecSpace::fence();

    const int remap_alg = boundary_cond::fortran_remap_alg;
    HostViewManaged<Real[NUM_PHYSICAL_LEV + 4]> f90_input("fortran dpo");
    HostViewManaged<Real[NUM_PHYSICAL_LEV + 2][10]> f90_result("fortra ppmdx");
    for (int var = 0; var < num_remap; ++var) {
      auto kokkos_result = Kokkos::create_mirror_view(remap.ppmdx);
      Kokkos::deep_copy(kokkos_result, remap.ppmdx);
      for (int ie = 0; ie < ne; ++ie) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            Kokkos::deep_copy(f90_input,
                              Homme::subview(remap.dpo, ie, igp, jgp));
            compute_ppm_grids_c_callable(f90_input.data(), f90_result.data(),
                                         remap_alg);
            for (int k = 0; k < f90_result.extent_int(0); ++k) {
              for (int stencil_idx = 0; stencil_idx < f90_result.extent_int(1);
                   ++stencil_idx) {
                REQUIRE(!std::isnan(f90_result(k, stencil_idx)));
                REQUIRE(
                    !std::isnan(kokkos_result(ie, igp, jgp, stencil_idx, k)));
                REQUIRE(f90_result(k, stencil_idx) ==
                        kokkos_result(ie, igp, jgp, stencil_idx, k));
              }
            }
          }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagGridTest &, TeamMember team) const {
    KernelVariables kv(team);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NP * NP), [&](const int &loop_idx) {
          const int igp = loop_idx / NP;
          const int jgp = loop_idx % NP;
          remap.compute_grids(kv, Homme::subview(remap.dpo, kv.ie, igp, jgp),
                              Homme::subview(remap.ppmdx, kv.ie, igp, jgp));
        });
  }

  void test_ppm() {
    std::random_device rd;
    rngAlg engine(rd());
    genRandArray(remap.ppmdx, engine,
                 std::uniform_real_distribution<Real>(0.125, 1000));
    for (int i = 0; i < num_remap; ++i) {
      genRandArray(remap.ao[i], engine,
                   std::uniform_real_distribution<Real>(0.125, 1000));
    }
    Kokkos::parallel_for(
        Homme::get_default_team_policy<ExecSpace, TagPPMTest>(ne), *this);
    ExecSpace::fence();

    const int remap_alg = boundary_cond::fortran_remap_alg;

    HostViewManaged<Real[_ppm_consts::AO_PHYSICAL_LEV]> f90_cellmeans_input(
        "fortran cell means");
    HostViewManaged<Real[_ppm_consts::PPMDX_PHYSICAL_LEV][10]> f90_dx_input(
        "fortran ppmdx");
    HostViewManaged<Real[NUM_PHYSICAL_LEV][3]> f90_result("fortra result");
    for (int var = 0; var < num_remap; ++var) {
      auto kokkos_result =
          Kokkos::create_mirror_view(remap.parabola_coeffs[var]);
      Kokkos::deep_copy(kokkos_result, remap.parabola_coeffs[var]);

      for (int ie = 0; ie < ne; ++ie) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            Kokkos::deep_copy(f90_cellmeans_input,
                              Homme::subview(remap.ao[var], ie, igp, jgp));
            sync_to_host(Homme::subview(remap.ppmdx, ie, igp, jgp),
                         f90_dx_input);

            auto tmp = Kokkos::create_mirror_view(remap.ppmdx);
            Kokkos::deep_copy(tmp, remap.ppmdx);

            compute_ppm_c_callable(f90_cellmeans_input.data(),
                                   f90_dx_input.data(), f90_result.data(),
                                   remap_alg);
            for (int k = 0; k < f90_result.extent_int(0); ++k) {
              for (int parabola_coeff = 0;
                   parabola_coeff < f90_result.extent_int(1);
                   ++parabola_coeff) {
                REQUIRE(!std::isnan(f90_result(k, parabola_coeff)));
                REQUIRE(!std::isnan(
                    kokkos_result(ie, igp, jgp, parabola_coeff, k)));
                REQUIRE(f90_result(k, parabola_coeff) ==
                        kokkos_result(ie, igp, jgp, parabola_coeff, k));
              }
            }
          }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagPPMTest &, TeamMember team) const {
    KernelVariables kv(team);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, num_remap * NP * NP),
        [&](const int &loop_idx) {
          const int var = loop_idx / NP / NP;
          const int igp = (loop_idx / NP) % NP;
          const int jgp = loop_idx % NP;
          remap.compute_ppm(
              kv, Homme::subview(remap.ao[var], kv.ie, igp, jgp),
              Homme::subview(remap.ppmdx, kv.ie, igp, jgp),
              Homme::subview(remap.dma[var], kv.ie, igp, jgp),
              Homme::subview(remap.ai[var], kv.ie, igp, jgp),
              Homme::subview(remap.parabola_coeffs[var], kv.ie, igp, jgp));
        });
  }

  // This ensures that the represented grid is not degenerate
  // The grid is represented by the interval widths, so they must all be
  // positive. The top of the grid must also be a fixed value,
  // so the sum of the intervals must be top, and the bottom is assumed to be 0
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>
  generate_grid_intervals(rngAlg &engine, const Real &top, std::string name) {
    HostViewManaged<Scalar * [NP][NP][NUM_LEV]> grid("grid", ne);
    genRandArray(grid, engine,
                 std::uniform_real_distribution<Real>(0.0625, top));
    for (int ie = 0; ie < ne; ++ie) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          auto grid_slice = Homme::subview(grid, ie, igp, jgp);
          auto start = reinterpret_cast<Real *>(grid_slice.data());
          auto end = start + grid_slice.size();
          std::sort(start, end);
          grid_slice(0)[0] = 0.0;
          grid_slice(NUM_LEV - 1)[VECTOR_SIZE - 1] = top;
          for (int k = NUM_PHYSICAL_LEV - 1; k > 0; --k) {
            const int vector_level = k / VECTOR_SIZE;
            const int vector = k % VECTOR_SIZE;
            const int lower_vector_level = (k - 1) / VECTOR_SIZE;
            const int lower_vector = (k - 1) % VECTOR_SIZE;
            grid_slice(vector_level)[vector] -=
                grid_slice(lower_vector_level)[lower_vector];
          }
        }
      }
    }

    ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> intervals(name, ne);
    Kokkos::deep_copy(intervals, grid);
    return intervals;
  }

  void initialize_layers(rngAlg &engine) {
    // Note that these must have the property that
    // sum(src_layer_thickness) = sum(tgt_layer_thickness)
    // To do this, we generate two grids and compute the interval lengths

    const Real top = std::uniform_real_distribution<Real>(1.0, 1024.0)(engine);

    src_layer_thickness_kokkos =
        generate_grid_intervals(engine, top, "kokkos source layer thickness");
    tgt_layer_thickness_kokkos =
        generate_grid_intervals(engine, top, "kokkos target layer thickness");
  }

  void test_remap() {
    std::random_device rd;
    rngAlg engine(rd());
    std::uniform_real_distribution<Real> dist(0.125, 1000.0);
    for (int i = 0; i < num_remap; ++i) {
      remap_vals[i] =
          ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("remap variable", ne);
      genRandArray(remap_vals[i], engine, dist);
    }

    // This must be initialize before remap_vals is updated
    HostViewManaged<Real * [num_remap][NUM_PHYSICAL_LEV][NP][NP]> f90_remap_qdp(
        "fortran qdp", ne);
    for (int var = 0; var < num_remap; ++var) {
      for (int ie = 0; ie < ne; ++ie) {
        sync_to_host(Homme::subview(remap_vals[var], ie),
                     Homme::subview(f90_remap_qdp, ie, var));
      }
    }

    initialize_layers(engine);

    Kokkos::parallel_for(
        Homme::get_default_team_policy<ExecSpace, TagRemapTest>(ne), *this);
    ExecSpace::fence();

    const int remap_alg = boundary_cond::fortran_remap_alg;
    const int np = NP;
    const int qsize = num_remap;

    HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]>
        f90_src_layer_thickness_input("fortran source layer thickness");
    HostViewManaged<Real[NUM_PHYSICAL_LEV][NP][NP]>
        f90_tgt_layer_thickness_input("fortran target layer thickness");

    Kokkos::Array<HostViewManaged<Scalar * [NP][NP][NUM_LEV]>, num_remap>
        kokkos_remapped;
    for (int var = 0; var < num_remap; ++var) {
      kokkos_remapped[var] = Kokkos::create_mirror_view(remap_vals[var]);
      Kokkos::deep_copy(kokkos_remapped[var], remap_vals[var]);
    }

    for (int ie = 0; ie < ne; ++ie) {
      sync_to_host(Homme::subview(src_layer_thickness_kokkos, ie),
                   f90_src_layer_thickness_input);
      sync_to_host(Homme::subview(tgt_layer_thickness_kokkos, ie),
                   f90_tgt_layer_thickness_input);

      remap_q_ppm_c_callable(Homme::subview(f90_remap_qdp, ie).data(), np,
                             qsize, f90_src_layer_thickness_input.data(),
                             f90_tgt_layer_thickness_input.data(), remap_alg);

      for (int var = 0; var < num_remap; ++var) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
              const int vector_level = k / VECTOR_SIZE;
              const int vector = k % VECTOR_SIZE;
              // TODO: Fix so that neither are returning NaN's or always do so
              // in the same place
              //
              // The fortran returns NaN's, so make certain we only return NaN's
              // when the Fortran does
              REQUIRE(std::isnan(f90_remap_qdp(ie, var, k, igp, jgp)) ==
                      std::isnan(kokkos_remapped[var](ie, igp, jgp,
                                                      vector_level)[vector]));
              if (!std::isnan(f90_remap_qdp(ie, var, k, igp, jgp)) &&
                  !std::isnan(kokkos_remapped[var](ie, igp, jgp,
                                                   vector_level)[vector])) {
                REQUIRE(
                    f90_remap_qdp(ie, var, k, igp, jgp) ==
                    kokkos_remapped[var](ie, igp, jgp, vector_level)[vector]);
              }
            }
          }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagRemapTest &, TeamMember team) const {
    KernelVariables kv(team);
    remap.compute_grids_phase(
        kv, Homme::subview(src_layer_thickness_kokkos, kv.ie),
        Homme::subview(tgt_layer_thickness_kokkos, kv.ie));
    for (int var = 0; var < num_remap; ++var) {
      remap.compute_remap_phase(kv, var,
                                Homme::subview(remap_vals[var], kv.ie));
    }
  }

  const int ne;
  PpmVertRemap<num_remap, boundary_cond> remap;
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> src_layer_thickness_kokkos;
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> tgt_layer_thickness_kokkos;
  Kokkos::Array<ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>, num_remap>
      remap_vals;
};

TEST_CASE("ppm_mirrored", "vertical remap") {
  constexpr int remap_dim = 3;
  constexpr int num_elems = 4;
  ppm_remap_functor_test<PpmMirrored, remap_dim> remap_test_mirrored(num_elems);
  SECTION("grid test") { remap_test_mirrored.test_grid(); }
  SECTION("ppm test") { remap_test_mirrored.test_ppm(); }
  SECTION("remap test") { remap_test_mirrored.test_remap(); }
}

TEST_CASE("ppm_fixed", "vertical remap") {
  constexpr int remap_dim = 3;
  constexpr int num_elems = 4;
  ppm_remap_functor_test<PpmFixed, remap_dim> remap_test_fixed(num_elems);
  SECTION("grid test") { remap_test_fixed.test_grid(); }
  SECTION("ppm test") { remap_test_fixed.test_ppm(); }
  SECTION("remap test") { remap_test_fixed.test_remap(); }
}

TEST_CASE("remap_interface", "vertical remap") {
  constexpr int num_elems = 4;
  Elements elements;
  elements.random_init(num_elems);

  // TODO: make dt random
  constexpr int np1 = 0;
  constexpr int n0_qdp = 0;
  constexpr Real dt = 0.0;

  HybridVCoord hvcoord;
  hvcoord.random_init(std::random_device()());
  SECTION("states_only") {
    constexpr int rsplit = 1;
    constexpr int qsize = 0;
    using _Remap = RemapFunctor<rsplit, PpmVertRemap, PpmMirrored>;
    _Remap remap(qsize, elements, hvcoord);
    remap.run_remap(np1,n0_qdp,dt);
  }
  SECTION("tracers_only") {
    constexpr int rsplit = 0;
    constexpr int qsize = QSIZE_D;
    using _Remap = RemapFunctor<rsplit, PpmVertRemap, PpmMirrored>;
    _Remap remap(qsize, elements, hvcoord);
    remap.run_remap(np1,n0_qdp,dt);
  }
  SECTION("states_tracers") {
    constexpr int remap_dim = 3 + QSIZE_D;
    constexpr int rsplit = 1;
    constexpr int qsize = QSIZE_D;
    using _Remap = RemapFunctor<remap_dim, PpmVertRemap, PpmMirrored>;
    _Remap remap(qsize, elements, hvcoord);
    remap.run_remap(np1,n0_qdp,dt);
  }
}
