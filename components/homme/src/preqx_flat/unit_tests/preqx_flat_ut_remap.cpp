#include <catch/catch.hpp>

#include "dimensions_remap_tests.hpp"

#include "remap.hpp"
#include "Utility.hpp"

#include "KernelVariables.hpp"
#include "Types.hpp"
#include "Control.hpp"
#include "RemapFunctor.hpp"

#include <assert.h>
#include <stdio.h>
#include <limits>
#include <random>

using namespace Homme;

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

template <typename boundary_cond> class ppm_remap_functor_test {
  static_assert(std::is_base_of<PPM_Boundary_Conditions, boundary_cond>::value,
                "PPM Remap test must have a valid boundary condition");

public:
  static constexpr int num_remap = boundary_cond::remap_dim;

  ppm_remap_functor_test(Control &data)
      : remap(data), ne(data.num_elems),
        src_layer_thickness("source layer thickness", ne),
        tgt_layer_thickness("target layer thickness", ne) {
    for (int var = 0; var < num_remap; ++var) {
      remap_vals[var] =
          ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("remap var", ne);
    }
  }

  struct TagGridTest {};
  struct TagPPMTest {};
  struct TagRemapTest {};

  void test_grid() {
    std::random_device rd;
    rngAlg engine(rd());
    for (int i = 0; i < num_remap; ++i) {
      genRandArray(remap.dpo[i], engine,
                   std::uniform_real_distribution<Real>(0.125, 0.875));
    }
    Kokkos::parallel_for(
        Homme::get_default_team_policy<ExecSpace, TagGridTest>(ne), *this);
    ExecSpace::fence();

    const int remap_alg = boundary_cond::fortran_remap_alg;
    HostViewManaged<Real[NUM_PHYSICAL_LEV + 4]> f90_input("fortran dpo");
    HostViewManaged<Real[NUM_PHYSICAL_LEV + 2][10]> f90_result("fortra ppmdx");
    for (int var = 0; var < num_remap; ++var) {
      auto kokkos_result = Kokkos::create_mirror_view(remap.ppmdx[var]);
      Kokkos::deep_copy(kokkos_result, remap.ppmdx[var]);
      for (int ie = 0; ie < ne; ++ie) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            Kokkos::deep_copy(f90_input,
                              Homme::subview(remap.dpo[var], ie, igp, jgp));
            compute_ppm_grids_c_callable(f90_input.data(), f90_result.data(),
                                         remap_alg);
            for (int k = 0; k < f90_result.extent(0); ++k) {
              for (int stencil_idx = 0; stencil_idx < f90_result.extent(1);
                   ++stencil_idx) {
                REQUIRE(!std::isnan(f90_result(k, stencil_idx)));
                REQUIRE(
                    !std::isnan(kokkos_result(ie, igp, jgp, k, stencil_idx)));
                REQUIRE(f90_result(k, stencil_idx) ==
                        kokkos_result(ie, igp, jgp, k, stencil_idx));
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
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, num_remap * NP * NP),
                         [&](const int &loop_idx) {
      const int var = loop_idx / NP / NP;
      const int igp = (loop_idx / NP) % NP;
      const int jgp = loop_idx % NP;
      remap.compute_grids(kv, Homme::subview(remap.dpo[var], kv.ie, igp, jgp),
                          Homme::subview(remap.ppmdx[var], kv.ie, igp, jgp));
    });
  }

  void test_ppm() {
    std::random_device rd;
    rngAlg engine(rd());
    for (int i = 0; i < num_remap; ++i) {
      genRandArray(remap.ao[i], engine,
                   std::uniform_real_distribution<Real>(0.125, 0.875));
      genRandArray(remap.ppmdx[i], engine,
                   std::uniform_real_distribution<Real>(0.125, 0.875));
    }
    Kokkos::parallel_for(
        Homme::get_default_team_policy<ExecSpace, TagPPMTest>(ne), *this);
    ExecSpace::fence();

    const int remap_alg = boundary_cond::fortran_remap_alg;

    HostViewManaged<Real[NUM_PHYSICAL_LEV + 4]> f90_cellmeans_input(
        "fortran cell means");
    HostViewManaged<Real[NUM_PHYSICAL_LEV + 2][10]> f90_dx_input(
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
            Kokkos::deep_copy(f90_dx_input,
                              Homme::subview(remap.ppmdx[var], ie, igp, jgp));
            compute_ppm_c_callable(f90_cellmeans_input.data(),
                                   f90_dx_input.data(), f90_result.data(),
                                   remap_alg);
            for (int k = 0; k < f90_result.extent(0); ++k) {
              for (int stencil_idx = 0; stencil_idx < f90_result.extent(1);
                   ++stencil_idx) {
                REQUIRE(!std::isnan(f90_result(k, stencil_idx)));
                REQUIRE(
                    !std::isnan(kokkos_result(ie, igp, jgp, k, stencil_idx)));
                DEBUG_PRINT(
                    "%s results ppm: %d %d %d %d %d %d -> % .17e vs % .17e\n",
                    boundary_cond::name(), var, ie, igp, jgp, k, stencil_idx,
                    f90_result(k, stencil_idx),
                    kokkos_result(ie, igp, jgp, k, stencil_idx));
                REQUIRE(f90_result(k, stencil_idx) ==
                        kokkos_result(ie, igp, jgp, k, stencil_idx));
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
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, num_remap * NP * NP),
                         [&](const int &loop_idx) {
      const int var = loop_idx / NP / NP;
      const int igp = (loop_idx / NP) % NP;
      const int jgp = loop_idx % NP;
      remap.compute_ppm(
          kv, Homme::subview(remap.ao[var], kv.ie, igp, jgp),
          Homme::subview(remap.ppmdx[var], kv.ie, igp, jgp),
          Homme::subview(remap.dma[var], kv.ie, igp, jgp),
          Homme::subview(remap.ai[var], kv.ie, igp, jgp),
          Homme::subview(remap.parabola_coeffs[var], kv.ie, igp, jgp));
    });
  }

  void test_remap() {
    std::random_device rd;
    rngAlg engine(rd());
    for (int i = 0; i < num_remap; ++i) {
      genRandArray(remap.dpo[i], engine,
                   std::uniform_real_distribution<Real>(0.125, 0.875));
    }
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

    HostViewManaged<Real[num_remap][NUM_PHYSICAL_LEV][NP][NP]> f90_remapped_qdp(
        "fortran qdp");

    Kokkos::Array<HostViewManaged<Scalar * [NP][NP][NUM_LEV]>, num_remap>
    kokkos_remapped;
    for (int var = 0; var < num_remap; ++var) {
      kokkos_remapped[var] = Kokkos::create_mirror_view(remap_vals[var]);
      Kokkos::deep_copy(kokkos_remapped[var], remap_vals[var]);
    }

    for (int ie = 0; ie < ne; ++ie) {
      sync_to_host(Homme::subview(src_layer_thickness, ie),
                   f90_src_layer_thickness_input);
      sync_to_host(Homme::subview(tgt_layer_thickness, ie),
                   f90_tgt_layer_thickness_input);

      remap_q_ppm_c_callable(f90_remapped_qdp.data(), np, qsize,
                             f90_src_layer_thickness_input.data(),
                             f90_tgt_layer_thickness_input.data(), remap_alg);

      for (int var = 0; var < num_remap; ++var) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
              const int vector_level = k / VECTOR_SIZE;
              const int vector = k % VECTOR_SIZE;
              REQUIRE(!std::isnan(f90_remapped_qdp(var, k, igp, jgp)));
              REQUIRE(!std::isnan(kokkos_remapped[var](ie, igp, jgp,
                                                       vector_level)[vector]));
              REQUIRE(f90_remapped_qdp(var, k, igp, jgp) ==
                      kokkos_remapped[var](ie, igp, jgp, vector_level)[vector]);
            }
          }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagRemapTest &, TeamMember team) const {
    KernelVariables kv(team);
    Kokkos::Array<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>, num_remap>
    elem_remap;
    for (int var = 0; var < num_remap; ++var) {
      elem_remap[var] = Homme::subview(remap_vals[var], kv.ie);
    }
    remap.remap(kv, num_remap, Homme::subview(src_layer_thickness, kv.ie),
                Homme::subview(tgt_layer_thickness, kv.ie), elem_remap);
  }

  const int ne;
  PPM_Vert_Remap<boundary_cond> remap;
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> src_layer_thickness;
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> tgt_layer_thickness;
  Kokkos::Array<ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>, num_remap>
  remap_vals;
};

TEST_CASE("ppm_mirrored", "vertical remap") {
  constexpr int remap_dim = 1;
  constexpr int num_elems = 1;
  Control data;
  data.random_init(num_elems, std::random_device()());
  ppm_remap_functor_test<PPM_Mirrored<remap_dim> > remap_test_mirrored(data);
  SECTION("grid test") { remap_test_mirrored.test_grid(); }
  SECTION("ppm test") { remap_test_mirrored.test_ppm(); }
  SECTION("remap test") { remap_test_mirrored.test_remap(); }
}

TEST_CASE("ppm_fixed", "vertical remap") {
  constexpr int remap_dim = 3 + QSIZE_D;
  constexpr int num_elems = 10;
  Control data;
  data.random_init(num_elems, std::random_device()());
  ppm_remap_functor_test<PPM_Fixed<remap_dim> > remap_test_fixed(data);
  SECTION("grid test") { remap_test_fixed.test_grid(); }
  SECTION("ppm test") { remap_test_fixed.test_ppm(); }
  SECTION("remap test") { remap_test_fixed.test_remap(); }
}
