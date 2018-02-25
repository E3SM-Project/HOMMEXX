#include <catch/catch.hpp>

#include <iostream>
#include <random>

#include "Elements.hpp"
#include "Tracers.hpp"
#include "Types.hpp"
#include "utilities/TestUtils.hpp"
#include "utilities/SubviewUtils.hpp"

using namespace Homme;

// ====================== RANDOM INITIALIZATION ====================== //

TEST_CASE("dp3d_intervals", "Testing Elements::random_init") {
  constexpr int num_elems = 5;
  constexpr Real max_pressure = 32.0;
  constexpr Real rel_threshold = 128.0 * std::numeric_limits<Real>::epsilon();
  Elements elements;
  elements.random_init(num_elems, max_pressure);
  HostViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp3d("host dp3d",
                                                                    num_elems);
  auto h_elements = elements.get_elements_host();
  for (int ie = 0; ie < num_elems; ++ie) {
    Kokkos::deep_copy(Homme::subview(dp3d,ie), h_elements(ie).m_dp3d);
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          HostViewUnmanaged<Scalar[NUM_LEV]> dp3d_col =
              Homme::subview(dp3d, ie, tl, igp, jgp);
          Real sum = 0.0;
          for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
            const int ilev = level / VECTOR_SIZE;
            const int vec_lev = level % VECTOR_SIZE;
            sum += dp3d_col(ilev)[vec_lev];
            REQUIRE(dp3d_col(ilev)[vec_lev] > 0.0);
          }
          Real rel_error = compare_answers(max_pressure, sum);
          REQUIRE(rel_threshold >= rel_error);
        }
      }
    }
  }
}

TEST_CASE("d_dinv_check", "Testing Elements::random_init") {
  constexpr int num_elems = 5;
  constexpr Real rel_threshold = 1e-10;
  Elements elements;
  elements.random_init(num_elems);
  HostViewManaged<Real * [2][2][NP][NP]> d("host d", num_elems);
  HostViewManaged<Real * [2][2][NP][NP]> dinv("host dinv", num_elems);
  auto h_elements = elements.get_elements_host();
  for (int ie = 0; ie < num_elems; ++ie) {
    Kokkos::deep_copy(Homme::subview(d,ie),    h_elements(ie).m_d);
    Kokkos::deep_copy(Homme::subview(dinv,ie), h_elements(ie).m_dinv);
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        const Real det_1 = d(ie, 0, 0, igp, jgp) * d(ie, 1, 1, igp, jgp) -
                           d(ie, 0, 1, igp, jgp) * d(ie, 1, 0, igp, jgp);
        REQUIRE(det_1 > 0.0);
        const Real det_2 = dinv(ie, 0, 0, igp, jgp) * dinv(ie, 1, 1, igp, jgp) -
                           dinv(ie, 0, 1, igp, jgp) * dinv(ie, 1, 0, igp, jgp);
        REQUIRE(det_2 > 0.0);
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j) {
            Real pt_product = 0.0;
            for (int k = 0; k < 2; ++k) {
              pt_product += d(ie, i, k, igp, jgp) * dinv(ie, k, j, igp, jgp);
            }
            const Real expected = (i == j) ? 1.0 : 0.0;
            const Real rel_error = compare_answers(expected, pt_product);
            REQUIRE(rel_threshold >= rel_error);
          }
        }
      }
    }
  }
}
#if 0
TEST_CASE("tracers_check", "Testing Tracers::Tracers(int, int)") {
  // Ensures three things - that genRandArray results in an array starting and
  // ending with values between the specified bounds, that it does not exceed
  // the bounds specified, and that the tracers aren't accidentally overwriting
  // each other
  constexpr int num_elems = 3;
  constexpr int num_tracers = 5;
  constexpr Real min_val = 5.3, max_val = 9.3;
  constexpr Real signature = min_val - 1.0;
  std::random_device rd;
  std::mt19937_64 engine(rd());
  std::uniform_real_distribution<Real> dist(min_val, max_val);
  Tracers tracers(num_elems, num_tracers);
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int iq = 0; iq < num_tracers; ++iq) {
      Tracers::Tracer t = tracers.device_tracers()(ie, iq);
      genRandArray(t.qtens, engine, dist);

      auto qtens = Kokkos::create_mirror_view(t.qtens);
      Kokkos::deep_copy(qtens,t.qtens);
      REQUIRE(qtens(0, 0, 0)[0] >= min_val);
      REQUIRE(qtens(0, 0, 0)[0] <= max_val);
      REQUIRE(qtens(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] >= min_val);
      REQUIRE(qtens(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] <= max_val);
      qtens(0, 0, 0)[0] = signature;
      qtens(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] = signature;
      Kokkos::deep_copy(t.qtens,qtens);

      genRandArray(t.vstar_qdp, engine, dist);
      auto vstar_qdp = Kokkos::create_mirror_view(t.vstar_qdp);
      Kokkos::deep_copy(vstar_qdp,t.vstar_qdp);
      REQUIRE(vstar_qdp(0, 0, 0, 0)[0] >= min_val);
      REQUIRE(vstar_qdp(0, 0, 0, 0)[0] <= max_val);
      REQUIRE(vstar_qdp(1, NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] >=
              min_val);
      REQUIRE(vstar_qdp(1, NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] <=
              max_val);
      vstar_qdp(0, 0, 0, 0)[0] = signature;
      vstar_qdp(1, NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] = signature;
      Kokkos::deep_copy(t.vstar_qdp,vstar_qdp);

      genRandArray(t.qlim, engine, dist);
      auto qlim = Kokkos::create_mirror_view(t.qlim);
      Kokkos::deep_copy(qlim,t.qlim);
      REQUIRE(qlim(0, 0)[0] >= min_val);
      REQUIRE(qlim(0, 0)[0] <= max_val);
      REQUIRE(qlim(1, NUM_LEV - 1)[VECTOR_SIZE - 1] >= min_val);
      REQUIRE(qlim(1, NUM_LEV - 1)[VECTOR_SIZE - 1] <= max_val);
      qlim(0, 0)[0] = signature;
      qlim(1, NUM_LEV - 1)[VECTOR_SIZE - 1] = signature;
      Kokkos::deep_copy(t.qlim,qlim);

      genRandArray(t.q, engine, dist);
      auto q = Kokkos::create_mirror_view(t.q);
      Kokkos::deep_copy(q,t.q);
      REQUIRE(q(0, 0, 0)[0] >= min_val);
      REQUIRE(q(0, 0, 0)[0] <= max_val);
      REQUIRE(q(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] >= min_val);
      REQUIRE(q(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] <= max_val);
      q(0, 0, 0)[0] = signature;
      q(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] = signature;
      Kokkos::deep_copy(t.q,q);

      genRandArray(t.qtens_biharmonic, engine, dist);
      auto qtens_biharmonic = Kokkos::create_mirror_view(t.qtens_biharmonic);
      Kokkos::deep_copy(qtens_biharmonic,t.qtens_biharmonic);
      REQUIRE(qtens_biharmonic(0, 0, 0)[0] >= min_val);
      REQUIRE(qtens_biharmonic(0, 0, 0)[0] <= max_val);
      REQUIRE(qtens_biharmonic(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] >= min_val);
      REQUIRE(qtens_biharmonic(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] <= max_val);
      qtens_biharmonic(0, 0, 0)[0] = signature;
      qtens_biharmonic(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] = signature;
      Kokkos::deep_copy(t.qtens_biharmonic,qtens_biharmonic);
    }
  }
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int iq = 0; iq < num_tracers; ++iq) {
      Tracers::Tracer t = tracers.device_tracers()(ie, iq);
      auto qtens = Kokkos::create_mirror_view(t.qtens);
      auto vstar_qdp = Kokkos::create_mirror_view(t.vstar_qdp);
      auto qlim = Kokkos::create_mirror_view(t.qlim);
      auto q = Kokkos::create_mirror_view(t.q);
      auto qtens_biharmonic = Kokkos::create_mirror_view(t.qtens_biharmonic);
      Kokkos::deep_copy(qtens,t.qtens);
      Kokkos::deep_copy(vstar_qdp,t.vstar_qdp);
      Kokkos::deep_copy(qlim,t.qlim);
      Kokkos::deep_copy(q,t.q);
      Kokkos::deep_copy(qtens_biharmonic,t.qtens_biharmonic);

      REQUIRE(qtens(0, 0, 0)[0] == signature);
      REQUIRE(qtens(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] ==
              signature);
      REQUIRE(vstar_qdp(0, 0, 0, 0)[0] == signature);
      REQUIRE(vstar_qdp(1, NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] ==
              signature);
      REQUIRE(qlim(0, 0)[0] == signature);
      REQUIRE(qlim(1, NUM_LEV - 1)[VECTOR_SIZE - 1] == signature);
      REQUIRE(q(0, 0, 0)[0] == signature);
      REQUIRE(q(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] == signature);
      REQUIRE(qtens_biharmonic(0, 0, 0)[0] == signature);
      REQUIRE(qtens_biharmonic(NP - 1, NP - 1, NUM_LEV - 1)[VECTOR_SIZE - 1] == signature);
    }
  }
}
#endif
