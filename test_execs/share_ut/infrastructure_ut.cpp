#include <catch/catch.hpp>

#include <random>
#include <iostream>

#include "ExecSpaceDefs.hpp"
#include "Types.hpp"
#include "Utility.hpp"

using namespace Homme;

// ====================== EXECUTION SPACE SETUP ====================== //

TEST_CASE("ExecSpaceDefs",
          "Test parallel machine parameterization.") {
  const int plev = 72;

  const auto test_basics = [=] (const ThreadPreferences& tp,
                                const std::pair<int, int>& tv) {
    REQUIRE(tv.first >= 1);
    REQUIRE(tv.second >= 1);
    if (tp.prefer_threads)
      REQUIRE((tv.first == tp.max_threads_usable || tv.second == 1));
    else
      REQUIRE((tv.first == 1 || tv.second > 1));
    if (tv.first * tv.second <= tp.max_threads_usable * tp.max_vectors_usable) {
      REQUIRE(tv.first <= tp.max_threads_usable);
      REQUIRE(tv.second <= tp.max_vectors_usable);
    }
  };

  SECTION("CPU/KNL") {
    for (int num_elem = 0; num_elem <= 3; ++num_elem) {
      for (int qsize = 1; qsize <= 30; ++qsize) {
        for (int pool = 1; pool <= 64; ++pool) {
          for (bool prefer_threads : {true, false}) {
            Homme::ThreadPreferences tp;
            tp.max_vectors_usable = plev;
            tp.prefer_threads = prefer_threads;
            const int npi = num_elem*qsize;
            const auto tv = Homme::Parallel::team_num_threads_vectors_from_pool(
              pool, npi, tp);
            // Correctness tests.
            test_basics(tp, tv);
            REQUIRE(tv.first * tv.second <= pool);
            // Tests for good behavior.
            if (npi >= pool)
              REQUIRE(tv.first * tv.second == 1);
          }
        }
      }
    }
    // Spot check some cases. Numbers are
    //     {#elem, qsize, pool, prefer_threads, #thread, #vector}.
    static const int cases[][6] = {{1, 1, 8, 1, 8, 1},
                                   {1, 30, 8, 1, 1, 1},
                                   {1, 1, 32, 1, 16, 2},
                                   {1, 1, 32, 0, 2, 16}};
    for (unsigned int i = 0; i < sizeof(cases)/sizeof(*cases); ++i) {
      const auto& c = cases[i];
      Homme::ThreadPreferences tp;
      tp.max_vectors_usable = plev/2;
      tp.prefer_threads = c[3];
      const auto tv = Homme::Parallel::team_num_threads_vectors_from_pool(
        c[2], c[0]*c[1], tp);
      REQUIRE(tv.first == c[4]);
      REQUIRE(tv.second == c[5]);
    }
  }

  SECTION("GPU") {
    static const int num_device_warps = 1792;
    static const int min_warps_per_team = 4, max_warps_per_team = 16;
    static const int num_threads_per_warp = 32;
    for (int num_elem = 0; num_elem <= 10000; num_elem += 10) {
      for (int qsize = 1; qsize <= 30; ++qsize) {
        for (bool prefer_threads : {true, false}) {
          Homme::ThreadPreferences tp;
          tp.prefer_threads = prefer_threads;
          tp.max_vectors_usable = plev;
          const int npi = num_elem*qsize;
          const auto tv = Homme::Parallel::team_num_threads_vectors_for_gpu(
            num_device_warps, num_threads_per_warp,
            min_warps_per_team, max_warps_per_team,
            npi, tp);
          // Correctness tests.
          test_basics(tp, tv);
          REQUIRE(Homme::Parallel::prevpow2(tv.second) == tv.second);
          REQUIRE(tv.first * tv.second >= num_threads_per_warp*min_warps_per_team);
          REQUIRE(tv.first * tv.second <= num_threads_per_warp*max_warps_per_team);
          // Tests for good behavior.
          REQUIRE(tv.first * tv.second >= min_warps_per_team * num_threads_per_warp);
          if (npi >= num_device_warps*num_threads_per_warp)
            REQUIRE(tv.first * tv.second == min_warps_per_team * num_threads_per_warp);
        }
      }
    }
    // Numbers are
    //     {#elem, qsize, prefer_threads, #thread, #vector}.
    static const int cases[][5] = {{1, 1, 1, 16, 32},
                                   {96, 30, 1, 16, 8},
                                   {96, 30, 0, 4, 32}};
    for (unsigned int i = 0; i < sizeof(cases)/sizeof(*cases); ++i) {
      const auto& c = cases[i];
      Homme::ThreadPreferences tp;
      tp.max_vectors_usable = plev/2;
      tp.prefer_threads = c[2];
      const auto tv = Homme::Parallel::team_num_threads_vectors_for_gpu(
        num_device_warps, num_threads_per_warp,
        min_warps_per_team, max_warps_per_team,
        c[0]*c[1], tp);
      REQUIRE(tv.first == c[3]);
      REQUIRE(tv.second == c[4]);
    }
  }
}
