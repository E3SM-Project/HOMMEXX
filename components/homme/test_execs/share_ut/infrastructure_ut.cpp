#include <catch/catch.hpp>

#include <iostream>

#include "ExecSpaceDefs.hpp"
#include "Types.hpp"

#include "utilities/SubviewUtils.hpp"
#include "utilities/TestUtils.hpp"

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

TEST_CASE("Parallel scan",
          "Test parallel vector scan at ThreadVectorRange level.") {

  constexpr int num_elems   = 10;
  constexpr int num_points = 16;
  constexpr int vector_length = 16;
  ExecViewManaged<Real*[num_points][vector_length]> input("",num_elems,vector_length);
  ExecViewManaged<Real*[num_points][vector_length]> output_seq("",num_elems,vector_length);
  ExecViewManaged<Real*[num_points][vector_length]> output_dispatch("",num_elems,vector_length);
  ExecViewManaged<Real*[num_points][vector_length]> output_kokkos("",num_elems,vector_length);

  // Policy used in all parallel for's below
  Kokkos::TeamPolicy<ExecSpace> policy(num_elems,num_points,vector_length);
  policy.set_chunk_size(1);

  // Fill the input view.
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team){
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,num_points),
                         [&](const int ip){
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,vector_length),
                           [&](const int iv) {
        // We fill the vector entry iv with 1./(iv+1) - 1./(iv+2).
        // This way, the sum of entries [0,N] should return 1 - 1./(N+2).
        input(ie,ip,iv) = 1.0/(iv+1) - 1.0/(iv+2);
      });
    });
  });

  // Note: we want to check whether Dispatch<ExecSpace>::parallel_scan
  //       computes a correct parallel scan. We do not know if the macro
  //       HOMMEXX_GPU_BFB_WITH_CPU is defined or not, so we do not know
  //       if we rely on the BFB version or the Kokkos one (which computes
  //       the sum with a parallel-friendly reorganization of sums).
  //       However, we know exactly the mathematical scan sum (see above),
  //       so we can check if, regardless of the HOMMEXX_GPU_BFB_WITH_CPU
  //       macro, Dispatch<ExecSpace>::parallel_scan computes a scan sum
  //       within a given tolerance of the correct value

  // Computing a scan sum with Dispatch<ExecSpace> version
  // Note: we do not know if the BFB version is used or not. It depends
  //       on whether HOMMEXX_GPU_BFB_WITH_CPU was defined BEFORE including
  //       the ExecSpaceDefs.hpp header.
  Kokkos::parallel_for(policy,KOKKOS_LAMBDA(const TeamMember& team){
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,num_points),
                         [&](const int ip) {
      auto input_vector  = Homme::subview(input,team.league_rank(),ip);
      auto output_vector = Homme::subview(output_dispatch,team.league_rank(),ip);
      auto my_lambda = [&](const int i, Real& accum, bool last) {
        accum += input_vector(i);
        if (last) {
          output_vector(i) = accum;
        }
      };
      Dispatch<ExecSpace>::parallel_scan(team,vector_length,my_lambda);
    });
  });

  // Computing the scan sum with Kokkos' standard routine.
  Kokkos::parallel_for(policy,KOKKOS_LAMBDA(const TeamMember& team){
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,num_points),
                         [&](const int ip) {
      auto input_vector  = Homme::subview(input,team.league_rank(),ip);
      auto output_vector = Homme::subview(output_kokkos,team.league_rank(),ip);
      auto my_lambda = [&](const int i, Real& accum, bool last) {
        accum += input_vector(i);
        if (last) {
          output_vector(i) = accum;
        }
      };
      Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team,vector_length),my_lambda);
    });
  });

  // The two versions *should* give the exact answer on CPU,
  // but differ on GPU. Either way, we should be within a good tolerance
  auto output_dispatch_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),output_dispatch);
  auto output_kokkos_h   = Kokkos::create_mirror_view_and_copy(HostMemSpace(),output_kokkos);

  const Real rel_threshold = std::numeric_limits<Real>::epsilon();
  for (int ie=0; ie<num_elems; ++ie) {
    for (int ip=0; ip<num_points; ++ip) {
      for (int iv=0; iv<vector_length; ++iv) {
        const Real computed_kokkos   = output_kokkos_h(ie,ip,iv);
        const Real computed_dispatch = output_dispatch_h(ie,ip,iv);
        const Real exact = 1.0 - 1.0/(iv+2);
        Real rel_error;

        rel_error = compare_answers(computed_kokkos,computed_dispatch);
        REQUIRE(rel_error<=rel_threshold);
        rel_error = compare_answers(exact,computed_dispatch);
        REQUIRE(rel_error<=rel_threshold);
        rel_error = compare_answers(exact,computed_dispatch);
        REQUIRE(rel_error<=rel_threshold);
      }
    }
  }
}
