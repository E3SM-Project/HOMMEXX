#include <catch/catch.hpp>
//?
#include <limits>

#include "dimensions_remap_tests.hpp"

#include "remap.hpp"
#include "Utility.hpp"

#include "KernelVariables.hpp"
#include "Types.hpp"
#include "Control.hpp"
#include "RemapFunctor.hpp"

#include <assert.h>
#include <stdio.h>
#include <random>

//Documenting failures with opt. flags.

/*test compute_ppm_grids (alg=1) finished. 
 * test compute_ppm_grids (alg=2) finished. 
 * -------------------------------------------------------------------------------
 *  Testing compute_ppm() with alg=1
 *  -------------------------------------------------------------------------------
 /home/onguba/acmexxremap/components/homme/src/preqx_flat/unit_tests/preqx_flat_ut_remap.cpp:382
...............................................................................

/home/onguba/acmexxremap/components/homme/src/preqx_flat/unit_tests/preqx_flat_ut_remap.cpp:293: FAILED:
  REQUIRE( std::numeric_limits<Real>::epsilon() >= compare_answers(fortran_output[_i][_j], coutput0, 128.0) )
with expansion:
  0.0 >= 0.015625

-------------------------------------------------------------------------------
Testing compute_ppm() with alg=2
-------------------------------------------------------------------------------
/home/onguba/acmexxremap/components/homme/src/preqx_flat/unit_tests/preqx_flat_ut_remap.cpp:388
...............................................................................

/home/onguba/acmexxremap/components/homme/src/preqx_flat/unit_tests/preqx_flat_ut_remap.cpp:293: FAILED:
  REQUIRE( std::numeric_limits<Real>::epsilon() >= compare_answers(fortran_output[_i][_j], coutput0, 128.0) )
with expansion:
  0.0 >= 0.015625

test remap_Q_ppm (alg=1) finished. 
-------------------------------------------------------------------------------
Testing remap_Q_ppm() with alg=2
-------------------------------------------------------------------------------
/home/onguba/acmexxremap/components/homme/src/preqx_flat/unit_tests/preqx_flat_ut_remap.cpp:400
...............................................................................

/home/onguba/acmexxremap/components/homme/src/preqx_flat/unit_tests/preqx_flat_ut_remap.cpp:361: FAILED:
  REQUIRE( std::numeric_limits<Real>::epsilon() >= compare_answers( fortran_output[_i][_j][_k][_l], coutput0, 128.0) )
with expansion:
  0.0 >= 0.0
*/



using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {

// sort out const here
void compute_ppm_grids_c_callable(const Real *dx,
                                  Real *rslt,
                                  const int &alg);

void compute_ppm_c_callable(const Real *a, const Real *dx,
                            Real *coefs, const int &alg);

// F.o object files have only small letters in names
void remap_q_ppm_c_callable(Real *Qdp, const int &nx,
                            const int &qsize,
                            const Real *dp1,
                            const Real *dp2,
                            const int &alg);

};  // extern C

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
  }

  void test_ppm() {
    std::random_device rd;
    rngAlg engine(rd());
    for (int i = 0; i < num_remap; ++i) {
      genRandArray(remap.dpo[i], engine,
                   std::uniform_real_distribution<Real>(0.125, 0.875));
    }
    Kokkos::parallel_for(
        Homme::get_default_team_policy<ExecSpace, TagPPMTest>(ne), *this);
    ExecSpace::fence();
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

TEST_CASE("ppm", "vertical remap") {
  constexpr int remap_dim = 38;
  constexpr int num_elems = 30;
  Control data;
  data.random_init(num_elems, std::random_device()());
  ppm_remap_functor_test<PPM_Mirrored<remap_dim> > remap_test_mirrored(data);
  ppm_remap_functor_test<PPM_Fixed<remap_dim> > remap_test_fixed(data);
  remap_test_mirrored.test_grid();
  remap_test_fixed.test_grid();
  remap_test_mirrored.test_ppm();
  remap_test_fixed.test_ppm();
  remap_test_mirrored.test_remap();
  remap_test_fixed.test_remap();
}

#if 0

class remap_test {
 public:
  remap_test(int _alg) : alg(_alg) {
    std::random_device rd;
    rngAlg engine(rd());
    // in case of routine compute_pm_grids dx should always
    // be positive  it is thickness of the grid.
    genRandArray(
        r1_dx, r1_dx_dim, engine,
        std::uniform_real_distribution<Real>(0.0, 100.0));

    // in case of routine compute_ppm it is not clear what
    // input is.
    genRandArray(
        r2_a, r2_a_dim, engine,
        std::uniform_real_distribution<Real>(0.0, 10.0));
    genRandArray(
        &(r2_dx[0][0]), r2_dx_dim1 * r2_dx_dim2, engine,
        std::uniform_real_distribution<Real>(0.0, 10.0));

    // for routine 3
    // it is bug prone to multiply 3-4 dims together

    // we need to satisfy sum(dp1) = sum(dp2), so, we will
    // first generate  bounds for p, then p1 and p2 within
    // bounds, then will compute dp1 and dp2  and place a
    // check to make sure sum(dp1) = sum(dp2)

    genRandArray(
        &(r3_Qdp[0][0][0][0]),
        r3_Qdp_dim1 * r3_Qdp_dim2 * r3_Qdp_dim3 *
            r3_Qdp_dim4,
        engine,
        std::uniform_real_distribution<Real>(0.0, 10.0));

    // what are the typical dp values?
    Real ptop, pbottom;
    genRandArray(
        &pbottom, 1, engine,
        std::uniform_real_distribution<Real>(10.0, 100.0));
    genRandArray(&ptop, 1, engine,
                 std::uniform_real_distribution<Real>(
                     10000.0, 20000.0));

    // now, generate p, for each column: we divide column
    // into NLEV-1 equal intervals  and generate p in each
    // interval. This way sum(dp) is always the same.
    Real dp0 = (ptop - pbottom) / (NLEV - 1);
    Real pend[NLEV];
    pend[0] = pbottom;
    pend[NLEV-1] = ptop;

    for(int _k = 0; _k < (NLEV - 2); _k++) {
      pend[_k + 1] = pend[_k] + dp0;
    }  // k loop

    for(int _i = 0; _i < r3_dp1_dim2; _i++)
      for(int _j = 0; _j < r3_dp1_dim3; _j++) {
        Real pinner1[NLEV + 1], pinner2[NLEV + 1];
        pinner1[0] = pbottom;
        pinner2[0] = pbottom;
        pinner1[NLEV] = ptop;
        pinner2[NLEV] = ptop;
        for(int _k = 0; _k < NLEV - 1; _k++) {
          genRandArray(&(pinner1[_k + 1]), 1, engine,
                       std::uniform_real_distribution<Real>(
                           pend[_k], pend[_k + 1]));
          genRandArray(&(pinner2[_k + 1]), 1, engine,
                       std::uniform_real_distribution<Real>(
                           pend[_k], pend[_k + 1]));
        }  // k loop,

        // now pinner is generated
        for(int _k = 0; _k < NLEV; _k++) {
          r3_dp1[_k][_i][_j] =
              pinner1[_k + 1] - pinner1[_k];
          r3_dp2[_k][_i][_j] =
              pinner2[_k + 1] - pinner2[_k];
        }
      }  // i,j

    for(int _i = 0; _i < r3_Qdp_dim1; _i++)
      for(int _j = 0; _j < r3_Qdp_dim2; _j++)
        for(int _k = 0; _k < r3_Qdp_dim3; _k++)
          for(int _l = 0; _l < r3_Qdp_dim4; _l++) {
            r3_Qdp_copy2[_i][_j][_k][_l] =
                r3_Qdp[_i][_j][_k][_l];
          };

    for(int _i = 0; _i < r1_rslt_dim1; ++_i)
      for(int _j = 0; _j < r1_rslt_dim2; ++_j)
        r1_rslt[_i][_j] = 0.0;

    for(int _i = 0; _i < r2_coefs_dim1; ++_i)
      for(int _j = 0; _j < r2_coefs_dim2; ++_j)
        r2_coefs[_i][_j] = 0.0;

    // for debugging, assign 1 to everything
    //  for(int _i = 0; _i < r2_dx_dim1; ++_i)
    //    for(int _j = 0; _j < r2_dx_dim2; ++_j)
    //       r2_dx[_i][_j] = 1.0;
    //    for(int _i = 0; _i < r2_a_dim; ++_i)
    //       r2_a[_i] = 1.0;

    /*
        for(int _i = 0; _i < r3_Qdp_dim1; _i++)
        for(int _j = 0; _j < r3_Qdp_dim2; _j++)
        for(int _k = 0; _k < r3_Qdp_dim3; _k++)
        for(int _l = 0; _l < r3_Qdp_dim4; _l++){
          r3_Qdp[_i][_j][_k][_l] = 1.0;
          r3_Qdp_copy2[_i][_j][_k][_l] = 1.0;
        };
        for(int _i = 0; _i < r3_dp1_dim1; _i++)
        for(int _j = 0; _j < r3_dp1_dim2; _j++)
        for(int _k = 0; _k < r3_dp1_dim3; _k++){
          r3_dp2[_i][_j][_k] = 1.0;
        };
    */

  }  // end of constructor

  // let's avoid static
  // Since there are many routines that take in/out vars
  // with same names,  and since we want to keep names
  // consistens, let's mark  each var with rN_ , where N is
  // number of routine under development.

  // routine compute_ppm_grids
  const int r1_dx_dim = NLEVP4;
  const int r1_rslt_dim1 = NLEVP2;
  const int r1_rslt_dim2 = DIM10;
  Real r1_dx[NLEVP4];
  Real r1_rslt[NLEVP2][DIM10];

  // routine compute_ppm_grids
  const int r2_a_dim = NLEVP4;
  const int r2_dx_dim1 = NLEVP2;
  const int r2_dx_dim2 = DIM10;
  const int r2_coefs_dim1 = NLEV;
  const int r2_coefs_dim2 = DIM3;
  Real r2_a[NLEVP4];
  Real r2_dx[NLEVP2][DIM10];
  Real r2_coefs[NLEV][DIM3];

  // routine remap_Q_ppm
  const int r3_Qdp_dim1 = QSIZETEST;
  const int r3_Qdp_dim2 = NLEV;
  const int r3_Qdp_dim3 = NP;
  const int r3_Qdp_dim4 = NP;
  const int r3_dp1_dim1 =
      NLEV;  // both dp1 and dp2 have same dims
  const int r3_dp1_dim2 = NP;
  const int r3_dp1_dim3 = NP;
  Real r3_Qdp[QSIZETEST][NLEV][NP][NP];
  Real r3_Qdp_copy2[QSIZETEST][NLEV][NP]
                   [NP];  // in r3 Qdp in input and output
  // we need an extra copy to save the input for F
  Real r3_dp1[NLEV][NP][NP];
  Real r3_dp2[NLEV][NP][NP];

  int alg;

  void run_compute_ppm_grids() {
    compute_ppm_grids(r1_dx, r1_rslt, alg);
  }

  void run_compute_ppm() {
    compute_ppm(r2_a, r2_dx, r2_coefs, alg);
  }

  void run_remap_Q_ppm() {
    remap_Q_ppm(r3_Qdp, QSIZETEST, r3_dp1, r3_dp2, alg);
  }

};  // end of class def compute_sphere_op_test_ml

void testbody_compute_ppm_grids(const int _alg) {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int iterations = 10;
  const int vertical_alg = _alg;
  remap_test test(vertical_alg);
  test.run_compute_ppm_grids();

  const int out_len1 = test.r1_rslt_dim1,
            out_len2 = test.r1_rslt_dim2,
            dx_len = test.r1_dx_dim;

  Real fortran_output[out_len1][out_len2];
  Real dxf[dx_len];

  for(int _i = 0; _i < dx_len; _i++) {
    dxf[_i] = test.r1_dx[_i];
  }

  compute_ppm_grids_c_callable(
      &(dxf[0]), &(fortran_output[0][0]), test.alg);

  for(int _i = 0; _i < out_len1; ++_i) {
    for(int _j = 0; _j < out_len2; ++_j) {
      Real coutput0 = test.r1_rslt[_i][_j];
      REQUIRE(!std::isnan(fortran_output[_i][_j]));
      REQUIRE(!std::isnan(coutput0));
      REQUIRE(std::numeric_limits<Real>::epsilon() >=
              compare_answers(fortran_output[_i][_j],
                              coutput0, 128.0));
    }  // _j
  }    // _i
  std::cout << "test compute_ppm_grids (alg=" << _alg
            << ") finished. \n";
};  // end fo testbody_compute_ppm_grids

void testbody_compute_ppm(const int _alg) {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int iterations = 10;
  const int vertical_alg = _alg;
  remap_test test(vertical_alg);
  test.run_compute_ppm();

  const int out_dim1 = test.r2_coefs_dim1,
            out_dim2 = test.r2_coefs_dim2,
            dx_dim1 = test.r2_dx_dim1,
            dx_dim2 = test.r2_dx_dim2,
            a_dim = test.r2_a_dim;

  Real fortran_output[out_dim1][out_dim2];
  Real dxf[dx_dim1][dx_dim2];
  Real af[a_dim];

  for(int _i = 0; _i < dx_dim1; _i++)
    for(int _j = 0; _j < dx_dim2; _j++) {
      dxf[_i][_j] = test.r2_dx[_i][_j];
    }

  for(int _i = 0; _i < a_dim; _i++) {
    af[_i] = test.r2_a[_i];
  }

  compute_ppm_c_callable(&(af[0]), &(dxf[0][0]),
                         &(fortran_output[0][0]), test.alg);

  for(int _i = 0; _i < out_dim1; ++_i) {
    for(int _j = 0; _j < out_dim2; ++_j) {
      Real coutput0 = test.r2_coefs[_i][_j];
      //     std::cout << std::setprecision(20)
      //<<"F result = " << fortran_output[_i][_j] << ", C
      //output = " << coutput0 << "\n";
      REQUIRE(!std::isnan(fortran_output[_i][_j]));
      REQUIRE(!std::isnan(coutput0));
      REQUIRE(std::numeric_limits<Real>::epsilon() >=
              compare_answers(fortran_output[_i][_j],
                              coutput0, 128.0));
    }  // _j
  }    // _i

  std::cout << "test compute_ppm (alg=" << _alg
            << ") finished. \n";
};  // end of testbody_compute_ppm

void testbody_remap_Q_ppm(const int _alg) {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int iterations = 10;
  const int vertical_alg = _alg;
  remap_test test(vertical_alg);
  test.run_remap_Q_ppm();

  const int out_dim1 = test.r3_Qdp_dim1,
            out_dim2 = test.r3_Qdp_dim2,
            out_dim3 = test.r3_Qdp_dim3,
            out_dim4 = test.r3_Qdp_dim4,
            dp_dim1 = test.r3_dp1_dim1,
            dp_dim2 = test.r3_dp1_dim2,
            dp_dim3 = test.r3_dp1_dim3;

  Real fortran_output[out_dim1][out_dim2][out_dim3]
                     [out_dim4];
  ;
  Real dp1f[dp_dim1][dp_dim2][dp_dim3];
  Real dp2f[dp_dim1][dp_dim2][dp_dim3];

  for(int _i = 0; _i < dp_dim1; _i++)
    for(int _j = 0; _j < dp_dim2; _j++)
      for(int _k = 0; _k < dp_dim3; _k++) {
        dp1f[_i][_j][_k] = test.r3_dp1[_i][_j][_k];
        dp2f[_i][_j][_k] = test.r3_dp2[_i][_j][_k];
      }

  for(int _i = 0; _i < out_dim1; _i++)
    for(int _j = 0; _j < out_dim2; _j++)
      for(int _k = 0; _k < out_dim3; _k++)
        for(int _l = 0; _l < out_dim4; _l++) {
          fortran_output[_i][_j][_k][_l] =
              test.r3_Qdp_copy2[_i][_j][_k][_l];
        }

  remap_q_ppm_c_callable(&(fortran_output[0][0][0][0]), NP,
                         out_dim1, &(dp1f[0][0][0]),
                         &(dp2f[0][0][0]), test.alg);

  for(int _i = 0; _i < out_dim1; _i++)
    for(int _j = 0; _j < out_dim2; _j++)
      for(int _k = 0; _k < out_dim3; _k++)
        for(int _l = 0; _l < out_dim4; _l++) {
          Real coutput0 = test.r3_Qdp[_i][_j][_k][_l];

          // std::cout <<" indices " << _i << " " << _j << "
          // " << _k << " " << _l << "\n";  std::cout <<
          // std::setprecision(20)
          //<<"F result = " <<
          //fortran_output[_i][_j][_k][_l]
          //<< ", C output = " << coutput0 << "\n";

          REQUIRE(
              !std::isnan(fortran_output[_i][_j][_k][_l]));
          REQUIRE(!std::isnan(coutput0));
          REQUIRE(std::numeric_limits<Real>::epsilon() >=
                  compare_answers(
                      fortran_output[_i][_j][_k][_l],
                      coutput0, 128.0));

        }  // _l

  std::cout << "test remap_Q_ppm (alg=" << _alg
            << ") finished. \n";
};  // end of testbody_compute_ppm

TEST_CASE("Testing compute_ppm_grids() with alg=1",
          "compute_ppm_grids, alg=1") {
  const int _alg = 1;
  testbody_compute_ppm_grids(_alg);
};  // end fo test compute_ppm_grids, alg=1

TEST_CASE("Testing compute_ppm_grids() with alg=2",
          "compute_ppm_grids, alg=2") {
  const int _alg = 2;
  testbody_compute_ppm_grids(_alg);
};  // end fo test compute_ppm_grids, alg=1

TEST_CASE("Testing compute_ppm() with alg=1",
          "compute_ppm, alg=1") {
  const int _alg = 1;
  testbody_compute_ppm(_alg);
};  // end fo test compute_ppm, alg=1

TEST_CASE("Testing compute_ppm() with alg=2",
          "compute_ppm, alg=2") {
  const int _alg = 2;
  testbody_compute_ppm(_alg);
};  // end fo test compute_ppm, alg=2

TEST_CASE("Testing remap_Q_ppm() with alg=1",
          "remap_Q_ppm, alg=1") {
  const int _alg = 1;
  testbody_remap_Q_ppm(_alg);
};  // end fo test compute_ppm, alg=1

TEST_CASE("Testing remap_Q_ppm() with alg=2",
          "remap_Q_ppm, alg=2") {
  const int _alg = 2;
  testbody_remap_Q_ppm(_alg);
};  // end fo test compute_ppm, alg=2
#endif // 0
