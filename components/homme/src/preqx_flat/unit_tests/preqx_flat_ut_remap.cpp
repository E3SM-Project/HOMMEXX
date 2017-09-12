#include <catch/catch.hpp>
//?
#include <limits>

#include "Dimensions.hpp"
//#include "RemapDimensions.hpp"

#include "KernelVariables.hpp"
#include "Types.hpp"

#include "utils_flat_ut.cpp"
#include "remap.cpp"

#include <assert.h>
#include <stdio.h>
#include <random>

using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {

void compute_ppm_grids_c_callable(
const Real * dx,
Real * rslt,
const int &alg);

};  // extern C


class remap_test {
 public:
  remap_test(int _alg): 
  alg(_alg)
  {
    std::random_device rd;
    rngAlg engine(rd());
    genRandArray(
        dx, dx_dim, engine,
        std::uniform_real_distribution<Real>(0.0,1000.0));
//    genRandArray(
//        &(rslt[0][0]), ( RSLT_DIM1*RSLT_DIM2 ), engine,
//        std::uniform_real_distribution<Real>(-1000.0,1000.0));

  for(int _i = 0; _i < rslt_dim1; ++_i) 
    for(int _j = 0; _j < rslt_dim2; ++_j) 
       rslt[_i][_j] = 0.0;

  }  // end of constructor

//let's avoid static
  const int dx_dim = NLEVP4;
  const int rslt_dim1 = NLEVP2;
  const int rslt_dim2 = DIM10;
  Real dx[NLEVP4];
  Real rslt [NLEVP2][DIM10];
  int  alg;

  void run_compute_ppm_grids(){
    compute_ppm_grids(dx,rslt,alg);
  }

};  // end of class def compute_sphere_op_test_ml

TEST_CASE("Testing compute_ppm_grids() with alg=1","compute_ppm_grids, alg=1") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int iterations = 10;

  constexpr const int vertical_alg = 1;

//put some iteration here?
  remap_test test(vertical_alg);

  test.run_compute_ppm_grids();

  // fortran output
  const int out_len1 = test.rslt_dim1,
            out_len2 = test.rslt_dim2,
            dx_dim = test.dx_dim;
                      
  Real fortran_output[out_len1][out_len2];
  // F input
  Real dxf[dx_dim];

  for(int _i = 0; _i < dx_dim; _i++){
    dxf[_i] = test.dx[_i];
  }

  // running F version of operator
  compute_ppm_grids_c_callable( &(dxf[0]), &(fortran_output[0][0]), test.alg );

  // compare with the part from C run
  for(int _i = 0; _i < out_len1; ++_i) {
    for(int _j = 0; _j < out_len2; ++_j) {
       Real coutput0 = test.rslt[_i][_j];
       
//     std::cout << "F result = " << fortran_output[_i][_j] << ", C output = " << coutput0 << "\n";

       REQUIRE(!std::isnan(fortran_output[_i][_j]));
       REQUIRE(!std::isnan(coutput0));
            // what is 128 here?
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        fortran_output[_i][_j],
                        coutput0, 128.0));
    }  // _j
  }    // _i

  std::cout << "test compute_ppm_grids (alg=1) finished. \n";
};  // end fo test compute_ppm_grids, alg=1


TEST_CASE("Testing compute_ppm_grids() with alg=2","compute_ppm_grids, alg=2") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int iterations = 10;

  constexpr const int vertical_alg = 2;

  remap_test test(vertical_alg);

  test.run_compute_ppm_grids();

  const int out_len1 = test.rslt_dim1,
            out_len2 = test.rslt_dim2,
            dx_len = test.dx_dim;

  Real fortran_output[out_len1][out_len2];

  Real dxf[dx_len];

  for(int _i = 0; _i < dx_len; _i++){
    dxf[_i] = test.dx[_i];
  }

  compute_ppm_grids_c_callable( &(dxf[0]), &(fortran_output[0][0]), test.alg );

  for(int _i = 0; _i < out_len1; ++_i) {
    for(int _j = 0; _j < out_len2; ++_j) {
       Real coutput0 = test.rslt[_i][_j];
       REQUIRE(!std::isnan(fortran_output[_i][_j]));
       REQUIRE(!std::isnan(coutput0));
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        fortran_output[_i][_j],
                        coutput0, 128.0));
    }  // _j
  }    // _i

  std::cout << "test compute_ppm_grids (alg=2) finished. \n";
};  // end fo test compute_ppm_grids, alg=1












