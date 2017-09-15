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

void compute_ppm_c_callable(
const Real * a,
const Real * dx,
Real * coefs,
const int &alg);


};  // extern C


class remap_test {
 public:
  remap_test(int _alg): 
  alg(_alg)
  {
    std::random_device rd;
    rngAlg engine(rd());
//in case of routine compute_pm_grids dx should always be positive
//it is thickness of the grid.
    genRandArray(
        r1_dx, r1_dx_dim, engine,
        std::uniform_real_distribution<Real>(0.0,1000.0));

//in case of routine compute_ppm it is not clear what input is.
    genRandArray(
        r2_a, r2_a_dim, engine,
        std::uniform_real_distribution<Real>(0.0,1000.0));
    genRandArray(
        &(r2_dx[0][0]), r2_dx_dim1*r2_dx_dim2, engine,
        std::uniform_real_distribution<Real>(0.0,1000.0));


//    genRandArray(
//        &(rslt[0][0]), ( RSLT_DIM1*RSLT_DIM2 ), engine,
//        std::uniform_real_distribution<Real>(-1000.0,1000.0));

  for(int _i = 0; _i < r1_rslt_dim1; ++_i) 
    for(int _j = 0; _j < r1_rslt_dim2; ++_j) 
       r1_rslt[_i][_j] = 0.0;

  for(int _i = 0; _i < r2_coefs_dim1; ++_i)
    for(int _j = 0; _j < r2_coefs_dim2; ++_j)
       r2_coefs[_i][_j] = 0.0;


//for debugging, assign 1 to everything
  for(int _i = 0; _i < r2_dx_dim1; ++_i)
    for(int _j = 0; _j < r2_dx_dim2; ++_j)
       r2_dx[_i][_j] = 1.0;

/*  for(int _i = 0; _i < r2_a_dim; ++_i)
       r2_a[_i] = 1.0;
*/

  }  // end of constructor

//let's avoid static
//Since there are many routines that take in/out vars with same names,
//and since we want to keep names consistens, let's mark
//each var with rN_ , where N is number of routine under development.

//routine compute_ppm_grids
  const int r1_dx_dim = NLEVP4;
  const int r1_rslt_dim1 = NLEVP2;
  const int r1_rslt_dim2 = DIM10;
  Real r1_dx[NLEVP4];
  Real r1_rslt [NLEVP2][DIM10];

//routine compute_ppm_grids
  const int r2_a_dim = NLEVP4;
  const int r2_dx_dim1 = NLEVP2;
  const int r2_dx_dim2 = DIM10;
  const int r2_coefs_dim1 = NLEV;
  const int r2_coefs_dim2 = DIM3;
  Real r2_a[NLEVP4];
  Real r2_dx[NLEVP2][DIM10];
  Real r2_coefs[NLEV][DIM3];

  int  alg;

  void run_compute_ppm_grids(){
    compute_ppm_grids(r1_dx,r1_rslt,alg);
  }
  void run_compute_ppm(){
    compute_ppm(r2_a,r2_dx,r2_coefs,alg);
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
  const int out_len1 = test.r1_rslt_dim1,
            out_len2 = test.r1_rslt_dim2,
            dx_dim = test.r1_dx_dim;
                      
  Real fortran_output[out_len1][out_len2];
  // F input
  Real dxf[dx_dim];

  for(int _i = 0; _i < dx_dim; _i++){
    dxf[_i] = test.r1_dx[_i];
  }

  // running F version of operator
  compute_ppm_grids_c_callable( &(dxf[0]), &(fortran_output[0][0]), test.alg );

  // compare with the part from C run
  for(int _i = 0; _i < out_len1; ++_i) {
    for(int _j = 0; _j < out_len2; ++_j) {
       Real coutput0 = test.r1_rslt[_i][_j];
       
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

  const int out_len1 = test.r1_rslt_dim1,
            out_len2 = test.r1_rslt_dim2,
            dx_len = test.r1_dx_dim;

  Real fortran_output[out_len1][out_len2];

  Real dxf[dx_len];

  for(int _i = 0; _i < dx_len; _i++){
    dxf[_i] = test.r1_dx[_i];
  }

  compute_ppm_grids_c_callable( &(dxf[0]), &(fortran_output[0][0]), test.alg );

  for(int _i = 0; _i < out_len1; ++_i) {
    for(int _j = 0; _j < out_len2; ++_j) {
       Real coutput0 = test.r1_rslt[_i][_j];
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


TEST_CASE("Testing compute_ppm() with alg=1","compute_ppm, alg=1") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int iterations = 10;

  constexpr const int vertical_alg = 1;

std::cout << "here 1 \n";

  remap_test test(vertical_alg);

std::cout << "here 2\n";
  test.run_compute_ppm();
std::cout << "here 3 \n";

  const int out_dim1 = test.r2_coefs_dim1,
            out_dim2 = test.r2_coefs_dim2,
            dx_dim1 = test.r2_dx_dim1,
            dx_dim2 = test.r2_dx_dim2,
            a_dim = test.r2_a_dim;

  Real fortran_output[out_dim1][out_dim2];
  Real dxf[dx_dim1][dx_dim2];
  Real af[a_dim];

std::cout << "here 4 \n";
std::cout << "dx dim 1, dx dim 1 " << dx_dim1 << " "<< dx_dim2 << "\n";
  for(int _i = 0; _i < dx_dim1; _i++)
  for(int _j = 0; _j < dx_dim2; _j++){

std::cout << "i,j " << _i << " " << _j << "\n";
    dxf[_i][_j] = test.r2_dx[_i][_j];
  }


std::cout << "here 5 \n";

  for(int _i = 0; _i < a_dim; _i++){
    af[_i] = test.r2_a[_i];
  }


std::cout << "here 6 \n";

  compute_ppm_c_callable( &(af[0]),&(dxf[0][0]),&(fortran_output[0][0]), test.alg );


std::cout << "here 7 \n";

  for(int _i = 0; _i < out_dim1; ++_i) {
    for(int _j = 0; _j < out_dim2; ++_j) {
       Real coutput0 = test.r2_coefs[_i][_j];
     std::cout << std::setprecision(20)
<<"F result = " << fortran_output[_i][_j] << ", C output = " << coutput0 << "\n";

       REQUIRE(!std::isnan(fortran_output[_i][_j]));
       REQUIRE(!std::isnan(coutput0));
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        fortran_output[_i][_j],
                        coutput0, 128.0));
    }  // _j
  }    // _i

  std::cout << "test compute_ppm (alg=2) finished. \n";
};  // end fo test compute_ppm_grids, alg=1













