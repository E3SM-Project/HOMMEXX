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
  remap_test()
  {
    std::random_device rd;
    rngAlg engine(rd());
    genRandArray(
        dx, DX_DIM, engine,
        std::uniform_real_distribution<Real>(0.0,1000.0));
//    genRandArray(
//        &(rslt[0][0]), ( RSLT_DIM1*RSLT_DIM2 ), engine,
//        std::uniform_real_distribution<Real>(-1000.0,1000.0));

  for(int _i = 0; _i < RSLT_DIM1; ++_i) 
    for(int _j = 0; _j < RSLT_DIM2; ++_j) 
       rslt[_i][_j] = 0.0;

  }  // end of constructor

  Real dx[DX_DIM];
  Real rslt [RSLT_DIM1][RSLT_DIM2];

  void run_compute_ppm_grids(){
    compute_ppm_grids(dx,rslt,2);
  }

};  // end of class def compute_sphere_op_test_ml

// SHMEM ????

TEST_CASE("Testing compute_ppm_grids()", "compute_ppm_grids") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int iterations = 10;

std::cout << "here 1 \n";

  remap_test test;

std::cout << "here 1a \n";

  test.run_compute_ppm_grids();

std::cout << "here 2 \n";

  // fortran output
  const int out_len1 = RSLT_DIM1,
            out_len2 = RSLT_DIM2;
                      
  Real fortran_output[out_len1][out_len2];
  // F input
  Real dxf[DX_DIM];

  for(int _i = 0; _i < DX_DIM; _i++){
std::cout << "dx=" << test.dx[_i] << "\n";
    dxf[_i] = test.dx[_i];
}

  // running F version of operator
  compute_ppm_grids_c_callable( &(dxf[0]), &(fortran_output[0][0]), 2 );

std::cout << "here after F \n";
  // compare with the part from C run
  for(int _i = 0; _i < out_len1; ++_i) {
    for(int _j = 0; _j < out_len2; ++_j) {
       Real coutput0 = test.rslt[_i][_j];
       
std::cout << "F result = " << fortran_output[_i][_j] << ", C output = " << coutput0 << "\n";

       REQUIRE(!std::isnan(fortran_output[_i][_j]));
       REQUIRE(!std::isnan(coutput0));
            // what is 128 here?
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        fortran_output[_i][_j],
                        coutput0, 128.0));
    }  // _j
  }    // _i

  std::cout << "test compute_ppm_grids finished. \n";
};  // end fo test compute_ppm_grids




