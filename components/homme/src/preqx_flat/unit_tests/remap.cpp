#ifndef REMAP_CPP
#define REMAP_CPP

#include <catch/catch.hpp>

#include <limits>

//#include "CaarControl.hpp"
//#include "CaarFunctor.hpp"
//#include "CaarRegion.hpp"
#include "Dimensions.hpp"
//#include "RemapDimensions.hpp"
//#include "KernelVariables.hpp"
#include "Types.hpp"

#include <assert.h>
#include <stdio.h>
#include <random>

using namespace Homme;

//extern "C" {
//void compute_ppm_grids_c_callable(const Real * dx,
//Real * rslt, const int alg);
//}//extern C


//save dims for all these intermediate arrays somewhere?
void compute_ppm_grids(const Real dx[DX_DIM], 
  Real rslt[RSLT_DIM1][RSLT_DIM2], 
  const int alg)
{


std::cout << "here in f() 1 \n";

if((alg != 1)&&(alg != 2)){
  //ABORT
}

std::cout << "here in f() 2 \n";
const int nlev = NUM_PHYSICAL_LEV;

//where to get nlev from?
int indB = 2;
int indE = nlev - 1;
if (alg == 2){
  indB = 0;
  indE = nlev+1;
}

std::cout << "here in f() 3 \n";
//compared to F: all addressing dx[j] should go to dx[j+1]
//all addressing of rslt(i,j) should go rslt(i-1,j)
//so, keep loop vars the same, but change addressing of dx, rslt only
for(int j = indB; j <= indE; j++){
  rslt[j][0] = dx[j+1] / ( dx[j] + dx[j+1] + dx[j+2] );
  rslt[j][1] = ( 2.0*dx[j] + dx[j+1] ) / ( dx[j+2] + dx[j+1] );
  rslt[j][2] = ( dx[j+1] + 2.0*dx[j+2] ) / ( dx[j] + dx[j+1] );
}

std::cout << "here in f() 4 \n";
if(alg == 2){
  indB = 2;
  indE = nlev-2;
}else{
  indB = 0;
  indE = nlev;
}

std::cout << "nlev=" << nlev << ", alg = " << alg <<" here in f() 5 \n";
for(int j = indB; j <= indE; j++){

std::cout << "in loop, j=" << j << "\n";

  rslt[j][3] = dx[j+1] / ( dx[j+1] + dx[j+2] );

  rslt[j][4] = 1.0/ ( dx[j] + dx[j+1] + dx[j+2] + dx[j+3] );

  rslt[j][5] = ( 2.*dx[j+2]*dx[j+1] ) / ( dx[j+1] + dx[j+2] );

  rslt[j][6] = ( dx[j] + dx[j+1] ) / ( 2.*dx[j+1] + dx[j+2] );

  rslt[j][7] = ( dx[j+3] + dx[j+2] ) / ( 2.*dx[j+2] + dx[j+1] );

  rslt[j][8] = dx[j+1]*( dx[j] + dx[j+1] ) / ( 2.*dx[j+1] + dx[j+2] );

  rslt[j][9] = dx[j+2]*( dx[j+2] + dx[j+3] ) / ( dx[j+1] + 2.*dx[j+2] );
}


}


#endif //REMAP_CPP


