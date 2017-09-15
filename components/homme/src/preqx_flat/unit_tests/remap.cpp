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
void compute_ppm_grids(const Real dx[NLEVP4], 
  Real rslt[NLEVP2][DIM10], 
  const int alg)
{

if((alg != 1)&&(alg != 2)){
  //ABORT
}

const int nlev = NLEV;

int indB = 2;
int indE = nlev - 1;
if (alg == 1){
  indB = 0;
  indE = nlev+1;
}

//compared to F: all addressing dx[j] should go to dx[j+1]
//all addressing of rslt(i,j) should go rslt(i-1,j)
//so, keep loop vars the same, but change addressing of dx, rslt only
for(int j = indB; j <= indE; j++){
  rslt[j][0] = dx[j+1] / ( dx[j] + dx[j+1] + dx[j+2] );
  rslt[j][1] = ( 2.0*dx[j] + dx[j+1] ) / ( dx[j+2] + dx[j+1] );
  rslt[j][2] = ( dx[j+1] + 2.0*dx[j+2] ) / ( dx[j] + dx[j+1] );
}

if(alg == 2){
  indB = 2;
  indE = nlev-2;
}else{
  indB = 0;
  indE = nlev;
}

for(int j = indB; j <= indE; j++){

  rslt[j][3] = dx[j+1] / ( dx[j+1] + dx[j+2] );

  rslt[j][4] = 1.0/ ( dx[j] + dx[j+1] + dx[j+2] + dx[j+3] );

  rslt[j][5] = ( 2.*dx[j+2]*dx[j+1] ) / ( dx[j+1] + dx[j+2] );

  rslt[j][6] = ( dx[j] + dx[j+1] ) / ( 2.*dx[j+1] + dx[j+2] );

  rslt[j][7] = ( dx[j+3] + dx[j+2] ) / ( 2.*dx[j+2] + dx[j+1] );

  rslt[j][8] = dx[j+1]*( dx[j] + dx[j+1] ) / ( 2.*dx[j+1] + dx[j+2] );

  rslt[j][9] = dx[j+2]*( dx[j+2] + dx[j+3] ) / ( dx[j+1] + 2.*dx[j+2] );
}//end of for j


}//end of compute_ppm_grids()


void compute_ppm(const Real a[NLEVP4], 
  const Real dx[NLEVP2][DIM10],
  Real coefs[NLEV][DIM3],
  const int alg)
{

  if((alg != 1)&&(alg != 2)){
     //ABORT
  }

  const int nlev = NLEV;
//compared to F: addressing changes like
//F a(i) -> C a(i+1)
//F dx(i,j) -> C dx(j,i-1)
//F coefs(i,j) -> C coefs(j-1,i)
//auxiliary vars:
//ai(i) -> ai(i), dma(i) -> dma(i)

Real dma[NLEVP2]; 
Real ai[NLEVP1];

int indB = 2;
int indE = nlev - 1;
if (alg == 1){
  indB = 0;
  indE = nlev+1;
}

for(int j = indB; j <= indE; j++){
  Real da = dx[j][0] * ( dx[j][1] * ( a[j+2] - a[j+1] ) + dx[j][2] * ( a[j+1] - a[j] ) );

  dma[j] = std::min( std::min( fabs(da),
                     2.0 * fabs( a[j+1] - a[j] )) ,
                2. * fabs( a[j+2] - a[j+1] )  ) * 
           std::copysign(1.0,da);
 
 if ( ( a[j+2] - a[j+1] ) * ( a[j+1] - a[j] ) <= 0. ) 
    dma[j] = 0.;
}//end of j loop

if(alg == 2){
  indB = 2;
  indE = nlev-2;
}else{
  indB = 0;
  indE = nlev;
}

for(int j = indB; j <= indE; j++){
ai[j] = a[j] + 
dx[j][3] * ( a[j+2] - a[j+1] ) 
+ dx[j][4] * (  dx[j][5] * ( dx[j][6] - dx[j][7] ) * ( a[j+2] - a[j+1] ) 
              - dx[j][8] * dma[j+1] + dx[j][9] * dma[j] );
}//end of j loop

if(alg == 2){
  indB = 3;
  indE = nlev-2;
}else{
  indB = 1;
  indE = nlev;
}

for(int j = indB; j <= indE; j++){
    Real al = ai[j-1], ar = ai[j];
    if( (ar - a[j+1] ) * (a[j+1] - al) <= 0. ){
      al = a[j+1];
      ar = a[j+1];
    }
    if ( (ar - al) * (a[j+1] - (al + ar)/2.0) >  (ar - al)*(ar - al) /6.0 ) 
       al = 3.0*a[j+1] - 2.0 * ar;
    if ( (ar - al) * (a[j+1] - (al + ar)/2.0) < -(ar - al)*(ar - al) /6.0 ) 
       ar = 3.0*a[j+1] - 2.0 * al;

//Computed these coefficients from the edge values and cell mean in Maple. Assumes normalized coordinates: xi=(x-x0)/dx
    coefs[j-1][0] = 1.5 * a[j+1] - ( al + ar ) / 4.0;
    coefs[j-1][1] = ar - al;
    coefs[j-1][2] = -6.0 * a[j+1] + 3.0 * ( al + ar );

}//end of j loop

//compared to F: addressing changes like
//F a(i) -> C a(i+1)
//F dx(i,j) -> C dx(j,i-1)
//F coefs(i,j) -> C coefs(j-1,i)
//auxiliary vars:
//ai(i) -> ai(i), dma(i) -> dma(i)

//If we're not using a mirrored boundary condition, then make the two cells bordering the top and bottom
//material boundaries piecewise constant. Zeroing out the first and second moments, and setting the zeroth
//moment to the cell mean is sufficient to maintain conservation.
  if (alg == 2){
//    coefs(0,1:2) = a(1:2)
coefs[0][0]=a[2];
coefs[1][0]=a[3];

//    coefs(1:2,1:2) = 0.
coefs[0][1] = 0.0;
coefs[1][1] = 0.0;
coefs[0][2] = 0.0;
coefs[1][2] = 0.0;

//    coefs(0,nlev-1:nlev) = a(nlev-1:nlev)
coefs[nlev-2][0] = a[nlev];
coefs[nlev-1][0] = a[nlev+1];

//    coefs(1:2,nlev-1:nlev) = 0.D0
coefs[nlev-2][1] = 0.0;
coefs[nlev-1][1] = 0.0;
coefs[nlev-2][2] = 0.0;
coefs[nlev-1][2] = 0.0;
  }

}//end of compute_ppm
  




#endif //REMAP_CPP


