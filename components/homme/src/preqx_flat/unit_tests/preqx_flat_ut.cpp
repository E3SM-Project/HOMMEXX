#include <catch/catch.hpp>

#include <limits>
#include <random>
#include <iostream>
#include <iomanip>

#include "Types.hpp"
#include "Dimensions.hpp"
#include "PhysicalConstants.hpp"

using namespace Homme;

extern "C"
{
void caar_compute_pressure_c (const int& nets, const int& nete, const int& nelemd, const int& n0,
                              const Real& hyai_ps0, RCPtr& p_ptr, RCPtr& dp_ptr);

void caar_compute_pressure_f90 (const int& nets, const int& nete, const int& nelemd, const int& n0,
                                const Real& hyai_ps0, RCPtr& p_ptr, RCPtr& dp_ptr);

void caar_compute_vort_and_div_c (const int& nets, const int& nete, const int& nelemd, const int& n0,
                                  const Real& eta_ave_w, CRCPtr& dvv_ptr, CRCPtr& D_ptr, CRCPtr& Dinv_ptr,
                                  CRCPtr& metdet_ptr, CRCPtr& rmetdet_ptr, CRCPtr& p_ptr, CRCPtr& dp_ptr,
                                  RCPtr& grad_p_ptr, RCPtr& vgrad_p_ptr, CRCPtr& v_ptr, RCPtr& vn0_ptr,
                                  RCPtr& vdp_ptr, RCPtr& div_vdp_ptr, RCPtr& vort_ptr);

void caar_compute_vort_and_div_f90 (const int& nets, const int& nete, const int& nelemd, const int& n0,
                                    const Real& eta_ave_w, CRCPtr& dvv_ptr, CRCPtr& D_ptr, CRCPtr& Dinv_ptr,
                                    CRCPtr& metdet_ptr, CRCPtr& rmetdet_ptr, CRCPtr& p_ptr, CRCPtr& dp_ptr,
                                    RCPtr& grad_p_ptr, RCPtr& vgrad_p_ptr, CRCPtr& v_ptr, RCPtr& vn0_ptr,
                                    RCPtr& vdp_ptr, RCPtr& div_vdp_ptr, RCPtr& vort_ptr);

void caar_compute_t_v_c (const int& nets, const int& nete, const int& nelemd,
                         const int& n0,   const int& qn0,  const int& use_cpstar,
                         RCPtr& T_v_ptr,  RCPtr& kappa_star_ptr,
                         CRCPtr& dp_ptr,  CRCPtr& Temp_ptr, CRCPtr& Qdp_ptr);

void caar_compute_t_v_f90 (const int& nets, const int& nete, const int& nelemd,
                           const int& n0,   const int& qn0,  const int& use_cpstar,
                           RCPtr& T_v_ptr,  RCPtr& kappa_star_ptr,
                           CRCPtr& dp_ptr,  CRCPtr& Temp_ptr, CRCPtr& Qdp_ptr);
} // extern "C"

template <typename rngAlg, typename PDF>
void genRandArray(Real* const x, int length, rngAlg& engine, PDF& pdf)
{
  for(int i=0; i<length; ++i)
  {
    x[i] = pdf(engine);
  }
}

void flip_array_np_np (RCPtr& arr_f, RCPtr& arr_c)
{
  // Note: the np x np block should not be flipped, that is,
  //       (igp,jgp)_f -> (igp,jgp)_c
  const int size     = NP * NP;
  for (int i=0; i<size; ++i)
  {
    const int igp  = i % NP;
    const int jgp  = i / NP;

    arr_c[ igp*NP + jgp ] = arr_f[i];
  }
}

void flip_array_np_np_nelems (RCPtr& arr_f, RCPtr& arr_c, const int num_elems)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,ielem)_f -> (ielem,igp,jgp)_c
  const int size     = NP * NP * num_elems;
  const int dim12    = NP * NP;
  for (int i=0; i<size; ++i)
  {
    const int igp  =  i % NP;
    const int jgp  = (i / NP) % NP;
    const int ie   = (i / NP) / NP;

    arr_c[ ie*dim12 + igp*NP + jgp ] = arr_f[i];
  }
}

void flip_array_np_np_2_2_nelems (RCPtr& arr_f, RCPtr& arr_c, const int num_elems)
{
  // Note: each np x np block and each 2x2 block should not be flipped, that is,
  //       (igp,jgp,idim,jdim,ielem)_f -> (ielem,idim,jdim,igp,jgp)_c
  const int size     = NP * NP * 2 * 2 * num_elems;
  const int dim1234  = NP * NP * 2 * 2;
  const int dim123   = NP * NP * 2;
  const int dim12    = NP * NP;
  for (int i=0; i<size; ++i)
  {
    const int igp  =      i % NP;
    const int jgp  =     (i / NP) % NP;
    const int idim =   ( (i / NP) / NP ) % 2;
    const int jdim = ( ( (i / NP) / NP ) / 2 ) % 2;
    const int ie   = ( ( (i / NP) / NP ) / 2 ) / 2;

    arr_c[ ie*dim1234 + idim*dim123 + jdim*dim12 + igp*NP + jgp ] = arr_f[i];
  }
}

void flip_array_np_np_nlev_nelems (RCPtr& arr_f, RCPtr& arr_c, const int num_elems)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,ilev,ielem)_f -> (ielem,ilev,igp,jgp)_c
  const int size    = NP * NP * NUM_LEV * num_elems;
  const int dim123  = NP * NP * NUM_LEV;
  const int dim12   = NP * NP;
  for (int i=0; i<size; ++i)
  {
    const int igp =     i % NP;
    const int jgp =    (i / NP) % NP;
    const int ilev = ( (i / NP) / NP ) % NUM_LEV;
    const int ie =   ( (i / NP) / NP ) / NUM_LEV;

    arr_c[ ie*dim123 + ilev*dim12 + igp*NP + jgp ] = arr_f[i];
  }
}

void flip_array_np_np_2_nlev_nelems (RCPtr& arr_f, RCPtr& arr_c, const int num_elems)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,idim,ilev,ielem)_f -> (ielem,ilev,idim,igp,jgp)_c
  const int size     = NP * NP * 2 * NUM_LEV * num_elems;
  const int dim1234  = NP * NP * 2 * NUM_LEV;
  const int dim123   = NP * NP * 2;
  const int dim12    = NP * NP;
  for (int i=0; i<size; ++i)
  {
    const int igp  =      i % NP;
    const int jgp  =     (i / NP) % NP;
    const int idim =   ( (i / NP) / NP ) % 2;
    const int ilev = ( ( (i / NP) / NP ) / 2 ) % NUM_LEV;
    const int ie   = ( ( (i / NP) / NP ) / 2 ) / NUM_LEV;

    arr_c[ ie*dim1234 + ilev*dim123 + idim*dim12 + igp*NP + jgp ] = arr_f[i];
  }
}

void flip_array_np_np_2_nlev_timelevels_nelems (RCPtr& arr_f, RCPtr& arr_c, const int num_elems)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,idim,ilev,itlev,ielem)_f -> (ielem,itlev,ilev,idim,igp,jgp)_c
  const int size     = NP * NP * 2 * NUM_LEV * NUM_TIME_LEVELS * num_elems;
  const int dim12345 = NP * NP * 2 * NUM_LEV * NUM_TIME_LEVELS;
  const int dim1234  = NP * NP * 2 * NUM_LEV;
  const int dim123   = NP * NP * 2;
  const int dim12    = NP * NP;
  for (int i=0; i<size; ++i)
  {
    const int igp  =        i % NP;
    const int jgp  =       (i / NP) % NP;
    const int dim  =     ( (i / NP) / NP ) % 2;
    const int ilev =   ( ( (i / NP) / NP ) / 2 ) % NUM_LEV;
    const int itl  = ( ( ( (i / NP) / NP ) / 2 ) / NUM_LEV ) % NUM_TIME_LEVELS;
    const int ie   = ( ( ( (i / NP) / NP ) / 2 ) / NUM_LEV ) / NUM_TIME_LEVELS;

    arr_c[ ie*dim12345 + itl*dim1234 + ilev*dim123 + dim*dim12 + igp*NP + jgp ] = arr_f[i];
  }
}

void flip_array_np_np_nlev_timelevels_nelems (RCPtr& arr_f, RCPtr& arr_c, const int num_elems)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,ilev,itlev,ielem)_f -> (ielem,itlev,ilev,igp,jgp)_c
  const int size    = NP * NP * NUM_LEV * NUM_TIME_LEVELS * num_elems;
  const int dim1234 = NP * NP * NUM_LEV * NUM_TIME_LEVELS;
  const int dim123  = NP * NP * NUM_LEV;
  const int dim12   = NP * NP;
  for (int i=0; i<size; ++i)
  {
    const int igp = i % NP;
    const int jgp = (i / NP) % NP;
    const int ilev = ( (i / NP) / NP ) % NUM_LEV;
    const int itl = ( ( (i / NP) / NP ) / NUM_LEV ) % NUM_TIME_LEVELS;
    const int ie = ( ( (i / NP) / NP ) / NUM_LEV ) / NUM_TIME_LEVELS;

    arr_c[ ie*dim1234 + itl*dim123 + ilev*dim12 + igp*NP + jgp ] = arr_f[i];
  }
}

void flip_array_Qdp (RCPtr& arr_f, RCPtr& arr_c, const int num_elems)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,ilev,iq,tlq,ielem)_f -> (ielem,tlq,iq,ilev,igp,jgp)_c
  const int size     = NP * NP * NUM_LEV * QSIZE_D * Q_NUM_TIME_LEVELS * num_elems;
  const int dim12345 = NP * NP * NUM_LEV * QSIZE_D * Q_NUM_TIME_LEVELS;
  const int dim1234  = NP * NP * NUM_LEV * QSIZE_D;
  const int dim123   = NP * NP * NUM_LEV;
  const int dim12    = NP * NP;
  for (int i=0; i<size; ++i)
  {
    const int igp  =        i % NP;
    const int jgp  =       (i / NP) % NP;
    const int ilev =     ( (i / NP) / NP ) % NUM_LEV;
    const int iq   =   ( ( (i / NP) / NP ) / NUM_LEV ) % QSIZE_D;
    const int tlq  = ( ( ( (i / NP) / NP ) / NUM_LEV ) / QSIZE_D ) % Q_NUM_TIME_LEVELS;
    const int ie   = ( ( ( (i / NP) / NP ) / NUM_LEV ) / QSIZE_D ) / Q_NUM_TIME_LEVELS;

    arr_c[ ie*dim12345 + tlq*dim1234 + iq*dim123 + ilev*dim12 + igp*NP + jgp ] = arr_f[i];
  }
}

Real compare_answers (Real target, Real computed,
                      Real relative_coeff = 1.0)
{
  Real denom = 1.0;
  if (relative_coeff>0.0 && target!=0.0)
  {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target-computed) / denom;
}

TEST_CASE ("compute_and_apply_rhs / 1", "compute_pressure")
{
  constexpr int num_elems = 10;
  constexpr int p_length = num_elems*NUM_LEV*NP*NP;
  constexpr int dp_length = num_elems*NUM_TIME_LEVELS*NUM_LEV*NP*NP;

  // We need two inputs, since C routines assume C ordering
  Real* dp_f90 = new Real[dp_length];
  Real* dp_cxx = new Real[dp_length];

  Real* p_f90 = new Real[p_length];
  Real* p_cxx = new Real[p_length];

  Real* p_f90_cxx = new Real[p_length];

  constexpr int num_rand_test = 10;

  SECTION ("random_test")
  {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    using udi_type = std::uniform_int_distribution<int>;

    udi_type dnets (1, num_elems);
    std::uniform_real_distribution<Real> dreal (0,1);

    for (int itest=0; itest<num_rand_test; ++itest)
    {
      const int nets = dnets(engine);
      const int nete = udi_type(std::min(nets+1, num_elems),num_elems)(engine);
      const int n0   = udi_type(1,3)(engine);
      const Real hyai_ps0 = dreal(engine);

      // To avoid false checks on elements not actually processed in this random test
      for (int i=0; i<p_length; ++i)
      {
        p_cxx[i] = p_f90[i] = 0.0;
      }

      // Initialize input(s)
      genRandArray(dp_f90, dp_length, engine, dreal);
      flip_array_np_np_nlev_timelevels_nelems(dp_f90, dp_cxx, num_elems);

      // Compute
      caar_compute_pressure_c   (nets, nete, num_elems, n0, hyai_ps0, p_cxx, dp_cxx);
      caar_compute_pressure_f90 (nets, nete, num_elems, n0, hyai_ps0, p_f90, dp_f90);

      // Check the answer
      flip_array_np_np_nlev_nelems (p_f90, p_f90_cxx, num_elems);
      for (int i=0; i<p_length; ++i)
      {
        REQUIRE(compare_answers(p_f90_cxx[i],p_cxx[i]) == 0.0);
      }
    }
  }

  delete[] dp_cxx;
  delete[] dp_f90;
  delete[] p_f90;
  delete[] p_cxx;
  delete[] p_f90_cxx;
}

TEST_CASE ("compute_and_apply_rhs / 2", "compute_vort_and_div")
{
  constexpr int num_elems = 10;
  constexpr int size2d    = num_elems*NP*NP;
  constexpr int size3d    = num_elems*NUM_LEV*NP*NP;
  constexpr int size4d    = num_elems*NUM_TIME_LEVELS*NUM_LEV*NP*NP;

  // We need two inputs, since C routines assume C ordering
  Real* dvv_f90     = new Real[NP*NP];
  Real* D_f90       = new Real[4*size2d];
  Real* Dinv_f90    = new Real[4*size2d];
  Real* metdet_f90  = new Real[size2d];
  Real* rmetdet_f90 = new Real[size2d];
  Real* dp_f90      = new Real[size4d];
  Real* v_f90       = new Real[2*size4d];
  Real* p_f90       = new Real[size3d];
  Real* vn0_f90     = new Real[2*size3d];
  Real* grad_p_f90  = new Real[2*size3d];
  Real* vgrad_p_f90 = new Real[size3d];
  Real* vdp_f90     = new Real[2*size3d];
  Real* div_vdp_f90 = new Real[size3d];
  Real* vort_f90    = new Real[size3d];

  Real* dvv_cxx     = new Real[NP*NP];
  Real* D_cxx       = new Real[4*size2d];
  Real* Dinv_cxx    = new Real[4*size2d];
  Real* metdet_cxx  = new Real[size2d];
  Real* rmetdet_cxx = new Real[size2d];
  Real* dp_cxx      = new Real[size4d];
  Real* v_cxx       = new Real[2*size4d];
  Real* p_cxx       = new Real[size3d];
  Real* vn0_cxx     = new Real[2*size3d];
  Real* grad_p_cxx  = new Real[2*size3d];
  Real* vgrad_p_cxx = new Real[size3d];
  Real* vdp_cxx     = new Real[2*size3d];
  Real* div_vdp_cxx = new Real[size3d];
  Real* vort_cxx    = new Real[size3d];

  Real* vn0_f90_cxx     = new Real[2*size3d];
  Real* grad_p_f90_cxx  = new Real[2*size3d];
  Real* vgrad_p_f90_cxx = new Real[size3d];
  Real* vdp_f90_cxx     = new Real[2*size3d];
  Real* div_vdp_f90_cxx = new Real[size3d];
  Real* vort_f90_cxx    = new Real[size3d];

  constexpr int num_rand_test = 10;

  SECTION ("random_test")
  {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    using udi_type = std::uniform_int_distribution<int>;

    udi_type dnets (1, num_elems);
    udi_type bernoulli(0,1);
    std::uniform_real_distribution<Real> dreal (0.000001,1);

    for (int itest=0; itest<num_rand_test; ++itest)
    {
      const int nets = dnets(engine);
      const int nete = udi_type(std::min(nets+1, num_elems),num_elems)(engine);
      const int n0   = udi_type(1,3)(engine);
      Real eta_ave_w = 0.0;
      if (bernoulli(engine)>0)
      {
        eta_ave_w = dreal(engine);
      }

      // To avoid false checks on elements not actually processed in this random test
      for (int i=0; i<size3d; ++i)
      {
        vort_cxx[i]    = vort_f90[i] = 0.0;
        vgrad_p_cxx[i] = vgrad_p_f90[i] = 0.0;
        div_vdp_cxx[i] = div_vdp_f90[i] = 0.0;
        for (int dim=0; dim<2; ++dim)
        {
          vn0_cxx[i+dim*size3d]     = vn0_f90[i+dim*size3d] = 0.0;
          grad_p_cxx[i+dim*size3d]  = grad_p_f90[i+dim*size3d] = 0.0;
          vdp_cxx[i+dim*size3d]     = vdp_f90[i+dim*size3d] = 0.0;
        }
      }

      // Initialize input(s)
      genRandArray(dvv_f90, NP*NP, engine, dreal);
      genRandArray(D_f90, 4*size2d, engine, dreal);
      genRandArray(Dinv_f90, 4*size2d, engine, dreal);
      genRandArray(metdet_f90, size2d, engine, dreal);
      genRandArray(rmetdet_f90, size2d, engine, dreal);
      genRandArray(p_f90, size3d, engine, dreal);
      genRandArray(dp_f90, size4d, engine, dreal);
      genRandArray(v_f90, 2*size4d, engine, dreal);

      flip_array_np_np(dvv_f90, dvv_cxx);
      flip_array_np_np_2_2_nelems(D_f90, D_cxx, num_elems);
      flip_array_np_np_2_2_nelems(Dinv_f90, Dinv_cxx, num_elems);
      flip_array_np_np_nelems(metdet_f90,  metdet_cxx, num_elems);
      flip_array_np_np_nelems(rmetdet_f90, rmetdet_cxx, num_elems);
      flip_array_np_np_nlev_nelems(p_f90, p_cxx, num_elems);
      flip_array_np_np_nlev_timelevels_nelems(dp_f90, dp_cxx, num_elems);
      flip_array_np_np_2_nlev_timelevels_nelems(v_f90, v_cxx, num_elems);

      // Compute
      caar_compute_vort_and_div_c   (nets, nete, num_elems, n0, eta_ave_w, dvv_cxx, D_cxx, Dinv_cxx,
                                     metdet_cxx, rmetdet_cxx, p_cxx, dp_cxx, grad_p_cxx, vgrad_p_cxx,
                                     v_cxx, vn0_cxx, vdp_cxx, div_vdp_cxx, vort_cxx);
      caar_compute_vort_and_div_f90 (nets, nete, num_elems, n0, eta_ave_w, dvv_f90, D_f90, Dinv_f90,
                                     metdet_f90, rmetdet_f90, p_f90, dp_f90, grad_p_f90, vgrad_p_f90,
                                     v_f90, vn0_f90, vdp_f90, div_vdp_f90, vort_f90);

      // Check the answers
      flip_array_np_np_nlev_nelems (vgrad_p_f90, vgrad_p_f90_cxx, num_elems);
      flip_array_np_np_nlev_nelems (div_vdp_f90, div_vdp_f90_cxx, num_elems);
      flip_array_np_np_nlev_nelems (vort_f90, vort_f90_cxx, num_elems);

      for (int i=0; i<size3d; ++i)
      {
        REQUIRE(compare_answers(vgrad_p_f90_cxx[i],vgrad_p_cxx[i]) == 0.0);
        REQUIRE(compare_answers(div_vdp_f90_cxx[i],div_vdp_cxx[i]) == 0.0);
        REQUIRE(compare_answers(vort_f90_cxx[i],vort_cxx[i]) == 0.0);
      }
      flip_array_np_np_2_nlev_nelems (vn0_f90, vn0_f90_cxx, num_elems);
      flip_array_np_np_2_nlev_nelems (grad_p_f90, grad_p_f90_cxx, num_elems);
      flip_array_np_np_2_nlev_nelems (vdp_f90, vdp_f90_cxx, num_elems);
      for (int i=0; i<2*size3d; ++i)
      {
        REQUIRE(compare_answers(vn0_f90_cxx[i],vn0_cxx[i]) == 0.0);
        REQUIRE(compare_answers(grad_p_f90_cxx[i],grad_p_cxx[i]) == 0.0);
        REQUIRE(compare_answers(vdp_f90_cxx[i],vdp_cxx[i]) == 0.0);
      }
    }
  }

  delete[] dvv_f90;
  delete[] D_f90;
  delete[] Dinv_f90;
  delete[] metdet_f90;
  delete[] rmetdet_f90;
  delete[] dp_f90;
  delete[] v_f90;
  delete[] p_f90;
  delete[] vn0_f90;
  delete[] grad_p_f90;
  delete[] vgrad_p_f90;
  delete[] vdp_f90;
  delete[] div_vdp_f90;
  delete[] vort_f90;

  delete[] dvv_cxx;
  delete[] D_cxx;
  delete[] Dinv_cxx;
  delete[] metdet_cxx;
  delete[] rmetdet_cxx;
  delete[] dp_cxx;
  delete[] v_cxx;
  delete[] p_cxx;
  delete[] vn0_cxx;
  delete[] grad_p_cxx;
  delete[] vgrad_p_cxx;
  delete[] vdp_cxx;
  delete[] div_vdp_cxx;
  delete[] vort_cxx;

  delete[] vn0_f90_cxx;
  delete[] grad_p_f90_cxx;
  delete[] vgrad_p_f90_cxx;
  delete[] vdp_f90_cxx;
  delete[] div_vdp_f90_cxx;
  delete[] vort_f90_cxx;
}

TEST_CASE ("compute_and_apply_rhs / 3", "compute_T_v")
{
  constexpr int num_elems = 10;
  constexpr int size3d    = num_elems*NUM_LEV*NP*NP;
  constexpr int size4d    = num_elems*NUM_TIME_LEVELS*NUM_LEV*NP*NP;
  constexpr int sizeQ     = num_elems*Q_NUM_TIME_LEVELS*QSIZE_D*NUM_LEV*NP*NP;

  // We need two inputs, since C routines assume C ordering
  Real* T_f90          = new Real[size4d];
  Real* dp_f90         = new Real[size4d];
  Real* Qdp_f90        = new Real[sizeQ];
  Real* T_v_f90        = new Real[size3d];
  Real* kappa_star_f90 = new Real[size3d];

  Real* T_cxx          = new Real[size4d];
  Real* dp_cxx         = new Real[size4d];
  Real* Qdp_cxx        = new Real[sizeQ];
  Real* T_v_cxx        = new Real[size3d];
  Real* kappa_star_cxx = new Real[size3d];

  Real* T_f90_cxx          = new Real[size4d];
  Real* dp_f90_cxx         = new Real[size4d];
  Real* Qdp_f90_cxx        = new Real[sizeQ];
  Real* T_v_f90_cxx        = new Real[size3d];
  Real* kappa_star_f90_cxx = new Real[size3d];

  constexpr int num_rand_test = 10;

  SECTION ("random_test")
  {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    using udi_type = std::uniform_int_distribution<int>;

    udi_type dnets (1, num_elems);
    udi_type bernoulli(0,1);
    std::uniform_real_distribution<Real> dreal (0.000001,1);

    for (int itest=0; itest<num_rand_test; ++itest)
    {
      const int nets = dnets(engine);
      const int nete = udi_type(std::min(nets+1, num_elems),num_elems)(engine);
      const int n0   = udi_type(1,NUM_TIME_LEVELS)(engine);
      const int qn0  = bernoulli(engine)==0 ? -1 : udi_type(1,Q_NUM_TIME_LEVELS)(engine);
      const int use_cpstar = bernoulli(engine);

      // To avoid false checks on elements not actually processed in this random test
      for (int i=0; i<size3d; ++i)
      {
        T_v_f90[i]        = T_v_cxx[i] = 0.0;
        kappa_star_f90[i] = kappa_star_cxx[i] = 0.0;
      }

      // Initialize input(s)
      genRandArray(Qdp_f90, sizeQ, engine, dreal);
      genRandArray(dp_f90, size4d, engine, dreal);
      genRandArray(T_f90, size4d, engine, dreal);

      flip_array_Qdp(Qdp_f90, Qdp_cxx, num_elems);
      flip_array_np_np_nlev_timelevels_nelems(dp_f90, dp_cxx, num_elems);
      flip_array_np_np_nlev_timelevels_nelems(T_f90, T_cxx, num_elems);

      // Compute
      caar_compute_t_v_c   (nets, nete, num_elems, n0, qn0, use_cpstar,
                            T_v_cxx, kappa_star_cxx, dp_cxx, T_cxx, Qdp_cxx);
      caar_compute_t_v_f90 (nets, nete, num_elems, n0, qn0, use_cpstar,
                            T_v_f90, kappa_star_f90, dp_f90, T_f90, Qdp_f90);

      // Check the answers
      flip_array_np_np_nlev_nelems (T_v_f90,        T_v_f90_cxx,        num_elems);
      flip_array_np_np_nlev_nelems (kappa_star_f90, kappa_star_f90_cxx, num_elems);

      for (int i=0; i<size3d; ++i)
      {
        REQUIRE(compare_answers(T_v_f90_cxx[i],T_v_cxx[i]) == 0.0);
        REQUIRE(compare_answers(kappa_star_f90_cxx[i],kappa_star_cxx[i]) == 0.0);
      }
    }
  }

  delete[] Qdp_f90;
  delete[] dp_f90;
  delete[] T_f90;
  delete[] T_v_f90;
  delete[] kappa_star_f90;

  delete[] Qdp_cxx;
  delete[] dp_cxx;
  delete[] T_cxx;
  delete[] T_v_cxx;
  delete[] kappa_star_cxx;

  delete[] T_v_f90_cxx;
  delete[] kappa_star_f90_cxx;
}
