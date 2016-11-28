
#include <catch/catch.hpp>

#include <cmath>
#include <iostream>
#include <random>

#include <dimensions.hpp>
#include <fortran_binding.hpp>
#include <kinds.hpp>
#include <ControlParameters.hpp>
#include <PointersPool.hpp>

using namespace Homme;

extern "C" {

namespace
{
  constexpr const int numelems     = 100;
  constexpr const int dim2d        = 2;
  constexpr const int dim3d        = 3;
  constexpr const int numRandTests = 10;
  constexpr const int rkstages     = 5;

  bool test_running = false;
}

void copy_timelevels_f90(const int &nets, const int &nete,
                         const int &numelems,
                         const int &n_src,
                         const int &n_dist, real *&p_ptr,
                         real *&v_ptr);

void copy_timelevels_c(const int &nets, const int &nete,
                       const int &numelems,
                       const int &n_src, const int &n_dist,
                       real *&p_ptr, real *&v_ptr);

void recover_q_f90(const int &nets, const int &nete,
                   const int &kmass, const int &n0,
                   const int &nelems, real *const &p);

void recover_q_c(const int &nets, const int &nete,
                 const int &kmass, const int &n0,
                 const int &nelems, real *const &p);

void contra2latlon_f90(const int &nets, const int &nete,
                       const int &n0, const int &nelems,
                       real *const &D, real *&v);

void contra2latlon_c(const int &nets, const int &nete,
                     const int &n0, const int &nelems,
                     real *const &D, real *&v);

void init_derivative_f90 ();

void init_physical_constants_f90 ();

void pick_random_control_parameters_f90 ();

void init_elem_f90 (const int& numelems);

void test_laplace_sphere_wk_f90 (const int& nets, const int& nete,
                                 const int& nelems, const int& var_coef_c,
                                 real* const& input_cptr, real*& output_cptr);

void test_vlaplace_sphere_wk_f90 (const int& nets, const int& nete,
                                  const int& nelems,
                                  const int& var_coef_c, const real& nu_ratio,
                                  real* const& input_cptr, real*& output_cptr);

void laplace_sphere_wk_c (const int& nets, const int& nete,
                          const int& numelems, const int& var_coef,
                          real* const& input, real*& output);

void vlaplace_sphere_wk_c (const int& nets, const int& nete,
                           const int& numelems,
                           const int& var_coef, const real& nu_ratio,
                           real* const& input, real*& output);

void test_lapl_pre_bndry_ex_f90 (const int& nets, const int& nete,
                                 const int& numelems, const int& n0,
                                 const bool& var_coef, const real& nu_ratio,
                                 real* &ptens_f90, real* &vtens_f90);

void test_lapl_post_bndry_ex_f90 (const int& nets, const int& nete,
                                  const int& numelems, const real& nu_ratio,
                                  real* &ptens_f90, real* &vtens_f90);

void cleanup_testing_f90 ();

void loop_lapl_pre_bndry_ex_c (const int &nets, const int &nete,
                               const int &nelems, const int& n0,
                               const int& var_coef, const real& nu_ratio,
                               real*& ptens_ptr, real*& vtens_ptr);

void loop_lapl_post_bndry_ex_c (const int &nets, const int &nete,
                                const int &nelems,const real& nu_ratio,
                                real*& ptens_ptr, real*& vtens_ptr);

void add_hv_f90(const int &nets, const int &nete,
               const int &nelems, real *const &spheremp,
               real *&ptens, real *&vtens);

void add_hv_c(const int &nets, const int &nete,
             const int &nelems, real *const &spheremp,
             real *&ptens, real *&vtens);

void recover_dpq_f90(const int &nets, const int &nete,
               const int &kmass, const int &n0,
               const int &nelems, real *const &p);

void recover_dpq_c(const int &nets, const int &nete,
             const int &kmass, const int &n0,
             const int &nelems, real *const &p);

void weighted_rhs_f90(const int &nets, const int &nete,
               const int &numelems,
               real *const &rspheremp_ptr,
               real *const &dinv_ptr, real *&ptens_ptr,
               real *&vtens_ptr);

void weighted_rhs_c(const int &nets, const int &nete,
             const int &numelems,
             real *const &rspheremp_ptr,
             real *const &dinv_ptr, real *&ptens_ptr,
             real *&vtens_ptr);

void rk_stage_f90(const int &nets, const int &nete,
               const int &n0, const int &np1, const int &s,
               const int &rkstages, const int &numelems,
               real *&v_ptr, real *&p_ptr,
               real *const &alpha0_ptr,
               real *const &alpha_ptr,
               real *const &ptens_ptr,
               real *const &vtens_ptr);

void rk_stage_c(const int &nets, const int &nete,
             const int &n0, const int &np1, const int &s,
             const int &rkstages, const int &numelems,
             real *&v_ptr, real *&p_ptr,
             real *const &alpha0_ptr,
             real *const &alpha_ptr, real *const &ptens_ptr,
             real *const &vtens_ptr);
}

template <typename rngAlg, typename dist, typename number>
void genRandArray(number *arr, int arr_len, rngAlg &engine,
                  dist &pdf) {
  for(int i = 0; i < arr_len; i++) {
    arr[i] = pdf(engine);
  }
}

template <typename rngAlg, typename dist, typename number>
void genRandArray(number *arr, int arr_len, rngAlg &engine,
                  dist &&pdf) {
  for(int i = 0; i < arr_len; i++) {
    arr[i] = pdf(engine);
  }
}

template <typename rngAlg, typename dist, typename number>
void genRandTheoryExper(number *arr_theory,
                        number *arr_exper, int arr_len,
                        rngAlg &engine, dist &pdf) {
  for(int i = 0; i < arr_len; i++) {
    arr_theory[i] = pdf(engine);
    arr_exper[i] = arr_theory[i];
  }
}

template <typename rngAlg, typename dist, typename number>
void genRandTheoryExper(number *arr_theory,
                        number *arr_exper, int arr_len,
                        rngAlg &engine, dist &&pdf) {
  for(int i = 0; i < arr_len; i++) {
    arr_theory[i] = pdf(engine);
    arr_exper[i] = arr_theory[i];
  }
}

template<typename data_type>
void flip_array_np_np_nelems (const data_type* arr_f, data_type* arr_c, int nelems)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,ielem)_f -> (ielem,igp,jgp)_c
  const int size    = np * np * nelems;
  const int dim12   = np * np;
  for (int i=0; i<size; ++i)
  {
    const int i1 = i % np;
    const int i2 = (i / np) % np;
    const int i3 = (i / np) / np;

    arr_c[ i3*dim12 + i1*np + i2 ] = arr_f[i];
  }
}

template<typename data_type>
void flip_array_np_np_nlev_nelems (const data_type* arr_f, data_type* arr_c, int nelems)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,ilev,ielem)_f -> (ielem,ilev,igp,jgp)_c
  const int size   = np * np * nlev * nelems;
  const int dim123 = np * np * nlev;
  const int dim12  = np * np;
  for (int i=0; i<size; ++i)
  {
    const int i1 = i % np;
    const int i2 = (i / np) % np;
    const int i3 = ( (i / np) / np ) % nlev;
    const int i4 = ( (i / np) / np ) / nlev;

    arr_c[ i4*dim123 + i3*dim12 + i1*np + i2 ] = arr_f[i];
  }
}

template<typename data_type>
void flip_array_np_np_ndims_nlev_nelems (data_type*& arr_f, data_type*& arr_c, int nelems, int ndims)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,idim,ilev,ielem)_f -> (ielem,ilev,idim,igp,jgp)_c
  const int size    = np * np * ndims * nlev * nelems;
  const int dim1234 = np * np * ndims * nlev;
  const int dim123  = np * np * ndims;
  const int dim12   = np * np;
  for (int i=0; i<size; ++i)
  {
    const int i1 = i % np;
    const int i2 = (i / np) % np;
    const int i3 = ( (i / np) / np ) % ndims;
    const int i4 = ( ( (i / np) / np ) / ndims ) % nlev;
    const int i5 = ( ( (i / np) / np ) / ndims ) / nlev;

    arr_c[ i5*dim1234 + i4*dim123 + i3*dim12 + i1*np + i2 ] = arr_f[i];
  }
}

template<typename data_type>
void flip_array_np_np_ndims1_ndims2_nelems (data_type*& arr_f, data_type*& arr_c, int nelems, int ndims1, int ndims2)
{
  // Note: each np x np and dim1 x dim2 block should not be flipped, that is,
  //       (igp,jgp,idim,jdim,ielem)_f -> (ielem,idim,jdim,igp,jgp)_c
  const int size    = np * np * ndims1 * ndims2 * nelems;
  const int dim1234 = np * np * ndims1 * ndims2;
  const int dim124  = np * np * ndims2;
  const int dim12   = np * np;
  for (int i=0; i<size; ++i)
  {
    const int i1 = i % np;
    const int i2 = (i / np) % np;
    const int i3 = ( (i / np) / np ) % ndims1;
    const int i4 = ( ( (i / np) / np ) / ndims1 ) % ndims2;
    const int i5 = ( ( (i / np) / np ) / ndims1 ) / ndims2;

    arr_c[ i5*dim1234 + i3*dim124 + i4*dim12 + i1*np + i2 ] = arr_f[i];
  }
}

template<typename data_type>
void flip_array_np_np_nlev_timelevels_nelems (data_type*& arr_f, data_type*& arr_c, int nelems)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,ilev,itlev,ielem)_f -> (ielem,itlev,ilev,igp,jgp)_c
  const int size    = np * np * nlev * timelevels * nelems;
  const int dim1234 = np * np * nlev * timelevels;
  const int dim123  = np * np * nlev;
  const int dim12   = np * np;
  for (int i=0; i<size; ++i)
  {
    const int i1 = i % np;
    const int i2 = (i / np) % np;
    const int i3 = ( (i / np) / np ) % nlev;
    const int i4 = ( ( (i / np) / np ) / nlev ) % timelevels;
    const int i5 = ( ( (i / np) / np ) / nlev ) / timelevels;

    arr_c[ i5*dim1234 + i4*dim123 + i3*dim12 + i1*np + i2 ] = arr_f[i];
  }
}

template<typename data_type>
void flip_array_np_np_ndims_nlev_timelevels_nelems (data_type*& arr_f, data_type*& arr_c, int nelems, int ndims)
{
  // Note: each np x np block should not be flipped, that is,
  //       (igp,jgp,idim,ilev,itlev,ielem)_f -> (ielem,itlev,ilev,idim,igp,jgp)_c
  const int size     = np * np * dim2d * nlev * timelevels * nelems;
  const int dim12345 = np * np * dim2d * nlev * timelevels;
  const int dim1234  = np * np * dim2d * nlev;
  const int dim123   = np * np * dim2d;
  const int dim12    = np * np;
  for (int i=0; i<size; ++i)
  {
    const int i1 = i % np;
    const int i2 = (i / np) % np;
    const int i3 = ( (i / np) / np ) % dim2d;
    const int i4 = ( ( (i / np) / np ) / dim2d ) % nlev;
    const int i5 = ( ( ( (i / np) / np ) / dim2d ) / nlev ) % timelevels;
    const int i6 = ( ( ( (i / np) / np ) / dim2d ) / nlev ) / timelevels;

    arr_c[ i6*dim12345 + i5*dim1234 + i4*dim123 + i3*dim12 + i1*np + i2 ] = arr_f[i];
  }
}

TEST_CASE("SETUP OF F90 STRUCTURES", "SETUP OF F90 STRUCTURES")
{
  std::cout << "num elements  : " << numelems << "\n"
            << "num levels    : " << nlev     << "\n"
            << "num gauss pts : " << np       << "\n";

  // Creating fake Derivative
  // This will call the c function that initializes
  // the Homme::Derivative static instance
  init_derivative_f90 ();

  // Setting up physical constants (mainly earth radius)
  // This will call the c function that initializes
  // the Homme::PhysicalConstant static instance
  init_physical_constants_f90 ();

  // Setting up the fortran elem structure
  // This will also call the c function that initializes
  // the Homme::PointersPool static instance
  init_elem_f90 (numelems);

  test_running = true;

  REQUIRE (numelems>0);
  REQUIRE (dim2d == 2);
  REQUIRE (numRandTests>0);
}

TEST_CASE("copy_timelevels", "advance_nonstag_rk_cxx")
{
  constexpr const int p_len =
      np * np * nlev * timelevels * numelems;
  real *p_theory = new real[p_len];
  real *p_exper = new real[p_len];
  constexpr const int v_len =
      np * np * dim2d * nlev * timelevels * numelems;
  real *v_theory = new real[v_len];
  real *v_exper = new real[v_len];

  SECTION("random_test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; i++) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);

      genRandTheoryExper(
          p_theory, p_exper, p_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandTheoryExper(
          v_theory, v_exper, v_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));

      const int n_src = (std::uniform_int_distribution<int>(
          1, timelevels))(engine);
      const int n_dist =
          (std::uniform_int_distribution<int>(
              1, timelevels))(engine);

      copy_timelevels_f90(nets, nete, numelems, n_src,
                          n_dist, p_theory, v_theory);
      copy_timelevels_c(nets, nete, numelems, n_src, n_dist,
                        p_exper, v_exper);

      for(int j = 0; j < p_len; j++) {
        if(p_exper[j] != p_theory[j]) {
          std::cout << "error at " << j << std::endl;
        }
        REQUIRE(p_exper[j] == p_theory[j]);
      }
      for(int j = 0; j < v_len; j++) {
        REQUIRE(v_exper[j] == v_theory[j]);
      }
    }
  }

  delete[] p_theory;
  delete[] p_exper;
  delete[] v_theory;
  delete[] v_exper;
}

TEST_CASE("q_tests", "advance_nonstag_rk_cxx")
{
  // real elem_state_p (np,np,nlevel,timelevels,nelemd)
  constexpr const int p_len =
      numelems * timelevels * nlev * np * np;
  real *p_theory = new real[p_len];
  real *p_exper = new real[p_len];

  SECTION("random test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; i++) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);
      const int n0 = (std::uniform_int_distribution<int>(
          1, timelevels))(engine);
      int kmass = (std::uniform_int_distribution<int>(
          0, nlev))(engine);
      // kmass needs to be in [1, nlev] or be -1 for the
      // Fortran implementation
      if(kmass == 0) {
        kmass = -1;
      }

      std::uniform_real_distribution<real> p_dist(0, 1.0);
      for(int j = 0; j < p_len; j++) {
        p_theory[j] = p_dist(engine);
        p_exper[j] = p_theory[j];
      }
      recover_q_f90(nets, nete, kmass, n0, numelems,
                    p_theory);
      recover_q_c(nets, nete, kmass, n0, numelems, p_exper);
      for(int j = 0; j < p_len; j++) {
        REQUIRE(p_exper[j] == p_theory[j]);
      }
    }
  }
  delete[] p_exper;
  delete[] p_theory;
}

TEST_CASE("recover_dpq", "advance_nonstag_rk_cxx")
{
  // real elem_state_p (np,np,nlevel,timelevels,nelemd)
  constexpr const int p_len =
      numelems * timelevels * nlev * np * np;
  real *p_theory = new real[p_len];
  real *p_exper = new real[p_len];

  SECTION("random test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; i++) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);
      const int n0 = (std::uniform_int_distribution<int>(
          1, timelevels))(engine);
      int kmass = (std::uniform_int_distribution<int>(
          0, nlev))(engine);
      // kmass needs to be in [1, nlev] or be -1 for the
      // Fortran implementation
      if(kmass == 0) {
        kmass = -1;
      }

      std::uniform_real_distribution<real> p_dist(0, 1.0);
      for(int j = 0; j < p_len; j++) {
        p_theory[j] = p_dist(engine);
        p_exper[j] = p_theory[j];
      }

      recover_dpq_c(nets, nete, kmass, n0, numelems, p_exper);
      recover_dpq_f90(nets, nete, kmass, n0, numelems, p_theory);

      for(int j = 0; j < p_len; j++) {
        REQUIRE(p_exper[j] == p_theory[j]);
      }
    }
  }
  delete[] p_exper;
  delete[] p_theory;
}

TEST_CASE("contra2latlon", "advance_nonstag_rk_cxx")
{
  // real elem_D (np,np,2,2,nelemd)
  // real elem_state_v (np,np,2,nlev,timelevels,nelemd)
  constexpr const int D_len =
      np * np * dim2d * dim2d * numelems;
  constexpr const int v_len =
      np * np * dim2d * nlev * timelevels * numelems;
  real *D = new real[D_len];
  real *v_theory = new real[v_len];
  real *v_exper = new real[v_len];

  SECTION("random test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; i++) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);
      const int n0 = (std::uniform_int_distribution<int>(
          1, timelevels))(engine);

      std::uniform_real_distribution<real> D_dist(0, 1.0);
      for(int j = 0; j < D_len; j++) {
        D[j] = D_dist(engine);
      }

      std::uniform_real_distribution<real> v_dist(0, 1.0);
      for(int j = 0; j < v_len; j++) {
        v_theory[j] = v_dist(engine);
        v_exper[j] = v_theory[j];
      }
      contra2latlon_f90(nets, nete, n0, numelems, D,
                        v_theory);
      contra2latlon_c(nets, nete, n0, numelems, D, v_exper);
      for(int j = 0; j < v_len; j++) {
        REQUIRE(v_exper[j] == v_theory[j]);
      }
    }
  }
  delete[] v_exper;
  delete[] v_theory;
  delete[] D;
}

TEST_CASE("laplace_sphere_wk", "advance_nonstag_rk_cxx")
{
  using udi_type = std::uniform_int_distribution<int>;

  udi_type dnets (1, numelems);
  udi_type bernoulli(0,1);
  std::uniform_real_distribution<real> dreal (0,1);

  real* tmp_ar;

  SECTION ("random test for laplace_sphere_wk")
  {
    real* input_c    = new real[numelems*nlev*np*np];
    real* input_f90  = new real[numelems*nlev*np*np];
    real* output_c   = new real[numelems*nlev*np*np];
    real* output_f90 = new real[numelems*nlev*np*np];

    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());

    for(int itest = 0; itest < numRandTests; ++itest)
    {
      const int nets     = dnets(engine);
      const int nete     = udi_type(std::min(nets + 1,numelems), numelems)(engine);
      const int var_coef = bernoulli(engine);

      // To avoid false checks on elements not actually processed in this random test.
      for (int i=0; i<numelems*nlev*np*np; ++i)
      {
        output_c[i] = output_f90[i] = 0;
      }

      // Randomize only the arrays that are used as inputs
      genRandArray (input_f90,                        numelems*nlev*np*np,        engine, dreal);
      genRandArray (get_pointers_pool_c()->hypervisc, numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->Dinv,      numelems*dim2d*dim2d*np*np, engine, dreal);
      genRandArray (get_pointers_pool_c()->spheremp,  numelems*np*np,             engine, dreal);

      // Compute laplacian with fortran routines
      test_laplace_sphere_wk_f90 (nets, nete, numelems, var_coef, input_f90, output_f90);

      // Need to flip the input arrays, so that they represent the same data when read as c arrays
      flip_array_np_np_nlev_nelems (input_f90, input_c, numelems);

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->hypervisc, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->hypervisc);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->Dinv, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->Dinv);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->spheremp, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->spheremp);
      delete[] tmp_ar;

      // Compute laplacian with kokkos kernels
      laplace_sphere_wk_c (nets-1, nete, numelems, var_coef, input_c, output_c);

      // Flip the f90 output array, so we can directly compare it with the C one
      tmp_ar = new real[numelems*nlev*np*np];
      flip_array_np_np_nlev_nelems (output_f90, tmp_ar, numelems);
      for (int i=0; i<numelems*nlev*np*np; ++i)
      {
        REQUIRE ( std::fabs(output_c[i]-tmp_ar[i]) <= 1e-10*std::fabs(tmp_ar[i]) );
      }
      delete[] tmp_ar;

    } // for numRandTests

    delete[] input_c;
    delete[] output_c;
    delete[] input_f90;
    delete[] output_f90;
  } // SECTION

  SECTION ("random test for vlaplace_sphere_wk")
  {
    real* input_c    = new real[numelems*nlev*dim2d*np*np];
    real* input_f90  = new real[numelems*nlev*dim2d*np*np];
    real* output_c   = new real[numelems*nlev*dim2d*np*np];
    real* output_f90 = new real[numelems*nlev*dim2d*np*np];

    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());

    for(int itest = 0; itest < numRandTests; ++itest)
    {
      const int nets      = dnets(engine);
      const int nete      = udi_type(std::min(nets + 1,numelems), numelems)(engine);
      const int var_coef  = bernoulli(engine);

      // Randomly set up the control parameters (mainly whether hypervisc is selected)
      // This will also call the c function that initializes the Homme::ControlParameters
      // static instance
      pick_random_control_parameters_f90 ();

      // Avoid abort in laplace routines: nu_ratio=1 if hypervisc is selected
      const real nu_ratio = (get_control_parameters_c()->hypervisc_scaling && var_coef ? 1 : dreal(engine));

      // To avoid false checks on elements not actually processed in this random test.
      for (int i=0; i<numelems*nlev*dim2d*np*np; ++i)
      {
        output_c[i] = output_f90[i] = 0;
      }

      // Randomize all the arrays that can (possibly) be used as inputs
      genRandArray (input_f90,                              numelems*nlev*dim2d*np*np,  engine, dreal);
      genRandArray (get_pointers_pool_c()->hypervisc,       numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->D,               numelems*dim2d*dim2d*np*np, engine, dreal);
      genRandArray (get_pointers_pool_c()->Dinv,            numelems*dim2d*dim2d*np*np, engine, dreal);
      genRandArray (get_pointers_pool_c()->mp,              numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->spheremp,        numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->metinv,          numelems*dim2d*dim2d*np*np, engine, dreal);
      genRandArray (get_pointers_pool_c()->metdet,          numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->rmetdet,         numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->tensor_visc,     numelems*dim2d*dim2d*np*np, engine, dreal);
      genRandArray (get_pointers_pool_c()->vec_sphere2cart, numelems*dim3d*dim2d*np*np, engine, dreal);

      // Compute laplacian with fortran routines
      test_vlaplace_sphere_wk_f90 (nets, nete, numelems, var_coef, nu_ratio, input_f90, output_f90);

      // Need to flip the input arrays, so that they represent the same data when read as c arrays
      flip_array_np_np_ndims_nlev_nelems (input_f90, input_c, numelems, dim2d);

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->hypervisc, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->hypervisc);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->D, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->D);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->Dinv, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->Dinv);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->mp, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->mp);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->spheremp, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->spheremp);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->metinv, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->metinv);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->metdet, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->metdet);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->rmetdet, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->rmetdet);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->tensor_visc, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->tensor_visc);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim3d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->vec_sphere2cart, tmp_ar, numelems, dim3d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim3d*dim2d*np*np, get_pointers_pool_c()->vec_sphere2cart);
      delete[] tmp_ar;

      // Compute laplacian with kokkos kernels
      vlaplace_sphere_wk_c (nets-1, nete, numelems, var_coef, nu_ratio, input_c, output_c);

      // Flip the f90 output array, so we can directly compare it with the C one
      tmp_ar = new real[numelems*nlev*dim2d*np*np];
      flip_array_np_np_ndims_nlev_nelems (output_f90, tmp_ar, numelems, dim2d);
      for (int i=0; i<numelems*nlev*dim2d*np*np; ++i)
      {
        REQUIRE ( std::fabs(output_c[i]-tmp_ar[i]) <= 1e-10*std::fabs(tmp_ar[i]) );
      }
      delete[] tmp_ar;

    } // for numRandTests

    delete[] input_c;
    delete[] output_c;
    delete[] input_f90;
    delete[] output_f90;
  } // SECTION
} // laplace_sphere_wk

TEST_CASE("lapl_loop_(pre/post)_bndry_ex", "advance_nonstag_rk_cxx")
{
  using udi_type = std::uniform_int_distribution<int>;

  udi_type dnets (1, numelems - 1);
  udi_type dn0 (1,3);
  udi_type bernoulli(0,1);
  std::uniform_real_distribution<real> dreal (0,1);

  real* tmp_ar;

  SECTION ("random test for loop_lapl_pre_bndry_ex")
  {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());

    real* output_p_f90 = new real[numelems*nlev*np*np];
    real* output_v_f90 = new real[numelems*nlev*dim2d*np*np];
    real* output_p_c   = new real[numelems*nlev*np*np];
    real* output_v_c   = new real[numelems*nlev*dim2d*np*np];

    for(int i = 0; i < numRandTests; i++)
    {
      const int nets      = dnets(engine);
      const int nete      = udi_type(std::min(nets + 1,numelems), numelems)(engine);
      const int n0        = dn0(engine);
      const int var_coef  = bernoulli(engine);

      // Randomly set up the control parameters (mainly whether hypervisc is selected)
      // This will also call the c function that initializes the Homme::ControlParameters
      // static instance
      pick_random_control_parameters_f90 ();

      // Avoid abort in laplace routines: nu_ratio=1 if hypervisc is selected
      const real nu_ratio = (get_control_parameters_c()->hypervisc_scaling && var_coef ? 1 : dreal(engine));

      // To avoid false checks on elements not actually processed in this random test.
      for (int i=0; i<numelems*nlev*np*np; ++i)
      {
        output_p_c[i] = output_p_f90[i] = 0;
      }
      for (int i=0; i<numelems*nlev*dim2d*np*np; ++i)
      {
        output_v_c[i] = output_v_f90[i] = 0;
      }

      // Randomize all the arrays that can (possibly) be used as inputs
      genRandArray (get_pointers_pool_c()->elem_state_ps,   numelems*np*np,                       engine, dreal);
      genRandArray (get_pointers_pool_c()->elem_state_p,    numelems*timelevels*nlev*np*np,       engine, dreal);
      genRandArray (get_pointers_pool_c()->elem_state_v,    numelems*timelevels*nlev*dim2d*np*np, engine, dreal);
      genRandArray (get_pointers_pool_c()->hypervisc,       numelems*np*np,                       engine, dreal);
      genRandArray (get_pointers_pool_c()->D,               numelems*dim2d*dim2d*np*np,           engine, dreal);
      genRandArray (get_pointers_pool_c()->Dinv,            numelems*dim2d*dim2d*np*np,           engine, dreal);
      genRandArray (get_pointers_pool_c()->mp,              numelems*np*np,                       engine, dreal);
      genRandArray (get_pointers_pool_c()->spheremp,        numelems*np*np,                       engine, dreal);
      genRandArray (get_pointers_pool_c()->metinv,          numelems*dim2d*dim2d*np*np,           engine, dreal);
      genRandArray (get_pointers_pool_c()->metdet,          numelems*np*np,                       engine, dreal);
      genRandArray (get_pointers_pool_c()->rmetdet,         numelems*np*np,                       engine, dreal);
      genRandArray (get_pointers_pool_c()->tensor_visc,     numelems*dim2d*dim2d*np*np,           engine, dreal);
      genRandArray (get_pointers_pool_c()->vec_sphere2cart, numelems*dim2d*dim2d*np*np,           engine, dreal);

      // Launch loop with fortran routine
      test_lapl_pre_bndry_ex_f90 (nets, nete, numelems, n0, var_coef, nu_ratio, output_p_f90, output_v_f90);

      // Need to flip the input arrays, so that they represent the same data when read as c arrays
      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->elem_state_ps, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->elem_state_ps);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*timelevels*nlev*np*np];
      flip_array_np_np_nlev_timelevels_nelems (get_pointers_pool_c()->elem_state_p, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*timelevels*nlev*np*np, get_pointers_pool_c()->elem_state_p);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*timelevels*nlev*dim2d*np*np];
      flip_array_np_np_ndims_nlev_timelevels_nelems (get_pointers_pool_c()->elem_state_v, tmp_ar, numelems, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*timelevels*nlev*dim2d*np*np, get_pointers_pool_c()->elem_state_v);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->hypervisc, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->hypervisc);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->D, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->D);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->Dinv, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->Dinv);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->mp, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->mp);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->spheremp, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->spheremp);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->metinv, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->metinv);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->metdet, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->metdet);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->rmetdet, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->rmetdet);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->tensor_visc, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->tensor_visc);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim3d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->vec_sphere2cart, tmp_ar, numelems, dim3d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim3d*dim2d*np*np, get_pointers_pool_c()->vec_sphere2cart);
      delete[] tmp_ar;

      // Launch loop with c routine
      loop_lapl_pre_bndry_ex_c (nets, nete, numelems, n0, var_coef, nu_ratio, output_p_c, output_v_c);

      // Flip the f90 output array, so we can directly compare it with the C one
      tmp_ar = new real[numelems*nlev*np*np];
      flip_array_np_np_nlev_nelems (output_p_f90, tmp_ar, numelems);
      for (int i=0; i<numelems*nlev*np*np; ++i)
      {
        REQUIRE ( std::fabs(output_p_c[i]-tmp_ar[i]) <= 1e-10*std::fabs(tmp_ar[i]) );
      }
      delete[] tmp_ar;

      tmp_ar = new real[numelems*nlev*dim2d*np*np];
      flip_array_np_np_ndims_nlev_nelems (output_v_f90, tmp_ar, numelems, dim2d);
      for (int i=0; i<numelems*nlev*dim2d*np*np; ++i)
      {
        REQUIRE ( std::fabs(output_v_c[i]-tmp_ar[i]) <= 1e-10*std::fabs(tmp_ar[i]) );
      }
      delete[] tmp_ar;
    } // for numRandTests

    delete[] output_p_f90;
    delete[] output_v_f90;
    delete[] output_p_c;
    delete[] output_v_c;
  } // SECTION

  SECTION ("random test for loop_lapl_post_bndry_ex")
  {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());

    real* input_p      = new real[numelems*nlev*np*np];
    real* input_v      = new real[numelems*nlev*dim2d*np*np];
    real* output_p_f90 = new real[numelems*nlev*np*np];
    real* output_v_f90 = new real[numelems*nlev*dim2d*np*np];
    real* output_p_c   = new real[numelems*nlev*np*np];
    real* output_v_c   = new real[numelems*nlev*dim2d*np*np];

    for(int i = 0; i < numRandTests; i++)
    {
      const int nets      = dnets(engine);
      const int nete      = udi_type(std::min(nets + 1,numelems), numelems)(engine);
      const int n0        = dn0(engine);

      // Avoid abort in laplace routines: nu_ratio=1 if hypervisc is selected
      const real nu_ratio = get_control_parameters_c()->hypervisc_scaling ? 1 : dreal(engine);

      // Randomize all the arrays that can (possibly) be used as inputs
      genRandArray (input_p,                                numelems*nlev*np*np,        engine, dreal);
      genRandArray (input_v,                                numelems*nlev*dim2d*np*np,  engine, dreal);
      genRandArray (get_pointers_pool_c()->hypervisc,       numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->D,               numelems*dim2d*dim2d*np*np, engine, dreal);
      genRandArray (get_pointers_pool_c()->Dinv,            numelems*dim2d*dim2d*np*np, engine, dreal);
      genRandArray (get_pointers_pool_c()->mp,              numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->spheremp,        numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->rspheremp,       numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->metinv,          numelems*dim2d*dim2d*np*np, engine, dreal);
      genRandArray (get_pointers_pool_c()->metdet,          numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->rmetdet,         numelems*np*np,             engine, dreal);
      genRandArray (get_pointers_pool_c()->tensor_visc,     numelems*dim2d*dim2d*np*np, engine, dreal);
      genRandArray (get_pointers_pool_c()->vec_sphere2cart, numelems*dim3d*dim2d*np*np, engine, dreal);

      // Launch loop with fortran routine
      std::copy (input_p, input_p+numelems*nlev*np*np,       output_p_f90);
      std::copy (input_v, input_v+numelems*nlev*dim2d*np*np, output_v_f90);
      test_lapl_post_bndry_ex_f90 (nets, nete, numelems, nu_ratio, output_p_f90, output_v_f90);

      // Need to flip the input arrays, so that they represent the same data when read as c arrays
      flip_array_np_np_nlev_nelems       (input_p, output_p_c, numelems);
      flip_array_np_np_ndims_nlev_nelems (input_v, output_v_c, numelems, dim2d);

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->hypervisc, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->hypervisc);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->D, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->D);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->Dinv, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->Dinv);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->mp, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->mp);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->spheremp, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->spheremp);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->rspheremp, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->rspheremp);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->metinv, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->metinv);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->metdet, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->metdet);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*np*np];
      flip_array_np_np_nelems (get_pointers_pool_c()->rmetdet, tmp_ar, numelems);
      std::copy(tmp_ar, tmp_ar+numelems*np*np, get_pointers_pool_c()->rmetdet);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim2d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->tensor_visc, tmp_ar, numelems, dim2d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim2d*dim2d*np*np, get_pointers_pool_c()->tensor_visc);
      delete[] tmp_ar;

      tmp_ar = new real[numelems*dim3d*dim2d*np*np];
      flip_array_np_np_ndims1_ndims2_nelems (get_pointers_pool_c()->vec_sphere2cart, tmp_ar, numelems, dim3d, dim2d);
      std::copy(tmp_ar, tmp_ar+numelems*dim3d*dim2d*np*np, get_pointers_pool_c()->vec_sphere2cart);
      delete[] tmp_ar;

      // Launch loop with c routine
      loop_lapl_post_bndry_ex_c (nets, nete, numelems, nu_ratio, output_p_c, output_v_c);

      // Flip the f90 output array, so we can directly compare it with the C one
      tmp_ar = new real[numelems*nlev*np*np];
      flip_array_np_np_nlev_nelems (output_p_f90, tmp_ar, numelems);
      for (int i=0; i<numelems*nlev*np*np; ++i)
      {
        REQUIRE ( std::fabs(output_p_c[i]-tmp_ar[i]) <= 1e-10*std::fabs(tmp_ar[i]) );
      }
      delete[] tmp_ar;

      tmp_ar = new real[numelems*nlev*2*np*np];
      flip_array_np_np_ndims_nlev_nelems (output_v_f90, tmp_ar, numelems, dim2d);
      for (int i=0; i<numelems*nlev*dim2d*np*np; ++i)
      {
        REQUIRE ( std::fabs(output_v_c[i]-tmp_ar[i]) <= 1e-10*std::fabs(tmp_ar[i]) );
      }
      delete[] tmp_ar;
    } // for numRandTests

    delete[] input_p;
    delete[] input_v;
    delete[] output_p_f90;
    delete[] output_v_f90;
    delete[] output_p_c;
    delete[] output_v_c;
  } // SECTION
} // lapl_loop_(pre/post)_bndry_ex

TEST_CASE("add_hv", "advance_nonstag_rk_cxx")
{
  constexpr const int spheremp_len = np * np * numelems;
  real *spheremp = new real[spheremp_len];

  SECTION("random test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; i++) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);

      // real ptens (np,np,nlev,nets:nete)
      // real vtens (np,np,2,nlev,nets:nete)
      const int ptens_len =
          np * np * nlev * (nete - nets + 1);
      real *ptens_theory = new real[ptens_len];
      real *ptens_exper = new real[ptens_len];
      const int vtens_len =
          np * np * dim2d * nlev * (nete - nets + 1);
      real *vtens_theory = new real[vtens_len];
      real *vtens_exper = new real[vtens_len];

      std::uniform_real_distribution<real> ptens_dist(0,
                                                      1.0);
      for(int j = 0; j < ptens_len; j++) {
        ptens_theory[j] = ptens_dist(engine);
        ptens_exper[j] = ptens_theory[j];
      }
      std::uniform_real_distribution<real> vtens_dist(0,
                                                      1.0);
      for(int j = 0; j < vtens_len; j++) {
        vtens_theory[j] = vtens_dist(engine);
        vtens_exper[j] = vtens_theory[j];
      }

      std::uniform_real_distribution<real> spheremp_dist(
          0, 1.0);
      for(int j = 0; j < spheremp_len; j++) {
        spheremp[j] = spheremp_dist(engine);
      }

      add_hv_f90(nets, nete, numelems, spheremp,
                ptens_theory, vtens_theory);
      add_hv_c(nets, nete, numelems, spheremp, ptens_exper,
              vtens_exper);
      for(int j = 0; j < ptens_len; j++) {
        if(ptens_exper[j] != ptens_theory[j]) {
          std::cout << ptens_exper[j] - ptens_theory[j];
        }
        REQUIRE(ptens_exper[j] - ptens_theory[j] == 0.0);
      }
      for(int j = 0; j < vtens_len; j++) {
        REQUIRE(vtens_exper[j] - vtens_theory[j] == 0.0);
      }

      delete[] ptens_theory;
      delete[] ptens_exper;
      delete[] vtens_theory;
      delete[] vtens_exper;
    }
  }
  delete[] spheremp;
}

TEST_CASE("weighted_rhs", "advance_nonstag_rk_cxx")
{
  constexpr const int rspheremp_len = np * np * numelems;
  real *rspheremp = new real[rspheremp_len];
  constexpr const int dinv_len =
      np * np * dim2d * dim2d * numelems;
  real *dinv = new real[dinv_len];

  SECTION("random_test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; i++) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);
      const int ptens_len =
          np * np * nlev * (nete - nets + 1);
      real *ptens_theory = new real[ptens_len];
      real *ptens_exper = new real[ptens_len];
      const int vtens_len =
          np * np * dim2d * nlev * (nete - nets + 1);
      real *vtens_theory = new real[vtens_len];
      real *vtens_exper = new real[vtens_len];

      genRandArray(
          rspheremp, rspheremp_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandArray(
          dinv, dinv_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandTheoryExper(
          ptens_theory, ptens_exper, ptens_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandTheoryExper(
          vtens_theory, vtens_exper, vtens_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      weighted_rhs_f90(nets, nete, numelems, rspheremp, dinv,
                ptens_theory, vtens_theory);
      weighted_rhs_c(nets, nete, numelems, rspheremp, dinv,
              ptens_exper, vtens_exper);
      for(int j = 0; j < ptens_len; j++) {
        REQUIRE(ptens_exper[j] == ptens_theory[j]);
      }
      for(int j = 0; j < vtens_len; j++) {
        REQUIRE(vtens_exper[j] == vtens_theory[j]);
      }

      delete[] ptens_theory;
      delete[] ptens_exper;
      delete[] vtens_theory;
      delete[] vtens_exper;
    }
  }

  delete[] rspheremp;
  delete[] dinv;
}

TEST_CASE("rk_stage", "advance_nonstag_rk_cxx")
{
  real *alpha0 = new real[rkstages];
  real *alpha = new real[rkstages];
  constexpr const int p_len =
      np * np * nlev * timelevels * numelems;
  real *p_theory = new real[p_len];
  real *p_exper = new real[p_len];
  constexpr const int v_len =
      np * np * dim2d * nlev * timelevels * numelems;
  real *v_theory = new real[v_len];
  real *v_exper = new real[v_len];

  SECTION("random_test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; i++) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);
      const int ptens_len =
          np * np * nlev * (nete - nets + 1);
      real *ptens = new real[ptens_len];
      const int vtens_len =
          np * np * dim2d * nlev * (nete - nets + 1);
      real *vtens = new real[vtens_len];

      genRandArray(
          alpha0, rkstages, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandArray(
          alpha, rkstages, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandArray(
          ptens, ptens_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandArray(
          vtens, vtens_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandTheoryExper(
          p_theory, p_exper, p_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandTheoryExper(
          v_theory, v_exper, v_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));

      const int n0 = (std::uniform_int_distribution<int>(
          1, timelevels))(engine);
      const int np1 = (std::uniform_int_distribution<int>(
          1, timelevels))(engine);
      const int s = (std::uniform_int_distribution<int>(
          1, rkstages))(engine);

      rk_stage_f90(nets, nete, n0, np1, s, rkstages, numelems,
                v_theory, p_theory, alpha0, alpha, ptens,
                vtens);
      rk_stage_c(nets, nete, n0, np1, s, rkstages, numelems,
              v_exper, p_exper, alpha0, alpha, ptens,
              vtens);

      for(int j = 0; j < p_len; j++) {
        REQUIRE(p_exper[j] == p_theory[j]);
      }
      for(int j = 0; j < v_len; j++) {
        REQUIRE(v_exper[j] == v_theory[j]);
      }

      delete[] ptens;
      delete[] vtens;
    }
  }

  delete[] alpha0;
  delete[] alpha;
  delete[] p_theory;
  delete[] p_exper;
  delete[] v_theory;
  delete[] v_exper;
}

TEST_CASE ("CLEANUP","CLEANUP")
{
  // This deallocates stuff allocated in the advance_rk_unit_test_mod module
  cleanup_testing_f90();

  test_running = false;

  REQUIRE (test_running == false);
}
