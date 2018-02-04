
#include <catch/catch.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>

#include <Types.hpp>
#include <fortran_binding.hpp>

using namespace Homme;

extern "C" {

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

void add_hv_f90(const int &nets, const int &nete,
                const int &nelems, real *const &sphere_mp,
                real *&ptens, real *&vtens);

void add_hv_c(const int &nets, const int &nete,
              const int &nelems, real *const &sphere_mp,
              real *&ptens, real *&vtens);

void recover_dpq_f90(const int &nets, const int &nete,
                     const int &kmass, const int &n0,
                     const int &nelems, real *const &p);

void recover_dpq_c(const int &nets, const int &nete,
                   const int &kmass, const int &n0,
                   const int &nelems, real *const &p);

void weighted_rhs_f90(const int &nets, const int &nete,
                      const int &numelems,
                      real *const &rsphere_mp_ptr,
                      real *const &dinv_ptr,
                      real *&ptens_ptr, real *&vtens_ptr);

void weighted_rhs_c(const int &nets, const int &nete,
                    const int &numelems,
                    real *const &rsphere_mp_ptr,
                    real *const &dinv_ptr, real *&ptens_ptr,
                    real *&vtens_ptr);

void rk_stage_f90(const int &nets, const int &nete,
                  const int &n0, const int &np1,
                  const int &s, const int &rkstages,
                  const int &numelems, real *&v_ptr,
                  real *&p_ptr, real *const &alpha0_ptr,
                  real *const &alpha_ptr,
                  real *const &ptens_ptr,
                  real *const &vtens_ptr);

void rk_stage_c(const int &nets, const int &nete,
                const int &n0, const int &np1, const int &s,
                const int &rkstages, const int &numelems,
                real *&v_ptr, real *&p_ptr,
                real *const &alpha0_ptr,
                real *const &alpha_ptr,
                real *const &ptens_ptr,
                real *const &vtens_ptr);

void loop7_f90(const int &nets, const int &nete,
               const int &n0, const int &nelemd,
               const int &tracer_advection_formulation,
               const real &pmean, const real &dtstage,
               real *&dvv_ptr, real *&d_ptr,
               real *&dinv_ptr, real *&metdet_ptr,
               real *&rmetdet_ptr, real *&fcor_ptr,
               real *&p_ptr, real *&ps_ptr, real *&v_ptr,
               real *&ptens_ptr, real *&vtens_ptr);

void loop7_c(const int &nets, const int &nete,
             const int &n0, const int &nelemd,
             const int &tracer_advection_formulation,
             const real &pmean, const real &dtstage,
             real *&dvv_ptr, real *&d_ptr, real *&dinv_ptr,
             real *&metdet_ptr, real *&rmetdet_ptr,
             real *&fcor_ptr, real *&p_ptr, real *&ps_ptr,
             real *&v_ptr, real *&ptens_ptr,
             real *&vtens_ptr);

void divergence_sphere_c_callable(real *v, real *dvv,
                                  real *metdet,
                                  real *rmetdet, real *dinv,
                                  real *div);

void gradient_sphere_c_callable(real *s, real *dvv,
                                real *dinv, real *grad);

void vorticity_sphere_c_callable(real *v, real *dvv,
                                 real *rmetdet, real *dinv,
                                 real *vort);

}  // extern "C"

namespace Homme {
template <typename Scalar_QP, typename Vector_QP>
void gradient_sphere_c(int ie, const Scalar_QP &s,
                       const HommeExecView2D &dvv,
                       const HommeExecView5D &dinv,
                       Vector_QP &grad);

template <typename Scalar_QP, typename Vector_QP>
void vorticity_sphere_c(int ie, const Vector_QP &v,
                        const HommeExecView2D &dvv,
                        const HommeExecView5D &d,
                        const HommeExecView3D &rmetdet,
                        Scalar_QP &grad);

template <typename Scalar_QP, typename Vector_QP>
void divergence_sphere_c(int ie, const Vector_QP &v,
                         const HommeExecView2D &dvv,
                         const HommeExecView3D &metdet,
                         const HommeExecView3D &rmetdet,
                         const HommeExecView5D &dinv,
                         Scalar_QP &divergence);

}

template <typename rngAlg, typename dist, typename number>
void genRandArray(number *arr, int arr_len, rngAlg &engine,
                  dist &pdf) {
  for(int i = 0; i < arr_len; ++i) {
    arr[i] = pdf(engine);
  }
}

template <typename rngAlg, typename dist, typename number>
void genRandArray(number *arr, int arr_len, rngAlg &engine,
                  dist &&pdf) {
  for(int i = 0; i < arr_len; ++i) {
    arr[i] = pdf(engine);
  }
}

template <typename rngAlg, typename dist, typename number>
void genRandTheoryExper(number *arr_theory,
                        number *arr_exper, int arr_len,
                        rngAlg &engine, dist &pdf) {
  for(int i = 0; i < arr_len; ++i) {
    arr_theory[i] = pdf(engine);
    arr_exper[i] = arr_theory[i];
  }
}

template <typename rngAlg, typename dist, typename number>
void genRandTheoryExper(number *arr_theory,
                        number *arr_exper, int arr_len,
                        rngAlg &engine, dist &&pdf) {
  for(int i = 0; i < arr_len; ++i) {
    arr_theory[i] = pdf(engine);
    arr_exper[i] = arr_theory[i];
  }
}

template <typename input_type>
void input_reader(std::map<std::string, input_type *> &data,
                  std::istream &in) {
  std::string varname;
  while(std::getline(in, varname)) {
    if(data.count(varname) > 0) {
      input_type *values = data[varname];
      int i = 0;
      while(true) {
        input_type buf;
        std::istream::pos_type pos = in.tellg();
        in >> buf;
        if(in.fail()) {
          in.clear();
          in.seekg(pos);
          break;
        }
        values[i] = buf;
        ++i;
      }
    }
  }
}

#if defined(KOKKOS_HAVE_DEFAULT_DEVICE_TYPE_CUDA)
real check_answer(real theory, real exper,
                  real epsilon_coeff = 4.0) {
  if(theory == 0.0) {
    if(epsilon_coeff == 0.0) {
      epsilon_coeff = 1.0;
    }
    const real max_abs_err =
        fabs(epsilon_coeff *
             std::numeric_limits<real>::epsilon());
    if(fabs(exper) > max_abs_err) {
      return fabs(exper);
    }
  } else {
    if(epsilon_coeff == 0.0) {
      epsilon_coeff = 1.0 / theory;
    }
    const real max_abs_err =
        fabs(epsilon_coeff *
             std::numeric_limits<real>::epsilon() * theory);
    const real abs_err = fabs(theory - exper);
    if(abs_err > max_abs_err) {
      return abs_err;
    }
  }
  return 0.0;
}
#else
real check_answer(real theory, real exper,
                  real epsilon_coeff = 0.0) {
  return std::fabs(theory - exper);
}
#endif

TEST_CASE("copy_timelevels", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 100;
  constexpr const int dim = 2;

  constexpr const int p_len =
      np * np * nlev * timelevels * numelems;
  real *p_theory = new real[p_len];
  real *p_exper = new real[p_len];
  constexpr const int v_len =
      np * np * dim * nlev * timelevels * numelems;
  real *v_theory = new real[v_len];
  real *v_exper = new real[v_len];

  constexpr const int numRandTests = 10;
  SECTION("random_test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; ++i) {
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
        REQUIRE(check_answer(p_theory[j], p_exper[j]) ==
                0.0);
      }
      for(int j = 0; j < v_len; j++) {
        REQUIRE(check_answer(v_theory[j], v_exper[j]) ==
                0.0);
      }
    }
  }

  delete[] p_theory;
  delete[] p_exper;
  delete[] v_theory;
  delete[] v_exper;
}

TEST_CASE("q_tests", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;

  // real elem_state_p (np,np,nlevel,timelevels,nelemd)
  constexpr const int p_len =
      numelems * timelevels * nlev * np * np;
  real *p_theory = new real[p_len];
  real *p_exper = new real[p_len];

  constexpr const int numRandTests = 10;

  SECTION("random test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; ++i) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);
      const int n0 = (std::uniform_int_distribution<int>(
          1, timelevels))(engine);
      int kmass = (std::uniform_int_distribution<int>(
          0, nlev))(engine);
      /* kmass needs to be in [1, nlev] or be -1 for the
       * Fortran implementation */
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
        REQUIRE(check_answer(p_theory[j], p_exper[j]) ==
                0.0);
      }
    }
  }
  delete[] p_exper;
  delete[] p_theory;
}

TEST_CASE("recover_dpq", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;

  // real elem_state_p (np,np,nlevel,timelevels,nelemd)
  constexpr const int p_len =
      numelems * timelevels * nlev * np * np;
  real *p_theory = new real[p_len];
  real *p_exper = new real[p_len];

  constexpr const int numRandTests = 10;

  SECTION("random test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; ++i) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);
      const int n0 = (std::uniform_int_distribution<int>(
          1, timelevels))(engine);
      int kmass = (std::uniform_int_distribution<int>(
          0, nlev))(engine);
      /* kmass needs to be in [1, nlev] or be -1 for the
       * Fortran implementation */
      if(kmass == 0) {
        kmass = -1;
      }

      std::uniform_real_distribution<real> p_dist(0, 1.0);
      for(int j = 0; j < p_len; j++) {
        p_theory[j] = p_dist(engine);
        p_exper[j] = p_theory[j];
      }
      recover_dpq_f90(nets, nete, kmass, n0, numelems,
                      p_theory);
      recover_dpq_c(nets, nete, kmass, n0, numelems,
                    p_exper);
      for(int j = 0; j < p_len; j++) {
        REQUIRE(check_answer(p_theory[j], p_exper[j]) ==
                0.0);
      }
    }
  }
  delete[] p_exper;
  delete[] p_theory;
}

TEST_CASE("contra2latlon", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;
  constexpr const int dim = 2;
  // real elem_D (np,np,2,2,nelemd)
  // real elem_state_v (np,np,2,nlev,timelevels,nelemd)
  constexpr const int D_len =
      np * np * dim * dim * numelems;
  constexpr const int v_len =
      np * np * dim * nlev * timelevels * numelems;
  real *D = new real[D_len];
  real *v_theory = new real[v_len];
  real *v_exper = new real[v_len];

  constexpr const int numRandTests = 10;

  SECTION("random test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; ++i) {
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
        REQUIRE(check_answer(v_theory[j], v_exper[j]) ==
                0.0);
      }
    }
  }
  delete[] v_exper;
  delete[] v_theory;
  delete[] D;
}

TEST_CASE("add_hv", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;
  constexpr const int dim = 2;

  constexpr const int sphere_mp_len = np * np * numelems;
  real *sphere_mp = new real[sphere_mp_len];

  constexpr const int numRandTests = 10;

  SECTION("random test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; ++i) {
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
          np * np * dim * nlev * (nete - nets + 1);
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

      std::uniform_real_distribution<real> sphere_mp_dist(
          0, 1.0);
      for(int j = 0; j < sphere_mp_len; j++) {
        sphere_mp[j] = sphere_mp_dist(engine);
      }

      add_hv_f90(nets, nete, numelems, sphere_mp,
                 ptens_theory, vtens_theory);
      add_hv_c(nets, nete, numelems, sphere_mp, ptens_exper,
               vtens_exper);
      for(int j = 0; j < ptens_len; j++) {
        REQUIRE(check_answer(ptens_theory[j],
                             ptens_exper[j]) == 0.0);
      }
      for(int j = 0; j < vtens_len; j++) {
        REQUIRE(check_answer(vtens_theory[j],
                             vtens_exper[j]) == 0.0);
      }

      delete[] ptens_theory;
      delete[] ptens_exper;
      delete[] vtens_theory;
      delete[] vtens_exper;
    }
  }
  delete[] sphere_mp;
}

TEST_CASE("weighted_rhs", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;
  constexpr const int dim = 2;

  constexpr const int rsphere_mp_len = np * np * numelems;
  real *rsphere_mp = new real[rsphere_mp_len];
  constexpr const int dinv_len =
      np * np * dim * dim * numelems;
  real *dinv = new real[dinv_len];

  constexpr const int numRandTests = 10;

  SECTION("random_test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; ++i) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);
      const int ptens_len =
          np * np * nlev * (nete - nets + 1);
      real *ptens_theory = new real[ptens_len];
      real *ptens_exper = new real[ptens_len];
      const int vtens_len =
          np * np * dim * nlev * (nete - nets + 1);
      real *vtens_theory = new real[vtens_len];
      real *vtens_exper = new real[vtens_len];

      genRandArray(
          rsphere_mp, rsphere_mp_len, engine,
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
      weighted_rhs_f90(nets, nete, numelems, rsphere_mp,
                       dinv, ptens_theory, vtens_theory);
      weighted_rhs_c(nets, nete, numelems, rsphere_mp, dinv,
                     ptens_exper, vtens_exper);
      for(int j = 0; j < ptens_len; j++) {
        REQUIRE(check_answer(ptens_theory[j],
                             ptens_exper[j]) == 0.0);
      }
      for(int j = 0; j < vtens_len; j++) {
        REQUIRE(check_answer(vtens_theory[j],
                             vtens_exper[j]) == 0.0);
      }

      delete[] ptens_theory;
      delete[] ptens_exper;
      delete[] vtens_theory;
      delete[] vtens_exper;
    }
  }

  delete[] rsphere_mp;
  delete[] dinv;
}

TEST_CASE("rk_stage", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;
  constexpr const int dim = 2;
  constexpr const int rkstages = 5;

  real *alpha0 = new real[rkstages];
  real *alpha = new real[rkstages];
  constexpr const int p_len =
      np * np * nlev * timelevels * numelems;
  real *p_theory = new real[p_len];
  real *p_exper = new real[p_len];
  constexpr const int v_len =
      np * np * dim * nlev * timelevels * numelems;
  real *v_theory = new real[v_len];
  real *v_exper = new real[v_len];

  constexpr const int numRandTests = 10;
  SECTION("random_test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; ++i) {
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);
      const int ptens_len =
          np * np * nlev * (nete - nets + 1);
      real *ptens = new real[ptens_len];
      const int vtens_len =
          np * np * dim * nlev * (nete - nets + 1);
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

      rk_stage_f90(nets, nete, n0, np1, s, rkstages,
                   numelems, v_theory, p_theory, alpha0,
                   alpha, ptens, vtens);
      rk_stage_c(nets, nete, n0, np1, s, rkstages, numelems,
                 v_exper, p_exper, alpha0, alpha, ptens,
                 vtens);

      for(int j = 0; j < p_len; j++) {
        REQUIRE(check_answer(p_theory[j], p_exper[j]) ==
                0.0);
      }
      for(int j = 0; j < v_len; j++) {
        REQUIRE(check_answer(v_theory[j], v_exper[j]) ==
                0.0);
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

/* TODO: Give this a better name */
TEST_CASE("loop7", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;
  constexpr const int dim = 2;

  constexpr const int dvv_len = np * np;
  real *dvv = new real[dvv_len];
  constexpr const int d_len =
      np * np * dim * dim * numelems;
  real *d = new real[d_len];
  real *dinv = new real[d_len];
  constexpr const int metdet_len = np * np * numelems;
  real *metdet = new real[metdet_len];
  real *rmetdet = new real[metdet_len];
  constexpr const int fcor_len = np * np * numelems;
  real *fcor = new real[fcor_len];
  constexpr const int p_len =
      np * np * nlev * timelevels * numelems;
  real *p = new real[p_len];
  constexpr const int ps_len = np * np * numelems;
  real *ps = new real[ps_len];
  constexpr const int v_len =
      np * np * dim * nlev * timelevels * numelems;
  real *v = new real[v_len];

  constexpr const int numRandTests = 10;
  SECTION("random_test") {
    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());
    for(int i = 0; i < numRandTests; ++i) {
      const real pmean =
          (std::uniform_real_distribution<real>(0, 1.0))(
              engine);
      const real dtstage =
          (std::uniform_real_distribution<real>(0, 1.0))(
              engine);
      const int n0 = (std::uniform_int_distribution<int>(
          1, timelevels))(engine);
      const int tadv = (std::uniform_int_distribution<int>(
          0, 1))(engine);
      const int nets = (std::uniform_int_distribution<int>(
          1, numelems - 1))(engine);
      const int nete = (std::uniform_int_distribution<int>(
          nets + 1, numelems))(engine);
      const int ptens_len =
          np * np * nlev * (nete - nets + 1);
      real *ptens_theory = new real[ptens_len];
      real *ptens_exper = new real[ptens_len];
      const int vtens_len =
          np * np * dim * nlev * (nete - nets + 1);
      real *vtens_theory = new real[vtens_len];
      real *vtens_exper = new real[vtens_len];

      genRandArray(
          dvv, dvv_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandArray(
          d, d_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandArray(
          dinv, d_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandArray(
          metdet, metdet_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      for(int j = 0; j < metdet_len; j++) {
        rmetdet[j] = 1.0 / metdet[j];
      }
      genRandArray(
          fcor, fcor_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandArray(
          p, p_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandArray(
          ps, ps_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandArray(
          v, v_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandTheoryExper(
          ptens_theory, ptens_exper, ptens_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      genRandTheoryExper(
          vtens_theory, vtens_exper, vtens_len, engine,
          std::uniform_real_distribution<real>(0, 1.0));

      loop7_f90(nets, nete, n0, numelems, tadv, pmean,
                dtstage, dvv, d, dinv, metdet, rmetdet,
                fcor, p, ps, v, ptens_theory, vtens_theory);
      loop7_c(nets, nete, n0, numelems, tadv, pmean,
              dtstage, dvv, d, dinv, metdet, rmetdet, fcor,
              p, ps, v, ptens_exper, vtens_exper);

      for(int j = 0; j < ptens_len; j++) {
        REQUIRE(check_answer(ptens_theory[j],
                             ptens_exper[j]) == 0.0);
      }
      for(int j = 0; j < vtens_len; j++) {
        REQUIRE(check_answer(vtens_theory[j],
                             vtens_exper[j]) == 0.0);
      }

      delete[] ptens_theory;
      delete[] ptens_exper;
      delete[] vtens_theory;
      delete[] vtens_exper;
    }
  }

  delete[] dvv;
  delete[] d;
  delete[] dinv;
  delete[] metdet;
  delete[] rmetdet;
  delete[] fcor;
  delete[] p;
  delete[] ps;
  delete[] v;
}

TEST_CASE("gradient_sphere_input",
          "advance_nonstag_rk_cxx") {
  constexpr const int dim = 2;

  constexpr const char *testinput =
      np == 4
          ? "gradient_sphere_np4.in"
          : np == 8 ? "gradient_sphere_np8.in" : nullptr;
  SECTION(testinput) {
    int grad_np;
    std::map<std::string, int *> intparams;
    intparams.insert({std::string("np"), &grad_np});
    std::ifstream input(testinput);
    REQUIRE(input);
    input_reader(intparams, input);
    REQUIRE(grad_np == np);

    input.clear();
    input.seekg(std::ifstream::beg);

    HommeHostView2D<MemoryManaged> s_host("Scalars", np,
                                          np);
    std::map<std::string, real *> data;
    data.insert({std::string("s"), s_host.ptr_on_device()});

    HommeHostView2D<MemoryManaged> dvv_host("dvv", np, np);
    data.insert({std::string("deriv_Dvv"),
                 dvv_host.ptr_on_device()});

    constexpr const int numelems = 1;
    HommeHostView5D<MemoryManaged> dinv_host(
        "dinv_host", np, np, dim, dim, numelems);
    data.insert({std::string("elem_Dinv"),
                 dinv_host.ptr_on_device()});

    HommeHostView3D<MemoryManaged> grad_theory(
        "Gradient Theory", np, np, dim);
    data.insert({std::string("Gradient Sphere result"),
                 grad_theory.ptr_on_device()});
    input_reader(data, input);

    HommeExecView2D s("Scalar Values", np, np);
    Kokkos::deep_copy(s, s_host);

    HommeExecView2D dvv("dvv", np, np);
    Kokkos::deep_copy(dvv, dvv_host);

    HommeExecView5D dinv("dinv", np, np, dim, dim,
                         numelems);
    Kokkos::deep_copy(dinv, dinv_host);

    HommeExecView3D grad_exper_device("Gradient Exper", np,
                                      np, dim);
    gradient_sphere_c(0, s, dvv, dinv, grad_exper_device);
    HommeHostView3D<MemoryManaged> grad_exper(
        "Gradient Exper", np, np, dim);
    Kokkos::deep_copy(grad_exper, grad_exper_device);

    for(int j = 0; j < dim; j++) {
      for(int k = 0; k < np; k++) {
        for(int l = 0; l < np; l++) {
          REQUIRE(check_answer(grad_theory(l, k, j),
                               grad_exper(l, k, j)) == 0.0);
        }
      }
    }
  }
}

TEST_CASE("gradient_sphere_random",
          "advance_nonstag_rk_cxx") {
  SECTION("random test") {
    constexpr const int numelems = 10;
    constexpr const int dim = 2;

    constexpr const int numRandTests = 10;

    std::random_device rd;
    using rngAlg = std::mt19937_64;
    rngAlg engine(rd());

    HommeHostView2D<MemoryManaged> s_fortran("", np, np);
    HommeExecView2D s_kokkos("", np, np);

    HommeHostView2D<MemoryManaged> dvv_fortran("", np, np);
    HommeExecView2D dvv_kokkos("", np, np);

    HommeHostView5D<MemoryManaged> dinv_fortran(
        "", np, np, dim, dim, numelems);
    HommeExecView5D dinv_kokkos("", np, np, dim, dim,
                                numelems);

    HommeHostView3D<MemoryManaged> grad_theory("", np, np,
                                               dim);
    HommeExecView3D grad_exper("", np, np, dim);
    HommeHostView3D<MemoryManaged> grad_exper_host("", np,
                                                   np, dim);

    for(int i = 0; i < numRandTests; ++i) {
      genRandArray(
          s_fortran.ptr_on_device(), np * np, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      Kokkos::deep_copy(s_kokkos, s_fortran);

      genRandArray(
          dvv_fortran.ptr_on_device(), np * np, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      Kokkos::deep_copy(dvv_kokkos, dvv_fortran);

      genRandArray(
          dinv_fortran.ptr_on_device(),
          np * np * dim * numelems, engine,
          std::uniform_real_distribution<real>(0, 1.0));
      Kokkos::deep_copy(dinv_kokkos, dinv_fortran);

      for(int ie = 0; ie < numelems; ie++) {
        real *offset_dinv = &dinv_fortran(0, 0, 0, 0, ie);
        gradient_sphere_c_callable(
            s_fortran.ptr_on_device(),
            dvv_fortran.ptr_on_device(),
            &dinv_fortran(0, 0, 0, 0, ie),
            grad_theory.ptr_on_device());
        gradient_sphere_c(ie, s_kokkos, dvv_kokkos,
                          dinv_kokkos, grad_exper);
        Kokkos::deep_copy(grad_exper_host, grad_exper);
        for(int k = 0; k < dim; k++) {
          for(int j = 0; j < np; j++) {
            for(int i = 0; i < np; ++i) {
              REQUIRE(check_answer(
                          grad_theory(i, j, k),
                          grad_exper_host(i, j, k)) == 0.0);
            }
          }
        }
      }
    }
  }
}

TEST_CASE("vorticity_sphere_input",
          "advance_nonstag_rk_cxx") {
  constexpr const int dim = 2;

  constexpr const char *testinput =
      np == 4
          ? "vorticity_sphere_np4.in"
          : np == 8 ? "vorticity_sphere_np8.in" : nullptr;
  SECTION(testinput) {
    int vort_np;
    std::map<std::string, int *> intparams;
    intparams.insert({std::string("np"), &vort_np});
    std::ifstream input(testinput);
    REQUIRE(input);
    input_reader(intparams, input);

    REQUIRE(vort_np == np);

    input.clear();
    input.seekg(std::ifstream::beg);

    HommeHostView3D<MemoryManaged> v_host("Velocity", np,
                                          np, dim);
    std::map<std::string, real *> data;
    data.insert({std::string("v"), v_host.ptr_on_device()});

    HommeHostView2D<MemoryManaged> dvv_host("dvv", np, np);
    data.insert({std::string("deriv_Dvv"),
                 dvv_host.ptr_on_device()});

    constexpr const int num_elems = 1;
    HommeHostView5D<MemoryManaged> d_host("d", np, np, dim,
                                          dim, num_elems);
    data.insert(
        {std::string("elem_D"), d_host.ptr_on_device()});

    HommeHostView3D<MemoryManaged> rmetdet_host(
        "rmetdet", np, np, num_elems);
    data.insert({std::string("elem_rmetdet"),
                 rmetdet_host.ptr_on_device()});

    HommeHostView2D<MemoryManaged> vort_theory(
        "Vorticity Theory", np, np);
    data.insert({std::string("Vorticity Sphere result"),
                 vort_theory.ptr_on_device()});
    input_reader(data, input);

    HommeExecView3D v("Velocity", np, np, dim);
    Kokkos::deep_copy(v, v_host);

    HommeExecView2D dvv("dvv", np, np);
    Kokkos::deep_copy(dvv, dvv_host);

    HommeExecView5D d("d", np, np, dim, dim, num_elems);
    Kokkos::deep_copy(d, d_host);

    HommeExecView3D rmetdet("rmetdet", np, np, num_elems);
    Kokkos::deep_copy(rmetdet, rmetdet_host);

    HommeExecView2D vort_exper_device("Vorticity Exper", np,
                                      np);
    vorticity_sphere_c(0, v, dvv, d, rmetdet,
                       vort_exper_device);
    HommeHostView2D<MemoryManaged> vort_exper(
        "Vorticity Exper", np, np);
    Kokkos::deep_copy(vort_exper, vort_exper_device);

    for(int k = 0; k < np; k++) {
      for(int l = 0; l < np; l++) {
        REQUIRE(check_answer(vort_theory(l, k),
                             vort_exper(l, k)) == 0.0);
      }
    }
  }
}

TEST_CASE("vorticity_sphere_random",
          "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;
  constexpr const int dim = 2;

  constexpr const int numRandTests = 10;

  constexpr real machine_precision = std::numeric_limits<real>::epsilon();

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());

  HommeHostView3D<MemoryManaged> v_fortran("", np, np, dim);
  HommeExecView3D v_kokkos("", np, np, dim);

  HommeHostView2D<MemoryManaged> dvv_fortran("", np, np);
  HommeExecView2D dvv_kokkos("", np, np);

  HommeHostView5D<MemoryManaged> d_fortran("", np, np, dim,
                                           dim, numelems);
  HommeExecView5D d_kokkos("", np, np, dim, dim, numelems);

  HommeHostView3D<MemoryManaged> rmetdet_fortran("", np, np,
                                                 numelems);
  HommeExecView3D rmetdet_kokkos("", np, np, numelems);

  HommeHostView2D<MemoryManaged> vort_theory("", np, np);
  HommeExecView2D vort_exper("", np, np);
  HommeHostView2D<MemoryManaged> vort_exper_host("", np,
                                                 np);

  for(int i = 0; i < numRandTests; ++i) {
    genRandArray(
        v_fortran.ptr_on_device(), np * np * dim, engine,
        std::uniform_real_distribution<real>(0, 1.0));
    Kokkos::deep_copy(v_kokkos, v_fortran);

    genRandArray(
        dvv_fortran.ptr_on_device(), np * np, engine,
        std::uniform_real_distribution<real>(0, 1.0));
    Kokkos::deep_copy(dvv_kokkos, dvv_fortran);

    genRandArray(
        d_fortran.ptr_on_device(),
        np * np * dim * dim * numelems, engine,
        std::uniform_real_distribution<real>(0, 1.0));
    Kokkos::deep_copy(d_kokkos, d_fortran);

    genRandArray(
        rmetdet_fortran.ptr_on_device(), np * np * numelems,
        engine,
        std::uniform_real_distribution<real>(0, 1.0));
    Kokkos::deep_copy(rmetdet_kokkos, rmetdet_fortran);

    for(int ie = 0; ie < numelems; ie++) {
      vorticity_sphere_c_callable(
          v_fortran.ptr_on_device(),
          dvv_fortran.ptr_on_device(),
          &rmetdet_fortran(0, 0, ie),
          &d_fortran(0, 0, 0, 0, ie),
          vort_theory.ptr_on_device());
      vorticity_sphere_c(ie, v_kokkos, dvv_kokkos, d_kokkos,
                         rmetdet_kokkos, vort_exper);
      Kokkos::deep_copy(vort_exper_host, vort_exper);
      for(int j = 0; j < np; j++) {
        for(int i = 0; i < np; ++i) {
          REQUIRE(check_answer(vort_theory(i, j),
                               vort_exper_host(i, j)) <=
                  machine_precision);
        }
      }
    }
  }
}

TEST_CASE("divergence_sphere_input",
          "advance_nonstag_rk_cxx") {
  constexpr const int dim = 2;

  constexpr const char *testinput =
      np == 4
          ? "divergence_sphere_np4.in"
          : np == 8 ? "divergence_sphere_np8.in" : nullptr;
  SECTION(testinput) {
    int div_np;
    std::map<std::string, int *> intparams;
    intparams.insert({std::string("np"), &div_np});
    std::ifstream input(testinput);
    REQUIRE(input);
    input_reader(intparams, input);
    REQUIRE(np == np);

    input.clear();
    input.seekg(std::ifstream::beg);

    HommeHostView3D<MemoryManaged> v_host("Velocity", np,
                                          np, dim);
    std::map<std::string, real *> data;
    data.insert({std::string("v"), v_host.ptr_on_device()});

    HommeHostView2D<MemoryManaged> dvv_host("dvv", np, np);
    data.insert({std::string("deriv_Dvv"),
                 dvv_host.ptr_on_device()});

    constexpr const int num_elems = 1;
    HommeHostView5D<MemoryManaged> dinv_host(
        "dinv_host", np, np, dim, dim, num_elems);
    data.insert({std::string("elem_Dinv"),
                 dinv_host.ptr_on_device()});

    HommeHostView3D<MemoryManaged> rmetdet_host(
        "rmetdet", np, np, num_elems);
    data.insert({std::string("elem_rmetdet"),
                 rmetdet_host.ptr_on_device()});

    HommeHostView3D<MemoryManaged> metdet_host(
        "metdet", np, np, num_elems);
    data.insert({std::string("elem_metdet"),
                 metdet_host.ptr_on_device()});

    HommeHostView2D<MemoryManaged> div_theory(
        "Divergence Theory", np, np);
    data.insert({std::string("Divergence Sphere result"),
                 div_theory.ptr_on_device()});
    input_reader(data, input);

    HommeExecView3D v("Velocity", np, np, dim);
    Kokkos::deep_copy(v, v_host);
    HommeExecView2D dvv("dvv", np, np);
    Kokkos::deep_copy(dvv, dvv_host);
    HommeExecView5D dinv("dinv", np, np, dim, dim,
                         num_elems);
    Kokkos::deep_copy(dinv, dinv_host);
    HommeExecView3D metdet("metdet", np, np, num_elems);
    Kokkos::deep_copy(metdet, metdet_host);

    HommeExecView3D rmetdet("rmetdet", np, np, num_elems);
    Kokkos::deep_copy(rmetdet, rmetdet_host);

    HommeExecView2D div_exper("Divergence Exper", np, np);
    divergence_sphere_c(0, v, dvv, metdet, rmetdet, dinv,
                        div_exper);
    HommeHostView2D<MemoryManaged> div_exper_host(
        "Divergence Exper", np, np);
    Kokkos::deep_copy(div_exper_host, div_exper);

    for(int k = 0; k < np; k++) {
      for(int l = 0; l < np; l++) {
        REQUIRE(check_answer(div_theory(l, k),
                             div_exper_host(l, k)) == 0.0);
      }
    }
  }
}

TEST_CASE("divergence_sphere_random",
          "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;
  constexpr const int dim = 2;

  constexpr const int numRandTests = 10;

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());

  HommeHostView3D<MemoryManaged> v_fortran("", np, np, dim);
  HommeExecView3D v_kokkos("", np, np, dim);

  HommeHostView2D<MemoryManaged> dvv_fortran("", np, np);
  HommeExecView2D dvv_kokkos("", np, np);

  HommeHostView5D<MemoryManaged> dinv_fortran(
      "", np, np, dim, dim, numelems);
  HommeExecView5D dinv_kokkos("", np, np, dim, dim,
                              numelems);

  HommeHostView3D<MemoryManaged> metdet_fortran("", np, np,
                                                numelems);
  HommeExecView3D metdet_kokkos("", np, np, numelems);

  HommeHostView3D<MemoryManaged> rmetdet_fortran("", np, np,
                                                 numelems);
  HommeExecView3D rmetdet_kokkos("", np, np, numelems);

  HommeHostView2D<MemoryManaged> div_theory("", np, np);
  HommeExecView2D div_exper("", np, np);
  HommeHostView2D<MemoryManaged> div_exper_host("", np, np);

  for(int i = 0; i < numRandTests; ++i) {
    genRandArray(
        v_fortran.ptr_on_device(), np * np * dim, engine,
        std::uniform_real_distribution<real>(0, 1.0));
    Kokkos::deep_copy(v_kokkos, v_fortran);

    genRandArray(
        dvv_fortran.ptr_on_device(), np * np, engine,
        std::uniform_real_distribution<real>(0, 1.0));
    Kokkos::deep_copy(dvv_kokkos, dvv_fortran);

    genRandArray(
        dinv_fortran.ptr_on_device(),
        np * np * dim * dim * numelems, engine,
        std::uniform_real_distribution<real>(0, 1.0));
    Kokkos::deep_copy(dinv_kokkos, dinv_fortran);

    genRandArray(
        metdet_fortran.ptr_on_device(), np * np * numelems,
        engine,
        std::uniform_real_distribution<real>(0, 1.0));
    Kokkos::deep_copy(metdet_kokkos, metdet_fortran);

    for (int ip=0; ip<np; ++ip)
      for (int jp=0; jp<np; ++jp)
        for (int ie=0; ie<numelems; ++ie)
        {
          rmetdet_fortran(ip,jp,ie) = 1./metdet_fortran(ip,jp,ie);
        }

    Kokkos::deep_copy(rmetdet_kokkos, rmetdet_fortran);

    for(int ie = 0; ie < numelems; ie++) {
      real *offset_dinv = &dinv_fortran(0, 0, 0, 0, ie);
      divergence_sphere_c_callable(
          v_fortran.ptr_on_device(),
          dvv_fortran.ptr_on_device(),
          &metdet_fortran(0, 0, ie),
          &rmetdet_fortran(0, 0, ie),
          &dinv_fortran(0, 0, 0, 0, ie),
          div_theory.ptr_on_device());
      divergence_sphere_c(ie, v_kokkos, dvv_kokkos,
                          metdet_kokkos, rmetdet_kokkos,
                          dinv_kokkos, div_exper);
      Kokkos::deep_copy(div_exper_host, div_exper);
      for(int j = 0; j < np; j++) {
        for(int i = 0; i < np; ++i) {
          REQUIRE(check_answer(div_theory(i, j),
                               div_exper_host(i, j)) ==
                  0.0);
        }
      }
    }
  }
}
