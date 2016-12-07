
#include <catch/catch.hpp>

#include <cmath>
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

}  // extern "C"

namespace Homme {
template <typename Scalar_QP, typename Vector_QP>
void gradient_sphere_c(int ie, const Scalar_QP &s,
                       const Dvv &dvv, const D &dinv,
                       Vector_QP &grad);

template <typename Scalar_QP, typename Vector_QP>
void vorticity_sphere_c(int ie, const Vector_QP &v,
                        const Dvv &dvv, const D &d,
                        const MetDet &rmetdet,
                        Scalar_QP &grad);

template <typename Scalar_QP, typename Vector_QP>
void divergence_sphere_c(int ie, const Vector_QP &v,
                         const Dvv &dvv,
                         const MetDet &metdet,
                         const MetDet &rmetdet,
                         const D &dinv,
                         Scalar_QP &divergence);
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
        i++;
      }
    }
  }
}

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
    for(int i = 0; i < numRandTests; i++) {
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
        REQUIRE(p_exper[j] == p_theory[j]);
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
    for(int i = 0; i < numRandTests; i++) {
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
        REQUIRE(p_exper[j] == p_theory[j]);
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
    for(int i = 0; i < numRandTests; i++) {
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
    for(int i = 0; i < numRandTests; i++) {
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

TEST_CASE("gradient_sphere", "advance_nonstag_rk_cxx") {
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

    Scalar_Field_Host s_host("Scalars", np, np);
    std::map<std::string, real *> data;
    data.insert({std::string("s"), s_host.ptr_on_device()});

    Dvv_Host dvv_host("dvv", np, np);
    data.insert({std::string("deriv_Dvv"),
                 dvv_host.ptr_on_device()});

    constexpr const int numelems = 1;
    D_Host dinv_host("dinv_host", np, np, dim, dim,
                     numelems);
    data.insert({std::string("elem_Dinv"),
                 dinv_host.ptr_on_device()});

    Homme_View_Host<real ***> grad_theory("Gradient Theory",
                                          np, np, dim);
    data.insert({std::string("Gradient Sphere result"),
                 grad_theory.ptr_on_device()});
    input_reader(data, input);

    Scalar_Field s("Scalar Values", np, np);
    Kokkos::deep_copy(s, s_host);

    Dvv dvv("dvv", np, np);
    Kokkos::deep_copy(dvv, dvv_host);

    D dinv("dinv", np, np, dim, dim, numelems);
    Kokkos::deep_copy(dinv, dinv_host);

    Vector_Field grad_exper_device("Gradient Exper", np, np,
                                   dim);
    gradient_sphere_c(0, s, dvv, dinv, grad_exper_device);
    Vector_Field_Host grad_exper("Gradient Exper", np, np,
                                 dim);
    Kokkos::deep_copy(grad_exper, grad_exper_device);

    for(int j = 0; j < dim; j++) {
      for(int k = 0; k < np; k++) {
        for(int l = 0; l < np; l++) {
          if(grad_theory(l, k, j) == 0.0) {
            /* Check the absolute error instead of the
             * relative error
             */
            REQUIRE(std::fabs(grad_exper(l, k, j)) <
                    std::numeric_limits<real>::epsilon());
          } else {
            const real rel_err =
                std::fabs((grad_exper(l, k, j) -
                           grad_theory(l, k, j)));
            const real max_rel_err = std::fabs(
                4.0 * std::numeric_limits<real>::epsilon() *
                grad_theory(l, k, j));
            REQUIRE(rel_err < max_rel_err);
          }
        }
      }
    }
  }
}

TEST_CASE("vorticity_sphere", "advance_nonstag_rk_cxx") {
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

    Vector_Field_Host v_host("Velocity", np, np, dim);
    std::map<std::string, real *> data;
    data.insert({std::string("v"), v_host.ptr_on_device()});

    Dvv_Host dvv_host("dvv", np, np);
    data.insert({std::string("deriv_Dvv"),
                 dvv_host.ptr_on_device()});

    constexpr const int num_elems = 1;
    D_Host d_host("d", np, np, dim, dim, num_elems);
    data.insert(
        {std::string("elem_D"), d_host.ptr_on_device()});

    MetDet_Host rmetdet_host("rmetdet", np, np, num_elems);
    data.insert({std::string("elem_rmetdet"),
                 rmetdet_host.ptr_on_device()});

    Scalar_Field_Host vort_theory("Vorticity Theory", np,
                                  np);
    data.insert({std::string("Vorticity Sphere result"),
                 vort_theory.ptr_on_device()});
    input_reader(data, input);

    Vector_Field v("Velocity", np, np, dim);
    Kokkos::deep_copy(v, v_host);

    Dvv dvv("dvv", np, np);
    Kokkos::deep_copy(dvv, dvv_host);

    D d("d", np, np, dim, dim, num_elems);
    Kokkos::deep_copy(d, d_host);

    MetDet rmetdet("rmetdet", np, np, num_elems);
    Kokkos::deep_copy(rmetdet, rmetdet_host);

    Scalar_Field vort_exper_device("Vorticity Exper", np,
                                   np);
    vorticity_sphere_c(0, v, dvv, d, rmetdet,
                       vort_exper_device);
    Scalar_Field_Host vort_exper("Vorticity Exper", np, np);
    Kokkos::deep_copy(vort_exper, vort_exper_device);

    for(int k = 0; k < np; k++) {
      for(int l = 0; l < np; l++) {
        if(vort_theory(l, k) == 0.0) {
          /* Check the absolute error instead of the
           * relative error
           */
          REQUIRE(std::fabs(vort_exper(l, k)) <
                  std::numeric_limits<real>::epsilon());
        } else {
          REQUIRE(std::fabs((vort_exper(l, k) -
                             vort_theory(l, k))) <
                  std::fabs(
                      4.0 *
                      std::numeric_limits<real>::epsilon() *
                      vort_theory(l, k)));
        }
      }
    }
  }
}

TEST_CASE("divergence_sphere", "advance_nonstag_rk_cxx") {
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

    Vector_Field_Host v_host("Velocity", np, np, dim);
    std::map<std::string, real *> data;
    data.insert({std::string("v"), v_host.ptr_on_device()});

    Dvv_Host dvv_host("dvv", np, np);
    data.insert({std::string("deriv_Dvv"),
                 dvv_host.ptr_on_device()});

    constexpr const int num_elems = 1;
    D_Host dinv_host("dinv_host", np, np, dim, dim,
                     num_elems);
    data.insert({std::string("elem_Dinv"),
                 dinv_host.ptr_on_device()});

    MetDet_Host rmetdet_host("rmetdet", np, np, num_elems);
    data.insert({std::string("elem_rmetdet"),
                 rmetdet_host.ptr_on_device()});

    MetDet_Host metdet_host("metdet", np, np, num_elems);
    data.insert({std::string("elem_metdet"),
                 metdet_host.ptr_on_device()});

    Scalar_Field_Host div_theory("Divergence Theory", np,
                                 np);
    data.insert({std::string("Divergence Sphere result"),
                 div_theory.ptr_on_device()});
    input_reader(data, input);

    Vector_Field v("Velocity", np, np, dim);
    Kokkos::deep_copy(v, v_host);
    Dvv dvv("dvv", np, np);
    Kokkos::deep_copy(dvv, dvv_host);
    D dinv("dinv", np, np, dim, dim, num_elems);
    Kokkos::deep_copy(dinv, dinv_host);
    MetDet metdet("metdet", np, np, num_elems);
    Kokkos::deep_copy(metdet, metdet_host);

    MetDet rmetdet("rmetdet", np, np, num_elems);
    Kokkos::deep_copy(rmetdet, rmetdet_host);

    Scalar_Field div_exper("Divergence Exper", np, np);
    divergence_sphere_c(0, v, dvv, metdet, rmetdet, dinv,
                        div_exper);
    Homme_View_Host<real **> div_exper_host(
        "Divergence Exper", np, np);
    Kokkos::deep_copy(div_exper_host, div_exper);

    for(int k = 0; k < np; k++) {
      for(int l = 0; l < np; l++) {
        if(div_theory(l, k) == 0.0) {
          /* Check the absolute error instead of the
           * relative error
           */
          REQUIRE(std::fabs(div_exper_host(l, k)) <
                  std::numeric_limits<real>::epsilon());
        } else {
          REQUIRE(std::fabs(div_exper_host(l, k) -
                            div_theory(l, k)) <
                  std::fabs(
                      4.0 *
                      std::numeric_limits<real>::epsilon() *
                      div_theory(l, k)));
        }
      }
    }
  }
}
