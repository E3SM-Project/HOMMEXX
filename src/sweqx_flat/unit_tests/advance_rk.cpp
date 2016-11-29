
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

void loop5_f90(const int &nets, const int &nete,
               const int &nelems, real *const &spheremp,
               real *&ptens, real *&vtens);

void loop5_c(const int &nets, const int &nete,
             const int &nelems, real *const &spheremp,
             real *&ptens, real *&vtens);

void loop6_f90(const int &nets, const int &nete,
               const int &kmass, const int &n0,
               const int &nelems, real *const &p);

void loop6_c(const int &nets, const int &nete,
             const int &kmass, const int &n0,
             const int &nelems, real *const &p);

void loop8_f90(const int &nets, const int &nete,
               const int &numelems,
               real *const &rspheremp_ptr,
               real *const &dinv_ptr, real *&ptens_ptr,
               real *&vtens_ptr);

void loop8_c(const int &nets, const int &nete,
             const int &numelems,
             real *const &rspheremp_ptr,
             real *const &dinv_ptr, real *&ptens_ptr,
             real *&vtens_ptr);

void loop9_f90(const int &nets, const int &nete,
               const int &n0, const int &np1, const int &s,
               const int &rkstages, const int &numelems,
               real *&v_ptr, real *&p_ptr,
               real *const &alpha0_ptr,
               real *const &alpha_ptr,
               real *const &ptens_ptr,
               real *const &vtens_ptr);

void loop9_c(const int &nets, const int &nete,
             const int &n0, const int &np1, const int &s,
             const int &rkstages, const int &numelems,
             real *&v_ptr, real *&p_ptr,
             real *const &alpha0_ptr,
             real *const &alpha_ptr, real *const &ptens_ptr,
             real *const &vtens_ptr);
}

namespace Homme {
template <typename ScalarQP>
void gradient_sphere_c(int ie, const ScalarQP &s,
                       const Dvv &dvv, const D &dinv,
                       VectorField &grad);

void vorticity_sphere_c(int ie, const VectorField &v,
                        const Dvv &dvv, const D &d,
                        const MetDet &rmetdet,
                        ScalarField &grad);

void divergence_sphere_c(int ie, const VectorField &v,
                         const Dvv &dvv,
                         const MetDet &metdet,
                         const MetDet &rmetdet,
                         const D &dinv,
                         ScalarField &divergence);

void team_parallel_ex(
    const int &nets, const int &nete, const int &n0,
    const int &nelemd,
    const int &tracer_advection_formulation,
    const real &pmean, const real &dtstage, real *&dvv_ptr,
    real *&d_ptr, real *&dinv_ptr, real *&metdet_ptr,
    real *&rmetdet_ptr, real *&fcor_ptr, real *&p_ptr,
    real *&ps_ptr, real *&v_ptr, real *&ptens_ptr,
    real *&vtens_ptr);
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
      loop6_f90(nets, nete, kmass, n0, numelems, p_theory);
      loop6_c(nets, nete, kmass, n0, numelems, p_exper);
      for(int j = 0; j < p_len; j++) {
        REQUIRE(p_exper[j] == p_theory[j]);
      }
    }
  }
  delete[] p_exper;
  delete[] p_theory;
}

TEST_CASE("loop6", "advance_nonstag_rk_cxx") {
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
      loop6_f90(nets, nete, kmass, n0, numelems, p_theory);
      loop6_c(nets, nete, kmass, n0, numelems, p_exper);
      for(int j = 0; j < p_len; j++) {
        REQUIRE(p_exper[j] == p_theory[j]);
      }
      loop6_f90(nets, nete, kmass, n0, numelems, p_theory);
      loop6_c(nets, nete, kmass, n0, numelems, p_exper);
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

TEST_CASE("loop5", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;
  constexpr const int dim = 2;

  constexpr const int spheremp_len = np * np * numelems;
  real *spheremp = new real[spheremp_len];

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

      std::uniform_real_distribution<real> spheremp_dist(
          0, 1.0);
      for(int j = 0; j < spheremp_len; j++) {
        spheremp[j] = spheremp_dist(engine);
      }

      loop5_f90(nets, nete, numelems, spheremp,
                ptens_theory, vtens_theory);
      loop5_c(nets, nete, numelems, spheremp, ptens_exper,
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

TEST_CASE("loop8", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 10;
  constexpr const int dim = 2;

  constexpr const int rspheremp_len = np * np * numelems;
  real *rspheremp = new real[rspheremp_len];
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
      loop8_f90(nets, nete, numelems, rspheremp, dinv,
                ptens_theory, vtens_theory);
      loop8_c(nets, nete, numelems, rspheremp, dinv,
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

TEST_CASE("loop9", "advance_nonstag_rk_cxx") {
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

      loop9_f90(nets, nete, n0, np1, s, rkstages, numelems,
                v_theory, p_theory, alpha0, alpha, ptens,
                vtens);
      loop9_c(nets, nete, n0, np1, s, rkstages, numelems,
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

TEST_CASE("vorticity_sphere", "advance_nonstag_rk_cxx") {
  constexpr const int dim = 2;

  constexpr const int numRandTests = 10;
#if NP == 4
  constexpr const char *testinput =
      "vorticity_sphere_np4.in";
#endif  // NP == 4
#if NP == 8
  constexpr const char *testinput =
      "vorticity_sphere_np8.in";
#endif  // NP == 8
  SECTION(testinput) {
    int vort_np;
    std::map<std::string, int *> intparams;
    intparams.insert({std::string("np"), &vort_np});
    std::ifstream input(testinput);
    REQUIRE(input);
    input_reader(intparams, input);

    input.clear();
    input.seekg(std::ifstream::beg);

    VectorField v("Velocity", vort_np, vort_np, dim);
    std::map<std::string, real *> data;
    data.insert({std::string("v"), v.ptr_on_device()});

    const int Dvv_len = vort_np * vort_np;
    Dvv dvv(new real[Dvv_len], vort_np, vort_np);
    data.insert(
        {std::string("deriv_Dvv"), dvv.ptr_on_device()});

    constexpr const int numelems = 1;
    const int D_len =
        vort_np * vort_np * dim * dim * numelems;
    D d(new real[D_len], vort_np, vort_np, dim, dim,
        numelems);
    data.insert({std::string("elem_D"), d.ptr_on_device()});

    const int rmetdet_len = vort_np * vort_np * numelems;
    MetDet rmetdet(new real[rmetdet_len], vort_np, vort_np,
                   numelems);
    data.insert({std::string("elem_rmetdet"),
                 rmetdet.ptr_on_device()});

    ScalarField vort_theory("Vorticity Theory", vort_np,
                            vort_np);
    data.insert({std::string("Vorticity Sphere result"),
                 vort_theory.ptr_on_device()});
    input_reader(data, input);

    ScalarField vort_exper("Vorticity Exper", vort_np,
                           vort_np);
    vorticity_sphere_c(0, v, dvv, d, rmetdet, vort_exper);

    for(int k = 0; k < vort_np; k++) {
      for(int l = 0; l < vort_np; l++) {
        REQUIRE(
            std::fabs(vort_exper(l, k) -
                      vort_theory(l, k)) <
            (4.0 * std::numeric_limits<real>::epsilon()));
      }
    }
    delete[] d.ptr_on_device();
    delete[] rmetdet.ptr_on_device();
  }
}
