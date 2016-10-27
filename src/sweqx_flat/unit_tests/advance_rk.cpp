
#include <catch/catch.hpp>

#include <iostream>
#include <random>

#include <dimensions.hpp>
#include <fortran_binding.hpp>
#include <kinds.hpp>

using namespace Homme;

extern "C" {

void recover_q_f90(const int &nets, const int &nete,
                   const int &kmass, const int &n0,
                   const int &nelems, real *const &p);

void recover_q_c(const int &nets, const int &nete,
                 const int &kmass, const int &n0,
                 const int &nelems, real *const &p);

void loop3_f90(const int &nets, const int &nete,
               const int &kmass, const int &n0,
               const int &nelems, real *const &D,
               real *const &v);

void loop3_c(const int &nets, const int &nete,
             const int &kmass, const int &n0,
             const int &nelems, real *const &D,
             real *const &v);
}

extern int nelemd FORTRAN_VAR(dimensions_mod, nelemd);

TEST_CASE("recover_q", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 100;
  nelemd = numelems;

  std::uniform_real_distribution<real> p_dist(0, 1.0);
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
