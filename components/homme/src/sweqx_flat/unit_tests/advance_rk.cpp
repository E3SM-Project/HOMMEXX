
#include <catch/catch.hpp>

#include <random>

#include <dimensions.hpp>
#include <fortran_binding.hpp>
#include <kinds.hpp>

using namespace Homme;

extern "C" {

void recover_q_f90(const int &nets, const int &nete,
                   const int &kmass, const int &nelems,
                   const int &n0, real *&p)
    FORTRAN_C(recover_q_f90);
void recover_q_c(const int &nets, const int &nete,
                 const int &kmass, const int &nelems,
                 const int &n0, real *&p);

void loop3_f90(const int &nets, const int &nete,
               const int &kmass, const int &nelems,
               const int &n0, real *const &D, real *&v)
    FORTRAN_C(loop3_f90);
void loop3_c(const int &nets, const int &nete,
             const int &kmass, const int &nelems,
             const int &n0, real *const &D, real *&v);
}

TEST_CASE("recover_q", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 100;

  std::uniform_real_distribution<real> p_dist(0, 1.0);
  constexpr const int p_len =
      numelems * timelevels * nlev * np * np;
  real *p_theory = new real[p_len];
  real *p_exper = new real[p_len];

  constexpr const int numRandTests = 10;

  SECTION("random tests") {
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
      if(kmass == 0) {
        kmass = -1;
      }

      for(int i = 0; i < p_len; i++) {
        p_theory[i] = p_dist(engine);
        p_exper[i] = p_theory[i];
      }
      recover_q_f90(nets, nete, kmass, numelems, n0,
                    p_theory);
      recover_q_c(nets, nete, kmass, numelems, n0,
                  p_theory);
      for(int i = 0; i < p_len; i++) {
        REQUIRE(p_exper[i] == p_theory[i]);
      }
    }
  }
  delete[] p_exper;
  delete[] p_theory;
}
