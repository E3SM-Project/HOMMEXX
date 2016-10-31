
#include <catch/catch.hpp>

#include <iostream>
#include <random>
#include <cmath>

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
               const int &n0, const int &nelems,
               real *const &D, real *&v);

void loop3_c(const int &nets, const int &nete,
             const int &n0, const int &nelems,
             real *const &D, real *&v);

void loop5_f90(const int &nets, const int &nete,
               const int &nelems, real *const &spheremp,
							 real *&ptens, real *&vtens);

void loop5_c(const int &nets, const int &nete,
             const int &nelems, real *const &spheremp,
						 real *&ptens, real *&vtens);
}

TEST_CASE("recover_q", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 100;

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

TEST_CASE("loop3", "advance_nonstag_rk_cxx") {
  constexpr const int numelems = 100;
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
      loop3_f90(nets, nete, n0, numelems, D, v_theory);
      loop3_c(nets, nete, n0, numelems, D, v_exper);
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
  constexpr const int numelems = 100;
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

      std::uniform_real_distribution<real> ptens_dist(0, 1.0);
      for(int j = 0; j < ptens_len; j++) {
        ptens_theory[j] = ptens_dist(engine);
        ptens_exper[j] = ptens_theory[j];
      }
      std::uniform_real_distribution<real> vtens_dist(0, 1.0);
      for(int j = 0; j < vtens_len; j++) {
        vtens_theory[j] = vtens_dist(engine);
        vtens_exper[j] = vtens_theory[j];
      }

      std::uniform_real_distribution<real> spheremp_dist(0, 1.0);
      for(int j = 0; j < spheremp_len; j++) {
        spheremp[j] = spheremp_dist(engine);
      }

      loop5_f90(nets, nete, numelems, spheremp, ptens_theory, vtens_theory);
      loop5_c(nets, nete, numelems, spheremp, ptens_exper, vtens_exper);
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
